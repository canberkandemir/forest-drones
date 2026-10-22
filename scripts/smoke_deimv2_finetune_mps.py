"""Minimal DEIMv2-S one-class gradient smoke on ForestPersons train400."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.deimv2_adapter import (
    build_deimv2_preprocess,
    load_official_deimv2_one_class_for_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=20261015)
    return parser.parse_args()


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["images"] if isinstance(payload, dict) and "images" in payload else payload


def collect_two_samples(root: Path, manifest: Path) -> list[Any]:
    dataset = ForestPersonsDataset(root, split="train")
    by_id = {sample.image_id: sample for sample in dataset.samples}
    selected = []
    for item in load_items(manifest):
        sample = by_id[item["image_id"]]
        if sample.file_name != item["file_name"]:
            raise ValueError(f"image_id {sample.image_id} filename mismatch")
        if sample.annotations and sample.image_path.is_file():
            selected.append(sample)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError("Need two readable train400 images with annotations")
    return selected


def input_and_target(sample: Any, device: torch.device, input_size: int) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    image = Image.open(sample.image_path).convert("RGB")
    image_tensor = build_deimv2_preprocess(input_size)(image).unsqueeze(0).to(device)
    boxes = []
    for annotation in sample.annotations:
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        boxes.append(
            [
                (x + width / 2.0) / image.width,
                (y + height / 2.0) / image.height,
                width / image.width,
                height / image.height,
            ]
        )
    target = {
        "labels": torch.zeros(len(boxes), dtype=torch.int64, device=device),
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        "image_id": torch.tensor([sample.image_id], dtype=torch.int64, device=device),
        "orig_size": torch.tensor([image.width, image.height], dtype=torch.int64, device=device),
    }
    return image_tensor, [target]


def make_criterion(official_repo: Path, device: torch.device) -> torch.nn.Module:
    import sys

    repo = str(official_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from engine.deim import DEIMCriterion, HungarianMatcher

    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
        change_matcher=True,
        iou_order_alpha=4.0,
        matcher_change_epoch=100,
    )
    return DEIMCriterion(
        matcher=matcher,
        weight_dict={
            "loss_mal": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        losses=["mal", "boxes", "local"],
        alpha=0.75,
        gamma=1.5,
        num_classes=1,
        reg_max=32,
    ).to(device)


def parameter_groups(model: torch.nn.Module) -> list[dict[str, Any]]:
    backbone_transformer = []
    no_decay = []
    default = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone.dinov3" in name:
            backbone_transformer.append(parameter)
        elif any(token in name.lower() for token in ("norm", "bn", "bias")):
            no_decay.append(parameter)
        else:
            default.append(parameter)
    groups = []
    if backbone_transformer:
        groups.append({"params": backbone_transformer, "lr": 2.5e-5, "weight_decay": 1e-4})
    if no_decay:
        groups.append({"params": no_decay, "lr": 5e-4, "weight_decay": 0.0})
    if default:
        groups.append({"params": default, "lr": 5e-4, "weight_decay": 1e-4})
    return groups


def gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    groups: dict[str, list[float]] = {"backbone_dinov3": [], "spatial_adapter": [], "encoder": [], "decoder": []}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("backbone.dinov3"):
            key = "backbone_dinov3"
        elif name.startswith("backbone"):
            key = "spatial_adapter"
        elif name.startswith("encoder"):
            key = "encoder"
        else:
            key = "decoder"
        groups[key].append(float(parameter.grad.detach().float().norm().cpu()))
    return {
        key: {
            "tensor_count": len(values),
            "finite": all(math.isfinite(value) for value in values),
            "l2_sum": float(sum(values)),
            "maximum": float(max(values, default=0.0)),
        }
        for key, values in groups.items()
    }


def run_arm(
    mode: str,
    sample: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    arm_seed = args.seed + (0 if mode == "frozen_transformer" else 1)
    torch.manual_seed(arm_seed)
    if device.type == "mps":
        torch.mps.empty_cache()
    model, load_metadata = load_official_deimv2_one_class_for_training(
        args.official_repo, args.checkpoint_dir, device
    )
    if mode == "frozen_transformer":
        model.backbone.dinov3.requires_grad_(False)
    elif mode != "full_model":
        raise ValueError(mode)
    model.train()
    criterion = make_criterion(args.official_repo, device)
    optimizer = torch.optim.AdamW(parameter_groups(model), betas=(0.9, 0.999), eps=1e-8)
    images, targets = input_and_target(sample, device, args.input_size)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    if device.type == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    outputs = model.forward_raw(images, targets=targets)
    if device.type == "mps":
        torch.mps.synchronize()
    forward_seconds = time.perf_counter() - started
    loss_started = time.perf_counter()
    loss_dict = criterion(outputs, targets, epoch=0, step=0, global_step=0, epoch_step=1)
    loss = sum(loss_dict.values())
    if device.type == "mps":
        torch.mps.synchronize()
    loss_seconds = time.perf_counter() - loss_started
    backward_started = time.perf_counter()
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()
    backward_seconds = time.perf_counter() - backward_started
    gradients = gradient_summary(model)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 0.1
    )
    optimizer.step()
    changed = sum(
        not torch.equal(before[name].cpu(), parameter.detach().cpu())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    losses = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
    finite = math.isfinite(float(loss.detach().cpu())) and all(math.isfinite(value) for value in losses.values())
    result = {
        "mode": mode,
        "seed": arm_seed,
        "image_id": sample.image_id,
        "file_name": sample.file_name,
        "ground_truth_count": len(targets[0]["labels"]),
        "input_shape": list(images.shape),
        "parameter_count": load_metadata["parameter_count"],
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "copied_person_row_count": load_metadata["copied_person_row_count"],
        "mps_integral_workaround": load_metadata["mps_integral_workaround"],
        "loss_total": float(loss.detach().cpu()),
        "losses": losses,
        "finite": finite,
        "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
        "gradient_groups": gradients,
        "changed_parameter_tensors": changed,
        "forward_seconds": forward_seconds,
        "loss_seconds": loss_seconds,
        "backward_seconds": backward_seconds,
        "step_seconds": time.perf_counter() - started,
        "mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()) if device.type == "mps" else 0,
        "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory()) if device.type == "mps" else 0,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    del model, criterion, optimizer, outputs, loss, loss_dict, images, targets, before
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    samples = collect_two_samples(args.root, args.manifest)
    arms = [
        run_arm("frozen_transformer", samples[0], args, device),
        run_arm("full_model", samples[1], args, device),
    ]
    result = {
        "label": "gradient smoke only; not a training experiment",
        "device": str(device),
        "input_size": args.input_size,
        "precision": "float32; no automatic mixed precision",
        "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "seed": args.seed,
        "manifest": str(args.manifest),
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
