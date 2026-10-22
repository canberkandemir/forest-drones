"""Run bounded real-image optimizer steps for the stride-8 LoRA FCOS model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    DINOv3Stride8FCOSHead,
    annotations_to_ground_truth,
    build_fcos_targets,
    choose_device,
    decode_fcos_predictions,
    load_compatible_fcos_weights_into_stride8,
)
from forestpersons.dinov3_lora import (
    extract_lora_patch_features,
    load_lora_fcos_checkpoint,
    save_lora_fcos_checkpoint,
    trainable_parameter_summary,
)


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = logits.sigmoid()
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = 0.75 * targets + 0.25 * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(2.0) * bce).mean()


def fcos_loss(
    logits: torch.Tensor,
    raw_box: torch.Tensor,
    center_logits: torch.Tensor,
    classification: torch.Tensor,
    targets: torch.Tensor,
    centerness: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    classification_loss = sigmoid_focal_loss(logits.reshape(-1), classification.reshape(-1))
    if mask.any():
        predicted = F.softplus(raw_box[mask])
        box_loss = F.smooth_l1_loss(torch.log1p(predicted), torch.log1p(targets[mask]), beta=0.25)
        centerness_loss = F.binary_cross_entropy_with_logits(center_logits[mask], centerness[mask])
    else:
        box_loss = torch.zeros((), device=logits.device)
        centerness_loss = torch.zeros((), device=logits.device)
    total = classification_loss + 2.0 * box_loss + centerness_loss
    return total, {
        "classification": float(classification_loss.detach().cpu()),
        "box": float(box_loss.detach().cpu()),
        "centerness": float(centerness_loss.detach().cpu()),
    }


def summarize_gradients(parameters: list[nn.Parameter]) -> dict[str, Any]:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return {
        "parameters": len(parameters),
        "with_gradient": len(gradients),
        "nonzero_gradient": sum(bool((gradient.detach().abs() > 0).any().item()) for gradient in gradients),
        "all_finite": all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients),
        "max_abs": max((float(gradient.detach().abs().max().cpu()) for gradient in gradients), default=0.0),
    }


def mps_memory() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    processor, backbone, source_head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    head = DINOv3Stride8FCOSHead(input_dim=int(metadata["head_input_dim"])).to(device)
    reuse = load_compatible_fcos_weights_into_stride8(head, source_head.state_dict())
    backbone.train()
    head.train()

    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = (manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data)[: args.max_images]
    dataset = ForestPersonsDataset(args.root, split="train")
    by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [by_file[item["file_name"]] for item in items]
    lora_parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    adapter_parameters = list(head.stride8_adapter.parameters())
    reused_head_parameters = [
        parameter
        for name, parameter in head.named_parameters()
        if not name.startswith("stride8_adapter.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": 1e-4},
            {"params": list(head.parameters()), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    rows: list[dict[str, Any]] = []
    peak_driver = 0
    start = time.perf_counter()

    for sample in samples:
        optimizer.zero_grad(set_to_none=True)
        with Image.open(sample.image_path) as handle:
            image = handle.convert("RGB")
        ground_truth = annotations_to_ground_truth(sample.annotations)
        features, info = extract_lora_patch_features(
            image,
            processor=processor,
            model=backbone,
            device=device,
            short_side=args.short_side,
        )
        logits, raw_box, center_logits = head(features)
        classification, targets, centerness, mask, stats = build_fcos_targets(
            ground_truth,
            info,
            stride=8,
            grid_height=int(logits.shape[0]),
            grid_width=int(logits.shape[1]),
        )
        total_loss, parts = fcos_loss(
            logits,
            raw_box,
            center_logits,
            classification.to(device),
            targets.to(device),
            centerness.to(device),
            mask.to(device),
        )
        if not bool(torch.isfinite(total_loss).item()):
            raise RuntimeError(f"Non-finite loss for {sample.file_name}")
        total_loss.backward()
        gradients = {
            "lora": summarize_gradients(lora_parameters),
            "adapter": summarize_gradients(adapter_parameters),
            "reused_fcos": summarize_gradients(reused_head_parameters),
        }
        nn.utils.clip_grad_norm_([*lora_parameters, *head.parameters()], max_norm=5.0)
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        memory = mps_memory()
        if memory is not None:
            peak_driver = max(peak_driver, memory["driver_allocated"])
        predictions = decode_fcos_predictions(
            logits.detach().cpu(),
            raw_box.detach().cpu(),
            center_logits.detach().cpu(),
            info,
            conf_threshold=0.50,
            stride=8,
        )
        rows.append(
            {
                "file_name": sample.file_name,
                "loss": float(total_loss.detach().cpu()),
                "loss_parts": parts,
                "positive_cells": int(stats["positive_cells"]),
                "decoded_predictions_at_conf_050": len(predictions),
                "gradients": gradients,
                "mps_memory_bytes": memory,
            }
        )

    elapsed = time.perf_counter() - start
    checkpoint_path = args.output_dir / "dinov3_lora_stride8_fcos_smoke.pt"
    settings = {
        "purpose": "implementation smoke only; not a trained experiment",
        "source_checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "images": len(samples),
        "short_side": args.short_side,
        "seed": args.seed,
        "optimizer": "AdamW",
        "lora_lr": 1e-4,
        "head_lr": 3e-4,
        "weight_decay": 1e-4,
        "batch_size": 1,
    }
    save_lora_fcos_checkpoint(
        checkpoint_path,
        backbone=backbone,
        head=head,
        config=metadata["config"],
        target_names=metadata["target_names"],
        settings=settings,
        history=rows,
        head_kind="stride8_fcos",
        model_id=metadata["model_id"],
    )
    summary = {
        "settings": settings,
        "device_requested": args.device,
        "device_resolved": str(device),
        "weight_reuse": reuse,
        "new_parameters": sorted(name for name in head.state_dict() if name.startswith("stride8_adapter.")),
        "parameter_summary": trainable_parameter_summary(backbone, head),
        "component_trainable_parameters": {
            "lora": sum(parameter.numel() for parameter in lora_parameters),
            "stride8_adapter": sum(parameter.numel() for parameter in adapter_parameters),
            "reused_fcos": sum(parameter.numel() for parameter in reused_head_parameters),
        },
        "seconds_total": elapsed,
        "seconds_per_image": elapsed / max(len(samples), 1),
        "peak_observed_mps_driver_memory_bytes": peak_driver or None,
        "checkpoint": str(checkpoint_path),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
