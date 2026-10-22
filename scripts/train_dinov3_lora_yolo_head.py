"""Train DINOv3 LoRA with the YOLO-style DFL/IoU detection head.

This script is scoped to the YOLO-style head pilot. It loads an existing
DINOv3 ViT-S LoRA + FCOS checkpoint only to reuse the backbone LoRA weights,
then initializes the YOLO-style detection head from scratch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import annotations_to_ground_truth, choose_device
from forestpersons.dinov3_lora import (
    extract_lora_patch_features,
    load_lora_fcos_checkpoint,
    trainable_parameter_summary,
)
from forestpersons.dinov3_yolo import (
    DINOv3YOLOHead,
    build_task_aligned_yolo_targets,
    dfl_expectation,
    load_lora_yolo_checkpoint,
    save_lora_yolo_checkpoint,
    yolo_detection_loss,
)


class OnlineForestPersonsDataset(Dataset):
    def __init__(self, root: Path, split: str, manifest: Path, max_images: int | None = None) -> None:
        self.dataset = ForestPersonsDataset(root, split=split)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        items = manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data
        if max_images is not None:
            items = items[:max_images]
        by_file = {sample.file_name: sample for sample in self.dataset.samples}
        self.samples = [by_file[item["file_name"]] for item in items]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "file_name": sample.file_name,
            "image": Image.open(sample.image_path).convert("RGB"),
            "ground_truth": annotations_to_ground_truth(sample.annotations),
        }


def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("DINOv3 YOLO trainer currently expects batch_size=1.")
    return batch[0]


def mps_memory() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def gradient_summary(backbone: nn.Module, head: nn.Module) -> dict[str, Any]:
    def summarize(parameters: list[nn.Parameter]) -> dict[str, Any]:
        with_grad = [p for p in parameters if p.grad is not None]
        nonzero = [p for p in with_grad if bool((p.grad.detach().abs() > 0).any().item())]
        max_abs = max((float(p.grad.detach().abs().max().cpu()) for p in with_grad), default=0.0)
        return {"with_grad": len(with_grad), "nonzero_grad": len(nonzero), "max_abs_grad": max_abs}

    return {
        "lora": summarize([p for p in backbone.parameters() if p.requires_grad]),
        "head": summarize([p for p in head.parameters() if p.requires_grad]),
    }


def first_trainable_clone(module: nn.Module) -> torch.Tensor | None:
    for parameter in module.parameters():
        if parameter.requires_grad:
            return parameter.detach().cpu().clone()
    return None


def first_trainable_diff(module: nn.Module, reference: torch.Tensor | None) -> float | None:
    if reference is None:
        return None
    for parameter in module.parameters():
        if parameter.requires_grad:
            return float((parameter.detach().cpu() - reference).abs().max())
    return None


def json_safe_settings(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    settings = vars(args).copy()
    settings["device_resolved"] = str(device)
    for key, value in list(settings.items()):
        if isinstance(value, Path):
            settings[key] = str(value)
    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json"),
    )
    parser.add_argument(
        "--init-lora-fcos-checkpoint",
        type=Path,
        default=Path("outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt"),
    )
    parser.add_argument(
        "--init-lora-yolo-checkpoint",
        type=Path,
        help="Continue from an existing DINOv3 LoRA YOLO-style checkpoint instead of initializing from FCOS LoRA.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-loss-weight", type=float, default=7.5)
    parser.add_argument("--dfl-loss-weight", type=float, default=1.5)
    parser.add_argument("--assignment-topk", type=int, default=10)
    parser.add_argument("--assignment-alpha", type=float, default=1.0)
    parser.add_argument("--assignment-beta", type=float, default=6.0)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--reg-max", type=int, default=32)
    parser.add_argument("--include-quality", action="store_true")
    parser.add_argument("--quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--quality-negative-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    if device.type == "mps":
        torch.mps.empty_cache()

    if args.init_lora_yolo_checkpoint is not None:
        processor, backbone, source_head, metadata = load_lora_yolo_checkpoint(args.init_lora_yolo_checkpoint, device)
        input_dim = int(metadata["head_input_dim"])
        if args.include_quality and not source_head.include_quality:
            head = DINOv3YOLOHead(
                input_dim=input_dim,
                hidden_dim=int(source_head.stem.out_channels),
                reg_max=int(source_head.reg_max),
                include_quality=True,
            ).to(device)
            missing, unexpected = head.load_state_dict(source_head.state_dict(), strict=False)
            allowed_missing = {key for key in missing if key.startswith("quality.")}
            if set(missing) != allowed_missing or unexpected:
                raise RuntimeError(
                    "Unexpected YOLO quality-branch initialization mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            yolo_head_initialization = (
                "loaded class/regression/stem weights from existing YOLO checkpoint; "
                "quality branch freshly initialized"
            )
        elif args.include_quality == source_head.include_quality:
            head = source_head.to(device)
            yolo_head_initialization = "loaded from existing YOLO checkpoint"
        else:
            raise ValueError(
                "Cannot disable a quality branch from a quality-enabled checkpoint in this trainer. "
                "Use a class-only checkpoint or omit --include-quality."
            )
    else:
        processor, backbone, _source_head, metadata = load_lora_fcos_checkpoint(args.init_lora_fcos_checkpoint, device)
        if metadata.get("head_kind") != "fcos":
            raise ValueError("--init-lora-fcos-checkpoint must be a single-scale FCOS checkpoint")
        input_dim = int(metadata["head_input_dim"])
        head = DINOv3YOLOHead(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            reg_max=args.reg_max,
            include_quality=args.include_quality,
        ).to(device)
        yolo_head_initialization = "fresh random initialization; FCOS head weights intentionally not reused"
    backbone.train()
    head.train()

    lora_config = metadata["config"]
    target_names = metadata["target_names"]
    settings = json_safe_settings(args, device)
    settings["model_id"] = metadata["model_id"]
    settings["initialization"] = {
        "backbone_lora": (
            "loaded from existing YOLO checkpoint"
            if args.init_lora_yolo_checkpoint is not None
            else "loaded from existing ViT-S LoRA + FCOS train5000 checkpoint"
        ),
        "yolo_head": yolo_head_initialization,
        "quality_branch": "enabled" if args.include_quality else "disabled",
        "quality_loss_normalization": (
            "positive mean + quality_negative_weight * negative mean"
            if args.include_quality
            else "not used"
        ),
    }

    dataset = OnlineForestPersonsDataset(args.root, args.split, args.manifest, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_one)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": args.lr_lora, "name": "lora"},
            {"params": [p for p in head.parameters() if p.requires_grad], "lr": args.lr_head, "name": "head"},
        ],
        weight_decay=args.weight_decay,
    )

    initial_lora = first_trainable_clone(backbone)
    initial_head = first_trainable_clone(head)
    parameter_summary = trainable_parameter_summary(backbone, head)
    history: list[dict[str, Any]] = []
    global_step = 0
    peak_driver_memory = 0
    start_total = time.perf_counter()

    (args.output_dir / "settings.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "parameter_summary": parameter_summary,
                "lora_target_modules": target_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    optimizer.zero_grad(set_to_none=True)
    last_gradient_summary: dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        totals = {
            "loss": 0.0,
            "classification_loss": 0.0,
            "iou_loss": 0.0,
            "dfl_loss": 0.0,
            "quality_loss": 0.0,
            "quality_positive_loss": 0.0,
            "quality_negative_loss": 0.0,
            "quality_positive_count": 0.0,
            "quality_negative_count": 0.0,
            "quality_target_mean": 0.0,
            "quality_target_max": 0.0,
            "positive_cells": 0,
            "annotations_with_positive": 0,
            "center_fallbacks": 0,
            "multi_gt_overwrites": 0,
            "clipped_target_distances": 0,
            "images": 0,
        }
        start_epoch = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            features, info = extract_lora_patch_features(
                batch["image"],
                processor=processor,
                model=backbone,
                device=device,
                short_side=args.short_side,
            )
            outputs = head(features)
            with torch.no_grad():
                predicted_distances = dfl_expectation(outputs["distance_logits"].detach())
                targets, stats = build_task_aligned_yolo_targets(
                    batch["ground_truth"],
                    info,
                    class_logits=outputs["class_logits"].detach(),
                    predicted_distances=predicted_distances,
                    topk=args.assignment_topk,
                    alpha=args.assignment_alpha,
                    beta=args.assignment_beta,
                    reg_max=args.reg_max,
                )
            loss, loss_parts = yolo_detection_loss(
                outputs,
                targets,
                box_weight=args.box_loss_weight,
                dfl_weight=args.dfl_loss_weight,
                quality_weight=args.quality_loss_weight if args.include_quality else 0.0,
                quality_negative_weight=args.quality_negative_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_index} file={batch['file_name']}")
            (loss / args.gradient_accumulation).backward()

            totals["loss"] += float(loss.detach().cpu())
            totals["classification_loss"] += loss_parts["classification_loss"]
            totals["iou_loss"] += loss_parts["iou_loss"]
            totals["dfl_loss"] += loss_parts["dfl_loss"]
            totals["quality_loss"] += loss_parts["quality_loss"]
            totals["quality_positive_loss"] += loss_parts["quality_positive_loss"]
            totals["quality_negative_loss"] += loss_parts["quality_negative_loss"]
            totals["quality_positive_count"] += loss_parts["quality_positive_count"]
            totals["quality_negative_count"] += loss_parts["quality_negative_count"]
            totals["quality_target_mean"] += loss_parts["quality_target_mean"]
            totals["quality_target_max"] = max(totals["quality_target_max"], loss_parts["quality_target_max"])
            totals["positive_cells"] += stats.positive_cells
            totals["annotations_with_positive"] += stats.annotations_with_positive
            totals["center_fallbacks"] += stats.center_fallbacks
            totals["multi_gt_overwrites"] += stats.multi_gt_overwrites
            totals["clipped_target_distances"] += stats.clipped_target_distances
            totals["images"] += 1

            should_step = batch_index % args.gradient_accumulation == 0 or batch_index == len(loader)
            if should_step:
                last_gradient_summary = gradient_summary(backbone, head)
                nn.utils.clip_grad_norm_(
                    [p for p in [*backbone.parameters(), *head.parameters()] if p.requires_grad],
                    max_norm=5.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                synchronize_device(device)
                if device.type == "mps":
                    peak_driver_memory = max(peak_driver_memory, int(torch.mps.driver_allocated_memory()))

        images = max(1, totals["images"])
        row = {
            "epoch": epoch,
            "images": totals["images"],
            "optimizer_steps_total": global_step,
            "loss": totals["loss"] / images,
            "classification_loss": totals["classification_loss"] / images,
            "iou_loss": totals["iou_loss"] / images,
            "dfl_loss": totals["dfl_loss"] / images,
            "quality_loss": totals["quality_loss"] / images,
            "quality_positive_loss": totals["quality_positive_loss"] / images,
            "quality_negative_loss": totals["quality_negative_loss"] / images,
            "quality_positive_count": totals["quality_positive_count"],
            "quality_negative_count": totals["quality_negative_count"],
            "quality_positive_count_per_image": totals["quality_positive_count"] / images,
            "quality_negative_count_per_image": totals["quality_negative_count"] / images,
            "quality_target_mean": totals["quality_target_mean"] / images,
            "quality_target_max": totals["quality_target_max"],
            "positive_cells": totals["positive_cells"],
            "positive_cells_per_image": totals["positive_cells"] / images,
            "annotations_with_positive": totals["annotations_with_positive"],
            "center_fallbacks": totals["center_fallbacks"],
            "multi_gt_overwrites": totals["multi_gt_overwrites"],
            "clipped_target_distances": totals["clipped_target_distances"],
            "seconds": time.perf_counter() - start_epoch,
            "gradient_summary": last_gradient_summary,
            "mps_memory_bytes": mps_memory(),
        }
        history.append(row)
        print(json.dumps(row))
        (args.output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_lora_yolo_checkpoint(
            args.output_dir / "latest.pt",
            backbone=backbone,
            head=head,
            settings=settings,
            history=history,
            model_id=metadata["model_id"],
            lora_config=lora_config,
            target_names=target_names,
        )
        if args.save_every_epoch:
            save_lora_yolo_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt",
                backbone=backbone,
                head=head,
                settings=settings,
                history=history,
                model_id=metadata["model_id"],
                lora_config=lora_config,
                target_names=target_names,
            )

    summary = {
        "settings": settings,
        "head": "yolo_dfl_iou_task_aligned",
        "head_input_dim": input_dim,
        "lora_target_modules": target_names,
        "parameter_summary": trainable_parameter_summary(backbone, head),
        "parameter_update_check": {
            "first_lora_max_abs_diff": first_trainable_diff(backbone, initial_lora),
            "first_head_max_abs_diff": first_trainable_diff(head, initial_head),
        },
        "history": history,
        "seconds_total": time.perf_counter() - start_total,
        "peak_mps_driver_memory_bytes": peak_driver_memory or None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_lora_yolo_checkpoint(
        args.output_dir / "dinov3_lora_yolo.pt",
        backbone=backbone,
        head=head,
        settings=settings,
        history=history,
        model_id=metadata["model_id"],
        lora_config=lora_config,
        target_names=target_names,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved DINOv3 LoRA YOLO checkpoint: {args.output_dir / 'dinov3_lora_yolo.pt'}")


if __name__ == "__main__":
    main()
