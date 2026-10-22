"""Smoke-test DINOv3 LoRA with a YOLO-style DFL/IoU detection head.

This is a bounded implementation check, not a training experiment. It loads the
best existing ViT-S LoRA + FCOS checkpoint to reuse the DINOv3 backbone and LoRA
weights, initializes a new YOLO-style head from scratch, and runs a few
forward/backward optimizer steps on real ForestPersons train images.
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
    decode_yolo_predictions,
    dfl_expectation,
    yolo_detection_loss,
)


class SmokeDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, max_images: int) -> None:
        dataset = ForestPersonsDataset(root, split="train")
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        items = manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data
        by_file = {sample.file_name: sample for sample in dataset.samples}
        self.samples = [by_file[item["file_name"]] for item in items[:max_images]]

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
        raise ValueError("Smoke script expects batch size 1.")
    return batch[0]


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dinov3_lora_yolo_head/smoke_train2_seed20261013"),
    )
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--reg-max", type=int, default=32)
    parser.add_argument("--include-quality", action="store_true")
    parser.add_argument("--quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--score-mode", choices=["class", "class_quality"], default="class")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    processor, backbone, _source_head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    input_dim = int(metadata["head_input_dim"])
    head = DINOv3YOLOHead(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        reg_max=args.reg_max,
        include_quality=args.include_quality,
    ).to(device)
    backbone.train()
    head.train()

    dataset = SmokeDataset(args.root, args.manifest, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_one)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 1e-4},
            {"params": [p for p in head.parameters() if p.requires_grad], "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    peak_driver_memory = 0
    if device.type == "mps":
        torch.mps.empty_cache()

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
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
                topk=10,
                reg_max=args.reg_max,
            )
        loss, parts = yolo_detection_loss(
            outputs,
            targets,
            quality_weight=args.quality_loss_weight if args.include_quality else 0.0,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke loss for {batch['file_name']}: {parts}")
        loss.backward()
        grad_summary = gradient_summary(backbone, head)
        nn.utils.clip_grad_norm_(
            [p for p in [*backbone.parameters(), *head.parameters()] if p.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
            peak_driver_memory = max(peak_driver_memory, int(torch.mps.driver_allocated_memory()))
        decoded = decode_yolo_predictions(
            outputs["class_logits"].detach().cpu(),
            outputs["distance_logits"].detach().cpu(),
            info,
            conf_threshold=0.25,
            quality_logits=outputs.get("quality_logits", None).detach().cpu()
            if "quality_logits" in outputs
            else None,
            score_mode=args.score_mode,
        )
        rows.append(
            {
                "file_name": batch["file_name"],
                "loss": float(loss.detach().cpu()),
                **parts,
                "positive_cells": stats.positive_cells,
                "annotations_with_positive": stats.annotations_with_positive,
                "center_fallbacks": stats.center_fallbacks,
                "multi_gt_overwrites": stats.multi_gt_overwrites,
                "clipped_target_distances": stats.clipped_target_distances,
                "decoded_predictions_at_conf_025": len(decoded),
                "gradient_summary": grad_summary,
                "mps_memory_bytes": mps_memory(),
            }
        )

    summary = {
        "settings": {
            "root": str(args.root),
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
            "max_images": args.max_images,
            "short_side": args.short_side,
            "seed": args.seed,
            "device_requested": args.device,
            "device_resolved": str(device),
            "hidden_dim": args.hidden_dim,
            "reg_max": args.reg_max,
            "include_quality": args.include_quality,
            "quality_loss_weight": args.quality_loss_weight,
            "score_mode": args.score_mode,
        },
        "model_id": metadata["model_id"],
        "head_input_dim": input_dim,
        "source_checkpoint_head_kind": metadata["head_kind"],
        "initialization": {
            "backbone_lora": "loaded from existing ViT-S LoRA + FCOS train5000 checkpoint",
            "yolo_head": "fresh random initialization; no FCOS head weights reused",
            "quality_branch": "enabled" if args.include_quality else "disabled",
        },
        "parameter_summary": trainable_parameter_summary(backbone, head),
        "rows": rows,
        "seconds_total": time.perf_counter() - start,
        "peak_mps_driver_memory_bytes": peak_driver_memory or None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "head": "dinov3_yolo_dfl_iou_task_aligned",
            "head_state": head.state_dict(),
            "settings": summary["settings"],
            "parameter_summary": summary["parameter_summary"],
        },
        args.output_dir / "smoke_yolo_head.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
