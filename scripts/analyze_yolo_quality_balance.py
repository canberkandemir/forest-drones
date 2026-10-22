"""Analyze YOLO-style quality-loss balance on a small ForestPersons subset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import annotations_to_ground_truth, choose_device
from forestpersons.dinov3_lora import extract_lora_patch_features
from forestpersons.dinov3_yolo import (
    DINOv3YOLOHead,
    build_task_aligned_yolo_targets,
    dfl_expectation,
    load_lora_yolo_checkpoint,
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
        raise ValueError("This diagnostic expects batch_size=1.")
    return batch[0]


def make_quality_head(source_head: DINOv3YOLOHead, input_dim: int, device: torch.device) -> DINOv3YOLOHead:
    head = DINOv3YOLOHead(
        input_dim=input_dim,
        hidden_dim=int(source_head.stem.out_channels),
        reg_max=int(source_head.reg_max),
        include_quality=True,
    ).to(device)
    missing, unexpected = head.load_state_dict(source_head.state_dict(), strict=False)
    allowed_missing = {key for key in missing if key.startswith("quality.")}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"Unexpected quality init mismatch: missing={missing}, unexpected={unexpected}")
    return head


def summarize_gradients(head: DINOv3YOLOHead) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, parameter in head.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        group = "quality" if name.startswith("quality.") else "shared_head"
        stats = rows.setdefault(group, {"with_grad": 0, "nonzero_grad": 0, "max_abs_grad": 0.0})
        stats["with_grad"] += 1
        grad = parameter.grad.detach()
        if bool((grad.abs() > 0).any().item()):
            stats["nonzero_grad"] += 1
        stats["max_abs_grad"] = max(stats["max_abs_grad"], float(grad.abs().max().cpu()))
    return rows


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
        "--checkpoint",
        type=Path,
        default=Path("outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/dinov3_lora_yolo.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--assignment-topk", type=int, default=10)
    parser.add_argument("--assignment-alpha", type=float, default=1.0)
    parser.add_argument("--assignment-beta", type=float, default=6.0)
    parser.add_argument("--quality-negative-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "mps":
        torch.mps.empty_cache()

    processor, backbone, source_head, metadata = load_lora_yolo_checkpoint(args.checkpoint, device)
    head = make_quality_head(source_head, int(metadata["head_input_dim"]), device)
    backbone.train()
    head.train()

    dataset = OnlineForestPersonsDataset(args.root, args.split, args.manifest, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_one)

    totals = {
        "images": 0,
        "positive_cells": 0.0,
        "negative_cells": 0.0,
        "quality_positive_loss": 0.0,
        "quality_negative_loss": 0.0,
        "quality_loss_balanced": 0.0,
        "quality_target_mean": 0.0,
        "quality_target_max": 0.0,
    }
    gradient_sample: dict[str, Any] | None = None
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
            targets, _stats = build_task_aligned_yolo_targets(
                batch["ground_truth"],
                info,
                class_logits=outputs["class_logits"].detach(),
                predicted_distances=predicted_distances,
                topk=args.assignment_topk,
                alpha=args.assignment_alpha,
                beta=args.assignment_beta,
                reg_max=head.reg_max,
            )
        loss, parts = yolo_detection_loss(
            outputs,
            targets,
            quality_weight=1.0,
            quality_negative_weight=args.quality_negative_weight,
        )
        if batch_index == 1:
            loss.backward()
            gradient_sample = summarize_gradients(head)
            head.zero_grad(set_to_none=True)
            backbone.zero_grad(set_to_none=True)
        totals["images"] += 1
        totals["positive_cells"] += parts["quality_positive_count"]
        totals["negative_cells"] += parts["quality_negative_count"]
        totals["quality_positive_loss"] += parts["quality_positive_loss"]
        totals["quality_negative_loss"] += parts["quality_negative_loss"]
        totals["quality_loss_balanced"] += parts["quality_loss"]
        totals["quality_target_mean"] += parts["quality_target_mean"]
        totals["quality_target_max"] = max(totals["quality_target_max"], parts["quality_target_max"])

    images = max(1, int(totals["images"]))
    pos_count = totals["positive_cells"]
    neg_count = totals["negative_cells"]
    pos_loss = totals["quality_positive_loss"] / images
    neg_loss = totals["quality_negative_loss"] / images
    raw_full_grid_loss = (pos_loss * pos_count + neg_loss * neg_count) / max(1.0, pos_count + neg_count)
    raw_negative_fraction = (neg_loss * neg_count) / max(1e-12, pos_loss * pos_count + neg_loss * neg_count)

    summary = {
        "checkpoint": str(args.checkpoint),
        "images": images,
        "short_side": args.short_side,
        "positive_cells_total": pos_count,
        "negative_cells_total": neg_count,
        "positive_cells_per_image": pos_count / images,
        "negative_cells_per_image": neg_count / images,
        "quality_positive_loss_mean": pos_loss,
        "quality_negative_loss_mean": neg_loss,
        "quality_balanced_loss_mean": totals["quality_loss_balanced"] / images,
        "quality_raw_full_grid_loss_estimate": raw_full_grid_loss,
        "quality_raw_negative_contribution_fraction": raw_negative_fraction,
        "quality_target_mean_per_image": totals["quality_target_mean"] / images,
        "quality_target_max": totals["quality_target_max"],
        "gradient_sample_first_image": gradient_sample,
        "normalization_used_for_quality_training": "positive mean + quality_negative_weight * negative mean",
        "quality_negative_weight": args.quality_negative_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
