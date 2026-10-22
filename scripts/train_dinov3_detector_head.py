"""Train a lightweight detection head on cached frozen DINOv3 features."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    DINOv3DetectionHead,
    DINOv3FCOSHead,
    DINOv3GridDetectionHead,
    DINOv3LinearPatchHead,
    annotations_to_ground_truth,
    build_dense_targets,
    build_fcos_targets,
    cache_path_for_file,
    load_feature_cache,
)


class CachedFeatureDataset(Dataset):
    def __init__(self, root: Path, split: str, manifest: Path, cache_root: Path) -> None:
        self.dataset = ForestPersonsDataset(root, split=split)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        self.items = manifest_data["images"]
        by_file = {sample.file_name: sample for sample in self.dataset.samples}
        self.samples = [by_file[item["file_name"]] for item in self.items]
        self.cache_root = cache_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        features, info = load_feature_cache(cache_path_for_file(self.cache_root, sample.file_name))
        ground_truth = annotations_to_ground_truth(sample.annotations)
        objectness, box_targets, box_mask, collisions = build_dense_targets(ground_truth, info)
        fcos_classification, fcos_box_targets, fcos_centerness, fcos_mask, fcos_stats = (
            build_fcos_targets(ground_truth, info)
        )
        return {
            "file_name": sample.file_name,
            "features": features,
            "objectness": objectness,
            "box_targets": box_targets,
            "box_mask": box_mask,
            "collisions": collisions,
            "fcos_classification": fcos_classification,
            "fcos_box_targets": fcos_box_targets,
            "fcos_centerness": fcos_centerness,
            "fcos_mask": fcos_mask,
            "fcos_stats": fcos_stats,
        }


def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("This trainer expects batch_size=1 because feature grids vary by image.")
    return batch[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov3_detector/head_v1"))
    parser.add_argument("--head", choices=["linear", "mlp", "grid", "fcos"], default="mlp")
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=150.0)
    parser.add_argument("--box-loss-weight", type=float, default=5.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def json_safe_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = vars(args).copy()
    for key, value in list(settings.items()):
        if isinstance(value, Path):
            settings[key] = str(value)
    return settings


def make_head(name: str) -> nn.Module:
    if name == "linear":
        return DINOv3LinearPatchHead()
    if name == "mlp":
        return DINOv3DetectionHead()
    if name == "grid":
        return DINOv3GridDetectionHead()
    if name == "fcos":
        return DINOv3FCOSHead()
    raise ValueError(f"Unknown head: {name}")


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def fcos_box_loss(raw_box: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return torch.zeros((), device=raw_box.device)
    pred = torch.nn.functional.softplus(raw_box[mask])
    target = targets[mask]
    return F.smooth_l1_loss(torch.log1p(pred), torch.log1p(target), beta=0.25)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = CachedFeatureDataset(args.root, args.split, args.manifest, args.cache_root)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        collate_fn=collate_one,
    )

    head = make_head(args.head).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        totals = {
            "loss": 0.0,
            "objectness": 0.0,
            "box": 0.0,
            "centerness": 0.0,
            "positives": 0,
            "collisions": 0,
            "center_fallbacks": 0,
            "multi_gt_overwrites": 0,
        }
        for batch in loader:
            features = batch["features"].to(device)
            if args.head == "fcos":
                classification = batch["fcos_classification"].to(device)
                box_targets = batch["fcos_box_targets"].to(device)
                centerness = batch["fcos_centerness"].to(device)
                box_mask = batch["fcos_mask"].to(device)
                logits, raw_box, centerness_logits = head(features)
                objectness_loss = sigmoid_focal_loss(
                    logits.reshape(-1),
                    classification.reshape(-1),
                    alpha=args.focal_alpha,
                    gamma=args.focal_gamma,
                )
                box_loss = fcos_box_loss(raw_box, box_targets, box_mask)
                if box_mask.any():
                    centerness_loss = F.binary_cross_entropy_with_logits(
                        centerness_logits[box_mask],
                        centerness[box_mask],
                    )
                else:
                    centerness_loss = torch.zeros((), device=device)
                loss = objectness_loss + args.box_loss_weight * box_loss + centerness_loss
                totals["center_fallbacks"] += int(batch["fcos_stats"]["center_fallbacks"])
                totals["multi_gt_overwrites"] += int(batch["fcos_stats"]["multi_gt_overwrites"])
            else:
                objectness = batch["objectness"].to(device)
                box_targets = batch["box_targets"].to(device)
                box_mask = batch["box_mask"].to(device)
                logits, raw_box = head(features)
                if args.loss == "focal":
                    objectness_loss = sigmoid_focal_loss(
                        logits.reshape(-1),
                        objectness.reshape(-1),
                        alpha=args.focal_alpha,
                        gamma=args.focal_gamma,
                    )
                else:
                    objectness_loss = F.binary_cross_entropy_with_logits(
                        logits.reshape(-1),
                        objectness.reshape(-1),
                        pos_weight=pos_weight,
                    )
                if box_mask.any():
                    box_loss = F.smooth_l1_loss(raw_box[box_mask], box_targets[box_mask], beta=0.5)
                else:
                    box_loss = torch.zeros((), device=device)
                centerness_loss = torch.zeros((), device=device)
                loss = objectness_loss + args.box_loss_weight * box_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()

            totals["loss"] += float(loss.detach().cpu())
            totals["objectness"] += float(objectness_loss.detach().cpu())
            totals["box"] += float(box_loss.detach().cpu())
            totals["centerness"] += float(centerness_loss.detach().cpu())
            totals["positives"] += int(box_mask.sum().detach().cpu())
            totals["collisions"] += int(batch["collisions"])

        row = {
            "epoch": epoch,
            "loss": totals["loss"] / len(loader),
            "objectness_loss": totals["objectness"] / len(loader),
            "box_loss": totals["box"] / len(loader),
            "centerness_loss": totals["centerness"] / len(loader),
            "positive_cells": totals["positives"],
            "assignment_collisions": totals["collisions"],
            "center_fallbacks": totals["center_fallbacks"],
            "multi_gt_overwrites": totals["multi_gt_overwrites"],
        }
        history.append(row)
        print(json.dumps(row))
        checkpoint = {
            "model_state": head.state_dict(),
            "settings": json_safe_settings(args),
            "history": history,
            "head": args.head,
        }
        torch.save(checkpoint, args.output_dir / "latest.pt")
        (args.output_dir / "train_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    checkpoint = {
        "model_state": head.state_dict(),
        "settings": json_safe_settings(args),
        "history": history,
        "head": args.head,
    }
    checkpoint_path = args.output_dir / "dinov3_head.pt"
    torch.save(checkpoint, checkpoint_path)
    (args.output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
