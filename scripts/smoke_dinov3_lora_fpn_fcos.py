"""Smoke-test DINOv3 LoRA with the lightweight FPN FCOS head.

This script is intentionally small and bounded. It verifies that the new
multi-scale head can reuse compatible single-scale FCOS weights, run
forward/backward on real ForestPersons images, and save a reproducible summary.
It is not a training experiment.
"""

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
from torch.utils.data import DataLoader, Dataset

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    DINOv3FPNFCOSHead,
    annotations_to_ground_truth,
    build_fpn_fcos_targets,
    choose_device,
    decode_fpn_fcos_predictions,
    load_compatible_fcos_weights_into_fpn,
)
from forestpersons.dinov3_lora import (
    extract_lora_patch_features,
    load_lora_fcos_checkpoint,
    trainable_parameter_summary,
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


def sigmoid_focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    processor, backbone, single_head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    input_dim = int(metadata["head_input_dim"])
    fpn_head = DINOv3FPNFCOSHead(input_dim=input_dim).to(device)
    weight_reuse = load_compatible_fcos_weights_into_fpn(fpn_head, single_head.state_dict())
    backbone.train()
    fpn_head.train()

    dataset = SmokeDataset(args.root, args.manifest, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_one)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": 1e-4},
            {"params": [p for p in fpn_head.parameters() if p.requires_grad], "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        features, info = extract_lora_patch_features(
            batch["image"],
            processor=processor,
            model=backbone,
            device=device,
            short_side=args.short_side,
        )
        outputs = fpn_head(features)
        level_shapes = [
            (
                str(output["level"]),
                int(output["stride"]),
                int(output["class_logits"].shape[0]),
                int(output["class_logits"].shape[1]),
            )
            for output in outputs
        ]
        targets, stats = build_fpn_fcos_targets(batch["ground_truth"], info, level_shapes)
        total_loss = torch.zeros((), device=device)
        loss_parts = {"classification": 0.0, "box": 0.0, "centerness": 0.0}
        for output, target in zip(outputs, targets):
            classification = target["classification"].to(device)
            box_targets = target["box_targets"].to(device)
            centerness = target["centerness"].to(device)
            mask = target["positive_mask"].to(device)
            classification_loss = sigmoid_focal_loss(output["class_logits"].reshape(-1), classification.reshape(-1))
            box_loss = fcos_box_loss(output["box_raw"], box_targets, mask)
            if mask.any():
                centerness_loss = F.binary_cross_entropy_with_logits(output["centerness_logits"][mask], centerness[mask])
            else:
                centerness_loss = torch.zeros((), device=device)
            total_loss = total_loss + classification_loss + 2.0 * box_loss + centerness_loss
            loss_parts["classification"] += float(classification_loss.detach().cpu())
            loss_parts["box"] += float(box_loss.detach().cpu())
            loss_parts["centerness"] += float(centerness_loss.detach().cpu())
        total_loss.backward()
        grad_summary = gradient_summary(backbone, fpn_head)
        nn.utils.clip_grad_norm_(
            [p for p in [*backbone.parameters(), *fpn_head.parameters()] if p.requires_grad],
            max_norm=5.0,
        )
        optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()
        decoded = decode_fpn_fcos_predictions(
            [{key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in output.items()} for output in outputs],
            info,
            conf_threshold=0.50,
        )
        rows.append(
            {
                "file_name": batch["file_name"],
                "loss": float(total_loss.detach().cpu()),
                "classification_loss": loss_parts["classification"],
                "box_loss": loss_parts["box"],
                "centerness_loss": loss_parts["centerness"],
                "positive_cells": int(stats["positive_cells"]),
                "center_fallbacks": int(stats["center_fallbacks"]),
                "multi_gt_overwrites": int(stats["multi_gt_overwrites"]),
                "decoded_predictions_at_conf_050": len(decoded),
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
        },
        "model_id": metadata["model_id"],
        "head_input_dim": input_dim,
        "weight_reuse": weight_reuse,
        "parameter_summary": trainable_parameter_summary(backbone, fpn_head),
        "rows": rows,
        "seconds_total": time.perf_counter() - start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
