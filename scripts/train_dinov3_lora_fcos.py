"""Train or smoke-test DINOv3 LoRA adapters with the existing FCOS head.

This script optimizes from images, not detached frozen feature caches. It keeps
the original DINOv3 backbone weights frozen and trains only LoRA adapters plus
the FCOS detection head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    DINOv3FCOSHead,
    DINOv3FCOSQualityHead,
    DINOv3FPNFCOSHead,
    DINOv3Stride8FCOSHead,
    MODEL_ID,
    annotations_to_ground_truth,
    build_fpn_fcos_targets,
    build_fcos_targets,
    choose_device,
    load_compatible_fcos_weights_into_fpn,
    load_compatible_fcos_weights_into_stride8,
)
from forestpersons.dinov3_lora import (
    LoRAConfig,
    extract_lora_patch_features,
    load_lora_fcos_checkpoint,
    save_lora_fcos_checkpoint,
    load_dinov3_with_lora,
    trainable_parameter_summary,
)


class OnlineForestPersonsDataset(Dataset):
    def __init__(self, root: Path, split: str, manifest: Path, max_images: int | None = None) -> None:
        self.dataset = ForestPersonsDataset(root, split=split)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        items = manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data
        if max_images is not None:
            items = items[:max_images]
        self.samples = resolve_manifest_samples(self.dataset.samples, items)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        ground_truth = annotations_to_ground_truth(sample.annotations)
        return {
            "image_id": sample.image_id,
            "file_name": sample.file_name,
            "image": image,
            "ground_truth": ground_truth,
        }


def resolve_manifest_samples(samples: list[Any], items: list[dict[str, Any]]) -> list[Any]:
    """Resolve manifest entries by image ID and verify the paired filename."""

    by_id: dict[int | str, Any] = {}
    for sample in samples:
        if sample.image_id in by_id:
            raise ValueError(f"Duplicate COCO image id: {sample.image_id!r}")
        by_id[sample.image_id] = sample

    resolved = []
    for item in items:
        if "image_id" not in item:
            raise ValueError("Every training manifest entry must contain image_id")
        image_id = item["image_id"]
        if image_id not in by_id:
            raise KeyError(f"Manifest image_id {image_id!r} is absent from the {len(samples)} COCO images")
        sample = by_id[image_id]
        expected_file = str(item.get("file_name", ""))
        if sample.file_name != expected_file:
            raise ValueError(
                f"Manifest image_id {image_id!r} filename mismatch: "
                f"manifest={expected_file!r}, annotations={sample.file_name!r}"
            )
        resolved.append(sample)
    return resolved


def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("DINOv3 LoRA trainer currently expects batch_size=1.")
    return batch[0]


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
    normalization: str = "mean",
    positive_count: int | float | torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if normalization == "mean":
        return loss.mean()
    if normalization == "positive_count":
        if positive_count is None:
            positive_count = targets.sum()
        denominator = torch.as_tensor(positive_count, dtype=loss.dtype, device=loss.device).clamp(min=1.0)
        return loss.sum() / denominator
    raise ValueError(f"Unsupported focal normalization: {normalization!r}")


def _ltrb_to_centered_xyxy(distances: torch.Tensor) -> torch.Tensor:
    left, top, right, bottom = distances.unbind(dim=1)
    return torch.stack([-left, -top, right, bottom], dim=1)


def generalized_iou_loss_xyxy(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    if pred_boxes.numel() == 0:
        return torch.zeros((), device=pred_boxes.device)

    inter_x1 = torch.maximum(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.maximum(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.minimum(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.minimum(pred_boxes[:, 3], target_boxes[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    intersection = inter_w * inter_h

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps) * (
        pred_boxes[:, 3] - pred_boxes[:, 1]
    ).clamp(min=eps)
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps) * (
        target_boxes[:, 3] - target_boxes[:, 1]
    ).clamp(min=eps)
    union = (pred_area + target_area - intersection).clamp(min=eps)
    iou = intersection / union

    enclosing_x1 = torch.minimum(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_y1 = torch.minimum(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_x2 = torch.maximum(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_y2 = torch.maximum(pred_boxes[:, 3], target_boxes[:, 3])
    enclosing_area = (
        (enclosing_x2 - enclosing_x1).clamp(min=eps)
        * (enclosing_y2 - enclosing_y1).clamp(min=eps)
    )
    giou = iou - (enclosing_area - union) / enclosing_area
    return (1.0 - giou).mean()


def generalized_iou_loss_from_ltrb(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    if pred_distances.numel() == 0:
        return torch.zeros((), device=pred_distances.device)
    pred_boxes = _ltrb_to_centered_xyxy(pred_distances.clamp(min=eps))
    target_boxes = _ltrb_to_centered_xyxy(target_distances.clamp(min=eps))
    return generalized_iou_loss_xyxy(pred_boxes, target_boxes, eps=eps)


def fcos_box_loss(
    raw_box: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    giou_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not mask.any():
        zero = torch.zeros((), device=raw_box.device)
        return zero, {"smooth_l1": 0.0, "giou": 0.0}
    pred = torch.nn.functional.softplus(raw_box[mask])
    target = targets[mask]
    smooth_l1 = F.smooth_l1_loss(torch.log1p(pred), torch.log1p(target), beta=0.25)
    if giou_loss_weight <= 0:
        giou = torch.zeros((), device=raw_box.device)
    else:
        giou = generalized_iou_loss_from_ltrb(pred, target)
    loss = smooth_l1 + float(giou_loss_weight) * giou
    return loss, {
        "smooth_l1": float(smooth_l1.detach().cpu()),
        "giou": float(giou.detach().cpu()),
    }


def aligned_iou_from_ltrb(
    pred_distances: torch.Tensor,
    target_distances: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Return aligned IoU for L/T/R/B boxes sharing a feature location."""

    pred = pred_distances.clamp(min=0.0)
    target = target_distances.clamp(min=0.0)
    inter_width = torch.minimum(pred[:, 0], target[:, 0]) + torch.minimum(pred[:, 2], target[:, 2])
    inter_height = torch.minimum(pred[:, 1], target[:, 1]) + torch.minimum(pred[:, 3], target[:, 3])
    intersection = inter_width.clamp(min=0.0) * inter_height.clamp(min=0.0)
    pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    target_area = (target[:, 0] + target[:, 2]) * (target[:, 1] + target[:, 3])
    union = (pred_area + target_area - intersection).clamp(min=eps)
    return (intersection / union).clamp(min=0.0, max=1.0)


def fcos_quality_loss(
    quality_logits: torch.Tensor,
    raw_box: torch.Tensor,
    box_targets: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Positive-only BCE against detached current-box IoU targets."""

    if not positive_mask.any():
        zero = quality_logits.sum() * 0.0
        return zero, {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    pred_distances = F.softplus(raw_box[positive_mask]).clamp(max=128.0)
    targets = aligned_iou_from_ltrb(pred_distances, box_targets[positive_mask]).detach()
    loss = F.binary_cross_entropy_with_logits(
        quality_logits[positive_mask],
        targets,
        reduction="sum",
    ) / positive_mask.sum().clamp(min=1)
    return loss, {
        "mean": float(targets.mean().cpu()),
        "std": float(targets.std(unbiased=False).cpu()),
        "min": float(targets.min().cpu()),
        "max": float(targets.max().cpu()),
        "count": int(targets.numel()),
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-loss-weight", type=float, default=2.0)
    parser.add_argument("--quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--giou-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--focal-normalization",
        choices=["mean", "positive_count"],
        default="mean",
        help="Reduce focal loss by all-location mean or by the number of positive locations.",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="attention.q_proj,attention.v_proj")
    parser.add_argument(
        "--head-init-seed",
        type=int,
        help="Optional seed used immediately before random FCOS head initialization.",
    )
    parser.add_argument(
        "--head-type",
        choices=["fcos", "fcos_quality", "fpn_fcos", "stride8_fcos"],
        default="fcos",
    )
    parser.add_argument(
        "--init-head-checkpoint",
        type=Path,
        help="Optional existing DINOv3 FCOS checkpoint whose head state initializes the FCOS head.",
    )
    parser.add_argument(
        "--init-lora-fcos-checkpoint",
        type=Path,
        help=(
            "Optional existing LoRA+single-scale-FCOS checkpoint. Loads LoRA adapters "
            "and reuses compatible FCOS head weights. For --head-type fpn_fcos or "
            "stride8_fcos, the new feature adapters remain newly initialized."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def json_safe_settings(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    settings = vars(args).copy()
    settings["device_resolved"] = str(device)
    for key, value in list(settings.items()):
        if isinstance(value, Path):
            settings[key] = str(value)
    return settings


def split_trainable_parameters(backbone: nn.Module, head: nn.Module) -> list[dict[str, Any]]:
    lora_parameters = [p for p in backbone.parameters() if p.requires_grad]
    head_parameters = [p for p in head.parameters() if p.requires_grad]
    return [
        {"params": lora_parameters, "name": "lora"},
        {"params": head_parameters, "name": "head"},
    ]


def gradient_summary(backbone: nn.Module, head: nn.Module) -> dict[str, Any]:
    def summarize(parameters: list[nn.Parameter]) -> dict[str, Any]:
        with_grad = [p for p in parameters if p.grad is not None]
        nonzero = [p for p in with_grad if bool((p.grad.detach().abs() > 0).any().item())]
        max_abs = max((float(p.grad.detach().abs().max().cpu()) for p in with_grad), default=0.0)
        squared_norm = sum(float(p.grad.detach().float().pow(2).sum().cpu()) for p in with_grad)
        all_finite = all(bool(torch.isfinite(p.grad).all().item()) for p in with_grad)
        return {
            "with_grad": len(with_grad),
            "nonzero_grad": len(nonzero),
            "max_abs_grad": max_abs,
            "l2_norm": math.sqrt(squared_norm),
            "all_finite": all_finite,
        }

    summary = {
        "lora": summarize([p for p in backbone.parameters() if p.requires_grad]),
        "head": summarize([p for p in head.parameters() if p.requires_grad]),
    }
    for name in ("classification", "box", "centerness", "quality"):
        module = getattr(head, name, None)
        if isinstance(module, nn.Module):
            summary[name] = summarize(list(module.parameters()))
    return summary


def initialize_fcos_head(head: DINOv3FCOSHead, checkpoint_path: Path | None, device: torch.device) -> dict[str, Any]:
    if checkpoint_path is None:
        return {
            "strategy": "random",
            "checkpoint": None,
            "loaded": False,
            "note": "FCOS head initialized randomly because --init-head-checkpoint was not provided.",
        }
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("head") != "fcos":
        raise ValueError(f"Expected an FCOS checkpoint, got head={checkpoint.get('head')!r}")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a model_state dictionary: {checkpoint_path}")
    head.load_state_dict(state, strict=True)
    return {
        "strategy": "loaded_fcos_head",
        "checkpoint": str(checkpoint_path),
        "loaded": True,
        "source_settings": checkpoint.get("settings", {}),
    }


def compute_fcos_losses(
    logits: torch.Tensor,
    raw_box: torch.Tensor,
    centerness_logits: torch.Tensor,
    classification: torch.Tensor,
    box_targets: torch.Tensor,
    centerness: torch.Tensor,
    box_mask: torch.Tensor,
    args: argparse.Namespace,
    quality_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    classification_loss = sigmoid_focal_loss(
        logits.reshape(-1),
        classification.reshape(-1),
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        normalization=getattr(args, "focal_normalization", "mean"),
        positive_count=box_mask.sum(),
    )
    box_loss, box_parts = fcos_box_loss(
        raw_box,
        box_targets,
        box_mask,
        giou_loss_weight=args.giou_loss_weight,
    )
    if box_mask.any():
        centerness_loss = F.binary_cross_entropy_with_logits(
            centerness_logits[box_mask],
            centerness[box_mask],
        )
    else:
        centerness_loss = torch.zeros((), device=logits.device)
    quality_loss = torch.zeros((), device=logits.device)
    quality_target_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    if quality_logits is not None:
        quality_loss, quality_target_stats = fcos_quality_loss(
            quality_logits,
            raw_box,
            box_targets,
            box_mask,
        )
    loss = (
        classification_loss
        + args.box_loss_weight * box_loss
        + centerness_loss
        + float(getattr(args, "quality_loss_weight", 1.0)) * quality_loss
    )
    return loss, {
        "classification": float(classification_loss.detach().cpu()),
        "box": float(box_loss.detach().cpu()),
        "box_smooth_l1": box_parts["smooth_l1"],
        "box_giou": box_parts["giou"],
        "centerness": float(centerness_loss.detach().cpu()),
        "quality": float(quality_loss.detach().cpu()),
        "quality_target_mean": quality_target_stats["mean"],
        "quality_target_std": quality_target_stats["std"],
        "quality_target_min": quality_target_stats["min"],
        "quality_target_max": quality_target_stats["max"],
        "quality_target_count": quality_target_stats["count"],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.head_type == "fpn_fcos" and args.focal_normalization != "mean":
        raise ValueError("positive_count focal normalization is currently scoped to single-level FCOS")

    if args.init_lora_fcos_checkpoint is not None:
        processor, backbone, source_head, source_metadata = load_lora_fcos_checkpoint(args.init_lora_fcos_checkpoint, device)
        if source_metadata["head_kind"] != "fcos":
            raise ValueError("--init-lora-fcos-checkpoint currently expects a single-scale FCOS checkpoint")
        lora_config = source_metadata["config"]
        target_names = source_metadata["target_names"]
        head_input_dim = int(source_metadata["head_input_dim"])
        args.model_id = source_metadata["model_id"]
    else:
        lora_config = LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_suffixes=tuple(value.strip() for value in args.lora_targets.split(",") if value.strip()),
        )
        processor, backbone, target_names = load_dinov3_with_lora(device, lora_config, model_id=args.model_id)
        source_head = None
        head_input_dim = int(getattr(backbone.config, "hidden_size", 384))
    if args.head_init_seed is not None:
        torch.manual_seed(args.head_init_seed)
    if args.head_type == "fpn_fcos":
        head = DINOv3FPNFCOSHead(input_dim=head_input_dim).to(device)
        if source_head is not None:
            reuse_summary = load_compatible_fcos_weights_into_fpn(head, source_head.state_dict())
            head_initialization = {
                "strategy": "loaded_lora_checkpoint_plus_compatible_fcos_weights",
                "checkpoint": str(args.init_lora_fcos_checkpoint),
                "loaded": True,
                "weight_reuse": reuse_summary,
                "note": "LoRA adapters loaded from checkpoint; FPN reused compatible single-scale FCOS weights; pyramid adapters are newly initialized.",
            }
        elif args.init_head_checkpoint is not None:
            checkpoint = torch.load(args.init_head_checkpoint, map_location=device, weights_only=False)
            reuse_summary = load_compatible_fcos_weights_into_fpn(head, checkpoint["model_state"])
            head_initialization = {
                "strategy": "compatible_fcos_head_to_fpn",
                "checkpoint": str(args.init_head_checkpoint),
                "loaded": True,
                "weight_reuse": reuse_summary,
            }
        else:
            head_initialization = {
                "strategy": "random_fpn_fcos",
                "checkpoint": None,
                "loaded": False,
                "note": "FPN FCOS head initialized randomly.",
            }
    elif args.head_type == "stride8_fcos":
        head = DINOv3Stride8FCOSHead(input_dim=head_input_dim).to(device)
        if source_head is not None:
            reuse_summary = load_compatible_fcos_weights_into_stride8(head, source_head.state_dict())
            head_initialization = {
                "strategy": "loaded_lora_checkpoint_plus_compatible_stride8_fcos_weights",
                "checkpoint": str(args.init_lora_fcos_checkpoint),
                "loaded": True,
                "weight_reuse": reuse_summary,
                "note": "LoRA adapters and all compatible FCOS tensors were loaded; only the stride-8 adapter is new.",
            }
        elif args.init_head_checkpoint is not None:
            checkpoint = torch.load(args.init_head_checkpoint, map_location=device, weights_only=False)
            reuse_summary = load_compatible_fcos_weights_into_stride8(head, checkpoint["model_state"])
            head_initialization = {
                "strategy": "compatible_fcos_head_to_stride8",
                "checkpoint": str(args.init_head_checkpoint),
                "loaded": True,
                "weight_reuse": reuse_summary,
            }
        else:
            head_initialization = {
                "strategy": "random_stride8_fcos",
                "checkpoint": None,
                "loaded": False,
                "note": "Stride-8 FCOS head and adapter initialized without an FCOS checkpoint.",
            }
    elif args.head_type == "fcos_quality":
        head = DINOv3FCOSQualityHead(input_dim=head_input_dim).to(device)
        if source_head is None:
            raise ValueError("fcos_quality requires --init-lora-fcos-checkpoint")
        missing, unexpected = head.load_state_dict(source_head.state_dict(), strict=False)
        expected_missing = {"quality.weight", "quality.bias"}
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                f"Unexpected FCOS-to-quality initialization mismatch: missing={missing}, unexpected={unexpected}"
            )
        head_initialization = {
            "strategy": "loaded_lora_checkpoint_fcos_plus_new_quality_branch",
            "checkpoint": str(args.init_lora_fcos_checkpoint),
            "loaded": True,
            "reused_tensors": len(source_head.state_dict()),
            "new_tensors": sorted(missing),
        }
    else:
        head = DINOv3FCOSHead(input_dim=head_input_dim).to(device)
        if source_head is not None:
            head.load_state_dict(source_head.state_dict(), strict=True)
            head_initialization = {
                "strategy": "loaded_lora_checkpoint_fcos_head",
                "checkpoint": str(args.init_lora_fcos_checkpoint),
                "loaded": True,
            }
        else:
            head_initialization = initialize_fcos_head(head, args.init_head_checkpoint, device)
    print(json.dumps({"head_initialization": head_initialization}, indent=2))
    backbone.train()
    head.train()

    dataset = OnlineForestPersonsDataset(args.root, args.split, args.manifest, max_images=args.max_images)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_one,
        generator=shuffle_generator,
    )
    param_groups = split_trainable_parameters(backbone, head)
    param_groups[0]["lr"] = args.lr_lora
    param_groups[1]["lr"] = args.lr_head
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    settings = json_safe_settings(args, device)
    history: list[dict[str, Any]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start_total = time.perf_counter()
    last_gradient_summary: dict[str, Any] = {}

    for epoch in range(1, args.epochs + 1):
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "box": 0.0,
            "box_smooth_l1": 0.0,
            "box_giou": 0.0,
            "centerness": 0.0,
            "quality": 0.0,
            "quality_target_sum": 0.0,
            "quality_target_sq_sum": 0.0,
            "quality_target_min": float("inf"),
            "quality_target_max": float("-inf"),
            "quality_target_count": 0,
            "positive_cells": 0,
            "center_fallbacks": 0,
            "multi_gt_overwrites": 0,
            "images": 0,
        }
        epoch_file_names: list[str] = []
        epoch_sample_keys: list[str] = []
        clipping_steps = 0
        clipped_steps = 0
        max_pre_clip_norm = 0.0
        start_epoch = time.perf_counter()
        for batch_index, batch in enumerate(loader, start=1):
            epoch_file_names.append(str(batch["file_name"]))
            epoch_sample_keys.append(f"{batch['image_id']}\t{batch['file_name']}")
            features, info = extract_lora_patch_features(
                batch["image"],
                processor=processor,
                model=backbone,
                device=device,
                short_side=args.short_side,
            )
            if args.head_type == "fpn_fcos":
                outputs = head(features)
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
                loss = torch.zeros((), device=device)
                loss_values = {"classification": 0.0, "box": 0.0, "centerness": 0.0}
                positive_cells = 0
                for output, target in zip(outputs, targets):
                    classification = target["classification"].to(device)
                    box_targets = target["box_targets"].to(device)
                    centerness = target["centerness"].to(device)
                    box_mask = target["positive_mask"].to(device)
                    level_loss, level_values = compute_fcos_losses(
                        output["class_logits"],
                        output["box_raw"],
                        output["centerness_logits"],
                        classification,
                        box_targets,
                        centerness,
                        box_mask,
                        args,
                    )
                    loss = loss + level_loss
                    loss_values["classification"] += level_values["classification"]
                    loss_values["box"] += level_values["box"]
                    loss_values["box_smooth_l1"] += level_values["box_smooth_l1"]
                    loss_values["box_giou"] += level_values["box_giou"]
                    loss_values["centerness"] += level_values["centerness"]
                    positive_cells += int(box_mask.sum().detach().cpu())
            else:
                head_outputs = head(features)
                if args.head_type == "fcos_quality":
                    logits, raw_box, centerness_logits, quality_logits = head_outputs
                else:
                    logits, raw_box, centerness_logits = head_outputs
                    quality_logits = None
                feature_stride = 8 if args.head_type == "stride8_fcos" else 16
                classification, box_targets, centerness, box_mask, stats = build_fcos_targets(
                    batch["ground_truth"],
                    info,
                    stride=feature_stride,
                    grid_height=int(logits.shape[0]),
                    grid_width=int(logits.shape[1]),
                )
                classification = classification.to(device)
                box_targets = box_targets.to(device)
                centerness = centerness.to(device)
                box_mask = box_mask.to(device)
                loss, loss_values = compute_fcos_losses(
                    logits,
                    raw_box,
                    centerness_logits,
                    classification,
                    box_targets,
                    centerness,
                    box_mask,
                    args,
                    quality_logits=quality_logits,
                )
                positive_cells = int(box_mask.sum().detach().cpu())
            (loss / args.gradient_accumulation).backward()

            totals["loss"] += float(loss.detach().cpu())
            totals["classification"] += loss_values["classification"]
            totals["box"] += loss_values["box"]
            totals["box_smooth_l1"] += loss_values["box_smooth_l1"]
            totals["box_giou"] += loss_values["box_giou"]
            totals["centerness"] += loss_values["centerness"]
            totals["quality"] += loss_values.get("quality", 0.0)
            quality_count = int(loss_values.get("quality_target_count", 0))
            if quality_count:
                quality_mean = float(loss_values["quality_target_mean"])
                quality_std = float(loss_values["quality_target_std"])
                totals["quality_target_sum"] += quality_mean * quality_count
                totals["quality_target_sq_sum"] += (quality_std**2 + quality_mean**2) * quality_count
                totals["quality_target_min"] = min(
                    totals["quality_target_min"], float(loss_values["quality_target_min"])
                )
                totals["quality_target_max"] = max(
                    totals["quality_target_max"], float(loss_values["quality_target_max"])
                )
                totals["quality_target_count"] += quality_count
            totals["positive_cells"] += positive_cells
            totals["center_fallbacks"] += int(stats["center_fallbacks"])
            totals["multi_gt_overwrites"] += int(stats["multi_gt_overwrites"])
            totals["images"] += 1

            should_step = (
                batch_index % args.gradient_accumulation == 0
                or batch_index == len(loader)
            )
            if should_step:
                last_gradient_summary = gradient_summary(backbone, head)
                pre_clip_norm = nn.utils.clip_grad_norm_(
                    [p for group in param_groups for p in group["params"]],
                    max_norm=5.0,
                )
                pre_clip_value = float(pre_clip_norm.detach().cpu())
                if not math.isfinite(pre_clip_value):
                    raise FloatingPointError(f"Non-finite pre-clip gradient norm at optimizer step {global_step + 1}")
                clipping_steps += 1
                clipped_steps += int(pre_clip_value > 5.0)
                max_pre_clip_norm = max(max_pre_clip_norm, pre_clip_value)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        synchronize_device(device)
        images = max(1, totals["images"])
        row = {
            "epoch": epoch,
            "images": totals["images"],
            "optimizer_steps": global_step,
            "loss": totals["loss"] / images,
            "classification_loss": totals["classification"] / images,
            "box_loss": totals["box"] / images,
            "box_smooth_l1_loss": totals["box_smooth_l1"] / images,
            "box_giou_loss": totals["box_giou"] / images,
            "centerness_loss": totals["centerness"] / images,
            "quality_loss": totals["quality"] / images,
            "quality_target_distribution": {
                "count": totals["quality_target_count"],
                "mean": (
                    totals["quality_target_sum"] / totals["quality_target_count"]
                    if totals["quality_target_count"]
                    else 0.0
                ),
                "std": (
                    max(
                        0.0,
                        totals["quality_target_sq_sum"] / totals["quality_target_count"]
                        - (totals["quality_target_sum"] / totals["quality_target_count"]) ** 2,
                    ) ** 0.5
                    if totals["quality_target_count"]
                    else 0.0
                ),
                "min": totals["quality_target_min"] if totals["quality_target_count"] else 0.0,
                "max": totals["quality_target_max"] if totals["quality_target_count"] else 0.0,
            },
            "positive_cells": totals["positive_cells"],
            "center_fallbacks": totals["center_fallbacks"],
            "multi_gt_overwrites": totals["multi_gt_overwrites"],
            "image_order_sha256": hashlib.sha256("\n".join(epoch_file_names).encode("utf-8")).hexdigest(),
            "sample_order_sha256": hashlib.sha256("\n".join(epoch_sample_keys).encode("utf-8")).hexdigest(),
            "gradient_clipping": {
                "steps": clipping_steps,
                "clipped_steps": clipped_steps,
                "max_pre_clip_norm": max_pre_clip_norm,
                "max_norm": 5.0,
            },
            "seconds": time.perf_counter() - start_epoch,
            "gradient_summary": last_gradient_summary,
            "mps_memory_bytes": mps_memory(),
        }
        history.append(row)
        print(json.dumps(row))
        (args.output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_lora_fcos_checkpoint(
            args.output_dir / "latest.pt",
            backbone=backbone,
            head=head,
            config=lora_config,
            target_names=target_names,
            settings=settings,
            history=history,
            model_id=args.model_id,
            head_kind=args.head_type,
        )
        if args.save_every_epoch:
            save_lora_fcos_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt",
                backbone=backbone,
                head=head,
                config=lora_config,
                target_names=target_names,
                settings=settings,
                history=history,
                model_id=args.model_id,
                head_kind=args.head_type,
            )

    summary = {
        "settings": settings,
        "head_initialization": head_initialization,
        "head_type": args.head_type,
        "head_input_dim": head_input_dim,
        "lora_target_modules": target_names,
        "parameter_summary": trainable_parameter_summary(backbone, head),
        "history": history,
        "seconds_total": time.perf_counter() - start_total,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_lora_fcos_checkpoint(
        args.output_dir / "dinov3_lora_fcos.pt",
        backbone=backbone,
        head=head,
        config=lora_config,
        target_names=target_names,
        settings=settings,
        history=history,
        model_id=args.model_id,
        head_kind=args.head_type,
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved LoRA FCOS checkpoint: {args.output_dir / 'dinov3_lora_fcos.pt'}")


if __name__ == "__main__":
    main()
