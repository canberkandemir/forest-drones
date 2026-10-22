"""YOLO-style dense detector head for DINOv3 patch features.

This module is intentionally separate from the existing FCOS implementation.
It keeps the same stride-16 DINOv3 patch grid, but changes the detection
mechanics to a YOLO-inspired formulation:

* independent classification and regression towers,
* distributional L/T/R/B regression with Distribution Focal Loss,
* IoU box loss on decoded boxes,
* task-aligned positive assignment using classification score and IoU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms

from .dinov3_detector import PATCH_SIZE, PreprocessInfo


@dataclass(frozen=True)
class YOLOAssignmentStats:
    positive_cells: int
    annotations_with_positive: int
    center_fallbacks: int
    multi_gt_overwrites: int
    clipped_target_distances: int


class DINOv3YOLOHead(nn.Module):
    """Single-class YOLO-inspired head over a DINOv3 stride-16 feature map."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 160,
        reg_max: int = 32,
        include_quality: bool = False,
    ) -> None:
        super().__init__()
        if reg_max < 2:
            raise ValueError("reg_max must be at least 2")
        self.reg_max = int(reg_max)
        self.reg_bins = self.reg_max + 1
        self.include_quality = bool(include_quality)
        self.stem = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.classification = nn.Sequential(
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.regression = nn.Sequential(
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 4 * self.reg_bins, kernel_size=1),
        )
        if self.include_quality:
            self.quality = nn.Sequential(
                nn.GroupNorm(16, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(16, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
        else:
            self.quality = None

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape [H, W, C], got {tuple(features.shape)}")
        x = features.permute(2, 0, 1).unsqueeze(0)
        hidden = self.stem(x)
        class_logits = self.classification(hidden).squeeze(0).squeeze(0)
        distance_logits = self.regression(hidden).squeeze(0).permute(1, 2, 0)
        h, w = class_logits.shape
        distance_logits = distance_logits.reshape(h, w, 4, self.reg_bins)
        outputs = {"class_logits": class_logits, "distance_logits": distance_logits}
        if self.quality is not None:
            outputs["quality_logits"] = self.quality(hidden).squeeze(0).squeeze(0)
        return outputs


def make_grid_centers(
    height: int,
    width: int,
    stride: int = PATCH_SIZE,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    grid_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * float(stride)
    grid_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * float(stride)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def dfl_expectation(distance_logits: torch.Tensor) -> torch.Tensor:
    """Decode discrete distance logits to expected bin distances."""

    bins = torch.arange(
        distance_logits.shape[-1],
        device=distance_logits.device,
        dtype=distance_logits.dtype,
    )
    probs = F.softmax(distance_logits, dim=-1)
    return (probs * bins).sum(dim=-1)


def distribution_focal_loss(distance_logits: torch.Tensor, target_distances: torch.Tensor) -> torch.Tensor:
    """Distribution Focal Loss for continuous distance targets in bin units."""

    if distance_logits.numel() == 0:
        return distance_logits.sum() * 0.0
    reg_max = distance_logits.shape[-1] - 1
    targets = target_distances.clamp(min=0.0, max=float(reg_max) - 1e-4)
    left = targets.floor().long()
    right = (left + 1).clamp(max=reg_max)
    weight_right = targets - left.to(targets.dtype)
    weight_left = 1.0 - weight_right
    flat_logits = distance_logits.reshape(-1, reg_max + 1)
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    loss_left = F.cross_entropy(flat_logits, flat_left, reduction="none").reshape_as(targets)
    loss_right = F.cross_entropy(flat_logits, flat_right, reduction="none").reshape_as(targets)
    return (loss_left * weight_left + loss_right * weight_right).mean()


def ltrb_to_xyxy(centers: torch.Tensor, distances: torch.Tensor, stride: float = 1.0) -> torch.Tensor:
    scaled = distances * float(stride)
    return torch.stack(
        [
            centers[..., 0] - scaled[..., 0],
            centers[..., 1] - scaled[..., 1],
            centers[..., 0] + scaled[..., 2],
            centers[..., 1] + scaled[..., 3],
        ],
        dim=-1,
    )


def box_iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), device=boxes_a.device, dtype=boxes_a.dtype)
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=eps)


def iou_regression_loss(
    predicted_distances: torch.Tensor,
    target_distances: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    """IoU loss for L/T/R/B distances around the same grid centers."""

    if predicted_distances.numel() == 0:
        return predicted_distances.sum() * 0.0
    centers = torch.zeros((predicted_distances.shape[0], 2), device=predicted_distances.device, dtype=predicted_distances.dtype)
    pred_boxes = ltrb_to_xyxy(centers, predicted_distances)
    target_boxes = ltrb_to_xyxy(centers, target_distances)
    ious = box_iou_matrix(pred_boxes, target_boxes).diag().clamp(min=0.0, max=1.0)
    loss = 1.0 - ious
    if weights is None:
        return loss.mean()
    normalized = weights / weights.sum().clamp(min=eps)
    return (loss * normalized).sum()


def _gt_box_to_resized(gt: dict[str, Any], info: PreprocessInfo) -> list[float]:
    x1 = max(0.0, min(float(gt["box"][0]) * info.scale_x, info.resized_width - 1.0))
    y1 = max(0.0, min(float(gt["box"][1]) * info.scale_y, info.resized_height - 1.0))
    x2 = max(0.0, min(float(gt["box"][2]) * info.scale_x, info.resized_width - 1.0))
    y2 = max(0.0, min(float(gt["box"][3]) * info.scale_y, info.resized_height - 1.0))
    return [x1, y1, x2, y2]


@torch.no_grad()
def build_task_aligned_yolo_targets(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
    class_logits: torch.Tensor,
    predicted_distances: torch.Tensor,
    topk: int = 10,
    alpha: float = 1.0,
    beta: float = 6.0,
    reg_max: int = 32,
) -> tuple[dict[str, torch.Tensor], YOLOAssignmentStats]:
    """Assign positives using a simplified single-class task-aligned rule."""

    device = class_logits.device
    dtype = class_logits.dtype
    h, w = class_logits.shape
    centers = make_grid_centers(h, w, device=device, dtype=dtype)
    target_scores = torch.zeros((h, w), device=device, dtype=dtype)
    target_distances = torch.zeros((h, w, 4), device=device, dtype=dtype)
    positive_mask = torch.zeros((h, w), device=device, dtype=torch.bool)
    assigned_metric = torch.full((h, w), -1.0, device=device, dtype=dtype)
    stats = {
        "annotations_with_positive": 0,
        "center_fallbacks": 0,
        "multi_gt_overwrites": 0,
        "clipped_target_distances": 0,
    }

    flat_centers = centers.reshape(-1, 2)
    flat_scores = class_logits.sigmoid().reshape(-1)
    flat_pred_dist = predicted_distances.reshape(-1, 4)
    pred_boxes = ltrb_to_xyxy(flat_centers, flat_pred_dist, stride=PATCH_SIZE)

    for gt in ground_truth:
        x1, y1, x2, y2 = _gt_box_to_resized(gt, info)
        if x2 <= x1 or y2 <= y1:
            continue
        gt_box = torch.tensor([x1, y1, x2, y2], device=device, dtype=dtype)
        inside = (
            (flat_centers[:, 0] > x1)
            & (flat_centers[:, 0] < x2)
            & (flat_centers[:, 1] > y1)
            & (flat_centers[:, 1] < y2)
        )
        if not inside.any():
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            distances_to_center = (flat_centers[:, 0] - cx).pow(2) + (flat_centers[:, 1] - cy).pow(2)
            inside[distances_to_center.argmin()] = True
            stats["center_fallbacks"] += 1

        candidate_indices = torch.nonzero(inside, as_tuple=False).flatten()
        candidate_boxes = pred_boxes[candidate_indices]
        candidate_ious = box_iou_matrix(candidate_boxes, gt_box.unsqueeze(0)).squeeze(1)
        candidate_scores = flat_scores[candidate_indices]
        alignment = candidate_scores.pow(alpha) * candidate_ious.clamp(min=0.0).pow(beta)
        if float(alignment.max().detach().cpu()) <= 0.0:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            center_distance = (flat_centers[candidate_indices, 0] - cx).pow(2) + (
                flat_centers[candidate_indices, 1] - cy
            ).pow(2)
            order = torch.argsort(center_distance)
            selected = candidate_indices[order[: min(topk, candidate_indices.numel())]]
            selected_metric = torch.linspace(
                1.0,
                0.5,
                steps=selected.numel(),
                device=device,
                dtype=dtype,
            )
        else:
            selected_count = min(topk, candidate_indices.numel())
            selected_values, order = torch.topk(alignment, k=selected_count)
            selected = candidate_indices[order]
            selected_metric = selected_values / selected_values.max().clamp(min=1e-7)
            selected_metric = selected_metric.clamp(min=0.05, max=1.0)

        assigned_this_gt = False
        for flat_index, metric in zip(selected.tolist(), selected_metric):
            gy = flat_index // w
            gx = flat_index % w
            if metric <= assigned_metric[gy, gx]:
                stats["multi_gt_overwrites"] += 1
                continue
            cx, cy = centers[gy, gx]
            distances = torch.stack(
                [
                    (cx - x1) / PATCH_SIZE,
                    (cy - y1) / PATCH_SIZE,
                    (x2 - cx) / PATCH_SIZE,
                    (y2 - cy) / PATCH_SIZE,
                ]
            ).clamp(min=1e-4)
            clipped = int((distances > (float(reg_max) - 1e-4)).sum().item())
            stats["clipped_target_distances"] += clipped
            distances = distances.clamp(max=float(reg_max) - 1e-4)
            if positive_mask[gy, gx]:
                stats["multi_gt_overwrites"] += 1
            positive_mask[gy, gx] = True
            assigned_metric[gy, gx] = metric
            target_scores[gy, gx] = metric
            target_distances[gy, gx] = distances
            assigned_this_gt = True
        if assigned_this_gt:
            stats["annotations_with_positive"] += 1

    return {
        "target_scores": target_scores,
        "target_distances": target_distances,
        "positive_mask": positive_mask,
    }, YOLOAssignmentStats(
        positive_cells=int(positive_mask.sum().item()),
        annotations_with_positive=int(stats["annotations_with_positive"]),
        center_fallbacks=int(stats["center_fallbacks"]),
        multi_gt_overwrites=int(stats["multi_gt_overwrites"]),
        clipped_target_distances=int(stats["clipped_target_distances"]),
    )


def yolo_detection_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    box_weight: float = 7.5,
    dfl_weight: float = 1.5,
    quality_weight: float = 0.0,
    quality_negative_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    class_logits = outputs["class_logits"]
    distance_logits = outputs["distance_logits"]
    target_scores = targets["target_scores"].to(class_logits.device)
    target_distances = targets["target_distances"].to(class_logits.device)
    positive_mask = targets["positive_mask"].to(class_logits.device)
    normalizer = target_scores.sum().clamp(min=1.0)
    cls_loss = F.binary_cross_entropy_with_logits(class_logits, target_scores, reduction="sum") / normalizer

    pred_distances = dfl_expectation(distance_logits)
    if positive_mask.any():
        weights = target_scores[positive_mask].detach().clamp(min=0.05)
        positive_pred_distances = pred_distances[positive_mask]
        positive_target_distances = target_distances[positive_mask]
        box_loss = iou_regression_loss(positive_pred_distances, positive_target_distances, weights=weights)
        dfl_loss = distribution_focal_loss(distance_logits[positive_mask], target_distances[positive_mask])
        with torch.no_grad():
            centers = torch.zeros(
                (positive_pred_distances.shape[0], 2),
                device=positive_pred_distances.device,
                dtype=positive_pred_distances.dtype,
            )
            quality_targets_positive = box_iou_matrix(
                ltrb_to_xyxy(centers, positive_pred_distances),
                ltrb_to_xyxy(centers, positive_target_distances),
            ).diag().clamp(min=0.0, max=1.0)
    else:
        zero = distance_logits.sum() * 0.0
        box_loss = zero
        dfl_loss = zero
        quality_targets_positive = None
    quality_loss = distance_logits.sum() * 0.0
    quality_positive_loss = distance_logits.sum() * 0.0
    quality_negative_loss = distance_logits.sum() * 0.0
    quality_positive_count = int(positive_mask.sum().item())
    quality_negative_count = int((~positive_mask).sum().item())
    quality_target_mean = 0.0
    quality_target_max = 0.0
    if quality_weight > 0.0 and "quality_logits" in outputs:
        quality_logits = outputs["quality_logits"]
        quality_targets = torch.zeros_like(quality_logits)
        if positive_mask.any() and quality_targets_positive is not None:
            quality_targets[positive_mask] = quality_targets_positive.to(quality_targets.device)
            quality_target_mean = float(quality_targets_positive.detach().mean().cpu())
            quality_target_max = float(quality_targets_positive.detach().max().cpu())
        quality_element_loss = F.binary_cross_entropy_with_logits(quality_logits, quality_targets, reduction="none")
        if positive_mask.any():
            quality_positive_loss = quality_element_loss[positive_mask].mean()
        if (~positive_mask).any():
            quality_negative_loss = quality_element_loss[~positive_mask].mean()
        quality_loss = quality_positive_loss + float(quality_negative_weight) * quality_negative_loss
    total = cls_loss + box_weight * box_loss + dfl_weight * dfl_loss + quality_weight * quality_loss
    return total, {
        "classification_loss": float(cls_loss.detach().cpu()),
        "iou_loss": float(box_loss.detach().cpu()),
        "dfl_loss": float(dfl_loss.detach().cpu()),
        "quality_loss": float(quality_loss.detach().cpu()),
        "quality_positive_loss": float(quality_positive_loss.detach().cpu()),
        "quality_negative_loss": float(quality_negative_loss.detach().cpu()),
        "quality_positive_count": float(quality_positive_count),
        "quality_negative_count": float(quality_negative_count),
        "quality_target_mean": quality_target_mean,
        "quality_target_max": quality_target_max,
        "total_loss": float(total.detach().cpu()),
    }


def decode_yolo_predictions(
    class_logits: torch.Tensor,
    distance_logits: torch.Tensor,
    info: PreprocessInfo,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.50,
    topk: int = 300,
    quality_logits: torch.Tensor | None = None,
    score_mode: str = "class",
) -> list[dict[str, Any]]:
    scores = yolo_scores(class_logits, quality_logits=quality_logits, score_mode=score_mode).reshape(-1)
    keep = torch.nonzero(scores >= conf_threshold, as_tuple=False).flatten()
    if keep.numel() == 0:
        return []
    keep_scores = scores[keep]
    if keep.numel() > topk:
        keep_scores, order = torch.topk(keep_scores, k=topk)
        keep = keep[order]

    h, w = class_logits.shape
    centers = make_grid_centers(h, w, device=class_logits.device, dtype=class_logits.dtype).reshape(-1, 2)
    distances = dfl_expectation(distance_logits).reshape(-1, 4)[keep]
    boxes = ltrb_to_xyxy(centers[keep], distances, stride=PATCH_SIZE)
    boxes = torch.stack(
        [
            boxes[:, 0].clamp(0, info.resized_width - 1),
            boxes[:, 1].clamp(0, info.resized_height - 1),
            boxes[:, 2].clamp(0, info.resized_width - 1),
            boxes[:, 3].clamp(0, info.resized_height - 1),
        ],
        dim=1,
    )
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    keep_scores = keep_scores[valid]
    if boxes.numel() == 0:
        return []
    selected = nms(boxes, keep_scores, nms_threshold)
    boxes = boxes[selected]
    keep_scores = keep_scores[selected]
    scale = torch.tensor(
        [info.scale_x, info.scale_y, info.scale_x, info.scale_y],
        device=boxes.device,
        dtype=boxes.dtype,
    )
    boxes = boxes / scale
    predictions: list[dict[str, Any]] = []
    for box, score in zip(boxes.detach().cpu(), keep_scores.detach().cpu()):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        predictions.append(
            {
                "box": [
                    max(0.0, min(x1, info.original_width - 1.0)),
                    max(0.0, min(y1, info.original_height - 1.0)),
                    max(0.0, min(x2, info.original_width - 1.0)),
                    max(0.0, min(y2, info.original_height - 1.0)),
                ],
                "confidence": float(score),
            }
        )
    return predictions


def yolo_scores(
    class_logits: torch.Tensor,
    quality_logits: torch.Tensor | None = None,
    score_mode: str = "class",
) -> torch.Tensor:
    """Return inference scores while keeping class-only scoring as default."""

    class_scores = class_logits.sigmoid()
    if score_mode == "class":
        return class_scores
    if score_mode == "class_quality":
        if quality_logits is None:
            raise ValueError("quality_logits are required for score_mode='class_quality'")
        return class_scores * quality_logits.sigmoid()
    raise ValueError(f"Unsupported YOLO score mode: {score_mode}")


def save_lora_yolo_checkpoint(
    path: Path,
    backbone: nn.Module,
    head: DINOv3YOLOHead,
    settings: dict[str, Any],
    history: list[dict[str, Any]],
    model_id: str,
    lora_config: Any,
    target_names: list[str],
) -> None:
    """Save a LoRA-backbone + YOLO-head checkpoint without touching FCOS state."""

    from .dinov3_lora import lora_state_dict, trainable_parameter_summary

    path.parent.mkdir(parents=True, exist_ok=True)
    config_data = {
        "rank": int(lora_config.rank),
        "alpha": float(lora_config.alpha),
        "dropout": float(lora_config.dropout),
        "target_suffixes": tuple(lora_config.target_suffixes),
    }
    torch.save(
        {
            "head": "yolo_dfl_iou_task_aligned",
            "backbone_model_id": model_id,
            "lora_config": config_data,
            "lora_target_modules": target_names,
            "lora_state": lora_state_dict(backbone),
            "head_state": head.state_dict(),
            "head_input_dim": int(getattr(head.stem, "in_channels", 384)),
            "head_hidden_dim": int(getattr(head.stem, "out_channels", 160)),
            "reg_max": int(head.reg_max),
            "include_quality": bool(head.include_quality),
            "settings": settings,
            "history": history,
            "parameter_summary": trainable_parameter_summary(backbone, head),
        },
        path,
    )


def load_lora_yolo_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, nn.Module, DINOv3YOLOHead, dict[str, Any]]:
    """Load a DINOv3 LoRA backbone with the YOLO-style head."""

    from .dinov3_detector import MODEL_ID
    from .dinov3_lora import LoRAConfig, load_dinov3_with_lora, load_lora_state_dict

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("head") != "yolo_dfl_iou_task_aligned":
        raise ValueError(f"Expected YOLO-style checkpoint, got head={checkpoint.get('head')!r}")
    config_data = checkpoint["lora_config"]
    config = LoRAConfig(
        rank=int(config_data["rank"]),
        alpha=float(config_data["alpha"]),
        dropout=float(config_data["dropout"]),
        target_suffixes=tuple(config_data["target_suffixes"]),
    )
    model_id = str(checkpoint.get("backbone_model_id", MODEL_ID))
    processor, backbone, target_names = load_dinov3_with_lora(device, config, model_id=model_id)
    load_lora_state_dict(backbone, checkpoint["lora_state"])
    head = DINOv3YOLOHead(
        input_dim=int(checkpoint.get("head_input_dim", getattr(backbone.config, "hidden_size", 384))),
        hidden_dim=int(checkpoint.get("head_hidden_dim", 160)),
        reg_max=int(checkpoint.get("reg_max", 32)),
        include_quality=bool(checkpoint.get("include_quality", False)),
    ).to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    return processor, backbone, head, {
        "checkpoint": checkpoint,
        "target_names": target_names,
        "config": config,
        "model_id": model_id,
        "head_input_dim": int(checkpoint.get("head_input_dim", getattr(backbone.config, "hidden_size", 384))),
        "head_kind": "yolo_dfl_iou_task_aligned",
    }
