"""Lightweight RoI refinement for frozen DINOv3 + FCOS proposals."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_iou, nms, roi_align

from .dinov3_detector import PATCH_SIZE, PreprocessInfo


def ground_truth_boxes_resized(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert original-image GT boxes to the resized detector coordinate space."""

    if not ground_truth:
        return torch.empty((0, 4), dtype=torch.float32, device=device)
    boxes = torch.tensor([item["box"] for item in ground_truth], dtype=torch.float32, device=device)
    scale = boxes.new_tensor([info.scale_x, info.scale_y, info.scale_x, info.scale_y])
    boxes = boxes * scale
    boxes[:, 0::2].clamp_(0.0, float(info.resized_width - 1))
    boxes[:, 1::2].clamp_(0.0, float(info.resized_height - 1))
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid]


def decode_fcos_pre_nms_proposals(
    class_logits: torch.Tensor,
    box_raw: torch.Tensor,
    centerness_logits: torch.Tensor,
    info: PreprocessInfo,
    *,
    min_score: float = 0.001,
    pre_nms_topk: int = 1000,
) -> dict[str, torch.Tensor]:
    """Decode FCOS candidates in resized-image coordinates without applying NMS."""

    class_scores = class_logits.sigmoid().reshape(-1)
    centerness_scores = centerness_logits.sigmoid().reshape(-1)
    scores = torch.sqrt((class_scores * centerness_scores).clamp(min=0.0, max=1.0))
    raw = box_raw.reshape(-1, 4)
    indices = torch.nonzero(scores >= min_score, as_tuple=False).flatten()
    if indices.numel() == 0:
        empty_boxes = raw.new_empty((0, 4))
        empty_scores = scores.new_empty((0,))
        return {
            "boxes": empty_boxes,
            "scores": empty_scores,
            "class_scores": empty_scores,
            "centerness_scores": empty_scores,
            "location_indices": indices,
        }

    candidate_scores = scores[indices]
    if pre_nms_topk > 0 and indices.numel() > pre_nms_topk:
        candidate_scores, order = torch.topk(candidate_scores, k=pre_nms_topk)
        indices = indices[order]

    grid_y = torch.div(indices, info.grid_width, rounding_mode="floor").float()
    grid_x = (indices % info.grid_width).float()
    centers_x = (grid_x + 0.5) * PATCH_SIZE
    centers_y = (grid_y + 0.5) * PATCH_SIZE
    distances = F.softplus(raw[indices]).clamp(max=128.0) * PATCH_SIZE
    left, top, right, bottom = distances.unbind(dim=1)
    boxes = torch.stack(
        [
            (centers_x - left).clamp(0, info.resized_width - 1),
            (centers_y - top).clamp(0, info.resized_height - 1),
            (centers_x + right).clamp(0, info.resized_width - 1),
            (centers_y + bottom).clamp(0, info.resized_height - 1),
        ],
        dim=1,
    )
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return {
        "boxes": boxes[valid],
        "scores": candidate_scores[valid],
        "class_scores": class_scores[indices][valid],
        "centerness_scores": centerness_scores[indices][valid],
        "location_indices": indices[valid],
    }


def encode_box_deltas(
    proposals: torch.Tensor,
    targets: torch.Tensor,
    weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
) -> torch.Tensor:
    """Encode target boxes relative to proposals using Faster R-CNN deltas."""

    if proposals.shape != targets.shape or proposals.ndim != 2 or proposals.shape[-1] != 4:
        raise ValueError("proposals and targets must both have shape [N, 4]")
    eps = torch.finfo(proposals.dtype).eps
    proposal_w = (proposals[:, 2] - proposals[:, 0]).clamp(min=eps)
    proposal_h = (proposals[:, 3] - proposals[:, 1]).clamp(min=eps)
    proposal_cx = proposals[:, 0] + 0.5 * proposal_w
    proposal_cy = proposals[:, 1] + 0.5 * proposal_h
    target_w = (targets[:, 2] - targets[:, 0]).clamp(min=eps)
    target_h = (targets[:, 3] - targets[:, 1]).clamp(min=eps)
    target_cx = targets[:, 0] + 0.5 * target_w
    target_cy = targets[:, 1] + 0.5 * target_h
    wx, wy, ww, wh = weights
    return torch.stack(
        [
            wx * (target_cx - proposal_cx) / proposal_w,
            wy * (target_cy - proposal_cy) / proposal_h,
            ww * torch.log(target_w / proposal_w),
            wh * torch.log(target_h / proposal_h),
        ],
        dim=1,
    )


def decode_box_deltas(
    proposals: torch.Tensor,
    deltas: torch.Tensor,
    weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
    max_log_scale: float = math.log(1000.0 / 16.0),
) -> torch.Tensor:
    """Apply predicted center/size deltas to proposal boxes."""

    if proposals.shape != deltas.shape or proposals.ndim != 2 or proposals.shape[-1] != 4:
        raise ValueError("proposals and deltas must both have shape [N, 4]")
    eps = torch.finfo(proposals.dtype).eps
    width = (proposals[:, 2] - proposals[:, 0]).clamp(min=eps)
    height = (proposals[:, 3] - proposals[:, 1]).clamp(min=eps)
    center_x = proposals[:, 0] + 0.5 * width
    center_y = proposals[:, 1] + 0.5 * height
    wx, wy, ww, wh = weights
    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=max_log_scale)
    dh = (deltas[:, 3] / wh).clamp(max=max_log_scale)
    pred_cx = center_x + dx * width
    pred_cy = center_y + dy * height
    pred_w = width * torch.exp(dw)
    pred_h = height * torch.exp(dh)
    return torch.stack(
        [
            pred_cx - 0.5 * pred_w,
            pred_cy - 0.5 * pred_h,
            pred_cx + 0.5 * pred_w,
            pred_cy + 0.5 * pred_h,
        ],
        dim=1,
    )


def clip_boxes_to_resized_image(boxes: torch.Tensor, info: PreprocessInfo) -> torch.Tensor:
    clipped = boxes.clone()
    clipped[:, 0::2].clamp_(0.0, float(info.resized_width - 1))
    clipped[:, 1::2].clamp_(0.0, float(info.resized_height - 1))
    return clipped


def match_and_sample_proposals(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    *,
    positive_iou: float = 0.5,
    negative_iou: float = 0.3,
    samples_per_image: int = 128,
    positive_fraction: float = 0.25,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Match proposals to GT and sample positives/negatives for one image."""

    device = proposals.device
    if proposals.numel() == 0:
        empty_long = torch.empty((0,), dtype=torch.long, device=device)
        return {"indices": empty_long, "labels": empty_long, "matched_gt_indices": empty_long}
    if gt_boxes.numel() == 0:
        labels = torch.zeros((proposals.shape[0],), dtype=torch.long, device=device)
        matched = torch.zeros_like(labels)
    else:
        ious = box_iou(proposals, gt_boxes)
        max_iou, matched = ious.max(dim=1)
        labels = torch.full((proposals.shape[0],), -1, dtype=torch.long, device=device)
        labels[max_iou < negative_iou] = 0
        labels[max_iou >= positive_iou] = 1

    positive = torch.nonzero(labels == 1, as_tuple=False).flatten()
    negative = torch.nonzero(labels == 0, as_tuple=False).flatten()
    max_positive = min(int(samples_per_image * positive_fraction), int(positive.numel()))
    max_negative = min(samples_per_image - max_positive, int(negative.numel()))

    def choose(indices: torch.Tensor, count: int) -> torch.Tensor:
        if count >= indices.numel():
            return indices
        order = torch.randperm(indices.numel(), generator=generator, device="cpu")[:count].to(indices.device)
        return indices[order]

    positive = choose(positive, max_positive)
    negative = choose(negative, max_negative)
    sampled = torch.cat([positive, negative])
    return {
        "indices": sampled,
        "labels": labels[sampled],
        "matched_gt_indices": matched[sampled],
    }


class DINOv3RoIRefinementHead(nn.Module):
    """Small RoIAlign head for scoring and class-agnostic box refinement."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        output_size: int = 3,
        spatial_scale: float = 1.0 / PATCH_SIZE,
    ) -> None:
        super().__init__()
        self.output_size = int(output_size)
        self.spatial_scale = float(spatial_scale)
        roi_dim = 64
        self.roi_encoder = nn.Sequential(
            nn.Conv2d(input_dim, roi_dim, kernel_size=1),
            nn.GroupNorm(16, roi_dim),
            nn.GELU(),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(roi_dim * self.output_size * self.output_size, hidden_dim),
            nn.GELU(),
        )
        self.person_quality = nn.Linear(hidden_dim, 1)
        self.box_deltas = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.box_deltas.weight)
        nn.init.zeros_(self.box_deltas.bias)

    def forward(self, patch_features: torch.Tensor, proposals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_features.ndim == 3:
            feature_map = patch_features.permute(2, 0, 1).unsqueeze(0)
        elif patch_features.ndim == 4:
            feature_map = patch_features
        else:
            raise ValueError("patch_features must have shape [H,W,C] or [N,C,H,W]")
        if feature_map.shape[0] != 1:
            raise ValueError("The initial RoI refiner supports one image per batch")
        if proposals.numel() == 0:
            return feature_map.new_empty((0,)), feature_map.new_empty((0, 4))
        pooled = roi_align(
            feature_map,
            [proposals],
            output_size=(self.output_size, self.output_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
            aligned=True,
        )
        hidden = self.shared(self.roi_encoder(pooled))
        return self.person_quality(hidden).squeeze(1), self.box_deltas(hidden)


def roi_refinement_loss(
    score_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    sampled_proposals: torch.Tensor,
    labels: torch.Tensor,
    matched_gt_boxes: torch.Tensor,
    *,
    box_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if score_logits.numel() == 0:
        zero = box_deltas.sum() * 0.0
        return zero, {"classification": 0.0, "box": 0.0, "positive": 0, "negative": 0}
    classification = F.binary_cross_entropy_with_logits(score_logits, labels.float())
    positive = labels == 1
    if positive.any():
        targets = encode_box_deltas(sampled_proposals[positive], matched_gt_boxes[positive])
        box_loss = F.smooth_l1_loss(box_deltas[positive], targets, beta=1.0 / 9.0)
    else:
        box_loss = box_deltas.sum() * 0.0
    total = classification + float(box_loss_weight) * box_loss
    return total, {
        "classification": float(classification.detach().cpu()),
        "box": float(box_loss.detach().cpu()),
        "positive": int(positive.sum().item()),
        "negative": int((labels == 0).sum().item()),
    }


def refine_and_nms(
    proposals: torch.Tensor,
    proposal_scores: torch.Tensor,
    score_logits: torch.Tensor,
    box_deltas: torch.Tensor,
    info: PreprocessInfo,
    *,
    score_mode: str = "geometric_mean",
    confidence_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    detections_per_image: int = 100,
) -> list[dict[str, Any]]:
    """Decode refinements and return boxes in original-image coordinates."""

    quality = score_logits.sigmoid()
    if score_mode == "quality":
        scores = quality
    elif score_mode == "proposal":
        scores = proposal_scores
    elif score_mode == "multiply":
        scores = proposal_scores * quality
    elif score_mode == "geometric_mean":
        scores = torch.sqrt((proposal_scores * quality).clamp(min=0.0, max=1.0))
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")
    boxes = clip_boxes_to_resized_image(decode_box_deltas(proposals, box_deltas), info)
    valid = (
        (scores >= confidence_threshold)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes = boxes[valid]
    scores = scores[valid]
    if boxes.numel() == 0:
        return []
    keep = nms(boxes, scores, nms_threshold)[:detections_per_image]
    boxes = boxes[keep]
    scores = scores[keep]
    scale = boxes.new_tensor([info.scale_x, info.scale_y, info.scale_x, info.scale_y])
    boxes = boxes / scale
    results: list[dict[str, Any]] = []
    for box, score in zip(boxes.detach().cpu(), scores.detach().cpu()):
        results.append({"box": [float(value) for value in box], "confidence": float(score)})
    return results


def save_roi_refiner_checkpoint(
    path: Path,
    head: DINOv3RoIRefinementHead,
    *,
    source_detector_checkpoint: Path,
    settings: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "dinov3_fcos_roi_refiner",
            "source_detector_checkpoint": str(source_detector_checkpoint),
            "head_state": head.state_dict(),
            "head_config": {
                "input_dim": head.roi_encoder[0].in_channels,
                "hidden_dim": head.shared[1].out_features,
                "output_size": head.output_size,
                "spatial_scale": head.spatial_scale,
            },
            "settings": settings,
            "history": history or [],
        },
        path,
    )


def load_roi_refiner_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[DINOv3RoIRefinementHead, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "dinov3_fcos_roi_refiner":
        raise ValueError(f"Unsupported RoI refiner checkpoint: {checkpoint.get('architecture')!r}")
    config = checkpoint["head_config"]
    head = DINOv3RoIRefinementHead(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_size=int(config["output_size"]),
        spatial_scale=float(config["spatial_scale"]),
    ).to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    return head, checkpoint
