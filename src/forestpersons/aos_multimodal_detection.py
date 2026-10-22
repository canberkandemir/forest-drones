"""Shared helpers for the frozen-feature AOS modality comparison."""

from __future__ import annotations

import torch
from torch.nn import functional as F


ARM_NAMES = ("rgb", "thermal", "mean_fusion")
THERMAL_WEIGHTED_ARM_NAMES = ("thermal_weighted_075", "thermal_weighted_090")
SUPPORTED_ARM_NAMES = ARM_NAMES + THERMAL_WEIGHTED_ARM_NAMES


def verify_scene_disjoint(split_scenes: dict[str, set[str]]) -> None:
    """Raise when any official scene is assigned to more than one split."""
    names = sorted(split_scenes)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_scenes[left] & split_scenes[right]
            if overlap:
                raise ValueError(
                    f"Scene leakage between {left} and {right}: {sorted(overlap)}"
                )


def select_evenly(items: list, count: int) -> list:
    """Select deterministic, approximately even positions including endpoints."""

    if count <= 0:
        raise ValueError("count must be positive")
    if len(items) < count:
        raise ValueError(f"cannot select {count} items from {len(items)}")
    if count == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("even selection produced duplicate indices")
    return [items[index] for index in indices]


def feature_for_arm(
    rgb_features: torch.Tensor,
    thermal_features: torch.Tensor,
    arm: str,
) -> torch.Tensor:
    if rgb_features.shape != thermal_features.shape:
        raise ValueError("RGB and thermal feature grids must match")
    if arm == "rgb":
        return rgb_features
    if arm == "thermal":
        return thermal_features
    if arm == "mean_fusion":
        return F.normalize((rgb_features + thermal_features) * 0.5, dim=-1)
    if arm.startswith("thermal_weighted_"):
        suffix = arm.removeprefix("thermal_weighted_")
        try:
            thermal_weight = int(suffix) / 100.0
        except ValueError as error:
            raise ValueError(f"Invalid thermal-weighted arm: {arm}") from error
        if not 0.0 <= thermal_weight <= 1.0:
            raise ValueError(f"Thermal weight must be in [0, 1], got {thermal_weight}")
        fused = (1.0 - thermal_weight) * rgb_features + thermal_weight * thermal_features
        return F.normalize(fused, dim=-1)
    raise ValueError(f"Unknown comparison arm: {arm}")


def yolo_rows_to_ground_truth(
    rows: list[list[float]], width: int = 512, height: int = 512
) -> list[dict]:
    ground_truth = []
    for index, row in enumerate(rows):
        if len(row) != 5 or int(row[0]) != 0:
            raise ValueError(f"Invalid one-class YOLO row: {row}")
        _, xc, yc, box_width, box_height = row
        if not all(0.0 <= value <= 1.0 for value in (xc, yc, box_width, box_height)):
            raise ValueError(f"YOLO values must be normalized: {row}")
        box = [
            (xc - box_width / 2) * width,
            (yc - box_height / 2) * height,
            (xc + box_width / 2) * width,
            (yc + box_height / 2) * height,
        ]
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            raise ValueError(f"Decoded YOLO box leaves image: {box}")
        ground_truth.append(
            {
                "annotation_id": index,
                "box": box,
                "visibility_level": 100,
                "pose": "not_available",
            }
        )
    return ground_truth


def positive_normalized_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = logits.sigmoid()
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.sum() / positive_mask.sum().clamp(min=1).to(loss.dtype)


def fcos_head_loss(
    class_logits: torch.Tensor,
    raw_box: torch.Tensor,
    centerness_logits: torch.Tensor,
    classification_targets: torch.Tensor,
    box_targets: torch.Tensor,
    centerness_targets: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    box_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    classification = positive_normalized_focal_loss(
        class_logits, classification_targets, positive_mask
    )
    if positive_mask.any():
        predicted_distances = F.softplus(raw_box[positive_mask])
        regression = F.smooth_l1_loss(
            torch.log1p(predicted_distances),
            torch.log1p(box_targets[positive_mask]),
            beta=0.25,
        )
        centerness = F.binary_cross_entropy_with_logits(
            centerness_logits[positive_mask], centerness_targets[positive_mask]
        )
    else:
        regression = raw_box.sum() * 0.0
        centerness = centerness_logits.sum() * 0.0
    total = classification + box_weight * regression + centerness
    return total, {
        "total": total,
        "classification": classification,
        "regression": regression,
        "centerness": centerness,
    }
