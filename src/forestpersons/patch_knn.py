"""Patch-kNN anomaly utilities for frozen DINOv3 ForestPersons features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .dinov3_detector import PATCH_SIZE, PreprocessInfo


@dataclass(frozen=True)
class PatchLabels:
    person_mask: np.ndarray
    visibility: np.ndarray


def scale_box_to_resized(box: list[float], info: PreprocessInfo, margin: float = 0.0) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, x1 * info.scale_x - margin),
        max(0.0, y1 * info.scale_y - margin),
        min(float(info.resized_width), x2 * info.scale_x + margin),
        min(float(info.resized_height), y2 * info.scale_y + margin),
    ]


def patch_boxes(info: PreprocessInfo) -> np.ndarray:
    boxes = []
    for gy in range(info.grid_height):
        for gx in range(info.grid_width):
            x1 = gx * PATCH_SIZE
            y1 = gy * PATCH_SIZE
            boxes.append([x1, y1, x1 + PATCH_SIZE, y1 + PATCH_SIZE])
    return np.asarray(boxes, dtype=np.float32)


def boxes_intersect(boxes: np.ndarray, query: list[float]) -> np.ndarray:
    qx1, qy1, qx2, qy2 = query
    return (
        (boxes[:, 0] < qx2)
        & (boxes[:, 2] > qx1)
        & (boxes[:, 1] < qy2)
        & (boxes[:, 3] > qy1)
    )


def labels_from_ground_truth(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
    margin: float = 0.0,
) -> PatchLabels:
    boxes = patch_boxes(info)
    person_mask = np.zeros((boxes.shape[0],), dtype=bool)
    visibility = np.zeros((boxes.shape[0],), dtype=np.int16)
    for gt in ground_truth:
        scaled = scale_box_to_resized(gt["box"], info, margin=margin)
        overlap = boxes_intersect(boxes, scaled)
        if not overlap.any():
            continue
        gt_visibility = int(gt.get("visibility_level", 0) or 0)
        unset = overlap & (visibility == 0)
        visibility[unset] = gt_visibility
        person_mask |= overlap
    return PatchLabels(
        person_mask=person_mask.reshape(info.grid_height, info.grid_width),
        visibility=visibility.reshape(info.grid_height, info.grid_width),
    )


def background_patch_mask(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
    safety_margin: float,
) -> np.ndarray:
    labels = labels_from_ground_truth(ground_truth, info, margin=safety_margin)
    return ~labels.person_mask


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features.float(), dim=-1)


def cosine_knn_anomaly_scores(
    features: torch.Tensor,
    reference_bank: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    """Return per-patch anomaly scores as 1 - maximum cosine similarity."""

    flat = normalize_features(features).reshape(-1, features.shape[-1])
    bank = normalize_features(reference_bank)
    scores = []
    for start in range(0, flat.shape[0], batch_size):
        chunk = flat[start : start + batch_size]
        similarities = chunk @ bank.T
        nearest = similarities.max(dim=1).values
        scores.append(1.0 - nearest)
    return torch.cat(scores, dim=0).reshape(features.shape[0], features.shape[1])


def binary_classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int | None]:
    labels = labels.astype(bool).reshape(-1)
    scores = scores.astype(np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return {
            "positives": positives,
            "negatives": negatives,
            "auroc": None,
            "average_precision": None,
        }

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, dtype=np.float64)
    fp = np.cumsum(~sorted_labels, dtype=np.float64)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    recall_steps = np.r_[0.0, recall]
    precision_steps = np.r_[1.0, precision]
    average_precision = float(np.sum((recall_steps[1:] - recall_steps[:-1]) * precision_steps[1:]))

    rank_sum = np.flatnonzero(labels[np.argsort(scores, kind="mergesort")]) + 1
    auroc = (rank_sum.sum() - positives * (positives + 1) / 2.0) / (positives * negatives)
    return {
        "positives": positives,
        "negatives": negatives,
        "auroc": float(auroc),
        "average_precision": average_precision,
    }
