"""Shared detection utilities for ForestPersons experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image


def xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    x, y, width, height = box
    return [float(x), float(y), float(x + width), float(y + height)]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def greedy_match(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            iou = box_iou(gt["box"], pred["box"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append(
            {
                "gt_index": gt_index,
                "prediction_index": pred_index,
                "iou": iou,
                "visibility_level": ground_truth[gt_index].get("visibility_level"),
                "confidence": predictions[pred_index].get("confidence", 0.0),
            }
        )
    return matches, matched_gt, matched_pred


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_group(stats: dict[str, int]) -> dict[str, float | int]:
    true_positives = int(stats["true_positives"])
    false_positives = int(stats["false_positives"])
    false_negatives = int(stats["false_negatives"])
    ground_truth = true_positives + false_negatives
    return {
        "ground_truth": ground_truth,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "recall": safe_div(true_positives, ground_truth),
        "precision": safe_div(true_positives, true_positives + false_positives),
    }


def evaluate_image_matches(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], set[int], set[int], dict[str, int], dict[int, dict[str, int]]]:
    matches, matched_gt, matched_pred = greedy_match(ground_truth, predictions, iou_threshold)
    overall = defaultdict(int)
    by_visibility: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    overall["true_positives"] += len(matches)
    overall["false_positives"] += len(predictions) - len(matched_pred)
    overall["false_negatives"] += len(ground_truth) - len(matched_gt)

    for gt_index, gt in enumerate(ground_truth):
        visibility = int(gt["visibility_level"])
        if gt_index in matched_gt:
            by_visibility[visibility]["true_positives"] += 1
        else:
            by_visibility[visibility]["false_negatives"] += 1

    for pred_index, pred in enumerate(predictions):
        if pred_index in matched_pred or not ground_truth:
            continue
        best_gt = max(ground_truth, key=lambda gt: box_iou(gt["box"], pred["box"]))
        by_visibility[int(best_gt["visibility_level"])]["false_positives"] += 1

    return matches, matched_gt, matched_pred, overall, by_visibility


def draw_detection_comparison(
    image_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    output_path: Path,
    prediction_label: str,
) -> None:
    matched_gt = {match["gt_index"] for match in matches}
    matched_pred = {match["prediction_index"] for match in matches}

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(image)
    ax.set_title(image_path.name, fontsize=9)
    ax.axis("off")

    for index, gt in enumerate(ground_truth):
        x1, y1, x2, y2 = gt["box"]
        color = "lime" if index in matched_gt else "yellow"
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2.0,
                edgecolor=color,
                facecolor="none",
            )
        )
        ax.text(
            x1,
            max(0, y1 - 4),
            f"GT v={gt.get('visibility_level')} {gt.get('pose', 'unknown')}",
            color="black",
            fontsize=7,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1},
        )

    for index, pred in enumerate(predictions):
        x1, y1, x2, y2 = pred["box"]
        color = "cyan" if index in matched_pred else "red"
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.6,
                edgecolor=color,
                linestyle="--",
                facecolor="none",
            )
        )
        ax.text(
            x1,
            min(image.height - 4, y2 + 4),
            f"{prediction_label} {pred.get('confidence', 0.0):.2f}",
            color="black",
            fontsize=7,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 1},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
