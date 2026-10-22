"""Evaluation utilities for ForestPersons object detection experiments.

The module intentionally keeps two metric families separate:

* COCO-style AP/AR ranks detections by confidence and averages over IoU
  thresholds. This is the closest local equivalent to the ForestPersons paper
  protocol when pycocotools is not installed.
* Custom SAR metrics apply an explicit confidence threshold and report recall,
  precision, false positives per image, and recall by visibility level.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .detection import box_iou, evaluate_image_matches, safe_div, summarize_group, xywh_to_xyxy


DEFAULT_IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 0.96, 0.05))
DEFAULT_RECALL_THRESHOLDS = tuple(float(value) for value in np.linspace(0.0, 1.0, 101))
DEFAULT_MAX_DETS = (1, 10, 100)
PERSON_CATEGORY_ID = 1


@dataclass(frozen=True)
class EvaluationGroundTruth:
    image_id: int | str
    annotation_id: int | str | None
    box: list[float]
    visibility_level: int | None = None
    pose: str | None = None
    category_id: int = PERSON_CATEGORY_ID


@dataclass(frozen=True)
class EvaluationPrediction:
    image_id: int | str
    box: list[float]
    confidence: float
    category_id: int = PERSON_CATEGORY_ID


def coco_bbox_from_xyxy(box: list[float] | tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_xyxy_box(
    box: list[float] | tuple[float, float, float, float],
    width: int | float,
    height: int | float,
) -> list[float] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    clipped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def annotations_to_eval_ground_truth(
    image_id: int | str,
    annotations: Iterable[dict[str, Any]],
) -> list[EvaluationGroundTruth]:
    ground_truth: list[EvaluationGroundTruth] = []
    for annotation in annotations:
        metadata = annotation.get("metadata", {})
        visibility = metadata.get("visibility_level")
        if visibility is None:
            continue
        ground_truth.append(
            EvaluationGroundTruth(
                image_id=image_id,
                annotation_id=annotation.get("id"),
                box=xywh_to_xyxy([float(value) for value in annotation["bbox"]]),
                visibility_level=int(visibility),
                pose=str(metadata.get("pose", "unknown")),
            )
        )
    return ground_truth


def predictions_to_coco_results(
    predictions: Iterable[EvaluationPrediction],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for prediction in predictions:
        results.append(
            {
                "image_id": prediction.image_id,
                "category_id": prediction.category_id,
                "bbox": coco_bbox_from_xyxy(prediction.box),
                "score": float(prediction.confidence),
            }
        )
    return results


def ground_truth_to_coco_dataset(
    ground_truth: Iterable[EvaluationGroundTruth],
    image_ids: Iterable[int | str],
) -> dict[str, Any]:
    annotations = []
    for index, gt in enumerate(ground_truth, start=1):
        bbox = coco_bbox_from_xyxy(gt.box)
        annotations.append(
            {
                "id": int(gt.annotation_id) if isinstance(gt.annotation_id, int) else index,
                "image_id": gt.image_id,
                "category_id": gt.category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
        )
    return {
        "info": {"description": "ForestPersons one-class evaluation subset"},
        "licenses": [],
        "images": [{"id": image_id} for image_id in image_ids],
        "annotations": annotations,
        "categories": [{"id": PERSON_CATEGORY_ID, "name": "person"}],
    }


def evaluate_pycocotools(
    ground_truth: list[EvaluationGroundTruth],
    predictions: list[EvaluationPrediction],
    image_ids: list[int | str],
    max_dets: tuple[int, ...] = DEFAULT_MAX_DETS,
) -> dict[str, Any]:
    """Evaluate with pycocotools COCOeval when that dependency is available."""

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:  # pragma: no cover - exercised only without optional dep.
        return {
            "available": False,
            "error": repr(exc),
            "unsupported_without_pycocotools": [
                "Official COCOeval area-range summaries.",
                "Official ignore/crowd handling.",
                "Exact pycocotools precision tensor reproduction.",
            ],
        }

    coco_gt = COCO()
    coco_gt.dataset = ground_truth_to_coco_dataset(ground_truth, image_ids)
    coco_gt.createIndex()
    coco_results = predictions_to_coco_results(predictions)
    if coco_results:
        coco_dt = coco_gt.loadRes(coco_results)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "info": coco_gt.dataset["info"],
            "licenses": [],
            "images": coco_gt.dataset["images"],
            "annotations": [],
            "categories": coco_gt.dataset["categories"],
        }
        coco_dt.createIndex()
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(image_ids)
    coco_eval.params.catIds = [PERSON_CATEGORY_ID]
    coco_eval.params.maxDets = list(max_dets)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = [float(value) for value in coco_eval.stats.tolist()]
    return {
        "available": True,
        "metric_family": "pycocotools_cocoeval_bbox",
        "max_dets": list(max_dets),
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR@1": stats[6],
        "AR@10": stats[7],
        "AR@100": stats[8],
        "AR_small": stats[9],
        "AR_medium": stats[10],
        "AR_large": stats[11],
    }


def _ap_from_precision_recall(
    recall: np.ndarray,
    precision: np.ndarray,
    recall_thresholds: tuple[float, ...],
) -> float:
    if recall.size == 0:
        return 0.0
    precision_envelope = precision.copy()
    for index in range(precision_envelope.size - 2, -1, -1):
        precision_envelope[index] = max(precision_envelope[index], precision_envelope[index + 1])
    sampled: list[float] = []
    for threshold in recall_thresholds:
        valid = np.where(recall >= threshold)[0]
        sampled.append(float(precision_envelope[valid[0]]) if valid.size else 0.0)
    return float(np.mean(sampled))


def _evaluate_at_iou(
    ground_truth: list[EvaluationGroundTruth],
    predictions: list[EvaluationPrediction],
    image_ids: list[int | str],
    iou_threshold: float,
    max_det: int,
    recall_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    gt_by_image: dict[int | str, list[EvaluationGroundTruth]] = defaultdict(list)
    pred_by_image: dict[int | str, list[EvaluationPrediction]] = defaultdict(list)
    for gt in ground_truth:
        gt_by_image[gt.image_id].append(gt)
    for prediction in predictions:
        pred_by_image[prediction.image_id].append(prediction)

    detections: list[tuple[float, int, int]] = []
    total_gt = len(ground_truth)
    for image_id in image_ids:
        image_gt = gt_by_image.get(image_id, [])
        image_predictions = sorted(
            pred_by_image.get(image_id, []),
            key=lambda item: item.confidence,
            reverse=True,
        )[:max_det]
        matched_gt: set[int] = set()
        for prediction in image_predictions:
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, gt in enumerate(image_gt):
                if gt_index in matched_gt:
                    continue
                iou = box_iou(gt.box, prediction.box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            is_tp = int(best_gt_index >= 0 and best_iou >= iou_threshold)
            if is_tp:
                matched_gt.add(best_gt_index)
            detections.append((prediction.confidence, is_tp, 1 - is_tp))

    if total_gt == 0:
        return {"ap": 0.0, "ar": 0.0, "total_gt": 0}
    if not detections:
        return {"ap": 0.0, "ar": 0.0, "total_gt": total_gt}

    detections.sort(reverse=True, key=lambda item: item[0])
    tp = np.cumsum([item[1] for item in detections], dtype=np.float64)
    fp = np.cumsum([item[2] for item in detections], dtype=np.float64)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return {
        "ap": _ap_from_precision_recall(recall, precision, recall_thresholds),
        "ar": float(recall[-1]) if recall.size else 0.0,
        "total_gt": total_gt,
    }


def evaluate_coco_style(
    ground_truth: list[EvaluationGroundTruth],
    predictions: list[EvaluationPrediction],
    image_ids: list[int | str],
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    recall_thresholds: tuple[float, ...] = DEFAULT_RECALL_THRESHOLDS,
    max_dets: tuple[int, ...] = DEFAULT_MAX_DETS,
) -> dict[str, Any]:
    """Evaluate one-class detections with COCO-style AP/AR.

    This implementation follows the central COCO matching behavior for one
    category: detections are sorted by confidence, each ground-truth box can be
    matched once per IoU threshold, AP is averaged over 101 recall thresholds,
    and AR is averaged over IoU thresholds at fixed maxDets.
    """

    per_max_det: dict[str, Any] = {}
    for max_det in max_dets:
        per_iou: dict[str, Any] = {}
        aps: list[float] = []
        ars: list[float] = []
        for iou_threshold in iou_thresholds:
            result = _evaluate_at_iou(
                ground_truth,
                predictions,
                image_ids,
                iou_threshold,
                max_det,
                recall_thresholds,
            )
            per_iou[f"{iou_threshold:.2f}"] = result
            aps.append(float(result["ap"]))
            ars.append(float(result["ar"]))
        per_max_det[str(max_det)] = {
            "AP": float(np.mean(aps)) if aps else 0.0,
            "AR": float(np.mean(ars)) if ars else 0.0,
            "per_iou": per_iou,
        }

    primary_max_det = max(max_dets)
    primary = per_max_det[str(primary_max_det)]
    return {
        "metric_family": "local_coco_style_one_class",
        "category_id": PERSON_CATEGORY_ID,
        "iou_thresholds": list(iou_thresholds),
        "recall_threshold_count": len(recall_thresholds),
        "max_dets": list(max_dets),
        "primary_max_det": primary_max_det,
        "AP": primary["AP"],
        "AP50": primary["per_iou"]["0.50"]["ap"],
        "AP75": primary["per_iou"]["0.75"]["ap"],
        "AR": primary["AR"],
        "AR_by_max_dets": {
            str(max_det): per_max_det[str(max_det)]["AR"] for max_det in max_dets
        },
        "per_max_det": per_max_det,
    }


def evaluate_custom_metrics(
    ground_truth_by_image: dict[int | str, list[EvaluationGroundTruth]],
    predictions_by_image: dict[int | str, list[EvaluationPrediction]],
    image_ids: list[int | str],
    confidence_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    overall_stats = defaultdict(int)
    visibility_stats: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    predictions_total = 0

    for image_id in image_ids:
        ground_truth = [
            {
                "box": gt.box,
                "visibility_level": gt.visibility_level,
                "pose": gt.pose,
                "annotation_id": gt.annotation_id,
            }
            for gt in ground_truth_by_image.get(image_id, [])
            if gt.visibility_level is not None
        ]
        predictions = [
            {"box": pred.box, "confidence": pred.confidence}
            for pred in predictions_by_image.get(image_id, [])
            if pred.confidence >= confidence_threshold
        ]
        predictions_total += len(predictions)
        _, _, _, overall, by_visibility = evaluate_image_matches(
            ground_truth,
            predictions,
            iou_threshold,
        )
        for key, value in overall.items():
            overall_stats[key] += value
        for visibility, stats in by_visibility.items():
            for key, value in stats.items():
                visibility_stats[visibility][key] += value

    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "predictions": predictions_total,
        "overall": summarize_group(overall_stats),
        "false_positives_per_image": safe_div(overall_stats["false_positives"], len(image_ids)),
        "by_visibility": {
            str(visibility): summarize_group(visibility_stats[visibility])
            for visibility in sorted(visibility_stats)
        },
    }


def evaluate_confidence_sweep(
    ground_truth_by_image: dict[int | str, list[EvaluationGroundTruth]],
    predictions_by_image: dict[int | str, list[EvaluationPrediction]],
    image_ids: list[int | str],
    confidence_thresholds: Iterable[float],
    iou_threshold: float = 0.50,
    fp_budgets: Iterable[float] = (0.425,),
) -> dict[str, Any]:
    rows = [
        evaluate_custom_metrics(
            ground_truth_by_image,
            predictions_by_image,
            image_ids,
            confidence_threshold=float(threshold),
            iou_threshold=iou_threshold,
        )
        for threshold in confidence_thresholds
    ]
    if rows:
        max_recall_row = max(rows, key=lambda row: row["overall"]["recall"])
    else:
        max_recall_row = None

    matched_budgets: dict[str, Any] = {}
    for budget in fp_budgets:
        eligible = [row for row in rows if row["false_positives_per_image"] <= budget]
        chosen = max(eligible, key=lambda row: row["overall"]["recall"]) if eligible else None
        matched_budgets[str(budget)] = chosen

    return {
        "definition": {
            "custom_metrics": "Explicit confidence-thresholded one-to-one greedy matching.",
            "maximum_recall": (
                "Maximum custom recall over the evaluated confidence thresholds at "
                f"IoU {iou_threshold:.2f}; it is not COCO AR."
            ),
            "matched_fp_budget": (
                "For each FP/image budget, choose the threshold with the highest recall "
                "among rows whose FP/image is less than or equal to the budget."
            ),
        },
        "rows": rows,
        "maximum_recall": max_recall_row,
        "matched_false_positive_budgets": matched_budgets,
    }
