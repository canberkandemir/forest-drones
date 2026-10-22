"""Utilities for preparing and evaluating the ForestPersons dataset."""

from .dataset import ForestPersonsDataset, ForestPersonsSample, validate_coco_annotations
from .detection import box_iou, greedy_match, summarize_group, xywh_to_xyxy
from .evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_custom_metrics,
    evaluate_pycocotools,
    predictions_to_coco_results,
)

__all__ = [
    "EvaluationGroundTruth",
    "EvaluationPrediction",
    "ForestPersonsDataset",
    "ForestPersonsSample",
    "box_iou",
    "evaluate_coco_style",
    "evaluate_confidence_sweep",
    "evaluate_custom_metrics",
    "evaluate_pycocotools",
    "greedy_match",
    "predictions_to_coco_results",
    "summarize_group",
    "validate_coco_annotations",
    "xywh_to_xyxy",
]
