"""Evaluate frozen YOLO, DINOv3 FCOS, and Selective Fusion B on SARD."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from PIL import Image, ImageDraw

from forestpersons.detection import box_iou
from forestpersons.evaluation import EvaluationGroundTruth, EvaluationPrediction, evaluate_pycocotools
from forestpersons.selective_fusion import MonotonicPrecisionCalibrator, supported_yolo_rescue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _calibrator(data: dict[str, Any]) -> MonotonicPrecisionCalibrator:
    return MonotonicPrecisionCalibrator(
        tuple(data["upper_edges"]), tuple(data["probabilities"]), tuple(data["counts"])
    )


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["predictions"], payload


def _greedy_match(
    gt: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float, iou_threshold: float = 0.5
) -> tuple[set[int], set[int], list[tuple[int, int, float]]]:
    candidates = []
    for gt_index, annotation in enumerate(gt):
        for pred_index, prediction in enumerate(predictions):
            if float(prediction["fused_score"]) < threshold:
                continue
            overlap = box_iou(annotation["bbox_xyxy"], prediction["bbox"])
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_index, pred_index))
    candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches = []
    for overlap, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index); matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, overlap))
    return matched_gt, matched_pred, matches


def _evaluate_fixed(
    images: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float
) -> tuple[dict[str, Any], set[int], dict[int, set[int]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_image[int(prediction["image_id"])].append(prediction)
    tp = fp = fn = 0
    matched_ids: set[int] = set()
    matches_by_image: dict[int, set[int]] = {}
    positive_frame_tp = positive_frame_fp = negative_frame_fp = 0
    size_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for image in images:
        image_id = int(image["image_id"])
        gt = image["person_annotations"]
        image_predictions = by_image.get(image_id, [])
        matched_gt, matched_pred, _ = _greedy_match(gt, image_predictions, threshold)
        filtered = [item for item in image_predictions if float(item["fused_score"]) >= threshold]
        tp += len(matched_gt); fn += len(gt) - len(matched_gt); fp += len(filtered) - len(matched_pred)
        matches_by_image[image_id] = matched_gt
        if gt:
            positive_frame_tp += len(matched_gt)
            positive_frame_fp += len(filtered) - len(matched_pred)
        else:
            negative_frame_fp += len(filtered)
        for gt_index, annotation in enumerate(gt):
            area = float(annotation["area"])
            size = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            size_stats[size]["gt"] += 1
            if gt_index in matched_gt:
                size_stats[size]["tp"] += 1
                matched_ids.add(int(annotation["annotation_id"]))
    return {
        "threshold": threshold,
        "gt": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "fp_per_image": fp / len(images),
        "positive_frame_tp": positive_frame_tp,
        "positive_frame_fp": positive_frame_fp,
        "negative_frame_fp": negative_frame_fp,
        "recall_by_source_box_size": {
            key: {
                **value,
                "recall": value["tp"] / value["gt"] if value["gt"] else 0.0,
            }
            for key, value in sorted(size_stats.items())
        },
    }, matched_ids, matches_by_image


def _coco(images: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    gt = [
        EvaluationGroundTruth(
            image_id=image["image_id"], annotation_id=annotation["annotation_id"],
            box=annotation["bbox_xyxy"], visibility_level=None,
        )
        for image in images for annotation in image["person_annotations"]
    ]
    pred = [
        EvaluationPrediction(
            image_id=item["image_id"], box=item["bbox"], confidence=float(item["fused_score"])
        )
        for item in predictions
    ]
    return evaluate_pycocotools(gt, pred, [image["image_id"] for image in images])


def _save_curve(
    images: list[dict[str, Any]], predictions: list[dict[str, Any]], path: Path
) -> list[dict[str, float]]:
    # A fixed grid is deterministic across methods and avoids an expensive full
    # dataset pass for every distinct floating-point score.
    thresholds = np.linspace(1.0, 0.0, 201).tolist()
    rows = []
    for threshold in thresholds:
        metrics, _, _ = _evaluate_fixed(images, predictions, threshold)
        rows.append({
            "threshold": threshold, "recall": metrics["recall"],
            "precision": metrics["precision"], "fp_per_image": metrics["fp_per_image"],
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def _save_gallery(
    images: list[dict[str, Any]], image_root: Path,
    methods: dict[str, list[dict[str, Any]]], thresholds: dict[str, float],
    matches: dict[str, dict[int, set[int]]], output_dir: Path,
) -> None:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for name, predictions in methods.items():
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            by_image[int(prediction["image_id"])].append(prediction)
        grouped[name] = by_image
    examples = []
    for image in images:
        image_id = int(image["image_id"])
        yolo = matches["yolo"].get(image_id, set())
        dino = matches["dino"].get(image_id, set())
        fusion = matches["fusion_B"].get(image_id, set())
        if (dino - yolo) or (fusion - yolo) or (yolo - dino):
            examples.append(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image in examples[:12]:
        image_id = int(image["image_id"])
        with Image.open(image_root / image["file_name"]) as source:
            canvas = source.convert("RGB")
        canvas.thumbnail((1600, 1000))
        sx, sy = canvas.width / image["width"], canvas.height / image["height"]
        draw = ImageDraw.Draw(canvas)
        for index, annotation in enumerate(image["person_annotations"]):
            x1, y1, x2, y2 = annotation["bbox_xyxy"]
            draw.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline="lime", width=4)
            draw.text((x1*sx, max(0, y1*sy-14)), f"GT {index}", fill="lime")
        colors = {"yolo": "cyan", "dino": "magenta", "fusion_B": "red"}
        for name, predictions in grouped.items():
            for prediction in predictions.get(image_id, []):
                if float(prediction["fused_score"]) < thresholds[name]:
                    continue
                x1, y1, x2, y2 = prediction["bbox"]
                draw.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline=colors[name], width=2)
                draw.text((x1*sx, y2*sy), f"{name} {prediction['fused_score']:.2f}", fill=colors[name])
        canvas.save(output_dir / image["file_name"], quality=90)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest["images"]
    yolo, yolo_payload = _load_predictions(args.yolo)
    dino, dino_payload = _load_predictions(args.dino)
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    yolo_cal = _calibrator(source["calibration"]["yolo"])
    dino_cal = _calibrator(source["calibration"]["dino"])
    op = frozen["operating_points"]
    methods = {
        "yolo": [{**item, "fused_score": item["score"]} for item in yolo],
        "dino": [{**item, "fused_score": item["score"]} for item in dino],
        "fusion_B": supported_yolo_rescue(
            yolo, dino, yolo_cal, dino_cal,
            yolo_keep_threshold=op["selective_B"]["yolo_keep_threshold"],
            yolo_rescue_floor=op["selective_B"]["yolo_rescue_floor"],
            dino_support_threshold=op["selective_B"]["dino_support_threshold"],
            support_iou=op["selective_B"]["support_iou"],
            minimum_area_consistency=op["selective_B"]["minimum_area_consistency"],
            nms_iou=op["selective_B"]["fusion_nms_iou"],
        ),
    }
    thresholds = {
        "yolo": op["yolo"]["confidence_threshold"],
        "dino": 0.55,
        "fusion_B": op["selective_B"]["final_fused_score_threshold"],
    }
    results: dict[str, Any] = {}
    matched_ids: dict[str, set[int]] = {}
    matched_by_image: dict[str, dict[int, set[int]]] = {}
    for name, predictions in methods.items():
        fixed, ids, image_matches = _evaluate_fixed(images, predictions, thresholds[name])
        results[name] = {"fixed_operating_point": fixed, "coco": _coco(images, predictions)}
        matched_ids[name] = ids; matched_by_image[name] = image_matches
        _save_curve(images, predictions, args.output_dir / f"{name}_recall_fp_curve.csv")
        (args.output_dir / f"{name}_predictions.json").write_text(
            json.dumps(predictions, indent=2), encoding="utf-8"
        )
    results["complementarity"] = {
        "both_detect": len(matched_ids["yolo"] & matched_ids["dino"]),
        "yolo_only": len(matched_ids["yolo"] - matched_ids["dino"]),
        "dino_only": len(matched_ids["dino"] - matched_ids["yolo"]),
        "both_miss": manifest["summary"]["person_annotations"] - len(matched_ids["yolo"] | matched_ids["dino"]),
        "fusion_B_rescued_vs_yolo": len(matched_ids["fusion_B"] - matched_ids["yolo"]),
        "yolo_detections_lost_by_fusion_B": len(matched_ids["yolo"] - matched_ids["fusion_B"]),
    }
    summary = {
        "status": "SARD v1.0 mountain scenario zero-shot pilot; frozen ForestPersons models",
        "manifest_summary": manifest["summary"],
        "source_revision": manifest["source_revision"],
        "thresholds_frozen_before_sard": thresholds,
        "annotation_completeness_assumption": (
            "Official release provides one label file for every frame, including empty negatives; "
            "metrics assume those released labels are exhaustive, but no independent exhaustiveness audit exists."
        ),
        "visibility_metrics": "unavailable: SARD has no visibility-level attribute",
        "inference_timing": {"yolo": yolo_payload["timing"], "dino": dino_payload["timing"]},
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_gallery(images, args.image_root, methods, thresholds, matched_by_image, args.output_dir / "gallery")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
