"""Evaluate frozen YOLO, DINOv3 FCOS, and Selective Fusion B on Weitefeld."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from forestpersons.detection import box_iou
from forestpersons.evaluation import EvaluationGroundTruth, EvaluationPrediction, evaluate_pycocotools
from forestpersons.selective_fusion import (
    MonotonicPrecisionCalibrator, supported_yolo_rescue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/weitefeld/images/pilot200"))
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "tiled"), required=True)
    return parser.parse_args()


def calibrator(data: dict[str, Any]) -> MonotonicPrecisionCalibrator:
    return MonotonicPrecisionCalibrator(
        tuple(data["upper_edges"]), tuple(data["probabilities"]), tuple(data["counts"])
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["predictions"]


def greedy_matches(
    gt: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float, iou: float = 0.5
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for gt_index, annotation in enumerate(gt):
        for pred_index, prediction in enumerate(predictions):
            overlap = box_iou(annotation["bbox_xyxy"], prediction["bbox"])
            if overlap >= iou:
                candidates.append((overlap, gt_index, pred_index))
    candidates.sort(reverse=True)
    matched_gt, matched_pred, matches = set(), set(), []
    for overlap, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        if float(predictions[pred_index]["fused_score"]) < threshold:
            continue
        matched_gt.add(gt_index); matched_pred.add(pred_index)
        matches.append((gt_index, pred_index, overlap))
    return matches, matched_gt, matched_pred


def evaluate_method(
    images: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float
) -> tuple[dict[str, Any], set[int], dict[int, set[int]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_image[int(prediction["image_id"])].append(prediction)
    totals = defaultdict(int)
    tiers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matched_annotation_ids: set[int] = set()
    matched_by_image: dict[int, set[int]] = {}
    context_predictions = 0
    track_status: dict[str, bool] = {}
    for image in images:
        image_id = int(image["image_id"])
        gt = image["person_annotations"]
        image_predictions = by_image.get(image_id, [])
        matches, matched_gt, matched_pred = greedy_matches(gt, image_predictions, threshold)
        matched_by_image[image_id] = matched_gt
        filtered_count = sum(float(item["fused_score"]) >= threshold for item in image_predictions)
        totals["gt"] += len(gt); totals["tp"] += len(matched_gt)
        totals["fn"] += len(gt) - len(matched_gt); totals["fp"] += filtered_count - len(matched_pred)
        if image["selection_category"] == "context_unannotated_unknown":
            context_predictions += filtered_count
        for index, annotation in enumerate(gt):
            detected = index in matched_gt
            if detected:
                matched_annotation_ids.add(int(annotation["annotation_id"]))
            tier = annotation["verification_tier"]
            tiers[tier]["gt"] += 1
            tiers[tier]["tp"] += int(detected)
            tiers[tier]["fn"] += int(not detected)
            track_status.setdefault(annotation["track_id"], False)
            track_status[annotation["track_id"]] |= detected
    def summarize(values: dict[str, int]) -> dict[str, Any]:
        gt, tp, fp = values.get("gt", 0), values.get("tp", 0), values.get("fp", 0)
        return {
            **dict(values), "recall": tp / gt if gt else None,
            "precision": tp / (tp + fp) if tp + fp else None,
        }
    result = {
        "threshold": threshold, "overall": summarize(totals),
        "false_positives_per_image_nondefinitive": totals["fp"] / len(images),
        "predictions_on_unannotated_context_images": context_predictions,
        "by_verification_tier": {key: summarize(value) for key, value in sorted(tiers.items())},
        "strip_track_proxy_recall": sum(track_status.values()) / len(track_status) if track_status else None,
        "strip_tracks_detected": sum(track_status.values()), "strip_tracks": len(track_status),
        "false_positive_warning": "labels are non-exhaustive and class 3 includes suspected non-person anomalies",
    }
    return result, matched_annotation_ids, matched_by_image


def coco_metrics(images: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
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
    return evaluate_pycocotools(gt, pred, [item["image_id"] for item in images])


def save_gallery(
    images: list[dict[str, Any]], image_root: Path,
    methods: dict[str, list[dict[str, Any]]], thresholds: dict[str, float],
    matched: dict[str, dict[int, set[int]]], output_dir: Path,
) -> None:
    by_method: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for name, predictions in methods.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            grouped[int(prediction["image_id"])].append(prediction)
        by_method[name] = grouped
    candidates = []
    for image in images:
        image_id = int(image["image_id"])
        yolo = matched["yolo"].get(image_id, set())
        fusion = matched["fusion_B"].get(image_id, set())
        if fusion - yolo:
            candidates.append(image)
    for image in candidates[:8]:
        image_id = int(image["image_id"])
        path = image_root / f"strip{image['strip']}" / image["file_name"]
        with Image.open(path) as source:
            canvas = source.convert("RGB")
        scale = 1600 / canvas.width
        canvas.thumbnail((1600, 1200))
        draw = ImageDraw.Draw(canvas)
        for index, annotation in enumerate(image["person_annotations"]):
            box = [value * scale for value in annotation["bbox_xyxy"]]
            color = "lime" if index in matched["fusion_B"].get(image_id, set()) else "yellow"
            draw.rectangle(box, outline=color, width=4)
            draw.text((box[0], max(0, box[1] - 18)), f"GT {annotation['verification_tier']}", fill=color)
        colors = {"yolo": "cyan", "dino": "magenta", "fusion_B": "red"}
        for name, predictions in by_method.items():
            for prediction in predictions.get(image_id, []):
                if float(prediction["fused_score"]) < thresholds[name]:
                    continue
                box = [value * scale for value in prediction["bbox"]]
                draw.rectangle(box, outline=colors[name], width=2)
                draw.text((box[0], box[3]), f"{name} {prediction['fused_score']:.2f}", fill=colors[name])
        output_dir.mkdir(parents=True, exist_ok=True)
        canvas.save(output_dir / image["file_name"], quality=90)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest["images"]
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    yolo = load_predictions(args.yolo)
    dino = load_predictions(args.dino)
    yolo_cal = calibrator(source["calibration"]["yolo"])
    dino_cal = calibrator(source["calibration"]["dino"])
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
    thresholds = {"yolo": op["yolo"]["confidence_threshold"], "dino": 0.55, "fusion_B": 0.0}
    results, matched, matched_by_image = {}, {}, {}
    for name, predictions in methods.items():
        result, annotation_ids, image_matches = evaluate_method(images, predictions, thresholds[name])
        result["coco_class3_diagnostic"] = coco_metrics(images, predictions)
        results[name] = result; matched[name] = annotation_ids; matched_by_image[name] = image_matches
        (args.output_dir / f"{name}_predictions.json").write_text(
            json.dumps(predictions, indent=2), encoding="utf-8"
        )
    results["fusion_B"]["versus_yolo"] = {
        "class3_annotations_rescued": len(matched["fusion_B"] - matched["yolo"]),
        "yolo_class3_detections_lost": len(matched["yolo"] - matched["fusion_B"]),
        "net_detected_class3_change": len(matched["fusion_B"]) - len(matched["yolo"]),
    }
    summary = {
        "status": "Weitefeld zero-shot pilot; frozen ForestPersons models and thresholds",
        "mode": args.mode, "manifest_summary": manifest["summary"], "thresholds": thresholds,
        "annotation_semantics": {
            "primary": "official crowd-sourced class 3 candidate boxes",
            "not_confirmed_person_ground_truth": True,
            "false_positives_nondefinitive": True,
            "physical_identity": "strip-local contiguous proxy only",
        },
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_gallery(
        images, args.image_root, methods, thresholds, matched_by_image,
        args.output_dir / "fusion_rescue_gallery",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

