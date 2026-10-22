#!/usr/bin/env python3
"""Paired YOLO11s and DINOv3 LoRA FCOS analysis on ForestPersons val600.

The script consumes saved post-NMS COCO predictions only. It never runs model
inference, and therefore cannot diagnose candidates removed before NMS.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, greedy_match, xywh_to_xyxy
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    evaluate_confidence_sweep,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/error_analysis/yolo11s_vs_dinov3_lora_paired_val600_20261015"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/subsets/forestpersons_val_600_seed20261010_manifest.json",
    )
    parser.add_argument(
        "--yolo-predictions",
        type=Path,
        default=ROOT / "outputs/evaluation_framework/val600_yolo11s_finetuned/coco_results.json",
    )
    parser.add_argument(
        "--dino-predictions",
        type=Path,
        default=ROOT
        / "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600/coco_results.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yolo-threshold", type=float, default=0.05)
    parser.add_argument("--dino-threshold", type=float, default=0.55)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--meaningful-iou", type=float, default=0.10)
    parser.add_argument("--review-per-model", type=int, default=16)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_inputs(
    root: Path, manifest_path: Path
) -> tuple[list[Any], dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    dataset = ForestPersonsDataset(root, split="val")
    manifest = read_json(manifest_path)["images"]
    samples_by_id = {int(sample.image_id): sample for sample in dataset.samples}
    manifest_by_id = {int(item["image_id"]): item for item in manifest}
    samples = [samples_by_id[int(item["image_id"])] for item in manifest]
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        meta = manifest_by_id[int(sample.image_id)]
        for annotation in sample.annotations:
            attrs = annotation.get("metadata", {})
            visibility = attrs.get("visibility_level")
            if visibility is None:
                continue
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            max_side = max(width, height)
            if max_side < 96:
                size_bin = "small_max_side_lt96"
            elif max_side < 192:
                size_bin = "medium_max_side_96_192"
            else:
                size_bin = "large_max_side_ge192"
            gt_by_image[int(sample.image_id)].append(
                {
                    "image_id": int(sample.image_id),
                    "annotation_id": int(annotation["id"]),
                    "file_name": sample.file_name,
                    "image_path": str(sample.image_path),
                    "box": [x, y, x + width, y + height],
                    "bbox_width": width,
                    "bbox_height": height,
                    "bbox_area": width * height,
                    "bbox_area_fraction": width * height / (float(meta["width"]) * float(meta["height"])),
                    "size_bin": size_bin,
                    "visibility": int(visibility),
                    "pose": str(attrs.get("pose", "unknown")),
                    "camera": str(attrs.get("camera") or meta.get("camera", "unknown")),
                    "weather": str(attrs.get("weather") or meta.get("weather", "unknown")),
                    "season": str(attrs.get("season") or meta.get("season", "unknown")),
                    "place": str(attrs.get("place") or meta.get("place", "unknown")),
                    "image_width": int(meta["width"]),
                    "image_height": int(meta["height"]),
                }
            )
    return samples, manifest_by_id, gt_by_image


def load_predictions(
    path: Path, image_ids: set[int]
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    rows = read_json(path)
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    categories = Counter()
    invalid_boxes = 0
    for source_index, row in enumerate(rows):
        image_id = int(row["image_id"])
        if image_id not in image_ids:
            continue
        category_id = int(row.get("category_id", 1))
        categories[category_id] += 1
        if category_id != 1:
            continue
        box = xywh_to_xyxy([float(value) for value in row["bbox"]])
        if box[2] <= box[0] or box[3] <= box[1]:
            invalid_boxes += 1
            continue
        by_image[image_id].append(
            {
                "source_index": source_index,
                "image_id": image_id,
                "box": box,
                "score": float(row["score"]),
            }
        )
    for predictions in by_image.values():
        predictions.sort(key=lambda item: item["score"], reverse=True)
    scores = [prediction["score"] for predictions in by_image.values() for prediction in predictions]
    audit = {
        "path": str(path),
        "raw_rows": len(rows),
        "retained_person_rows": len(scores),
        "category_counts": dict(sorted(categories.items())),
        "invalid_boxes": invalid_boxes,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "images_with_predictions": len(by_image),
    }
    return by_image, audit


def match_model(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    image_ids: Iterable[int],
    threshold: float,
    iou_threshold: float,
) -> tuple[dict[int, dict[str, Any]], dict[int, set[int]], dict[int, set[int]]]:
    gt_matches: dict[int, dict[str, Any]] = {}
    matched_gt_indices: dict[int, set[int]] = {}
    matched_prediction_indices: dict[int, set[int]] = {}
    for image_id in image_ids:
        gts = gt_by_image.get(image_id, [])
        predictions = [
            {"box": item["box"], "confidence": item["score"], "_source": item}
            for item in predictions_by_image.get(image_id, [])
            if item["score"] >= threshold
        ]
        matches, matched_gt, matched_pred = greedy_match(gts, predictions, iou_threshold)
        matched_gt_indices[image_id] = matched_gt
        matched_prediction_indices[image_id] = matched_pred
        for match in matches:
            gt = gts[match["gt_index"]]
            prediction = predictions[match["prediction_index"]]
            gt_matches[int(gt["annotation_id"])] = {
                "iou": float(match["iou"]),
                "score": float(prediction["confidence"]),
                "box": prediction["box"],
            }
    return gt_matches, matched_gt_indices, matched_prediction_indices


def best_candidate(gt: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not predictions:
        return None
    prediction = max(predictions, key=lambda item: box_iou(gt["box"], item["box"]))
    return {
        "iou": box_iou(gt["box"], prediction["box"]),
        "score": float(prediction["score"]),
        "box": prediction["box"],
        "source_index": prediction["source_index"],
    }


def miss_reason(
    detected: bool,
    candidate: dict[str, Any] | None,
    threshold: float,
    iou_threshold: float,
    meaningful_iou: float,
) -> str:
    if detected:
        return "detected"
    if candidate is None or candidate["iou"] < meaningful_iou:
        return "no_useful_saved_post_nms_candidate"
    if candidate["iou"] < iou_threshold:
        return "localization_below_iou50"
    if candidate["score"] < threshold:
        return "confidence_below_threshold"
    return "one_to_one_competition"


def build_per_gt_rows(
    gt_by_image: dict[int, list[dict[str, Any]]],
    yolo_predictions: dict[int, list[dict[str, Any]]],
    dino_predictions: dict[int, list[dict[str, Any]]],
    yolo_matches: dict[int, dict[str, Any]],
    dino_matches: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id, gts in gt_by_image.items():
        for gt in gts:
            annotation_id = int(gt["annotation_id"])
            yolo_detected = annotation_id in yolo_matches
            dino_detected = annotation_id in dino_matches
            if yolo_detected and dino_detected:
                outcome = "both"
            elif yolo_detected:
                outcome = "yolo_only"
            elif dino_detected:
                outcome = "dino_only"
            else:
                outcome = "missed_by_both"
            yolo_best = best_candidate(gt, yolo_predictions.get(image_id, []))
            dino_best = best_candidate(gt, dino_predictions.get(image_id, []))
            row = {key: value for key, value in gt.items() if key not in {"box", "image_path"}}
            row.update(
                {
                    "gt_x1": gt["box"][0],
                    "gt_y1": gt["box"][1],
                    "gt_x2": gt["box"][2],
                    "gt_y2": gt["box"][3],
                    "paired_outcome": outcome,
                    "yolo_detected": yolo_detected,
                    "yolo_match_iou": yolo_matches.get(annotation_id, {}).get("iou", ""),
                    "yolo_match_score": yolo_matches.get(annotation_id, {}).get("score", ""),
                    "yolo_best_post_nms_iou": yolo_best["iou"] if yolo_best else 0.0,
                    "yolo_best_post_nms_score": yolo_best["score"] if yolo_best else "",
                    "yolo_miss_reason": miss_reason(
                        yolo_detected,
                        yolo_best,
                        args.yolo_threshold,
                        args.iou_threshold,
                        args.meaningful_iou,
                    ),
                    "dino_detected": dino_detected,
                    "dino_match_iou": dino_matches.get(annotation_id, {}).get("iou", ""),
                    "dino_match_score": dino_matches.get(annotation_id, {}).get("score", ""),
                    "dino_best_post_nms_iou": dino_best["iou"] if dino_best else 0.0,
                    "dino_best_post_nms_score": dino_best["score"] if dino_best else "",
                    "dino_miss_reason": miss_reason(
                        dino_detected,
                        dino_best,
                        args.dino_threshold,
                        args.iou_threshold,
                        args.meaningful_iou,
                    ),
                }
            )
            rows.append(row)
    return rows


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_fields = ("overall", "visibility", "size_bin", "pose", "camera", "weather", "place")
    for field in group_fields:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups["all" if field == "overall" else str(row[field])].append(row)
        for value, group in sorted(groups.items()):
            counts = Counter(row["paired_outcome"] for row in group)
            total = len(group)
            output.append(
                {
                    "group_field": field,
                    "group_value": value,
                    "ground_truth": total,
                    "both": counts["both"],
                    "yolo_only": counts["yolo_only"],
                    "dino_only": counts["dino_only"],
                    "missed_by_both": counts["missed_by_both"],
                    "yolo_recall": (counts["both"] + counts["yolo_only"]) / total,
                    "dino_recall": (counts["both"] + counts["dino_only"]) / total,
                    "oracle_recall": (total - counts["missed_by_both"]) / total,
                }
            )
    return output


def classify_false_positives(
    model: str,
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    matched_prediction_indices: dict[int, set[int]],
    image_ids: list[int],
    threshold: float,
    meaningful_iou: float,
    manifest_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    false_positive_index = 0
    for image_id in image_ids:
        predictions = [item for item in predictions_by_image.get(image_id, []) if item["score"] >= threshold]
        matched = matched_prediction_indices.get(image_id, set())
        gts = gt_by_image.get(image_id, [])
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched:
                continue
            false_positive_index += 1
            nearest = max(gts, key=lambda gt: box_iou(gt["box"], prediction["box"])) if gts else None
            maximum_iou = box_iou(nearest["box"], prediction["box"]) if nearest else 0.0
            if maximum_iou >= 0.50:
                geometric_category = "duplicate_or_one_to_one_competition"
            elif maximum_iou >= meaningful_iou:
                geometric_category = "localization_failure_iou_0.10_to_0.50"
            elif maximum_iou > 0:
                geometric_category = "weak_gt_overlap_below_0.10"
            else:
                geometric_category = "no_gt_overlap"
            box = prediction["box"]
            rows.append(
                {
                    "false_positive_id": f"{model}_{false_positive_index:04d}",
                    "model": model,
                    "image_id": image_id,
                    "file_name": manifest_by_id[image_id]["file_name"],
                    "prediction_source_index": prediction["source_index"],
                    "score": prediction["score"],
                    "pred_x1": box[0],
                    "pred_y1": box[1],
                    "pred_x2": box[2],
                    "pred_y2": box[3],
                    "max_gt_iou": maximum_iou,
                    "nearest_annotation_id": nearest["annotation_id"] if nearest else "",
                    "nearest_visibility": nearest["visibility"] if nearest else "",
                    "geometric_category": geometric_category,
                    "visual_label": "not_manually_reviewed",
                    "visual_note": "",
                }
            )
    return rows


def to_evaluation_inputs(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
) -> tuple[dict[int, list[EvaluationGroundTruth]], dict[int, list[EvaluationPrediction]]]:
    eval_gt = {
        image_id: [
            EvaluationGroundTruth(
                image_id=image_id,
                annotation_id=gt["annotation_id"],
                box=gt["box"],
                visibility_level=gt["visibility"],
                pose=gt["pose"],
            )
            for gt in gts
        ]
        for image_id, gts in gt_by_image.items()
    }
    eval_predictions = {
        image_id: [
            EvaluationPrediction(image_id=image_id, box=item["box"], confidence=item["score"])
            for item in predictions
        ]
        for image_id, predictions in predictions_by_image.items()
    }
    return eval_gt, eval_predictions


def dense_sweep(
    gt_by_image: dict[int, list[dict[str, Any]]],
    predictions_by_image: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
) -> dict[str, Any]:
    eval_gt, eval_predictions = to_evaluation_inputs(gt_by_image, predictions_by_image)
    thresholds = sorted(
        set([0.001, 0.005, 0.01, 0.025] + [round(float(value), 3) for value in np.arange(0.05, 0.951, 0.005)])
    )
    return evaluate_confidence_sweep(
        eval_gt,
        eval_predictions,
        image_ids,
        thresholds,
        iou_threshold=0.50,
        fp_budgets=(0.25, 0.3467, 0.425, 0.50, 1.0),
    )


def sweep_rows(model: str, sweep: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model": model,
            "confidence_threshold": row["confidence_threshold"],
            "recall": row["overall"]["recall"],
            "precision": row["overall"]["precision"],
            "false_positives_per_image": row["false_positives_per_image"],
            **{
                f"recall_visibility_{level}": row["by_visibility"][str(level)]["recall"]
                for level in (20, 40, 70, 100)
            },
        }
        for row in sweep["rows"]
    ]


def matched_budget_rows(sweeps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model, sweep in sweeps.items():
        for budget, row in sweep["matched_false_positive_budgets"].items():
            output.append(
                {
                    "model": model,
                    "fp_budget": float(budget),
                    "confidence_threshold": row["confidence_threshold"],
                    "recall": row["overall"]["recall"],
                    "precision": row["overall"]["precision"],
                    "false_positives_per_image": row["false_positives_per_image"],
                    **{
                        f"recall_visibility_{level}": row["by_visibility"][str(level)]["recall"]
                        for level in (20, 40, 70, 100)
                    },
                }
            )
    return output


def matched_recall_rows(sweeps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for target in (0.70, 0.75, 0.80, 0.85, 0.90, 0.92):
        for model, sweep in sweeps.items():
            eligible = [row for row in sweep["rows"] if row["overall"]["recall"] >= target]
            if not eligible:
                output.append(
                    {
                        "target_recall": target,
                        "model": model,
                        "available": False,
                        "confidence_threshold": "",
                        "actual_recall": "",
                        "precision": "",
                        "false_positives_per_image": "",
                    }
                )
                continue
            row = min(eligible, key=lambda item: (item["false_positives_per_image"], -item["overall"]["precision"]))
            output.append(
                {
                    "target_recall": target,
                    "model": model,
                    "available": True,
                    "confidence_threshold": row["confidence_threshold"],
                    "actual_recall": row["overall"]["recall"],
                    "precision": row["overall"]["precision"],
                    "false_positives_per_image": row["false_positives_per_image"],
                }
            )
    return output


def plot_curves(sweeps: dict[str, dict[str, Any]], output_dir: Path) -> None:
    styles = {"YOLO11s": ("#1677b8", "YOLO11s"), "DINOv3_LoRA_FCOS": ("#d24a38", "DINOv3 LoRA + FCOS")}
    figures = [
        ("precision_recall.png", "recall", "precision", "Recall", "Precision"),
        ("recall_vs_fp_per_image.png", "false_positives_per_image", "recall", "False positives per image", "Recall"),
        ("r20_vs_fp_per_image.png", "false_positives_per_image", "recall_visibility_20", "False positives per image", "Recall at visibility 20%"),
    ]
    rows_by_model = {model: sweep_rows(model, sweep) for model, sweep in sweeps.items()}
    for filename, x_key, y_key, x_label, y_label in figures:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for model, rows in rows_by_model.items():
            color, label = styles[model]
            rows = sorted(rows, key=lambda row: float(row[x_key]))
            ax.plot([row[x_key] for row in rows], [row[y_key] for row in rows], color=color, linewidth=2, label=label)
        if x_key == "false_positives_per_image":
            ax.set_xlim(0, 1.05)
            ax.axvline(0.425, color="black", linestyle="--", linewidth=1, label="0.425 FP/image")
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def choose_review_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    categories = (
        "no_gt_overlap",
        "weak_gt_overlap_below_0.10",
        "localization_failure_iou_0.10_to_0.50",
        "duplicate_or_one_to_one_competition",
    )
    selected: list[dict[str, Any]] = []
    quota = max(1, count // len(categories))
    for category in categories:
        candidates = sorted(
            [row for row in rows if row["geometric_category"] == category],
            key=lambda row: float(row["score"]),
            reverse=True,
        )
        selected.extend(candidates[:quota])
    if len(selected) < count:
        used = {row["false_positive_id"] for row in selected}
        remainder = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
        selected.extend(row for row in remainder if row["false_positive_id"] not in used)
    return selected[:count]


def render_fp_crop(row: dict[str, Any], root: Path, output_path: Path, gt_rows: list[dict[str, Any]]) -> None:
    image = Image.open(root / "images" / row["file_name"]).convert("RGB")
    box = [float(row[key]) for key in ("pred_x1", "pred_y1", "pred_x2", "pred_y2")]
    width = max(box[2] - box[0], 1)
    height = max(box[3] - box[1], 1)
    margin_x = max(width * 1.5, 100)
    margin_y = max(height * 1.5, 100)
    crop_box = [
        max(0, int(box[0] - margin_x)),
        max(0, int(box[1] - margin_y)),
        min(image.width, int(box[2] + margin_x)),
        min(image.height, int(box[3] + margin_y)),
    ]
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop)

    def local(value: list[float]) -> list[float]:
        return [value[0] - crop_box[0], value[1] - crop_box[1], value[2] - crop_box[0], value[3] - crop_box[1]]

    draw.rectangle(local(box), outline="red", width=4)
    for gt in gt_rows:
        gt_box = gt["box"]
        if gt_box[2] >= crop_box[0] and gt_box[0] <= crop_box[2] and gt_box[3] >= crop_box[1] and gt_box[1] <= crop_box[3]:
            draw.rectangle(local(gt_box), outline="lime", width=3)
    label = f"{row['false_positive_id']} {row['geometric_category']} score={float(row['score']):.3f} maxIoU={float(row['max_gt_iou']):.3f}"
    draw.rectangle((0, 0, min(crop.width, 780), 26), fill="black")
    draw.text((4, 5), label, fill="white", font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=92)


def make_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((420, 300))
        canvas = Image.new("RGB", (420, 300), "white")
        canvas.paste(image, ((420 - image.width) // 2, (300 - image.height) // 2))
        thumbs.append(canvas)
    columns = 4
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 420, rows * 300), "white")
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * 420, (index // columns) * 300))
    sheet.save(output_path, quality=92)


def render_paired_gt_crop(
    gt: dict[str, Any],
    outcome: str,
    yolo_candidate: dict[str, Any] | None,
    dino_candidate: dict[str, Any] | None,
    root: Path,
    output_path: Path,
) -> None:
    image = Image.open(root / "images" / gt["file_name"]).convert("RGB")
    boxes = [gt["box"]]
    if yolo_candidate:
        boxes.append(yolo_candidate["box"])
    if dino_candidate:
        boxes.append(dino_candidate["box"])
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    margin_x = max(width, 100)
    margin_y = max(height, 100)
    crop_box = [
        max(0, int(x1 - margin_x)),
        max(0, int(y1 - margin_y)),
        min(image.width, int(x2 + margin_x)),
        min(image.height, int(y2 + margin_y)),
    ]
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop)

    def local(box: list[float]) -> list[float]:
        return [box[0] - crop_box[0], box[1] - crop_box[1], box[2] - crop_box[0], box[3] - crop_box[1]]

    draw.rectangle(local(gt["box"]), outline="lime", width=4)
    if yolo_candidate:
        draw.rectangle(local(yolo_candidate["box"]), outline="magenta", width=3)
    if dino_candidate:
        draw.rectangle(local(dino_candidate["box"]), outline="cyan", width=3)
    label = (
        f"ann={gt['annotation_id']} {outcome} v={gt['visibility']} | "
        f"YOLO magenta IoU={yolo_candidate['iou']:.2f} s={yolo_candidate['score']:.2f} "
        if yolo_candidate
        else f"ann={gt['annotation_id']} {outcome} v={gt['visibility']} | YOLO none "
    )
    label += (
        f"| DINO cyan IoU={dino_candidate['iou']:.2f} s={dino_candidate['score']:.2f}"
        if dino_candidate
        else "| DINO none"
    )
    draw.rectangle((0, 0, min(crop.width, 940), 26), fill="black")
    draw.text((4, 5), label, fill="white", font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=92)


def save_paired_examples(
    per_gt: list[dict[str, Any]],
    gt_by_image: dict[int, list[dict[str, Any]]],
    yolo_predictions: dict[int, list[dict[str, Any]]],
    dino_predictions: dict[int, list[dict[str, Any]]],
    root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    gt_lookup = {
        int(gt["annotation_id"]): gt
        for image_ground_truth in gt_by_image.values()
        for gt in image_ground_truth
    }
    manifest_rows: list[dict[str, Any]] = []
    for outcome in ("yolo_only", "dino_only", "missed_by_both"):
        candidates = sorted(
            [row for row in per_gt if row["paired_outcome"] == outcome],
            key=lambda row: (int(row["visibility"]), float(row["bbox_area"])),
        )[:4]
        paths = []
        for row in candidates:
            annotation_id = int(row["annotation_id"])
            gt = gt_lookup[annotation_id]
            yolo_candidate = best_candidate(gt, yolo_predictions.get(int(gt["image_id"]), []))
            dino_candidate = best_candidate(gt, dino_predictions.get(int(gt["image_id"]), []))
            path = output_dir / outcome / f"ann_{annotation_id}_v{gt['visibility']}.jpg"
            render_paired_gt_crop(gt, outcome, yolo_candidate, dino_candidate, root, path)
            paths.append(path)
            manifest_rows.append(
                {
                    "annotation_id": annotation_id,
                    "outcome": outcome,
                    "visibility": gt["visibility"],
                    "file_name": gt["file_name"],
                    "crop_path": str(path.relative_to(ROOT)),
                }
            )
        make_contact_sheet(paths, output_dir / f"{outcome}_contact_sheet.jpg")
    return manifest_rows


def verify_box_bounds(
    predictions: dict[int, list[dict[str, Any]]], manifest_by_id: dict[int, dict[str, Any]]
) -> dict[str, int]:
    out_of_bounds = 0
    total = 0
    for image_id, image_predictions in predictions.items():
        width = float(manifest_by_id[image_id]["width"])
        height = float(manifest_by_id[image_id]["height"])
        for prediction in image_predictions:
            total += 1
            x1, y1, x2, y2 = prediction["box"]
            if x1 < -1e-4 or y1 < -1e-4 or x2 > width + 1e-4 or y2 > height + 1e-4:
                out_of_bounds += 1
    return {"total": total, "out_of_bounds": out_of_bounds}


def post_nms_overlap_audit(
    predictions: dict[int, list[dict[str, Any]]], nms_iou: float = 0.50
) -> dict[str, Any]:
    maximum_iou = 0.0
    pairs_above_threshold = 0
    for image_predictions in predictions.values():
        for left_index, left in enumerate(image_predictions):
            for right in image_predictions[left_index + 1 :]:
                iou = box_iou(left["box"], right["box"])
                maximum_iou = max(maximum_iou, iou)
                if iou > nms_iou + 1e-6:
                    pairs_above_threshold += 1
    return {
        "expected_nms_iou": nms_iou,
        "maximum_pairwise_iou": maximum_iou,
        "pairs_above_expected_nms_iou": pairs_above_threshold,
    }


def candidate_quality_summary(per_gt: list[dict[str, Any]], model: str) -> dict[str, Any]:
    values = [float(row[f"{model}_best_post_nms_iou"]) for row in per_gt]
    misses = [row for row in per_gt if not bool(row[f"{model}_detected"])]
    return {
        "mean_best_post_nms_iou": float(np.mean(values)),
        "median_best_post_nms_iou": float(np.median(values)),
        "best_iou_ge_0.50": sum(value >= 0.50 for value in values),
        "best_iou_ge_0.75": sum(value >= 0.75 for value in values),
        "operating_point_miss_reasons": dict(
            sorted(Counter(row[f"{model}_miss_reason"] for row in misses).items())
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, manifest_by_id, gt_by_image = load_inputs(args.root, args.manifest)
    image_ids = [int(sample.image_id) for sample in samples]
    image_id_set = set(image_ids)
    yolo_predictions, yolo_audit = load_predictions(args.yolo_predictions, image_id_set)
    dino_predictions, dino_audit = load_predictions(args.dino_predictions, image_id_set)

    yolo_matches, _, yolo_matched_predictions = match_model(
        gt_by_image, yolo_predictions, image_ids, args.yolo_threshold, args.iou_threshold
    )
    dino_matches, _, dino_matched_predictions = match_model(
        gt_by_image, dino_predictions, image_ids, args.dino_threshold, args.iou_threshold
    )
    per_gt = build_per_gt_rows(
        gt_by_image,
        yolo_predictions,
        dino_predictions,
        yolo_matches,
        dino_matches,
        args,
    )
    group_summary = summarize_groups(per_gt)
    write_csv(args.output_dir / "per_gt_paired_comparison.csv", per_gt)
    write_csv(args.output_dir / "paired_group_summary.csv", group_summary)

    yolo_false_positives = classify_false_positives(
        "yolo",
        gt_by_image,
        yolo_predictions,
        yolo_matched_predictions,
        image_ids,
        args.yolo_threshold,
        args.meaningful_iou,
        manifest_by_id,
    )
    dino_false_positives = classify_false_positives(
        "dino",
        gt_by_image,
        dino_predictions,
        dino_matched_predictions,
        image_ids,
        args.dino_threshold,
        args.meaningful_iou,
        manifest_by_id,
    )
    false_positive_rows = [*yolo_false_positives, *dino_false_positives]
    write_csv(args.output_dir / "false_positive_comparison.csv", false_positive_rows)

    sweeps = {
        "YOLO11s": dense_sweep(gt_by_image, yolo_predictions, image_ids),
        "DINOv3_LoRA_FCOS": dense_sweep(gt_by_image, dino_predictions, image_ids),
    }
    curve_rows = [row for model, sweep in sweeps.items() for row in sweep_rows(model, sweep)]
    write_csv(args.output_dir / "confidence_sweep_comparison.csv", curve_rows)
    budget_rows = matched_budget_rows(sweeps)
    write_csv(args.output_dir / "matched_fp_budget_comparison.csv", budget_rows)
    recall_rows = matched_recall_rows(sweeps)
    write_csv(args.output_dir / "matched_recall_comparison.csv", recall_rows)
    plot_curves(sweeps, args.output_dir)

    review_rows = []
    for model, model_rows in (("yolo", yolo_false_positives), ("dino", dino_false_positives)):
        selected = choose_review_rows(model_rows, args.review_per_model)
        paths = []
        for row in selected:
            output_path = args.output_dir / "false_positive_review" / model / f"{row['false_positive_id']}.jpg"
            render_fp_crop(row, args.root, output_path, gt_by_image[int(row["image_id"])])
            paths.append(output_path)
            review_rows.append({**row, "crop_path": str(output_path.relative_to(ROOT))})
        make_contact_sheet(paths, args.output_dir / "false_positive_review" / f"{model}_contact_sheet.jpg")
    write_csv(args.output_dir / "false_positive_review_manifest.csv", review_rows)
    paired_examples = save_paired_examples(
        per_gt,
        gt_by_image,
        yolo_predictions,
        dino_predictions,
        args.root,
        args.output_dir / "paired_examples",
    )
    write_csv(args.output_dir / "paired_examples_manifest.csv", paired_examples)

    outcomes = Counter(row["paired_outcome"] for row in per_gt)
    oracle_by_visibility = {}
    for level in (20, 40, 70, 100):
        group = [row for row in per_gt if row["visibility"] == level]
        oracle_by_visibility[str(level)] = {
            "ground_truth": len(group),
            "detected_by_either": sum(row["paired_outcome"] != "missed_by_both" for row in group),
            "oracle_recall": sum(row["paired_outcome"] != "missed_by_both" for row in group) / len(group),
        }
    yolo_only_reasons = Counter(
        row["dino_miss_reason"] for row in per_gt if row["paired_outcome"] == "yolo_only"
    )
    fp_categories = {
        model: dict(sorted(Counter(row["geometric_category"] for row in rows).items()))
        for model, rows in (("yolo", yolo_false_positives), ("dino", dino_false_positives))
    }
    summary = {
        "scope": {
            "images": len(image_ids),
            "ground_truth": len(per_gt),
            "iou_threshold": args.iou_threshold,
            "yolo_threshold": args.yolo_threshold,
            "dino_threshold": args.dino_threshold,
            "saved_predictions_are_post_nms": True,
        },
        "prediction_audit": {
            "yolo": yolo_audit,
            "dino": dino_audit,
            "yolo_box_bounds": verify_box_bounds(yolo_predictions, manifest_by_id),
            "dino_box_bounds": verify_box_bounds(dino_predictions, manifest_by_id),
            "yolo_post_nms_overlap": post_nms_overlap_audit(yolo_predictions),
            "dino_post_nms_overlap": post_nms_overlap_audit(dino_predictions),
        },
        "candidate_quality": {
            "yolo": candidate_quality_summary(per_gt, "yolo"),
            "dino": candidate_quality_summary(per_gt, "dino"),
        },
        "paired_outcomes": dict(sorted(outcomes.items())),
        "paired_outcome_percentages": {
            key: value / len(per_gt) for key, value in sorted(outcomes.items())
        },
        "yolo_only_dino_failure_reasons": dict(sorted(yolo_only_reasons.items())),
        "false_positive_counts": {"yolo": len(yolo_false_positives), "dino": len(dino_false_positives)},
        "false_positive_geometric_categories": fp_categories,
        "oracle": {
            "detected_by_either": len(per_gt) - outcomes["missed_by_both"],
            "recall": (len(per_gt) - outcomes["missed_by_both"]) / len(per_gt),
            "by_visibility": oracle_by_visibility,
            "note": "Uses ground truth to choose the successful detector per instance; not an achievable fusion metric.",
        },
        "matched_fp_budget_rows": budget_rows,
        "matched_recall_rows": recall_rows,
        "person_absent_images": {
            "verified_local_set_available": False,
            "val600_annotation_free_images": 0,
            "note": "Annotation-free train images are not automatically verified person-absent.",
        },
    }
    with (args.output_dir / "analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
