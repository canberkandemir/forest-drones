#!/usr/bin/env python3
"""Analyze complementary detections and FCOS localization failures on val600.

This analysis consumes existing COCO predictions, the paired per-GT table, and
the preserved DINOv3 FCOS pre-NMS candidate dump. It does not run either model.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, xywh_to_xyxy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRED_DIR = ROOT / "outputs/error_analysis/yolo11s_vs_dinov3_lora_paired_val600_20261015"
DEFAULT_OUTPUT = ROOT / "outputs/error_analysis/dinov3_only_localization_val600_20261015"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/subsets/forestpersons_val_600_seed20261010_manifest.json",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=DEFAULT_PAIRED_DIR / "per_gt_paired_comparison.csv",
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
    parser.add_argument(
        "--dino-pre-nms",
        type=Path,
        default=ROOT
        / "outputs/dinov3_lora_scoring_diagnostic/train400_calib_val600_20261013"
        / "pre_nms_candidates/val600_candidates.jsonl.gz",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--yolo-threshold", type=float, default=0.05)
    parser.add_argument("--dino-threshold", type=float, default=0.55)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--nms-topk", type=int, default=300)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(path: Path, image_ids: set[int]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_index, row in enumerate(read_json(path)):
        image_id = int(row["image_id"])
        if image_id not in image_ids or int(row.get("category_id", 1)) != 1:
            continue
        box = xywh_to_xyxy([float(value) for value in row["bbox"]])
        by_image[image_id].append(
            {
                "source_index": source_index,
                "box": [float(value) for value in box],
                "score": float(row["score"]),
            }
        )
    for predictions in by_image.values():
        predictions.sort(key=lambda item: item["score"], reverse=True)
    scores = [item["score"] for predictions in by_image.values() for item in predictions]
    return by_image, {
        "path": str(path),
        "predictions": len(scores),
        "images": len(by_image),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }


def best_candidate(gt_box: list[float], predictions: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_iou = -1.0
    for prediction in predictions:
        iou = float(box_iou(gt_box, prediction["box"]))
        if iou > best_iou:
            best_iou = iou
            best = {**prediction, "iou": iou}
    return best


def candidate_score(candidate: dict[str, Any]) -> float:
    return math.sqrt(float(candidate["class_score"]) * float(candidate["centerness_score"]))


def box_geometry(gt_box: list[float], pred_box: list[float] | None) -> dict[str, Any]:
    if pred_box is None:
        return {
            "pred_x1": "", "pred_y1": "", "pred_x2": "", "pred_y2": "",
            "center_dx_over_gt_width": "", "center_dy_over_gt_height": "",
            "center_error_gt_normalized": "", "center_error_over_gt_diagonal": "",
            "width_ratio": "", "height_ratio": "", "area_ratio": "",
            "box_scale_tendency": "no_candidate",
        }
    gx1, gy1, gx2, gy2 = gt_box
    px1, py1, px2, py2 = pred_box
    gw, gh = max(gx2 - gx1, 1e-9), max(gy2 - gy1, 1e-9)
    pw, ph = max(px2 - px1, 1e-9), max(py2 - py1, 1e-9)
    dx = ((px1 + px2) - (gx1 + gx2)) / 2.0
    dy = ((py1 + py2) - (gy1 + gy2)) / 2.0
    width_ratio, height_ratio = pw / gw, ph / gh
    area_ratio = width_ratio * height_ratio
    if area_ratio < 0.75:
        tendency = "too_small"
    elif area_ratio > 1.33:
        tendency = "too_large"
    else:
        tendency = "similar_area_or_mixed_dimensions"
    return {
        "pred_x1": px1, "pred_y1": py1, "pred_x2": px2, "pred_y2": py2,
        "center_dx_over_gt_width": dx / gw,
        "center_dy_over_gt_height": dy / gh,
        "center_error_gt_normalized": math.hypot(dx / gw, dy / gh),
        "center_error_over_gt_diagonal": math.hypot(dx, dy) / math.hypot(gw, gh),
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "area_ratio": area_ratio,
        "box_scale_tendency": tendency,
    }


def error_group(
    geometry: dict[str, Any],
    target_iou: float,
    other_gt_max_iou: float,
) -> tuple[str, str]:
    """Apply explicit diagnostic rules; thresholds are not learned from val600."""
    if other_gt_max_iou >= 0.50 and other_gt_max_iou >= target_iou + 0.10:
        return "D_wrong_object_or_ambiguous_matching", "candidate overlaps another annotated GT more strongly"
    center_bad = float(geometry["center_error_gt_normalized"]) > 0.25
    scale_bad = (
        float(geometry["width_ratio"]) < 0.75
        or float(geometry["width_ratio"]) > 1.33
        or float(geometry["height_ratio"]) < 0.75
        or float(geometry["height_ratio"]) > 1.33
    )
    if center_bad and not scale_bad:
        return "A_center_displacement", "normalized center error > 0.25; width and height ratios within [0.75, 1.33]"
    if scale_bad and not center_bad:
        return "B_width_height_scale_mismatch", "center error <= 0.25; width or height ratio outside [0.75, 1.33]"
    if center_bad and scale_bad:
        return "C_mixed_center_and_scale", "center error > 0.25 and width or height ratio outside [0.75, 1.33]"
    return "E_uncertain", "IoU < 0.50 despite neither diagnostic threshold being exceeded"


def load_dataset_rows(root: Path, manifest_path: Path) -> tuple[dict[int, Any], dict[int, list[dict[str, Any]]]]:
    dataset = ForestPersonsDataset(root, split="val")
    manifest = read_json(manifest_path)["images"]
    by_id = {int(sample.image_id): sample for sample in dataset.samples}
    samples = {int(item["image_id"]): by_id[int(item["image_id"])] for item in manifest}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image_id, sample in samples.items():
        for annotation in sample.annotations:
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            gt_by_image[image_id].append(
                {
                    "annotation_id": int(annotation["id"]),
                    "box": [x, y, x + width, y + height],
                }
            )
    return samples, gt_by_image


def enrich_with_pre_nms(
    path: Path,
    gt_lookup: dict[int, list[dict[str, Any]]],
    localization_ids: set[int],
    dino_post: dict[int, list[dict[str, Any]]],
    topk: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    enriched: dict[int, dict[str, Any]] = {}
    score_min = float("inf")
    score_max = 0.0
    candidate_count = 0
    exact_post_matches = 0
    requested_images = {int(gt["image_id"]) for gts in gt_lookup.values() for gt in gts if int(gt["annotation_id"]) in localization_ids}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_id = int(row["image_id"])
            candidates = row["candidates"]
            candidate_count += len(candidates)
            if candidates:
                scores = np.sqrt(
                    np.asarray([item["class_score"] for item in candidates], dtype=np.float64)
                    * np.asarray([item["centerness_score"] for item in candidates], dtype=np.float64)
                )
                score_min = min(score_min, float(scores.min()))
                score_max = max(score_max, float(scores.max()))
            if image_id not in requested_images:
                continue
            boxes = np.asarray([item["box"] for item in candidates], dtype=np.float64)
            scores = np.asarray([candidate_score(item) for item in candidates], dtype=np.float64)
            order = np.argsort(-scores, kind="stable")
            final_predictions = dino_post.get(image_id, [])
            for gt in gt_lookup[image_id]:
                annotation_id = int(gt["annotation_id"])
                if annotation_id not in localization_ids:
                    continue
                gt_box = np.asarray(gt["box"], dtype=np.float64)
                xx1 = np.maximum(boxes[:, 0], gt_box[0])
                yy1 = np.maximum(boxes[:, 1], gt_box[1])
                xx2 = np.minimum(boxes[:, 2], gt_box[2])
                yy2 = np.minimum(boxes[:, 3], gt_box[3])
                inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
                box_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                ious = inter / np.maximum(box_area + gt_area - inter, 1e-12)
                best_index = int(np.argmax(ious))
                best = candidates[best_index]
                rank = int(np.flatnonzero(order == best_index)[0]) + 1
                final_match = best_candidate(best["box"], final_predictions)
                survived = bool(final_match and final_match["iou"] >= 0.999)
                if survived:
                    exact_post_matches += 1
                suppressor = None
                if not survived and rank <= topk:
                    suppressors = []
                    for final_prediction in final_predictions:
                        overlap = box_iou(best["box"], final_prediction["box"])
                        if overlap >= 0.50 and final_prediction["score"] + 1e-6 >= scores[best_index]:
                            suppressors.append({**final_prediction, "suppression_iou": overlap})
                    if suppressors:
                        suppressor = max(suppressors, key=lambda item: item["score"])
                if survived:
                    stage = "survived_post_nms"
                elif rank > topk:
                    stage = "removed_by_topk_before_nms"
                elif suppressor:
                    stage = "removed_by_nms_confirmed"
                else:
                    stage = "unresolved_postprocessing_difference"

                post_best = best_candidate(gt["box"], final_predictions)
                post_components: dict[str, Any] = {
                    "post_nms_component_match_iou": "",
                    "post_nms_best_class_score": "",
                    "post_nms_best_centerness_score": "",
                }
                if post_best:
                    raw_post_match_index = int(
                        np.argmax([box_iou(post_best["box"], item["box"]) for item in candidates])
                    )
                    raw_post_match = candidates[raw_post_match_index]
                    component_match_iou = box_iou(post_best["box"], raw_post_match["box"])
                    post_components = {
                        "post_nms_component_match_iou": component_match_iou,
                        "post_nms_best_class_score": float(raw_post_match["class_score"]),
                        "post_nms_best_centerness_score": float(raw_post_match["centerness_score"]),
                    }
                enriched[annotation_id] = {
                    "pre_nms_best_iou": float(ious[best_index]),
                    "pre_nms_best_score": float(scores[best_index]),
                    "pre_nms_best_class_score": float(best["class_score"]),
                    "pre_nms_best_centerness_score": float(best["centerness_score"]),
                    "pre_nms_best_rank": rank,
                    "pre_nms_best_stage": stage,
                    "pre_nms_best_x1": float(best["box"][0]),
                    "pre_nms_best_y1": float(best["box"][1]),
                    "pre_nms_best_x2": float(best["box"][2]),
                    "pre_nms_best_y2": float(best["box"][3]),
                    "nms_suppressor_score": suppressor["score"] if suppressor else "",
                    "nms_suppressor_overlap_with_best_raw": suppressor["suppression_iou"] if suppressor else "",
                    **post_components,
                }
    return enriched, {
        "path": str(path),
        "candidates": candidate_count,
        "score_min": None if score_min == float("inf") else score_min,
        "score_max": score_max,
        "localization_rows_with_pre_nms": len(enriched),
        "best_raw_candidates_surviving_as_exact_post_nms_boxes": exact_post_matches,
    }


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": "", "median": "", "q25": "", "q75": ""}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def comparative_rows(
    paired_rows: list[dict[str, str]],
    yolo_predictions: dict[int, list[dict[str, Any]]],
    dino_predictions: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_gt: list[dict[str, Any]] = []
    for row in paired_rows:
        gt_box = [float(row[key]) for key in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        image_id = int(row["image_id"])
        output: dict[str, Any] = {
            "image_id": image_id,
            "annotation_id": int(row["annotation_id"]),
            "visibility": int(row["visibility"]),
            "size_bin": row["size_bin"],
            "bbox_area": float(row["bbox_area"]),
        }
        for prefix, predictions in (("yolo", yolo_predictions), ("dino", dino_predictions)):
            candidate = best_candidate(gt_box, predictions.get(image_id, []))
            geometry = box_geometry(gt_box, candidate["box"] if candidate else None)
            output[f"{prefix}_best_iou"] = candidate["iou"] if candidate else 0.0
            output[f"{prefix}_best_score"] = candidate["score"] if candidate else ""
            for key in ("center_error_gt_normalized", "center_error_over_gt_diagonal", "width_ratio", "height_ratio", "area_ratio"):
                output[f"{prefix}_{key}"] = geometry[key]
        per_gt.append(output)

    grouped: list[dict[str, Any]] = []
    group_specs = [("overall", "all", per_gt)]
    for field in ("visibility", "size_bin"):
        for value in sorted({str(row[field]) for row in per_gt}):
            group_specs.append((field, value, [row for row in per_gt if str(row[field]) == value]))
    for group_field, group_value, rows in group_specs:
        for model in ("yolo", "dino"):
            ious = [float(row[f"{model}_best_iou"]) for row in rows]
            center = [float(row[f"{model}_center_error_gt_normalized"]) for row in rows if row[f"{model}_center_error_gt_normalized"] != ""]
            width_log = [abs(math.log(max(float(row[f"{model}_width_ratio"]), 1e-9))) for row in rows if row[f"{model}_width_ratio"] != ""]
            height_log = [abs(math.log(max(float(row[f"{model}_height_ratio"]), 1e-9))) for row in rows if row[f"{model}_height_ratio"] != ""]
            grouped.append(
                {
                    "group_field": group_field,
                    "group_value": group_value,
                    "model": model,
                    "ground_truth": len(rows),
                    "mean_best_iou": float(np.mean(ious)),
                    "median_best_iou": float(np.median(ious)),
                    "best_iou_ge_0.50": sum(value >= 0.50 for value in ious),
                    "best_iou_ge_0.50_fraction": sum(value >= 0.50 for value in ious) / len(rows),
                    "best_iou_ge_0.75": sum(value >= 0.75 for value in ious),
                    "best_iou_ge_0.75_fraction": sum(value >= 0.75 for value in ious) / len(rows),
                    "median_center_error_gt_normalized": median(center),
                    "median_abs_log_width_ratio": median(width_log),
                    "median_abs_log_height_ratio": median(height_log),
                }
            )
    return per_gt, grouped


def characteristic_rows(paired_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_name in ("dino_only", "yolo_only"):
        rows = [row for row in paired_rows if row["paired_outcome"] == group_name]
        for field in ("visibility", "size_bin", "pose", "camera", "weather", "season", "place"):
            counts = Counter(row[field] for row in rows)
            for value, count in sorted(counts.items()):
                output.append(
                    {
                        "paired_outcome": group_name,
                        "characteristic": field,
                        "value": value,
                        "count": count,
                        "fraction_within_outcome": count / len(rows),
                    }
                )
        output.append(
            {
                "paired_outcome": group_name,
                "characteristic": "bbox_area",
                "value": "median_pixels",
                "count": len(rows),
                "fraction_within_outcome": median(float(row["bbox_area"]) for row in rows),
            }
        )
    return output


def crop_bounds(gt_box: list[float], predictions: list[list[float]], width: int, height: int) -> tuple[int, int, int, int]:
    boxes = [gt_box] + predictions
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    span = max(x2 - x1, y2 - y1, 80.0)
    margin = 0.65 * span
    return (
        max(0, int(x1 - margin)),
        max(0, int(y1 - margin)),
        min(width, int(x2 + margin)),
        min(height, int(y2 + margin)),
    )


def draw_panel(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    gt_box: list[float],
    prediction: dict[str, Any] | None,
    model_name: str,
    color: tuple[int, int, int],
    title: str,
    tile_size: tuple[int, int] = (420, 290),
) -> Image.Image:
    font = ImageFont.load_default()
    cropped = image.crop(crop).convert("RGB")
    canvas = Image.new("RGB", tile_size, "white")
    target_h = tile_size[1] - 44
    scale = min(tile_size[0] / cropped.width, target_h / cropped.height)
    resized = cropped.resize((max(1, int(cropped.width * scale)), max(1, int(cropped.height * scale))))
    offset_x = (tile_size[0] - resized.width) // 2
    offset_y = 42
    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)

    def project(box: list[float]) -> tuple[float, float, float, float]:
        return tuple(
            [
                offset_x + (box[0] - crop[0]) * scale,
                offset_y + (box[1] - crop[1]) * scale,
                offset_x + (box[2] - crop[0]) * scale,
                offset_y + (box[3] - crop[1]) * scale,
            ]
        )

    draw.rectangle(project(gt_box), outline=(40, 220, 80), width=3)
    if prediction:
        draw.rectangle(project(prediction["box"]), outline=color, width=3)
        pred_label = f"{model_name} s={prediction['score']:.3f} IoU={prediction['iou']:.3f}"
    else:
        pred_label = f"{model_name}: no saved candidate"
    draw.text((6, 4), title, fill=(0, 0, 0), font=font)
    draw.text((6, 20), pred_label, fill=color, font=font)
    return canvas


def make_gallery(
    path: Path,
    rows: list[dict[str, str]],
    samples: dict[int, Any],
    yolo_predictions: dict[int, list[dict[str, Any]]],
    dino_predictions: dict[int, list[dict[str, Any]]],
    max_cases: int,
) -> list[dict[str, Any]]:
    selected = rows[:max_cases]
    if not selected:
        return []
    case_width, case_height = 840, 290
    sheet = Image.new("RGB", (case_width, case_height * len(selected)), "white")
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        image_id = int(row["image_id"])
        annotation_id = int(row["annotation_id"])
        gt_box = [float(row[key]) for key in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        yolo = best_candidate(gt_box, yolo_predictions.get(image_id, []))
        dino = best_candidate(gt_box, dino_predictions.get(image_id, []))
        image = Image.open(samples[image_id].image_path).convert("RGB")
        pred_boxes = [item["box"] for item in (yolo, dino) if item]
        crop = crop_bounds(gt_box, pred_boxes, image.width, image.height)
        title = f"img {image_id} ann {annotation_id} vis {row['visibility']} {row['paired_outcome']}"
        left = draw_panel(image, crop, gt_box, yolo, "YOLO", (235, 40, 190), title)
        right = draw_panel(image, crop, gt_box, dino, "DINO", (30, 180, 235), title)
        sheet.paste(left, (0, index * case_height))
        sheet.paste(right, (420, index * case_height))
        manifest.append(
            {
                "gallery": path.name,
                "row": index,
                "image_id": image_id,
                "annotation_id": annotation_id,
                "file_name": row["file_name"],
                "visibility": row["visibility"],
                "paired_outcome": row["paired_outcome"],
                "yolo_best_iou": yolo["iou"] if yolo else 0.0,
                "dino_best_iou": dino["iou"] if dino else 0.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)
    return manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_rows = read_csv(args.paired_csv)
    image_ids = {int(row["image_id"]) for row in paired_rows}
    samples, gt_by_image = load_dataset_rows(args.root, args.manifest)
    yolo_predictions, yolo_audit = load_predictions(args.yolo_predictions, image_ids)
    dino_predictions, dino_audit = load_predictions(args.dino_predictions, image_ids)

    dino_only = [row.copy() for row in paired_rows if row["paired_outcome"] == "dino_only"]
    for row in dino_only:
        if row["yolo_miss_reason"] == "confidence_below_threshold":
            row["yolo_failure_interpretation"] = "saved box has IoU >= 0.50 but score < 0.05"
        elif row["yolo_miss_reason"] == "localization_below_iou50":
            row["yolo_failure_interpretation"] = "best saved box overlaps but has IoU < 0.50"
        else:
            row["yolo_failure_interpretation"] = "no useful saved post-NMS candidate"
    write_csv(args.output_dir / "dino_only_diagnostics.csv", dino_only)

    localization_source = [row.copy() for row in paired_rows if row["dino_miss_reason"] == "localization_below_iou50"]
    localization_ids = {int(row["annotation_id"]) for row in localization_source}
    gt_lookup: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image_id, gts in gt_by_image.items():
        for gt in gts:
            gt_lookup[image_id].append({"image_id": image_id, **gt})
    pre_nms, pre_nms_audit = enrich_with_pre_nms(
        args.dino_pre_nms, gt_lookup, localization_ids, dino_predictions, args.nms_topk
    )

    localization_rows: list[dict[str, Any]] = []
    for row in localization_source:
        image_id = int(row["image_id"])
        annotation_id = int(row["annotation_id"])
        gt_box = [float(row[key]) for key in ("gt_x1", "gt_y1", "gt_x2", "gt_y2")]
        candidate = best_candidate(gt_box, dino_predictions.get(image_id, []))
        geometry = box_geometry(gt_box, candidate["box"] if candidate else None)
        other_iou = 0.0
        if candidate:
            other_iou = max(
                [
                    box_iou(candidate["box"], other_gt["box"])
                    for other_gt in gt_by_image[image_id]
                    if int(other_gt["annotation_id"]) != annotation_id
                ]
                or [0.0]
            )
        group, rule = error_group(geometry, candidate["iou"] if candidate else 0.0, float(other_iou))
        output: dict[str, Any] = dict(row)
        output.update(
            {
                "post_nms_best_iou": candidate["iou"] if candidate else 0.0,
                "post_nms_best_score": candidate["score"] if candidate else "",
                **geometry,
                "max_iou_with_other_gt": other_iou,
                "localization_group": group,
                "grouping_rule": rule,
                **pre_nms.get(annotation_id, {}),
            }
        )
        localization_rows.append(output)
    write_csv(args.output_dir / "dino_localization_failures.csv", localization_rows)

    breakdown: list[dict[str, Any]] = []
    for field in ("localization_group", "box_scale_tendency", "visibility", "size_bin", "pre_nms_best_stage"):
        counts = Counter(str(row.get(field, "missing")) for row in localization_rows)
        for value, count in sorted(counts.items()):
            breakdown.append(
                {
                    "breakdown": field,
                    "value": value,
                    "count": count,
                    "fraction": count / len(localization_rows),
                }
            )
    write_csv(args.output_dir / "localization_error_breakdown.csv", breakdown)

    per_gt_localization, comparative = comparative_rows(paired_rows, yolo_predictions, dino_predictions)
    write_csv(args.output_dir / "comparative_per_gt_localization.csv", per_gt_localization)
    write_csv(args.output_dir / "comparative_localization_by_group.csv", comparative)
    characteristics = characteristic_rows(paired_rows)
    write_csv(args.output_dir / "dino_only_vs_yolo_only_characteristics.csv", characteristics)

    gallery_dir = args.output_dir / "gallery"
    gallery_manifest: list[dict[str, Any]] = []
    dino_only_gallery = sorted(dino_only, key=lambda row: (int(row["visibility"]), -float(row["dino_match_iou"])))
    yolo_only_gallery = sorted(
        [row for row in paired_rows if row["paired_outcome"] == "yolo_only"],
        key=lambda row: (int(row["visibility"]), -float(row["yolo_match_iou"])),
    )
    localization_gallery = sorted(
        localization_source,
        key=lambda row: (int(row["visibility"]), float(row["dino_best_post_nms_iou"])),
    )
    ambiguous_ids = {
        int(row["annotation_id"])
        for row in localization_rows
        if row["localization_group"] in {"D_wrong_object_or_ambiguous_matching", "E_uncertain"}
    }
    ambiguous_gallery = [row for row in localization_source if int(row["annotation_id"]) in ambiguous_ids]
    if not ambiguous_gallery:
        ambiguous_gallery = sorted(
            localization_source,
            key=lambda row: float(row["dino_best_post_nms_iou"]),
        )[:12]
    gallery_manifest.extend(make_gallery(gallery_dir / "dino_only_examples.jpg", dino_only_gallery, samples, yolo_predictions, dino_predictions, 10))
    gallery_manifest.extend(make_gallery(gallery_dir / "yolo_only_examples.jpg", yolo_only_gallery, samples, yolo_predictions, dino_predictions, 10))
    gallery_manifest.extend(make_gallery(gallery_dir / "dino_localization_failures.jpg", localization_gallery, samples, yolo_predictions, dino_predictions, 16))
    gallery_manifest.extend(make_gallery(gallery_dir / "ambiguous_human_review.jpg", ambiguous_gallery, samples, yolo_predictions, dino_predictions, 12))
    write_csv(args.output_dir / "gallery_manifest.csv", gallery_manifest)

    loc_group_counts = Counter(row["localization_group"] for row in localization_rows)
    loc_scale_counts = Counter(row["box_scale_tendency"] for row in localization_rows)
    loc_stage_counts = Counter(row["pre_nms_best_stage"] for row in localization_rows)
    good_raw_stage_counts = Counter(
        row["pre_nms_best_stage"]
        for row in localization_rows
        if float(row["pre_nms_best_iou"]) >= args.iou_threshold
    )
    dino_only_yolo_reasons = Counter(row["yolo_miss_reason"] for row in dino_only)
    post_iou = [float(row["post_nms_best_iou"]) for row in localization_rows]
    pre_iou = [float(row["pre_nms_best_iou"]) for row in localization_rows]
    overall_rows = {(row["model"]): row for row in comparative if row["group_field"] == "overall"}
    summary = {
        "scope": {
            "images": len(image_ids),
            "ground_truth": len(paired_rows),
            "dino_only": len(dino_only),
            "dino_localization_failures": len(localization_rows),
            "yolo_threshold": args.yolo_threshold,
            "dino_threshold": args.dino_threshold,
            "iou_threshold": args.iou_threshold,
        },
        "prediction_audit": {"yolo": yolo_audit, "dino": dino_audit, "dino_pre_nms": pre_nms_audit},
        "dino_only": {
            "visibility": dict(sorted(Counter(row["visibility"] for row in dino_only).items())),
            "size": dict(sorted(Counter(row["size_bin"] for row in dino_only).items())),
            "pose": dict(sorted(Counter(row["pose"] for row in dino_only).items())),
            "yolo_failure_reasons": dict(sorted(dino_only_yolo_reasons.items())),
        },
        "localization_failures": {
            "groups": dict(sorted(loc_group_counts.items())),
            "scale_tendency": dict(sorted(loc_scale_counts.items())),
            "best_raw_candidate_postprocessing_stage": dict(sorted(loc_stage_counts.items())),
            "good_raw_iou_ge_0.50_postprocessing_stage": dict(sorted(good_raw_stage_counts.items())),
            "visibility": dict(sorted(Counter(row["visibility"] for row in localization_rows).items())),
            "size": dict(sorted(Counter(row["size_bin"] for row in localization_rows).items())),
            "post_nms_best_iou": describe(post_iou),
            "pre_nms_best_iou": describe(pre_iou),
            "pre_nms_iou_ge_0.50": sum(value >= 0.50 for value in pre_iou),
            "pre_nms_iou_ge_0.75": sum(value >= 0.75 for value in pre_iou),
            "post_nms_iou_ge_0.50": sum(value >= 0.50 for value in post_iou),
            "post_nms_iou_ge_0.75": sum(value >= 0.75 for value in post_iou),
            "median_selected_post_nms_components": {
                "combined_score": median(float(row["post_nms_best_score"]) for row in localization_rows),
                "class_score": median(float(row["post_nms_best_class_score"]) for row in localization_rows),
                "centerness_score": median(float(row["post_nms_best_centerness_score"]) for row in localization_rows),
            },
            "median_best_raw_components": {
                "combined_score": median(float(row["pre_nms_best_score"]) for row in localization_rows),
                "class_score": median(float(row["pre_nms_best_class_score"]) for row in localization_rows),
                "centerness_score": median(float(row["pre_nms_best_centerness_score"]) for row in localization_rows),
            },
        },
        "comparative_overall": overall_rows,
        "grouping_rules": {
            "center_bad": "sqrt((dx/gt_width)^2 + (dy/gt_height)^2) > 0.25",
            "scale_bad": "predicted width or height ratio outside [0.75, 1.33]",
            "wrong_object_or_ambiguous": "candidate IoU with another annotated GT >= 0.50 and >= target IoU + 0.10",
            "uncertain": "target IoU < 0.50 but none of the above diagnostic rules applies",
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
