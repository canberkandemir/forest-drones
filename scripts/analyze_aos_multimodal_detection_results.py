#!/usr/bin/env python3
"""Offline thermal-versus-mean-fusion analysis for the completed AOS pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forestpersons.detection import box_iou, greedy_match  # noqa: E402


DEFAULT_EXPERIMENT = ROOT / "outputs/aos/dinov3_rgb_thermal_detection_20261020"
DEFAULT_OUTPUT = DEFAULT_EXPERIMENT / "analysis_thermal_vs_mean_fusion_20261020"
DEFAULT_REPORT = ROOT / "docs/reports/aos_thermal_vs_mean_fusion_analysis_2026-10-20.md"


def yolo_box(row: list[float], width: int, height: int) -> list[float]:
    _, xc, yc, box_width, box_height = row
    return [
        (xc - box_width / 2) * width,
        (yc - box_height / 2) * height,
        (xc + box_width / 2) * width,
        (yc + box_height / 2) * height,
    ]


def load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(path.read_text()):
        grouped[int(row["image_id"])].append(
            {"box": [float(value) for value in row["bbox_xyxy"]], "confidence": float(row["score"])}
        )
    for predictions in grouped.values():
        predictions.sort(key=lambda item: item["confidence"], reverse=True)
    return grouped


def best_candidate(gt_box: list[float], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {
            "iou": 0.0,
            "confidence": None,
            "box": None,
            "center_error_px": None,
            "center_error_gt_diagonal": None,
            "width_ratio": None,
            "height_ratio": None,
            "area_ratio": None,
        }
    candidate = max(predictions, key=lambda pred: box_iou(gt_box, pred["box"]))
    pred_box = candidate["box"]
    gt_width, gt_height = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
    pred_width, pred_height = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]
    center_error = math.hypot(
        (pred_box[0] + pred_box[2] - gt_box[0] - gt_box[2]) / 2,
        (pred_box[1] + pred_box[3] - gt_box[1] - gt_box[3]) / 2,
    )
    diagonal = math.hypot(gt_width, gt_height)
    return {
        "iou": box_iou(gt_box, pred_box),
        "confidence": candidate["confidence"],
        "box": pred_box,
        "center_error_px": center_error,
        "center_error_gt_diagonal": center_error / diagonal if diagonal else None,
        "width_ratio": pred_width / gt_width if gt_width else None,
        "height_ratio": pred_height / gt_height if gt_height else None,
        "area_ratio": (pred_width * pred_height) / (gt_width * gt_height) if gt_width * gt_height else None,
    }


def match_at_threshold(
    boxes: list[list[float]], predictions: list[dict[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], set[int], set[int], list[dict[str, Any]]]:
    selected = [prediction for prediction in predictions if prediction["confidence"] >= threshold]
    ground_truth = [{"box": box} for box in boxes]
    matches, matched_gt, matched_pred = greedy_match(ground_truth, selected, 0.5)
    return matches, matched_gt, matched_pred, selected


def draw_box(ax: Any, box: list[float], color: str, label: str, linewidth: float = 1.5) -> None:
    x1, y1, x2, y2 = box
    ax.add_patch(
        patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth
        )
    )
    ax.text(
        x1,
        max(2, y1 - 3),
        label,
        color="black",
        fontsize=6,
        bbox={"facecolor": color, "alpha": 0.8, "pad": 1, "edgecolor": "none"},
    )


def render_scene(
    record: dict[str, Any],
    boxes: list[list[float]],
    thermal_predictions: list[dict[str, Any]],
    fusion_predictions: list[dict[str, Any]],
    thermal_threshold: float,
    fusion_threshold: float,
    lost_gt: set[int],
    output_path: Path,
) -> None:
    rgb = Image.open(record["rgb_path"]).convert("RGB")
    thermal = Image.open(record["thermal_path"]).convert("RGB")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = (
        (axes[0], rgb, [], "RGB context + GT"),
        (
            axes[1],
            thermal,
            [pred for pred in thermal_predictions if pred["confidence"] >= thermal_threshold],
            f"Thermal detections (score >= {thermal_threshold:.2f})",
        ),
        (
            axes[2],
            rgb,
            [pred for pred in fusion_predictions if pred["confidence"] >= fusion_threshold],
            f"Mean-fusion detections (score >= {fusion_threshold:.2f})",
        ),
    )
    for axis, image, predictions, title in panels:
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        for gt_index, box in enumerate(boxes):
            color = "magenta" if gt_index in lost_gt else "lime"
            draw_box(axis, box, color, f"GT{gt_index}{' LOST' if gt_index in lost_gt else ''}", 2.0)
        for prediction in predictions:
            draw_box(axis, prediction["box"], "cyan", f"P {prediction['confidence']:.2f}")
    figure.suptitle(f"{record['scene']} | image_id={record['image_id']}", fontsize=11)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    experiment = args.experiment_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty analysis directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    overlays = output / "overlays"
    overlays.mkdir()

    manifest = json.loads((experiment / "prepared/manifest.json").read_text())
    metrics = {row["arm"]: row for row in json.loads((experiment / "evaluation/metrics.json").read_text())}
    thermal_threshold = float(metrics["thermal"]["threshold_selected_on_valid_o6"])
    fusion_threshold = float(metrics["mean_fusion"]["threshold_selected_on_valid_o6"])
    thermal = load_predictions(experiment / "evaluation/thermal_predictions.json")
    fusion = load_predictions(experiment / "evaluation/mean_fusion_predictions.json")
    records = [record for record in manifest["records"] if record["split"] == "TEST"]
    if len(records) != 10 or sum(len(record["labels"]) for record in records) != 57:
        raise ValueError("Expected the completed canonical ten-scene/57-person TEST protocol")

    per_gt: list[dict[str, Any]] = []
    per_scene: list[dict[str, Any]] = []
    lost_causes: dict[str, int] = defaultdict(int)
    total_lost = 0
    restored = 0
    for record in records:
        image_id = int(record["image_id"])
        boxes = [yolo_box(row, record["width"], record["height"]) for row in record["labels"]]
        t_matches, t_matched_gt, t_matched_pred, t_selected = match_at_threshold(
            boxes, thermal.get(image_id, []), thermal_threshold
        )
        f_matches, f_matched_gt, f_matched_pred, f_selected = match_at_threshold(
            boxes, fusion.get(image_id, []), fusion_threshold
        )
        lost = t_matched_gt - f_matched_gt
        gained = f_matched_gt - t_matched_gt
        total_lost += len(lost)
        restored += len(gained)
        per_scene.append(
            {
                "scene": record["scene"],
                "image_id": image_id,
                "gt": len(boxes),
                "thermal_tp": len(t_matched_gt),
                "thermal_fp": len(t_selected) - len(t_matched_pred),
                "thermal_fn": len(boxes) - len(t_matched_gt),
                "fusion_tp": len(f_matched_gt),
                "fusion_fp": len(f_selected) - len(f_matched_pred),
                "fusion_fn": len(boxes) - len(f_matched_gt),
                "thermal_only_gt": len(lost),
                "fusion_only_gt": len(gained),
            }
        )
        for gt_index, box in enumerate(boxes):
            t_best = best_candidate(box, thermal.get(image_id, []))
            f_best = best_candidate(box, fusion.get(image_id, []))
            thermal_detected = gt_index in t_matched_gt
            fusion_detected = gt_index in f_matched_gt
            cause = "not_thermal_only"
            if thermal_detected and not fusion_detected:
                if f_best["iou"] >= 0.5 and (f_best["confidence"] or 0.0) < fusion_threshold:
                    cause = "valid_box_below_confidence"
                elif f_best["iou"] < 0.25:
                    cause = "no_nearby_post_nms_candidate"
                elif f_best["iou"] < 0.5:
                    cause = "post_nms_localization_below_iou50"
                else:
                    cause = "one_to_one_competition"
                lost_causes[cause] += 1
            per_gt.append(
                {
                    "scene": record["scene"],
                    "image_id": image_id,
                    "gt_index": gt_index,
                    "gt_x1": fmt(box[0]),
                    "gt_y1": fmt(box[1]),
                    "gt_x2": fmt(box[2]),
                    "gt_y2": fmt(box[3]),
                    "thermal_detected": thermal_detected,
                    "fusion_detected": fusion_detected,
                    "thermal_only": thermal_detected and not fusion_detected,
                    "fusion_only": fusion_detected and not thermal_detected,
                    "loss_cause": cause,
                    "thermal_best_iou": fmt(t_best["iou"]),
                    "thermal_best_score": fmt(t_best["confidence"]),
                    "fusion_best_iou": fmt(f_best["iou"]),
                    "fusion_best_score": fmt(f_best["confidence"]),
                    "best_score_delta_fusion_minus_thermal": fmt(
                        (f_best["confidence"] - t_best["confidence"])
                        if f_best["confidence"] is not None and t_best["confidence"] is not None
                        else None
                    ),
                    "thermal_center_error_norm": fmt(t_best["center_error_gt_diagonal"]),
                    "fusion_center_error_norm": fmt(f_best["center_error_gt_diagonal"]),
                    "thermal_width_ratio": fmt(t_best["width_ratio"]),
                    "thermal_height_ratio": fmt(t_best["height_ratio"]),
                    "fusion_width_ratio": fmt(f_best["width_ratio"]),
                    "fusion_height_ratio": fmt(f_best["height_ratio"]),
                    "thermal_area_ratio": fmt(t_best["area_ratio"]),
                    "fusion_area_ratio": fmt(f_best["area_ratio"]),
                }
            )
        render_scene(
            record,
            boxes,
            thermal.get(image_id, []),
            fusion.get(image_id, []),
            thermal_threshold,
            fusion_threshold,
            lost,
            overlays / f"{record['scene']}_thermal_vs_mean_fusion.png",
        )

    lost_rows = [row for row in per_gt if row["thermal_only"]]
    detected_rows = [row for row in per_gt if row["thermal_detected"]]
    write_csv(output / "per_gt_diagnostics.csv", per_gt)
    write_csv(output / "per_scene_comparison.csv", per_scene)
    summary = {
        "prediction_scope": "saved post-NMS/top-100 predictions with score floor 0.001",
        "thermal_threshold": thermal_threshold,
        "fusion_threshold": fusion_threshold,
        "thermal_metrics": metrics["thermal"],
        "mean_fusion_metrics": metrics["mean_fusion"],
        "thermal_only_true_positives": total_lost,
        "fusion_only_true_positives": restored,
        "thermal_only_failure_causes": dict(lost_causes),
        "mean_best_score_delta_on_thermal_detected_gt": (
            sum(float(row["best_score_delta_fusion_minus_thermal"]) for row in detected_rows)
            / len(detected_rows)
            if detected_rows
            else None
        ),
        "mean_fusion_best_iou_on_thermal_only_gt": (
            sum(float(row["fusion_best_iou"]) for row in lost_rows) / len(lost_rows)
            if lost_rows
            else None
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    scene_table = "\n".join(
        f"| {row['scene']} | {row['gt']} | {row['thermal_tp']} | {row['fusion_tp']} | "
        f"{row['thermal_only_gt']} | {row['fusion_only_gt']} | {row['thermal_fp']} | {row['fusion_fp']} |"
        for row in per_scene
    )
    cause_text = ", ".join(f"{key.replace('_', ' ')}: {value}" for key, value in sorted(lost_causes.items()))
    report = f"""# AOS Thermal vs Mean-Fusion Detection Analysis

Date: 2026-10-20  
Scope: offline analysis of saved canonical TEST predictions; no inference or training

## Result

At the validation-locked score 0.70, thermal detected 26/57 people with two
false positives (recall 45.61%, precision 92.86%, 0.20 FP/image). Mean fusion
detected 5/57 with two false positives (recall 8.77%, precision 71.43%, 0.20
FP/image). Fusion lost {total_lost} thermal true positives and gained {restored}.
AP50 fell from {metrics['thermal']['ap50']:.4f} to {metrics['mean_fusion']['ap50']:.4f};
AR@100 fell from {metrics['thermal']['ar100']:.4f} to {metrics['mean_fusion']['ar100']:.4f}.

| Scene | GT | Thermal TP | Fusion TP | Lost | Gained | Thermal FP | Fusion FP |
|---|---:|---:|---:|---:|---:|---:|---:|
{scene_table}

## Lost detections

The {total_lost} thermal-only GT instances divide into: {cause_text or 'none'}.
For each GT, `per_gt_diagnostics.csv` records the best available saved candidate
IoU, score, normalized center error, width/height ratios, and area ratio. Across
thermal-detected GT, the mean fusion-minus-thermal best-candidate score change
was {summary['mean_best_score_delta_on_thermal_detected_gt']:.3f}. The mean best
fusion IoU for thermal-only GT was {summary['mean_fusion_best_iou_on_thermal_only_gt']:.3f}.

The dominant mechanism is interpreted from the saved post-NMS candidates only:
valid fusion boxes below 0.70 indicate confidence suppression; IoU 0.25-0.50
indicates localization failure; IoU below 0.25 indicates no useful nearby saved
candidate. Box-scale and center diagnostics distinguish displaced from badly
sized candidates but cannot recover candidates removed before serialization.

## Limitations

Each prediction file contains exactly 100 post-NMS detections per image with a
0.001 score floor. Therefore this analysis cannot attribute losses to pre-NMS
thresholding, top-k ordering, or NMS suppression. Thresholds were selected on
VALID/O6, not TEST. The ten canonical test products contain 57 people across
eight positive scenes and two empty scenes. The detector heads were trained on
augmented renderings of only 37 unique training people, so conclusions remain
pilot-scale.

## Artifacts

- `per_scene_comparison.csv`
- `per_gt_diagnostics.csv`
- `summary.json`
- `overlays/`: all ten canonical scenes; magenta marks thermal TP lost by fusion
"""
    args.report.resolve().write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
