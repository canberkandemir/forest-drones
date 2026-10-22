"""Analyze Weitefeld class-3 localization from saved detector predictions only.

This script is deliberately diagnostic.  Weitefeld class 3 contains suspected
people rather than verified, exhaustive person ground truth, so unmatched
predictions are not interpreted as definitive false positives.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from forestpersons.detection import box_iou


REFERENCE_THRESHOLDS = {"yolo": 0.04, "dino": 0.55}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    data = json.loads(path.read_text(encoding="utf-8"))
    for prediction in data["predictions"]:
        grouped[int(prediction["image_id"])].append(prediction)
    return grouped


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center_distance_normalized(first: list[float], second: list[float]) -> float:
    first_x = (first[0] + first[2]) / 2.0
    first_y = (first[1] + first[3]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    second_y = (second[1] + second[3]) / 2.0
    diagonal = math.hypot(first[2] - first[0], first[3] - first[1])
    return math.hypot(first_x - second_x, first_y - second_y) / max(diagonal, 1e-9)


def contains_center(prediction: list[float], target: list[float]) -> bool:
    center_x = (target[0] + target[2]) / 2.0
    center_y = (target[1] + target[3]) / 2.0
    return (
        prediction[0] <= center_x <= prediction[2]
        and prediction[1] <= center_y <= prediction[3]
    )


def best_candidate(
    target: list[float], predictions: list[dict[str, Any]], threshold: float | None = None
) -> tuple[dict[str, Any] | None, float]:
    eligible = predictions
    if threshold is not None:
        eligible = [item for item in predictions if float(item["score"]) >= threshold]
    if not eligible:
        return None, 0.0
    candidate = max(eligible, key=lambda item: box_iou(target, item["bbox"]))
    return candidate, float(box_iou(target, candidate["bbox"]))


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": median(values) if values else None,
        "maximum": max(values) if values else None,
        "at_least_0.10": sum(value >= 0.10 for value in values),
        "at_least_0.25": sum(value >= 0.25 for value in values),
        "at_least_0.50": sum(value >= 0.50 for value in values),
        "at_least_0.75": sum(value >= 0.75 for value in values),
    }


def draw_gallery(
    rows: list[dict[str, Any]], images: dict[int, dict[str, Any]], image_root: Path,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prefer the strongest available overlap while retaining all three label tiers.
    selected: list[dict[str, Any]] = []
    for tier in ("explicit_person_view", "ambiguous_person_candidate", "police_nonperson"):
        tier_rows = [item for item in rows if item["verification_tier"] == tier]
        tier_rows.sort(
            key=lambda item: max(item["yolo_tiled_best_iou"], item["dino_tiled_best_iou"]),
            reverse=True,
        )
        selected.extend(tier_rows[:4])

    paths = []
    for index, row in enumerate(selected, 1):
        image = images[int(row["image_id"])]
        path = image_root / f"strip{image['strip']}" / image["file_name"]
        with Image.open(path) as source:
            source = source.convert("RGB")
            gt = row["gt_bbox"]
            candidates = [
                row.get("yolo_tiled_best_box"), row.get("dino_tiled_best_box"),
            ]
            visible_boxes = [gt] + [box for box in candidates if box is not None]
            x1 = min(box[0] for box in visible_boxes)
            y1 = min(box[1] for box in visible_boxes)
            x2 = max(box[2] for box in visible_boxes)
            y2 = max(box[3] for box in visible_boxes)
            center_x = (gt[0] + gt[2]) / 2.0
            center_y = (gt[1] + gt[3]) / 2.0
            span = max(420.0, x2 - x1, y2 - y1)
            span = min(span * 1.25, 1200.0)
            left = max(0, min(source.width - span, center_x - span / 2.0))
            top = max(0, min(source.height - span, center_y - span / 2.0))
            crop_box = (int(left), int(top), int(left + span), int(top + span))
            canvas = source.crop(crop_box)

        scale = 900.0 / max(canvas.width, canvas.height)
        canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)))
        draw = ImageDraw.Draw(canvas)

        def translated(box: list[float]) -> list[float]:
            return [
                (box[0] - crop_box[0]) * scale, (box[1] - crop_box[1]) * scale,
                (box[2] - crop_box[0]) * scale, (box[3] - crop_box[1]) * scale,
            ]

        draw.rectangle(translated(gt), outline="lime", width=5)
        draw.text((10, 10), f"GT class 3: {row['verification_tier']}", fill="lime")
        for label, color in (("yolo", "cyan"), ("dino", "magenta")):
            box = row.get(f"{label}_tiled_best_box")
            if box is None:
                continue
            draw.rectangle(translated(box), outline=color, width=3)
            draw.text(
                (10, 35 if label == "yolo" else 60),
                f"{label}: IoU {row[f'{label}_tiled_best_iou']:.3f}, "
                f"score {row[f'{label}_tiled_best_score']:.3f}",
                fill=color,
            )
        draw.text(
            (10, canvas.height - 24),
            f"strip {image['strip']} image {image['image_number']} ann {row['annotation_id']}",
            fill="white",
        )
        output = output_dir / f"{index:02d}_image_{row['image_id']}_ann_{row['annotation_id']}.jpg"
        canvas.save(output, quality=92)
        paths.append(str(output))
    return paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = {int(item["image_id"]): item for item in manifest["images"]}
    predictions = {
        f"{model}_{mode}": load_predictions(args.prediction_root / f"{model}_{mode}.json")
        for model in ("yolo", "dino") for mode in ("full", "tiled")
    }

    rows: list[dict[str, Any]] = []
    for image in manifest["images"]:
        image_id = int(image["image_id"])
        for annotation in image["person_annotations"]:
            target = [float(value) for value in annotation["bbox_xyxy"]]
            row: dict[str, Any] = {
                "image_id": image_id,
                "annotation_id": int(annotation["annotation_id"]),
                "strip": int(image["strip"]),
                "image_number": int(image["image_number"]),
                "verification_tier": annotation["verification_tier"],
                "track_id": annotation["track_id"],
                "gt_width": target[2] - target[0],
                "gt_height": target[3] - target[1],
                "gt_area": box_area(target),
                "gt_bbox": target,
            }
            for key, grouped in predictions.items():
                model = key.split("_", 1)[0]
                raw, raw_iou = best_candidate(target, grouped.get(image_id, []))
                fixed, fixed_iou = best_candidate(
                    target, grouped.get(image_id, []), REFERENCE_THRESHOLDS[model]
                )
                row[f"{key}_best_iou"] = raw_iou
                row[f"{key}_best_score"] = float(raw["score"]) if raw else None
                row[f"{key}_best_box"] = raw["bbox"] if raw else None
                row[f"{key}_fixed_best_iou"] = fixed_iou
                row[f"{key}_fixed_best_score"] = float(fixed["score"]) if fixed else None
                row[f"{key}_fixed_contains_gt_center"] = bool(
                    fixed and contains_center(fixed["bbox"], target)
                )
                if raw:
                    row[f"{key}_best_area_ratio"] = box_area(raw["bbox"]) / box_area(target)
                    row[f"{key}_best_center_error_gt_diag"] = center_distance_normalized(
                        target, raw["bbox"]
                    )
                else:
                    row[f"{key}_best_area_ratio"] = None
                    row[f"{key}_best_center_error_gt_diag"] = None
            rows.append(row)

    csv_path = args.output_dir / "per_annotation_localization.csv"
    fieldnames = [key for key in rows[0] if not key.endswith("_best_box") and key != "gt_bbox"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)

    result: dict[str, Any] = {
        "annotation_coordinate_authority": {
            "conversion": "xyxy = (x, y-height, x+width, y)",
            "verified_against": ["official web viewer", "official Anomaly_detection_benchmark.ipynb"],
        },
        "saved_prediction_floor": 0.001,
        "reference_thresholds": REFERENCE_THRESHOLDS,
        "class3_semantics_warning": "suspected-person category; includes police-confirmed non-person findings",
        "annotation_completeness_warning": "published anomaly labels are not exhaustive",
        "gt_box_size": {
            "width_median": median(row["gt_width"] for row in rows),
            "height_median": median(row["gt_height"] for row in rows),
            "area_median": median(row["gt_area"] for row in rows),
        },
        "methods": {},
    }
    for key in predictions:
        model = key.split("_", 1)[0]
        raw_ious = [float(row[f"{key}_best_iou"]) for row in rows]
        fixed_ious = [float(row[f"{key}_fixed_best_iou"]) for row in rows]
        overlapping_ratios = [
            float(row[f"{key}_best_area_ratio"]) for row in rows
            if float(row[f"{key}_best_iou"]) > 0.0
        ]
        result["methods"][key] = {
            "all_saved_candidates": summarize(raw_ious),
            "reference_threshold_candidates": summarize(fixed_ious),
            "reference_threshold": REFERENCE_THRESHOLDS[model],
            "reference_threshold_candidate_contains_gt_center": sum(
                bool(row[f"{key}_fixed_contains_gt_center"]) for row in rows
            ),
            "best_overlapping_candidate_area_ratio_median": (
                median(overlapping_ratios) if overlapping_ratios else None
            ),
        }

    plt.figure(figsize=(7.4, 4.8))
    bins = [index / 40 for index in range(21)]
    for key, color in (
        ("yolo_full", "#2b8cbe"), ("yolo_tiled", "#045a8d"),
        ("dino_full", "#dd1c77"), ("dino_tiled", "#980043"),
    ):
        plt.hist(
            [float(row[f"{key}_best_iou"]) for row in rows], bins=bins,
            histtype="step", linewidth=2, label=key.replace("_", " "), color=color,
        )
    plt.xlabel("Best IoU among saved candidates (floor 0.001)")
    plt.ylabel("Official class-3 annotations")
    plt.title("Weitefeld pilot200 localization diagnostic")
    plt.legend()
    plt.tight_layout()
    plot_path = args.output_dir / "best_candidate_iou_histogram.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    result["gallery"] = draw_gallery(
        rows, images, args.image_root, args.output_dir / "localization_gallery"
    )
    result["artifacts"] = {
        "per_annotation_csv": str(csv_path),
        "iou_histogram": str(plot_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
