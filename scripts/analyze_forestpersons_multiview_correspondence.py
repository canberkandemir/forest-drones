"""Post-selection target correspondence audit for a frozen multi-view manifest.

Ground truth is used only after group membership has been frozen. The analysis
does not remove groups or change detector metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/val.json"))
    parser.add_argument("--detector-diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def xywh_to_xyxy(annotation: dict[str, Any]) -> list[float]:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    return [x, y, x + width, y + height]


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def project_box(annotation: dict[str, Any], homography: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = xywh_to_xyxy(annotation)
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return [
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    ]


def greedy_correspondence(
    reference: list[dict[str, Any]],
    source: list[dict[str, Any]],
    homography: np.ndarray,
) -> dict[int, dict[str, Any]]:
    projected = [project_box(annotation, homography) for annotation in source]
    candidates = []
    for reference_index, reference_annotation in enumerate(reference):
        reference_box = xywh_to_xyxy(reference_annotation)
        for source_index, source_box in enumerate(projected):
            candidates.append((box_iou(reference_box, source_box), reference_index, source_index))
    candidates.sort(reverse=True)
    result: dict[int, dict[str, Any]] = {}
    used_source: set[int] = set()
    for iou, reference_index, source_index in candidates:
        if reference_index in result or source_index in used_source:
            continue
        used_source.add(source_index)
        reference_box = xywh_to_xyxy(reference[reference_index])
        source_box = projected[source_index]
        reference_center = ((reference_box[0] + reference_box[2]) / 2, (reference_box[1] + reference_box[3]) / 2)
        source_center = ((source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2)
        diagonal = max(np.hypot(reference_box[2] - reference_box[0], reference_box[3] - reference_box[1]), 1e-6)
        result[reference_index] = {
            "iou": float(iou),
            "source_annotation_id": int(source[source_index]["id"]),
            "source_visibility": int(source[source_index]["attributes"]["visibility_level"]),
            "center_error_normalized": float(np.hypot(source_center[0] - reference_center[0], source_center[1] - reference_center[1]) / diagonal),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["annotations"]
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    detector_rows = {
        int(row["annotation_id"]): row
        for row in csv.DictReader(args.detector_diagnostics.open(encoding="utf-8"))
    }

    rows = []
    for group in manifest["groups"]:
        reference_annotations = annotations_by_image[int(group["reference"]["image_id"])]
        source_matches = []
        for source in group["sources"]:
            source_annotations = annotations_by_image[int(source["image_id"])]
            source_matches.append(
                greedy_correspondence(
                    reference_annotations,
                    source_annotations,
                    np.asarray(source["homography_source_to_reference"], dtype=np.float64),
                )
            )
        for reference_index, annotation in enumerate(reference_annotations):
            matches = [mapping.get(reference_index) for mapping in source_matches]
            ious = [float(match["iou"]) if match else 0.0 for match in matches]
            detector = detector_rows[int(annotation["id"])]
            rows.append({
                "group_id": group["group_id"],
                "sequence": group["sequence"],
                "reference_image_id": int(group["reference"]["image_id"]),
                "annotation_id": int(annotation["id"]),
                "visibility": int(annotation["attributes"]["visibility_level"]),
                "earlier_projected_gt_iou": ious[0],
                "later_projected_gt_iou": ious[1],
                "minimum_source_gt_iou": min(ious),
                "maximum_source_gt_iou": max(ious),
                "both_sources_iou50": all(value >= 0.50 for value in ious),
                "any_source_iou50": any(value >= 0.50 for value in ious),
                "earlier_center_error_normalized": matches[0]["center_error_normalized"] if matches[0] else None,
                "later_center_error_normalized": matches[1]["center_error_normalized"] if matches[1] else None,
                "single_detected_fixed": detector["single_detected_fixed"],
                "fused_detected_fixed": detector["fused_detected_fixed"],
                "single_best_iou_all_scores": float(detector["single_best_iou_all_scores"]),
                "fused_best_iou_all_scores": float(detector["fused_best_iou_all_scores"]),
            })

    with (args.output_dir / "per_gt_correspondence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    categories = {
        "both_sources_iou50": [row for row in rows if row["both_sources_iou50"]],
        "one_source_iou50": [row for row in rows if row["any_source_iou50"] and not row["both_sources_iou50"]],
        "neither_source_iou50": [row for row in rows if not row["any_source_iou50"]],
    }
    category_summary = {}
    for name, values in categories.items():
        single_tp = sum(row["single_detected_fixed"] == "True" for row in values)
        fused_tp = sum(row["fused_detected_fixed"] == "True" for row in values)
        category_summary[name] = {
            "ground_truth": len(values),
            "single_detected": single_tp,
            "fused_detected": fused_tp,
            "single_recall": single_tp / len(values) if values else None,
            "fused_recall": fused_tp / len(values) if values else None,
            "net_detected_change": fused_tp - single_tp,
        }
    summary = {
        "selection_changed": False,
        "warning": "No person track IDs exist. Projected-box matching is a post-selection spatial diagnostic, not confirmed identity correspondence.",
        "ground_truth_people": len(rows),
        "both_sources_iou50": sum(row["both_sources_iou50"] for row in rows),
        "any_source_iou50": sum(row["any_source_iou50"] for row in rows),
        "median_minimum_source_gt_iou": float(np.median([row["minimum_source_gt_iou"] for row in rows])),
        "median_maximum_source_gt_iou": float(np.median([row["maximum_source_gt_iou"] for row in rows])),
        "by_visibility": {
            str(visibility): {
                "ground_truth": sum(row["visibility"] == visibility for row in rows),
                "both_sources_iou50": sum(row["visibility"] == visibility and row["both_sources_iou50"] for row in rows),
                "any_source_iou50": sum(row["visibility"] == visibility and row["any_source_iou50"] for row in rows),
            }
            for visibility in (20, 40, 70, 100)
        },
        "correspondence_categories": category_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    x = np.asarray([row["minimum_source_gt_iou"] for row in rows])
    y = np.asarray([row["fused_best_iou_all_scores"] - row["single_best_iou_all_scores"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(x, y, s=24, alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Minimum projected source-GT IoU across the two source views")
    ax.set_ylabel("Fused minus single best-candidate IoU")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "target_alignment_vs_localization_change.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
