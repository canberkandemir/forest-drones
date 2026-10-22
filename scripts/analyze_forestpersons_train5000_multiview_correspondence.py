"""Post-selection person-box correspondence audit for train5000 groups.

This diagnostic never changes group membership. ForestPersons has no person
track IDs, so projected annotation overlap is spatial evidence, not confirmed
cross-view identity.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--groups-key", help="Optional list key when --groups contains an object")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/train.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def xyxy(annotation: dict[str, Any]) -> list[float]:
    x, y, width, height = map(float, annotation["bbox"])
    return [x, y, x + width, y + height]


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def project(annotation: dict[str, Any], homography: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = xyxy(annotation)
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    points = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]


def match_projected(
    references: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    homography: np.ndarray,
) -> dict[int, dict[str, float | int]]:
    projected = [project(annotation, homography) for annotation in sources]
    candidates = sorted(
        (
            (iou(xyxy(reference), source_box), reference_index, source_index)
            for reference_index, reference in enumerate(references)
            for source_index, source_box in enumerate(projected)
        ),
        reverse=True,
    )
    result: dict[int, dict[str, float | int]] = {}
    used_sources: set[int] = set()
    for overlap, reference_index, source_index in candidates:
        if reference_index in result or source_index in used_sources:
            continue
        reference_box = xyxy(references[reference_index])
        source_box = projected[source_index]
        reference_center = np.asarray([(reference_box[0] + reference_box[2]) / 2, (reference_box[1] + reference_box[3]) / 2])
        source_center = np.asarray([(source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2])
        diagonal = max(float(np.hypot(reference_box[2] - reference_box[0], reference_box[3] - reference_box[1])), 1e-6)
        result[reference_index] = {
            "iou": float(overlap),
            "center_error_normalized": float(np.linalg.norm(source_center - reference_center) / diagonal),
            "source_annotation_id": int(sources[source_index]["id"]),
        }
        used_sources.add(source_index)
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_data = json.loads(args.groups.read_text(encoding="utf-8"))
    groups = groups_data[args.groups_key] if args.groups_key else groups_data
    if not isinstance(groups, list):
        raise ValueError("Selected groups payload must be a list")
    annotation_data = json.loads(args.annotations.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotation_data["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)

    rows = []
    groups_without_source_annotations = 0
    for group_index, group in enumerate(groups, start=1):
        reference_annotations = by_image[int(group["reference"]["image_id"])]
        mappings = []
        if any(not by_image[int(source["image_id"])] for source in group["sources"]):
            groups_without_source_annotations += 1
        for source in group["sources"]:
            mappings.append(match_projected(
                reference_annotations,
                by_image[int(source["image_id"])],
                np.asarray(source["homography_source_to_reference"], dtype=np.float64),
            ))
        for reference_index, annotation in enumerate(reference_annotations):
            matches = [mapping.get(reference_index) for mapping in mappings]
            overlaps = [float(match["iou"]) if match else 0.0 for match in matches]
            center_errors = [float(match["center_error_normalized"]) if match else None for match in matches]
            rows.append({
                "group_index": group_index,
                "sequence": group["sequence"],
                "reference_identity": group["reference"]["identity"],
                "annotation_id": int(annotation["id"]),
                "visibility": int(annotation["attributes"]["visibility_level"]),
                "earlier_projected_iou": overlaps[0],
                "later_projected_iou": overlaps[1],
                "minimum_source_iou": min(overlaps),
                "maximum_source_iou": max(overlaps),
                "both_sources_iou50": all(value >= 0.50 for value in overlaps),
                "any_source_iou50": any(value >= 0.50 for value in overlaps),
                "earlier_center_error_normalized": center_errors[0],
                "later_center_error_normalized": center_errors[1],
            })

    with (args.output_dir / "per_gt_correspondence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "selection_changed": False,
        "groups": len(groups),
        "sequences": len({group["sequence"] for group in groups}),
        "ground_truth_people": len(rows),
        "groups_without_annotations_in_at_least_one_source": groups_without_source_annotations,
        "both_sources_iou50": sum(row["both_sources_iou50"] for row in rows),
        "any_source_iou50": sum(row["any_source_iou50"] for row in rows),
        "neither_source_iou50": sum(not row["any_source_iou50"] for row in rows),
        "median_minimum_source_iou": float(np.median([row["minimum_source_iou"] for row in rows])),
        "median_maximum_source_iou": float(np.median([row["maximum_source_iou"] for row in rows])),
        "by_visibility": {
            str(level): {
                "ground_truth": sum(row["visibility"] == level for row in rows),
                "both_sources_iou50": sum(row["visibility"] == level and row["both_sources_iou50"] for row in rows),
                "any_source_iou50": sum(row["visibility"] == level and row["any_source_iou50"] for row in rows),
                "neither_source_iou50": sum(row["visibility"] == level and not row["any_source_iou50"] for row in rows),
            }
            for level in (20, 40, 70, 100)
        },
        "warning": "ForestPersons has no person track IDs. Projected annotation overlap is post-selection spatial evidence, not confirmed identity correspondence.",
        "visibility_counts": dict(Counter(str(row["visibility"]) for row in rows)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
