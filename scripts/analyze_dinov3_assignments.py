"""Analyze DINOv3 patch-center target assignment for a ForestPersons subset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    PATCH_SIZE,
    annotations_to_ground_truth,
    cache_path_for_file,
    load_feature_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ForestPersonsDataset(args.root, split=args.split)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
    by_file = {sample.file_name: sample for sample in dataset.samples}
    rows = []
    totals = Counter()
    collision_visibility = Counter()
    collision_pose = Counter()
    visibility_total = Counter()
    visibility_collision_candidates = Counter()
    small_area_by_visibility = defaultdict(list)

    for item in manifest:
        sample = by_file[item["file_name"]]
        _, info = load_feature_cache(cache_path_for_file(args.cache_root, sample.file_name))
        cell_map: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for gt in annotations_to_ground_truth(sample.annotations):
            x1 = gt["box"][0] * info.scale_x
            y1 = gt["box"][1] * info.scale_y
            x2 = gt["box"][2] * info.scale_x
            y2 = gt["box"][3] * info.scale_y
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            gx = min(info.grid_width - 1, max(0, int(cx // PATCH_SIZE)))
            gy = min(info.grid_height - 1, max(0, int(cy // PATCH_SIZE)))
            area_original = (gt["box"][2] - gt["box"][0]) * (gt["box"][3] - gt["box"][1])
            area_resized = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            visibility = int(gt["visibility_level"])
            visibility_total[visibility] += 1
            small_area_by_visibility[visibility].append(area_original)
            record = {
                "file_name": sample.file_name,
                "annotation_id": gt["annotation_id"],
                "visibility_level": visibility,
                "pose": gt["pose"],
                "grid_x": gx,
                "grid_y": gy,
                "bbox_area_original": area_original,
                "bbox_area_resized": area_resized,
            }
            cell_map[(gy, gx)].append(record)

        for group in cell_map.values():
            collided = len(group) > 1
            for record in group:
                record["cell_collision"] = collided
                rows.append(record)
                totals["annotations"] += 1
                if collided:
                    totals["annotations_in_collided_cells"] += 1
                    collision_visibility[record["visibility_level"]] += 1
                    collision_pose[record["pose"]] += 1
                    visibility_collision_candidates[record["visibility_level"]] += 1
            if collided:
                totals["collided_cells"] += 1
                totals["extra_annotations_lost_if_one_target_per_cell"] += len(group) - 1

    summary = {
        "split": args.split,
        "manifest": str(args.manifest),
        "images": len(manifest),
        "annotations": totals["annotations"],
        "collided_cells": totals["collided_cells"],
        "annotations_in_collided_cells": totals["annotations_in_collided_cells"],
        "extra_annotations_lost_if_one_target_per_cell": totals[
            "extra_annotations_lost_if_one_target_per_cell"
        ],
        "visibility_total": dict(sorted(visibility_total.items())),
        "collision_visibility": dict(sorted(collision_visibility.items())),
        "collision_pose": dict(sorted(collision_pose.items())),
        "median_bbox_area_by_visibility": {
            str(visibility): sorted(areas)[len(areas) // 2]
            for visibility, areas in sorted(small_area_by_visibility.items())
            if areas
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assignment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "assignment_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "file_name",
            "annotation_id",
            "visibility_level",
            "pose",
            "grid_x",
            "grid_y",
            "bbox_area_original",
            "bbox_area_resized",
            "cell_collision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
