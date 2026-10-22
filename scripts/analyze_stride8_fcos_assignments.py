"""Compare stride-16 and stride-8 FCOS assignment on an existing manifest."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    analyze_fcos_positive_locations,
    annotations_to_ground_truth,
    resize_to_patch_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=672)
    return parser.parse_args()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"ground_truth": 0}
    return {
        "ground_truth": len(rows),
        "stride16_positive_mean": statistics.fmean(row["stride16_positive"] for row in rows),
        "stride8_positive_mean": statistics.fmean(row["stride8_positive"] for row in rows),
        "stride16_positive_median": statistics.median(row["stride16_positive"] for row in rows),
        "stride8_positive_median": statistics.median(row["stride8_positive"] for row in rows),
        "stride16_zero_positive": sum(row["stride16_positive"] == 0 for row in rows),
        "stride8_zero_positive": sum(row["stride8_positive"] == 0 for row in rows),
        "stride16_fallbacks": sum(row["stride16_fallback"] for row in rows),
        "stride8_fallbacks": sum(row["stride8_fallback"] for row in rows),
        "stride8_more_positive": sum(row["stride8_positive"] > row["stride16_positive"] for row in rows),
    }


def size_group(row: dict[str, Any]) -> str:
    maximum = max(row["resized_width"], row["resized_height"])
    if maximum < 16:
        return "tiny_max_side_lt16"
    if maximum < 32:
        return "small_max_side_16_32"
    return "max_side_ge32"


def main() -> None:
    args = parse_args()
    dataset = ForestPersonsDataset(args.root, split="train")
    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data
    by_file = {sample.file_name: sample for sample in dataset.samples}
    rows: list[dict[str, Any]] = []

    for item in items:
        sample = by_file[item["file_name"]]
        with Image.open(sample.image_path) as image:
            _resized, info = resize_to_patch_grid(image, short_side=args.short_side)
        ground_truth = annotations_to_ground_truth(sample.annotations)
        stride16 = analyze_fcos_positive_locations(ground_truth, info, stride=16)
        stride8 = analyze_fcos_positive_locations(ground_truth, info, stride=8)
        for old, new in zip(stride16, stride8):
            row = {
                "file_name": sample.file_name,
                "annotation_id": old["annotation_id"],
                "visibility_level": int(old["visibility_level"]),
                "resized_width": old["resized_width"],
                "resized_height": old["resized_height"],
                "resized_area": old["resized_area"],
                "stride16_positive": old["positive_locations"],
                "stride8_positive": new["positive_locations"],
                "stride16_fallback": old["used_center_fallback"],
                "stride8_fallback": new["used_center_fallback"],
            }
            row["size_group"] = size_group(row)
            rows.append(row)

    by_visibility: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_visibility[row["visibility_level"]].append(row)
        by_size[row["size_group"]].append(row)
    summary = {
        "protocol": {
            "manifest": str(args.manifest),
            "images": len(items),
            "short_side": args.short_side,
            "assignment": "production FCOS center sampling with smaller-area overlap precedence and center fallback",
            "tiny_definition": "maximum resized box side < 16 pixels",
        },
        "overall": aggregate(rows),
        "by_visibility": {str(key): aggregate(value) for key, value in sorted(by_visibility.items())},
        "by_resized_size": {key: aggregate(value) for key, value in sorted(by_size.items())},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
