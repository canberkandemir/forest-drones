"""Create a deterministic local-only ForestPersons validation calibration subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from create_forestpersons_val_subset import image_records, select_subset, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/val.json"))
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--images-per-visibility", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    excluded_data = json.loads(args.exclude_manifest.read_text(encoding="utf-8"))
    excluded = {item["file_name"] for item in excluded_data["images"]}
    records = [
        record
        for record in image_records(coco)
        if record["file_name"] not in excluded and (args.image_root / record["file_name"]).is_file()
    ]
    selected, selection = select_subset(records, args.images_per_visibility, args.seed)
    if len(selected) != 4 * args.images_per_visibility:
        raise RuntimeError(f"Could select only {len(selected)} calibration images")
    selected_files = {item["file_name"] for item in selected}
    if selected_files & excluded:
        raise RuntimeError("Calibration selection overlaps the excluded manifest")

    output = {
        "source_annotations": str(args.annotations),
        "excluded_manifest": str(args.exclude_manifest),
        "selection": {
            **selection,
            "purpose": "YOLO11s and DINOv3 FCOS selective-fusion calibration only",
            "local_images_only": True,
        },
        "summary": summarize(selected, args.image_root),
        "images": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
