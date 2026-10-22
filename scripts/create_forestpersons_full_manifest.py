"""Create a manifest for all official ForestPersons images.

The manifest is intended for use with download_forestpersons_subset_verified.py.
It is derived only from the official COCO annotation JSON files already present
under data/annotations and preserves the official split and relative file path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/subsets/forestpersons_full_official_images_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_by_path: dict[str, dict] = {}
    counts: dict[str, int] = {}

    for split in args.splits:
        annotation_path = args.root / "annotations" / f"{split}.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        split_images = 0
        for image in data["images"]:
            file_name = image["file_name"]
            existing = images_by_path.get(file_name)
            if existing is None:
                images_by_path[file_name] = {
                    "id": image["id"],
                    "file_name": file_name,
                    "width": image["width"],
                    "height": image["height"],
                    "splits": [split],
                    "source_image_ids": {split: image["id"]},
                }
            else:
                if existing["width"] != image["width"] or existing["height"] != image["height"]:
                    raise RuntimeError(f"Duplicate path has inconsistent dimensions: {file_name}")
                existing["splits"].append(split)
                existing["source_image_ids"][split] = image["id"]
            split_images += 1
        counts[split] = split_images

    images = sorted(images_by_path.values(), key=lambda item: item["file_name"])
    manifest = {
        "dataset": "etri/ForestPersons",
        "description": "All official ForestPersons image paths from local train/val/test annotations.",
        "source_annotations": [str(args.root / "annotations" / f"{split}.json") for split in args.splits],
        "counts": counts,
        "total_annotation_image_entries": sum(counts.values()),
        "total_unique_image_files": len(images),
        "images": images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": counts,
                "total_annotation_image_entries": sum(counts.values()),
                "total_unique_image_files": len(images),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
