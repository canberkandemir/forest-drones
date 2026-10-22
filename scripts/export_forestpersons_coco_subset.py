"""Export a ForestPersons manifest subset as one-class COCO annotations.

The exported file is intended for detector frameworks such as MMDetection,
detrex, and Detectron2. It preserves ForestPersons annotation attributes while
mapping all pose/visibility categories to a single `person` category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-prefix", default="")
    parser.add_argument(
        "--include-unknown-visibility",
        action="store_true",
        help="Include annotations without a standard ForestPersons visibility level.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def export_subset(
    annotations_path: Path,
    manifest_path: Path,
    output_path: Path,
    image_prefix: str = "",
    include_unknown_visibility: bool = False,
) -> dict[str, Any]:
    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    manifest = load_manifest(manifest_path)
    selected_files = [item["file_name"] for item in manifest]
    selected_file_set = set(selected_files)
    selected_manifest_ids = {
        item.get("image_id", item.get("id"))
        for item in manifest
        if item.get("image_id", item.get("id")) is not None
    }

    images_by_id = {image["id"]: image for image in coco["images"]}
    if selected_manifest_ids:
        selected_images = [
            dict(image)
            for image in coco["images"]
            if image["id"] in selected_manifest_ids
        ]
    else:
        selected_images = [
            dict(image)
            for image in coco["images"]
            if image["file_name"] in selected_file_set
        ]
    selected_image_ids = {image["id"] for image in selected_images}
    if selected_manifest_ids:
        missing_ids = sorted(selected_manifest_ids - selected_image_ids)
        if len(selected_images) != len(selected_manifest_ids) or missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} manifest image ids in annotations, "
                f"e.g. {missing_ids[:3]}"
            )
    elif len(selected_images) != len(selected_files):
        found = {image["file_name"] for image in selected_images}
        missing = sorted(selected_file_set - found)
        raise ValueError(f"Missing {len(missing)} manifest images in annotations, e.g. {missing[:3]}")

    for image in selected_images:
        if image_prefix:
            image["file_name"] = str(Path(image_prefix) / image["file_name"])

    annotations: list[dict[str, Any]] = []
    skipped_unknown_visibility = 0
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in selected_image_ids:
            continue
        attributes = annotation.get("attributes", {})
        if not include_unknown_visibility and attributes.get("visibility_level") is None:
            skipped_unknown_visibility += 1
            continue
        image = images_by_id[annotation["image_id"]]
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        x1 = min(max(x, 0.0), float(image["width"]))
        y1 = min(max(y, 0.0), float(image["height"]))
        x2 = min(max(x + width, 0.0), float(image["width"]))
        y2 = min(max(y + height, 0.0), float(image["height"]))
        if x2 <= x1 or y2 <= y1:
            continue
        exported = dict(annotation)
        exported["category_id"] = 1
        exported["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        exported["area"] = (x2 - x1) * (y2 - y1)
        exported["iscrowd"] = int(exported.get("iscrowd", 0))
        annotations.append(exported)

    output = {
        "info": {
            "description": "ForestPersons one-class subset export",
            "source_annotations": str(annotations_path),
            "source_manifest": str(manifest_path),
        },
        "licenses": coco.get("licenses", []),
        "images": selected_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return {
        "output": str(output_path),
        "images": len(selected_images),
        "annotations": len(annotations),
        "skipped_unknown_visibility": skipped_unknown_visibility,
        "categories": output["categories"],
    }


def main() -> None:
    args = parse_args()
    summary = export_subset(
        args.annotations,
        args.manifest,
        args.output,
        image_prefix=args.image_prefix,
        include_unknown_visibility=args.include_unknown_visibility,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
