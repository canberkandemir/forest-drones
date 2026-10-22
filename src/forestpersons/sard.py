"""Utilities for the official SARD mountain-scenario evaluation set.

SARD v1.0 stores one YOLO-format text file per image. Category ``1`` is a
person and coordinates are normalized center-x, center-y, width, and height.
Empty files are released negative frames and are retained during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class SardAnnotation:
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]


def normalized_yolo_to_xyxy(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO coordinates to clipped source-image XYXY."""

    if width <= 0.0 or height <= 0.0:
        raise ValueError("SARD boxes must have positive width and height")
    if not all(0.0 <= value <= 1.0 for value in (center_x, center_y, width, height)):
        raise ValueError("SARD normalized coordinates must be in [0, 1]")
    x1 = max(0.0, (center_x - width / 2.0) * image_width)
    y1 = max(0.0, (center_y - height / 2.0) * image_height)
    x2 = min(float(image_width), (center_x + width / 2.0) * image_width)
    y2 = min(float(image_height), (center_y + height / 2.0) * image_height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("SARD box is degenerate after clipping")
    return x1, y1, x2, y2


def parse_sard_annotation(path: Path, image_width: int, image_height: int) -> list[SardAnnotation]:
    annotations: list[SardAnnotation] = []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return annotations
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, found {len(fields)}")
        category, center_x, center_y, width, height = (float(value) for value in fields)
        if not category.is_integer() or int(category) != 1:
            raise ValueError(f"{path}:{line_number}: expected official person category 1")
        annotations.append(
            SardAnnotation(
                category_id=1,
                bbox_xyxy=normalized_yolo_to_xyxy(
                    center_x, center_y, width, height, image_width, image_height
                ),
            )
        )
    return annotations


def build_sard_mountains_manifest(
    dataset_root: Path,
    *,
    source_revision: str,
) -> dict[str, Any]:
    """Build a deterministic manifest from the complete 200-frame scenario."""

    frames_root = dataset_root / "frames"
    labels_root = dataset_root / "annotations"
    image_paths = sorted(frames_root.glob("*.jpg"), key=lambda path: int(path.stem))
    if len(image_paths) != 200:
        raise ValueError(f"Expected 200 official mountain frames, found {len(image_paths)}")

    images: list[dict[str, Any]] = []
    annotation_id = 1
    total_annotations = 0
    for image_path in image_paths:
        label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        parsed = parse_sard_annotation(label_path, width, height)
        annotations = []
        for annotation in parsed:
            x1, y1, x2, y2 = annotation.bbox_xyxy
            annotations.append(
                {
                    "annotation_id": annotation_id,
                    "category_id": 1,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                }
            )
            annotation_id += 1
        total_annotations += len(annotations)
        images.append(
            {
                "image_id": int(image_path.stem),
                "file_name": image_path.name,
                "scenario": "mountains",
                "sequence_frame": int(image_path.stem),
                "width": width,
                "height": height,
                "sha256": sha256(image_path.read_bytes()).hexdigest(),
                "person_annotations": annotations,
            }
        )

    return {
        "dataset": "SARD v1.0",
        "source": "https://github.com/gvessio/uav-search-and-rescue",
        "source_revision": source_revision,
        "scenario": "mountains",
        "selection": "all 200 official mountain frames in sequence order; no GT-based sampling",
        "split_status": "official release has scenarios but no train/validation/test split",
        "annotation_semantics": {
            "category_1": "person",
            "format": "normalized YOLO center-x center-y width height",
            "empty_label_files": "official released negative frames",
        },
        "images": images,
        "summary": {
            "images": len(images),
            "positive_images": sum(bool(image["person_annotations"]) for image in images),
            "negative_images": sum(not image["person_annotations"] for image in images),
            "person_annotations": total_annotations,
        },
    }


def manifest_to_coco(manifest: dict[str, Any]) -> dict[str, Any]:
    annotations = []
    for image in manifest["images"]:
        for annotation in image["person_annotations"]:
            annotations.append(
                {
                    "id": annotation["annotation_id"],
                    "image_id": image["image_id"],
                    "category_id": 1,
                    "bbox": annotation["bbox_xywh"],
                    "area": annotation["area"],
                    "iscrowd": 0,
                }
            )
    return {
        "info": {
            "description": "SARD v1.0 mountain scenario, deterministic 200-image cross-dataset pilot"
        },
        "licenses": [],
        "images": [
            {
                "id": image["image_id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
            }
            for image in manifest["images"]
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
