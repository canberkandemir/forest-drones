"""PyTorch dataset utilities for COCO-style ForestPersons annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset


PERSON_CATEGORY_NAMES = {"person", "human", "missing_person"}
COCO_ANNOTATION_KEYS = {
    "id",
    "image_id",
    "category_id",
    "bbox",
    "area",
    "iscrowd",
    "segmentation",
    "attributes",
}


@dataclass(frozen=True)
class ForestPersonsSample:
    """One ForestPersons image and all COCO annotations attached to it."""

    image_id: int | str
    file_name: str
    image_path: Path
    width: int | None
    height: int | None
    annotations: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a COCO JSON object in {path}")
    return data


def _normalize_categories(categories: list[dict[str, Any]]) -> dict[int | str, dict[str, str]]:
    normalized: dict[int | str, dict[str, str]] = {}
    for category in categories:
        if "id" not in category:
            raise ValueError("Every COCO category must contain an 'id'")
        normalized[category["id"]] = {
            "name": str(category.get("name", "")).strip(),
            "supercategory": str(category.get("supercategory", "")).strip(),
        }
    return normalized


def _annotation_metadata(annotation: dict[str, Any]) -> dict[str, Any]:
    """Keep ForestPersons-specific fields and flatten official attributes."""

    metadata = {
        key: value
        for key, value in annotation.items()
        if key not in COCO_ANNOTATION_KEYS
    }
    attributes = annotation.get("attributes")
    if isinstance(attributes, dict):
        metadata["attributes"] = dict(attributes)
        metadata.update(attributes)
    return metadata


def validate_coco_annotations(coco: dict[str, Any]) -> list[str]:
    """Return non-fatal validation warnings for a COCO annotation dictionary.

    Structural errors raise ``ValueError``. Warnings flag suspicious annotations
    while still allowing inspection of partially messy research data.
    """

    images = coco.get("images")
    annotations = coco.get("annotations")
    categories = coco.get("categories", [])
    if not isinstance(images, list):
        raise ValueError("COCO annotations must contain an 'images' list")
    if not isinstance(annotations, list):
        raise ValueError("COCO annotations must contain an 'annotations' list")
    if not isinstance(categories, list):
        raise ValueError("COCO annotations field 'categories' must be a list")

    image_ids = set()
    warnings: list[str] = []
    for image in images:
        if "id" not in image:
            raise ValueError("Every COCO image must contain an 'id'")
        if "file_name" not in image:
            raise ValueError(f"Image {image['id']} is missing 'file_name'")
        if image["id"] in image_ids:
            raise ValueError(f"Duplicate image id: {image['id']}")
        image_ids.add(image["id"])

    normalized_categories = _normalize_categories(categories)
    person_category_ids = {
        category_id
        for category_id, category in normalized_categories.items()
        if (
            category["name"].lower() in PERSON_CATEGORY_NAMES
            or category["supercategory"].lower() in PERSON_CATEGORY_NAMES
        )
    }

    for annotation in annotations:
        if "image_id" not in annotation:
            raise ValueError("Every annotation must contain 'image_id'")
        if annotation["image_id"] not in image_ids:
            warnings.append(
                f"Annotation {annotation.get('id', '<no id>')} references missing "
                f"image_id {annotation['image_id']}"
            )
        bbox = annotation.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            raise ValueError(
                f"Annotation {annotation.get('id', '<no id>')} has an invalid bbox"
            )
        _, _, width, height = bbox
        if width <= 0 or height <= 0:
            warnings.append(
                f"Annotation {annotation.get('id', '<no id>')} has non-positive bbox size"
            )
        category_id = annotation.get("category_id")
        if category_id not in normalized_categories:
            warnings.append(
                f"Annotation {annotation.get('id', '<no id>')} references missing "
                f"category_id {category_id}"
            )
        if person_category_ids and category_id not in person_category_ids:
            warnings.append(
                f"Annotation {annotation.get('id', '<no id>')} is not a person category"
            )

    return warnings


class ForestPersonsDataset(Dataset):
    """Load ForestPersons COCO annotations and RGB images.

    The loader preserves annotation metadata such as visibility, pose, and
    difficulty by placing every non-standard COCO field under ``metadata``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_dir: str | Path = "images",
        annotation_dir: str | Path = "annotations",
        transform: Callable[[Image.Image, dict[str, Any]], Any] | None = None,
        target_transform: Callable[[dict[str, Any]], Any] | None = None,
        validate: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.image_root = self.root / image_dir
        self.annotation_path = self.root / annotation_dir / f"{split}.json"
        self.transform = transform
        self.target_transform = target_transform

        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Missing ForestPersons annotation file: {self.annotation_path}"
            )

        self.coco = _read_json(self.annotation_path)
        self.validation_warnings = (
            validate_coco_annotations(self.coco) if validate else []
        )

        images = self.coco["images"]
        annotations = self.coco["annotations"]
        self.categories = _normalize_categories(self.coco.get("categories", []))

        annotations_by_image: dict[int | str, list[dict[str, Any]]] = {}
        for annotation in annotations:
            normalized = dict(annotation)
            normalized["metadata"] = _annotation_metadata(annotation)
            annotations_by_image.setdefault(annotation["image_id"], []).append(normalized)

        self.samples: list[ForestPersonsSample] = []
        for image in images:
            file_name = str(image["file_name"])
            self.samples.append(
                ForestPersonsSample(
                    image_id=image["id"],
                    file_name=file_name,
                    image_path=self.image_root / file_name,
                    width=image.get("width"),
                    height=image.get("height"),
                    annotations=annotations_by_image.get(image["id"], []),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        sample = self.samples[index]
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image file: {sample.image_path}")

        image = Image.open(sample.image_path).convert("RGB")
        target: dict[str, Any] = {
            "image_id": sample.image_id,
            "file_name": sample.file_name,
            "image_path": str(sample.image_path),
            "width": sample.width or image.width,
            "height": sample.height or image.height,
            "annotations": sample.annotations,
        }

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            return self.transform(image, target), target
        return image, target
