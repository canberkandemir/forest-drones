import json
from pathlib import Path

import pytest
from PIL import Image

from forestpersons import ForestPersonsDataset, validate_coco_annotations


def _write_fixture(root: Path) -> None:
    image_dir = root / "images" / "scene_a"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    Image.new("RGB", (64, 48), color=(20, 80, 40)).save(image_dir / "sample.jpg")
    coco = {
        "images": [
            {
                "id": 10,
                "file_name": "scene_a/sample.jpg",
                "width": 64,
                "height": 48,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 1,
                "bbox": [5, 6, 12, 18],
                "area": 216,
                "iscrowd": 0,
                "attributes": {
                    "visibility_level": 40,
                    "pose": "sitting",
                    "difficulty": "hard",
                },
            }
        ],
        "categories": [{"id": 1, "name": "sitting_40", "supercategory": "person"}],
    }
    (annotation_dir / "train.json").write_text(json.dumps(coco), encoding="utf-8")


def test_dataset_loads_image_annotations_and_metadata(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    dataset = ForestPersonsDataset(tmp_path, split="train")

    assert len(dataset) == 1
    image, target = dataset[0]

    assert image.size == (64, 48)
    assert target["image_id"] == 10
    assert target["annotations"][0]["bbox"] == [5, 6, 12, 18]
    assert target["annotations"][0]["metadata"] == {
        "attributes": {
            "visibility_level": 40,
            "pose": "sitting",
            "difficulty": "hard",
        },
        "visibility_level": 40,
        "pose": "sitting",
        "difficulty": "hard",
    }
    assert dataset.validation_warnings == []


def test_validation_rejects_missing_bbox() -> None:
    coco = {
        "images": [{"id": 1, "file_name": "x.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1}],
        "categories": [{"id": 1, "name": "person"}],
    }
    with pytest.raises(ValueError, match="invalid bbox"):
        validate_coco_annotations(coco)
