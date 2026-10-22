from pathlib import Path

import pytest

from forestpersons import ForestPersonsDataset


ANNOTATION_ROOT = Path("data")
ANNOTATION_DIR = ANNOTATION_ROOT / "annotations"
EXPECTED_COUNTS = {
    "train": (67686, 145816),
    "val": (18243, 37395),
    "test": (10553, 20867),
}


@pytest.mark.skipif(
    not (ANNOTATION_DIR / "train.json").exists(),
    reason="Official ForestPersons annotations are not available locally",
)
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_official_annotations_validate(split: str) -> None:
    dataset = ForestPersonsDataset(ANNOTATION_ROOT, split=split)
    expected_images, expected_annotations = EXPECTED_COUNTS[split]

    assert len(dataset) == expected_images
    assert len(dataset.coco["annotations"]) == expected_annotations
    assert dataset.validation_warnings == []
    assert {category["supercategory"] for category in dataset.categories.values()} == {
        "person"
    }
