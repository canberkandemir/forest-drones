import json
from pathlib import Path

from PIL import Image
import pytest

from forestpersons.sard import (
    build_sard_mountains_manifest,
    manifest_to_coco,
    normalized_yolo_to_xyxy,
    parse_sard_annotation,
)
from scripts.evaluate_sard_pilot import _greedy_match


def test_normalized_yolo_coordinate_conversion() -> None:
    assert normalized_yolo_to_xyxy(0.5, 0.5, 0.25, 0.5, 400, 200) == (150.0, 50.0, 250.0, 150.0)


def test_empty_negative_annotation_file(tmp_path: Path) -> None:
    path = tmp_path / "1.txt"
    path.write_text("", encoding="utf-8")
    assert parse_sard_annotation(path, 3840, 2160) == []


def test_nonperson_category_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "1.txt"
    path.write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="category 1"):
        parse_sard_annotation(path, 3840, 2160)


def test_manifest_requires_complete_official_scenario(tmp_path: Path) -> None:
    (tmp_path / "frames").mkdir()
    (tmp_path / "annotations").mkdir()
    with pytest.raises(ValueError, match="Expected 200"):
        build_sard_mountains_manifest(tmp_path, source_revision="test")


def test_coco_export_preserves_empty_images() -> None:
    manifest = {
        "images": [
            {"image_id": 1, "file_name": "1.jpg", "width": 10, "height": 10, "person_annotations": []},
            {"image_id": 2, "file_name": "2.jpg", "width": 10, "height": 10, "person_annotations": [
                {"annotation_id": 1, "category_id": 1, "bbox_xywh": [1, 2, 3, 4], "area": 12}
            ]},
        ]
    }
    coco = manifest_to_coco(manifest)
    assert [item["id"] for item in coco["images"]] == [1, 2]
    assert coco["annotations"][0]["image_id"] == 2


def test_greedy_match_is_one_to_one() -> None:
    gt = [{"bbox_xyxy": [0, 0, 10, 10]}, {"bbox_xyxy": [1, 1, 11, 11]}]
    predictions = [{"bbox": [0, 0, 10, 10], "fused_score": 0.9}]
    matched_gt, matched_pred, matches = _greedy_match(gt, predictions, threshold=0.5)
    assert len(matched_gt) == len(matched_pred) == len(matches) == 1
