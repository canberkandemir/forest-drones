import json
from pathlib import Path

from scripts.export_forestpersons_coco_subset import export_subset


def test_export_subset_prefers_manifest_image_ids_with_duplicate_file_names(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "subset.json"
    coco = {
        "images": [
            {"id": 1, "file_name": "same/path.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "same/path.jpg", "width": 100, "height": 80},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 3,
                "bbox": [10, 10, 20, 20],
                "attributes": {"visibility_level": 20},
            },
            {
                "id": 20,
                "image_id": 2,
                "category_id": 4,
                "bbox": [30, 30, 10, 10],
                "attributes": {"visibility_level": 40},
            },
        ],
        "categories": [
            {"id": 3, "name": "person", "supercategory": "person"},
            {"id": 4, "name": "person", "supercategory": "person"},
        ],
    }
    annotations.write_text(json.dumps(coco), encoding="utf-8")
    manifest.write_text(
        json.dumps({"images": [{"image_id": 2, "file_name": "same/path.jpg"}]}),
        encoding="utf-8",
    )

    summary = export_subset(annotations, manifest, output)
    exported = json.loads(output.read_text(encoding="utf-8"))

    assert summary["images"] == 1
    assert exported["images"][0]["id"] == 2
    assert [annotation["id"] for annotation in exported["annotations"]] == [20]
    assert exported["annotations"][0]["category_id"] == 1


def test_export_subset_writes_valid_one_class_coco(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "subset.json"
    coco = {
        "images": [{"id": 5, "file_name": "a.jpg", "width": 100, "height": 80}],
        "annotations": [
            {
                "id": 6,
                "image_id": 5,
                "category_id": 9,
                "bbox": [-5, 10, 20, 30],
                "attributes": {"visibility_level": 100},
            }
        ],
        "categories": [{"id": 9, "name": "person", "supercategory": "person"}],
    }
    annotations.write_text(json.dumps(coco), encoding="utf-8")
    manifest.write_text(json.dumps([{"file_name": "a.jpg"}]), encoding="utf-8")

    summary = export_subset(annotations, manifest, output)
    exported = json.loads(output.read_text(encoding="utf-8"))

    assert summary["images"] == 1
    assert summary["annotations"] == 1
    assert exported["categories"] == [{"id": 1, "name": "person", "supercategory": "person"}]
    assert exported["annotations"][0]["bbox"] == [0.0, 10.0, 15.0, 30.0]
