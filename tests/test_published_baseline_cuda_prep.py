import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.mmdet_infer_forestpersons import result_to_array


def test_mmdet_result_to_array_extracts_person_class() -> None:
    result = [
        np.array([[1, 2, 11, 22, 0.9]], dtype=np.float32),
        np.array([[5, 6, 7, 8, 0.8]], dtype=np.float32),
    ]

    boxes = result_to_array(result, class_index=0)

    assert boxes.shape == (1, 5)
    assert boxes[0].tolist() == [1.0, 2.0, 11.0, 22.0, 0.8999999761581421]


def test_mmdet_result_to_array_handles_empty_or_tuple_results() -> None:
    result = ([np.zeros((0, 5), dtype=np.float32)], None)

    boxes = result_to_array(result, class_index=0)

    assert boxes.shape == (0, 5)


def test_mmdet_result_to_array_handles_mmdet3_datasample_shape() -> None:
    result = SimpleNamespace(
        pred_instances=SimpleNamespace(
            bboxes=torch.tensor([[1.0, 2.0, 11.0, 22.0], [5.0, 6.0, 7.0, 8.0]]),
            scores=torch.tensor([0.9, 0.8]),
            labels=torch.tensor([0, 2]),
        )
    )

    boxes = result_to_array(result, class_index=0)

    assert boxes.shape == (1, 5)
    assert boxes[0].tolist() == [1.0, 2.0, 11.0, 22.0, 0.8999999761581421]


def test_prepared_val600_coco_export_is_one_class_and_image_ids_are_unique() -> None:
    path = Path("outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    image_ids = [image["id"] for image in data["images"]]
    annotation_image_ids = {annotation["image_id"] for annotation in data["annotations"]}

    assert len(data["images"]) == 600
    assert len(image_ids) == len(set(image_ids))
    assert len(data["annotations"]) == 749
    assert data["categories"] == [{"id": 1, "name": "person", "supercategory": "person"}]
    assert {annotation["category_id"] for annotation in data["annotations"]} == {1}
    assert annotation_image_ids <= set(image_ids)


def test_colab_smoke30_manifest_matches_first_val600_coco_images() -> None:
    coco_path = Path("outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json")
    smoke_path = Path("outputs/baseline_feasibility/forestpersons_val600_smoke30_manifest.json")
    if not coco_path.exists() or not smoke_path.exists():
        return
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    expected_files = [image["file_name"] for image in coco["images"][:30]]
    smoke_files = [image["file_name"] for image in smoke["images"]]

    assert len(smoke_files) == 30
    assert smoke_files == expected_files
    assert all((Path("data/images") / file_name).exists() for file_name in smoke_files)


def test_prepared_train5000_coco_export_is_exact_manifest_size() -> None:
    path = Path("outputs/baseline_feasibility/forestpersons_train5000_oneclass_coco.json")
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    image_ids = [image["id"] for image in data["images"]]

    assert len(data["images"]) == 5000
    assert len(image_ids) == len(set(image_ids))
    assert len(data["annotations"]) == 6415
    assert data["categories"] == [{"id": 1, "name": "person", "supercategory": "person"}]
    assert {annotation["category_id"] for annotation in data["annotations"]} == {1}
