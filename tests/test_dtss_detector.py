from __future__ import annotations

import torch

from forestpersons.dtss_detector import (
    DTSSDetectorConfig,
    DTSSPatchDetector,
    assign_boxes_to_patches,
    decode_patch_boxes,
    scale_xywh_to_normalized_xyxy,
    set_dtss_learning,
)
from forestpersons.evaluation import annotations_to_eval_ground_truth
from scripts.run_dtss_forestpersons_detection_pilot import select_records
from scripts.train_dtss_forestpersons_full import epoch_order, load_manifest_records


def test_independent_xy_scaling_and_normalization() -> None:
    box = scale_xywh_to_normalized_xyxy([100, 50, 200, 100], (1000, 500), (200, 200))
    assert torch.allclose(box, torch.tensor([0.1, 0.1, 0.3, 0.3]))


def test_assignment_gives_each_box_a_distinct_positive_patch() -> None:
    boxes = torch.tensor([[0.10, 0.10, 0.20, 0.20], [0.11, 0.11, 0.21, 0.21]])
    objectness, targets, positive = assign_boxes_to_patches(boxes, 4)
    assert int(positive.sum()) == 2
    assert int(objectness.sum()) == 2
    assert torch.count_nonzero(targets[positive]) == 8


def test_zero_raw_offsets_decode_around_patch_centers() -> None:
    raw = torch.zeros(1, 4, 4)
    boxes = decode_patch_boxes(raw, 2)
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    assert torch.allclose(centers[0, 0], torch.tensor([0.25, 0.25]))
    assert torch.all(boxes >= 0) and torch.all(boxes <= 1)


def test_detector_shapes_losses_and_gradients() -> None:
    torch.manual_seed(4)
    config = DTSSDetectorConfig(image_size=32, patch_size=8, embed_dim=16, num_heads=4)
    model = DTSSPatchDetector(config)
    images = torch.randn(2, 3, 32, 32)
    boxes = [torch.tensor([[0.2, 0.2, 0.5, 0.6]]), torch.tensor([[0.5, 0.4, 0.8, 0.9]])]
    outputs = model(images)
    assert outputs["objectness_logits"].shape == (2, 16)
    assert outputs["boxes"].shape == (2, 16, 4)
    losses = model.compute_loss(outputs, boxes)
    losses["detection"].backward()
    assert torch.isfinite(losses["detection"])
    assert model.patch_embedding.weight.grad is not None
    assert model.box_regression.weight.grad is not None


def test_fixed_t4_freezes_prefix_masks() -> None:
    model = DTSSPatchDetector(
        DTSSDetectorConfig(image_size=32, patch_size=8, embed_dim=16, num_heads=4)
    )
    set_dtss_learning(model, False)
    assert all(module().active_timesteps == 4 for module in model.timestep_masks().values())
    assert all(
        not module.raw_timestep_parameters.requires_grad
        for module in model.timestep_masks().values()
    )


def test_pilot_selection_normalizes_attributes_for_evaluation(tmp_path) -> None:
    images = []
    annotations = []
    for index, visibility in enumerate((20, 40, 70, 100), start=1):
        images.append({"image_id": index, "file_name": f"{index}.jpg"})
        annotations.append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [1, 2, 3, 4],
                "attributes": {"visibility_level": visibility, "pose": "standing"},
            }
        )
    manifest = tmp_path / "manifest.json"
    annotation_path = tmp_path / "annotations.json"
    manifest.write_text(__import__("json").dumps({"images": images}))
    annotation_path.write_text(__import__("json").dumps({"annotations": annotations}))
    records = select_records(manifest, annotation_path, 1)
    ground_truth = annotations_to_eval_ground_truth(
        records[0]["image_id"], records[0]["annotations"]
    )
    assert len(ground_truth) == 1
    assert ground_truth[0].visibility_level == 20


def test_full_records_resolve_duplicate_filenames_by_image_id(tmp_path) -> None:
    from PIL import Image
    import json

    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (16, 16)).save(image_root / "duplicate.jpg")
    manifest = tmp_path / "manifest.json"
    annotations = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps(
            {
                "images": [
                    {"image_id": 10, "file_name": "duplicate.jpg", "width": 16, "height": 16},
                    {"image_id": 11, "file_name": "duplicate.jpg", "width": 16, "height": 16},
                ]
            }
        )
    )
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 10, "file_name": "duplicate.jpg", "width": 16, "height": 16},
                    {"id": 11, "file_name": "duplicate.jpg", "width": 16, "height": 16},
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "bbox": [1, 2, 3, 4],
                        "attributes": {"visibility_level": 20},
                    },
                    {
                        "id": 2,
                        "image_id": 11,
                        "bbox": [5, 6, 7, 8],
                        "attributes": {"visibility_level": 100},
                    },
                    {"id": 3, "image_id": 11, "bbox": [0, 0, 2, 2], "attributes": {}},
                ],
            }
        )
    )
    records, stats = load_manifest_records(manifest, annotations, image_root)
    assert [record["image_id"] for record in records] == [10, 11]
    assert records[0]["annotations"][0]["bbox"] == [1, 2, 3, 4]
    assert records[1]["annotations"][0]["bbox"] == [5, 6, 7, 8]
    assert stats == {
        "images": 2,
        "annotations": 2,
        "excluded_annotations": 1,
        "distinct_filenames": 1,
    }


def test_epoch_order_is_matched_and_resume_safe() -> None:
    first = epoch_order(20, seed=17, epoch=3)
    repeated = epoch_order(20, seed=17, epoch=3)
    next_epoch = epoch_order(20, seed=17, epoch=4)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_epoch)
    assert sorted(first.tolist()) == list(range(20))
