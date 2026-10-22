import json

import pytest
import torch
from PIL import Image

from scripts.train_dinov3_lora_fcos import (
    OnlineForestPersonsDataset,
    sigmoid_focal_loss,
)


def test_mean_focal_normalization_matches_legacy_reduction() -> None:
    logits = torch.tensor([0.3, -0.7, 1.1, -1.4])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    actual = sigmoid_focal_loss(logits, targets, alpha=0.75, gamma=2.0, normalization="mean")

    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = logits.sigmoid()
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = 0.75 * targets + 0.25 * (1.0 - targets)
    expected = (alpha_t * (1.0 - p_t).pow(2.0) * bce).mean()
    assert torch.allclose(actual, expected)


def test_positive_count_normalization_sums_then_divides_once() -> None:
    logits = torch.tensor([0.3, -0.7, 1.1, -1.4])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    mean = sigmoid_focal_loss(logits, targets, alpha=0.75, gamma=2.0, normalization="mean")
    normalized = sigmoid_focal_loss(
        logits,
        targets,
        alpha=0.75,
        gamma=2.0,
        normalization="positive_count",
        positive_count=2,
    )

    assert torch.allclose(normalized, mean * 2.0)


def test_positive_count_normalization_handles_zero_positives_with_finite_gradients() -> None:
    logits = torch.tensor([-0.5, 0.5], requires_grad=True)
    targets = torch.zeros(2)

    loss = sigmoid_focal_loss(
        logits,
        targets,
        alpha=0.75,
        gamma=2.0,
        normalization="positive_count",
        positive_count=0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_unknown_focal_normalization_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported focal normalization"):
        sigmoid_focal_loss(
            torch.zeros(1),
            torch.zeros(1),
            alpha=0.75,
            gamma=2.0,
            normalization="unknown",
        )


def test_online_dataset_resolves_duplicate_filename_by_image_id(tmp_path) -> None:
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images").mkdir()
    Image.new("RGB", (16, 16)).save(tmp_path / "images" / "duplicate.jpg")
    coco = {
        "images": [
            {"id": 1, "file_name": "duplicate.jpg", "width": 16, "height": 16},
            {"id": 2, "file_name": "duplicate.jpg", "width": 16, "height": 16},
        ],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3], "attributes": {"visibility_level": 20}},
            {"id": 22, "image_id": 2, "category_id": 1, "bbox": [2, 2, 7, 7], "attributes": {"visibility_level": 100}},
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    (tmp_path / "annotations" / "train.json").write_text(json.dumps(coco))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": [{"image_id": 1, "file_name": "duplicate.jpg"}]}))

    dataset = OnlineForestPersonsDataset(tmp_path, "train", manifest)
    sample = dataset[0]

    assert sample["image_id"] == 1
    assert sample["ground_truth"] == [
        {"annotation_id": 11, "box": [1.0, 1.0, 4.0, 4.0], "visibility_level": 20, "pose": "unknown"}
    ]


def test_online_dataset_verifies_manifest_filename_for_image_id(tmp_path) -> None:
    (tmp_path / "annotations").mkdir()
    coco = {
        "images": [{"id": 1, "file_name": "actual.jpg", "width": 16, "height": 16}],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    (tmp_path / "annotations" / "train.json").write_text(json.dumps(coco))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"images": [{"image_id": 1, "file_name": "wrong.jpg"}]}))

    with pytest.raises(ValueError, match="filename mismatch"):
        OnlineForestPersonsDataset(tmp_path, "train", manifest)
