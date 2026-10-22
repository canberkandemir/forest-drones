from __future__ import annotations

import torch
from PIL import Image

from forestpersons.deimv2_adapter import (
    DEIMV2_COCO_PERSON_LABEL,
    build_deimv2_preprocess,
    mps_compatible_distribution_integral,
    official_result_to_person_predictions,
)


def test_deimv2_preprocess_has_expected_shape_and_finite_values() -> None:
    image = Image.new("RGB", (1920, 1080), color=(10, 20, 30))
    tensor = build_deimv2_preprocess(640)(image)
    assert tensor.shape == (3, 640, 640)
    assert torch.isfinite(tensor).all()


def test_deimv2_person_mapping_clips_and_filters_outputs() -> None:
    result = {
        "labels": torch.tensor([DEIMV2_COCO_PERSON_LABEL, 1, DEIMV2_COCO_PERSON_LABEL]),
        "boxes": torch.tensor([[-5.0, 2.0, 110.0, 90.0], [1.0, 1.0, 2.0, 2.0], [5.0, 5.0, 5.0, 8.0]]),
        "scores": torch.tensor([0.8, 0.9, 0.7]),
    }
    predictions = official_result_to_person_predictions(
        result,
        image_id=42,
        width=100,
        height=80,
        min_score=0.1,
    )
    assert len(predictions) == 1
    assert predictions[0].image_id == 42
    assert predictions[0].category_id == 1
    assert predictions[0].box == [0.0, 2.0, 100.0, 80.0]


def test_deimv2_conversion_rejects_mismatched_output_lengths() -> None:
    result = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]]),
        "scores": torch.tensor([0.5]),
    }
    try:
        official_result_to_person_predictions(result, image_id=1, width=10, height=10)
    except ValueError as exc:
        assert "equal lengths" in str(exc)
    else:
        raise AssertionError("Expected mismatched DEIMv2 outputs to raise ValueError")


def test_forestpersons_single_class_mapping_is_zero_based() -> None:
    official_category_id = 1
    internal_label = official_category_id - 1
    assert internal_label == DEIMV2_COCO_PERSON_LABEL


def test_mps_distribution_integral_matches_linear_and_gradients() -> None:
    torch.manual_seed(7)
    reg_max = 32
    project = torch.linspace(-4.0, 4.0, reg_max + 1)
    official_logits = torch.randn(2, 300, 4 * (reg_max + 1), requires_grad=True)
    compatible_logits = official_logits.detach().clone().requires_grad_(True)

    shape = official_logits.shape
    probabilities = torch.softmax(official_logits.reshape(-1, reg_max + 1), dim=1)
    expected = torch.nn.functional.linear(probabilities, project).reshape(-1, 4)
    expected = expected.reshape(list(shape[:-1]) + [-1])
    actual = mps_compatible_distribution_integral(compatible_logits, project, reg_max)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert official_logits.grad is not None
    assert compatible_logits.grad is not None
    assert torch.allclose(compatible_logits.grad, official_logits.grad, atol=1e-7, rtol=1e-5)
