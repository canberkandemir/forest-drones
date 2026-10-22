from __future__ import annotations

import pytest
import torch

from forestpersons.aos_multimodal_detection import (
    feature_for_arm,
    fcos_head_loss,
    select_evenly,
    verify_scene_disjoint,
    yolo_rows_to_ground_truth,
)


def test_even_selection_is_deterministic_and_includes_endpoints() -> None:
    assert select_evenly(list(range(10)), 4) == [0, 3, 6, 9]


def test_scene_disjointness_rejects_leakage() -> None:
    verify_scene_disjoint({"TRAIN": {"F8"}, "VALID": {"O6"}, "TEST": {"F0"}})
    with pytest.raises(ValueError, match="Scene leakage"):
        verify_scene_disjoint({"TRAIN": {"F8"}, "VALID": {"F8"}, "TEST": {"F0"}})


def test_mean_fusion_is_normalized_and_arms_preserve_shape() -> None:
    rgb = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1)
    thermal = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1)
    assert feature_for_arm(rgb, thermal, "rgb") is rgb
    assert feature_for_arm(rgb, thermal, "thermal") is thermal
    fused = feature_for_arm(rgb, thermal, "mean_fusion")
    assert fused.shape == rgb.shape
    assert torch.allclose(fused.norm(dim=-1), torch.ones(3, 4), atol=1e-5)
    weighted = feature_for_arm(rgb, thermal, "thermal_weighted_075")
    expected = torch.nn.functional.normalize(0.25 * rgb + 0.75 * thermal, dim=-1)
    assert torch.allclose(weighted, expected)
    assert torch.allclose(weighted.norm(dim=-1), torch.ones(3, 4), atol=1e-5)
    with pytest.raises(ValueError, match="Unknown"):
        feature_for_arm(rgb, thermal, "attention")


def test_yolo_conversion_and_fcos_loss_have_finite_gradients() -> None:
    ground_truth = yolo_rows_to_ground_truth([[0, 0.5, 0.5, 0.25, 0.5]])
    assert ground_truth[0]["box"] == [192.0, 128.0, 320.0, 384.0]
    logits = torch.zeros(2, 2, requires_grad=True)
    raw_box = torch.zeros(2, 2, 4, requires_grad=True)
    center_logits = torch.zeros(2, 2, requires_grad=True)
    positive = torch.tensor([[False, False], [False, True]])
    classification = positive.float()
    box_targets = torch.ones(2, 2, 4)
    centerness = positive.float()
    loss, components = fcos_head_loss(
        logits,
        raw_box,
        center_logits,
        classification,
        box_targets,
        centerness,
        positive,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert raw_box.grad is not None and torch.isfinite(raw_box.grad).all()
