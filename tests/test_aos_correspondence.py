from __future__ import annotations

import torch

from forestpersons.aos_correspondence import (
    global_patch_retrieval,
    local_cosine_correspondence,
    yolo_person_patch_mask,
)


def test_yolo_patch_mask_uses_patch_centers() -> None:
    mask = yolo_person_patch_mask([[0, 0.5, 0.5, 0.5, 0.5]], 4, 4)
    assert mask.sum().item() == 4
    assert mask[1:3, 1:3].all()


def test_local_correspondence_recovers_one_patch_shift() -> None:
    rgb = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1)
    thermal = torch.zeros_like(rgb)
    thermal[:, 1:] = rgb[:, :-1]
    same, local, displacement = local_cosine_correspondence(rgb, thermal, radius=1)
    assert same.shape == (3, 4)
    assert local.shape == (3, 4)
    assert displacement.shape == (3, 4, 2)
    assert torch.allclose(local[:, :-1], torch.ones(3, 3), atol=1e-5)


def test_global_retrieval_is_exact_for_identical_unique_features() -> None:
    features = torch.eye(6).reshape(2, 3, 6)
    metrics = global_patch_retrieval(features, features, batch_size=2)
    assert metrics["exact_patch_fraction"] == 1.0
    assert metrics["within_one_patch_fraction"] == 1.0
    assert metrics["median_chebyshev_patch_error"] == 0.0
