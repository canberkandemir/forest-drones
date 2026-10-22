from __future__ import annotations

import torch

from forestpersons.dinov3_detector import PreprocessInfo
from forestpersons.multiview import (
    TrainableReferencePreservingFusion,
    fuse_aligned_features,
    fuse_selective_aligned_features,
    resized_homography,
    warp_patch_features,
)


def test_trainable_fusion_parameter_count_and_weight_cap() -> None:
    fusion = TrainableReferencePreservingFusion(feature_dim=384, hidden_dim=32)
    assert sum(parameter.numel() for parameter in fusion.parameters()) == 37377
    reference = torch.randn(3, 4, 384)
    sources = [
        (reference + 0.01 * torch.randn_like(reference), torch.ones(3, 4)),
        (reference + 0.01 * torch.randn_like(reference), torch.ones(3, 4)),
    ]
    _, diagnostics = fusion(reference, sources)
    assert torch.all(diagnostics["total_source_weight"] <= 0.500001)


def test_trainable_fusion_exact_reference_fallback() -> None:
    fusion = TrainableReferencePreservingFusion(feature_dim=8, hidden_dim=8)
    reference = torch.randn(3, 4, 8)
    source = torch.randn_like(reference)
    fused, diagnostics = fusion(reference, [(source, torch.zeros(3, 4))])
    assert torch.equal(fused, reference)
    assert diagnostics["fallback_mask"].all()


def test_trainable_fusion_has_finite_gate_gradients() -> None:
    torch.manual_seed(7)
    fusion = TrainableReferencePreservingFusion(feature_dim=8, hidden_dim=8)
    reference = torch.randn(3, 4, 8)
    source = torch.randn_like(reference)
    fused, _ = fusion(reference, [(source, torch.ones(3, 4))])
    loss = fused.square().mean() + fused[..., 0].mean()
    loss.backward()
    gradients = [parameter.grad for parameter in fusion.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None)


def test_trainable_fusion_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(11)
    fusion = TrainableReferencePreservingFusion(feature_dim=8, hidden_dim=8)
    reference = torch.randn(3, 4, 8)
    source = torch.randn_like(reference)
    inputs = [(source, torch.ones(3, 4))]
    expected, _ = fusion(reference, inputs)
    checkpoint = tmp_path / "fusion.pt"
    torch.save(fusion.state_dict(), checkpoint)
    restored = TrainableReferencePreservingFusion(feature_dim=8, hidden_dim=8)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual, _ = restored(reference, inputs)
    assert torch.equal(actual, expected)


def info(width: int = 64, height: int = 48) -> PreprocessInfo:
    return PreprocessInfo(
        original_width=width,
        original_height=height,
        resized_width=width,
        resized_height=height,
        scale=1.0,
        grid_width=width // 16,
        grid_height=height // 16,
    )


def test_resized_homography_accounts_for_independent_scales() -> None:
    source = PreprocessInfo(100, 100, 200, 300, 2.0, 12, 18)
    reference = PreprocessInfo(100, 100, 400, 500, 4.0, 25, 31)
    result = resized_homography(torch.eye(3), source, reference)
    expected = torch.tensor([[2.0, 0.0, 0.0], [0.0, 5.0 / 3.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(result, expected)


def test_identity_warp_preserves_patch_features() -> None:
    metadata = info()
    features = torch.randn(metadata.grid_height, metadata.grid_width, 8)
    warped, validity = warp_patch_features(features, metadata, metadata, torch.eye(3))
    assert torch.allclose(warped, features, atol=1e-5)
    assert torch.allclose(validity, torch.ones_like(validity), atol=1e-5)


def test_patch_translation_uses_patch_center_coordinates() -> None:
    metadata = info()
    features = torch.arange(metadata.grid_height * metadata.grid_width, dtype=torch.float32).reshape(
        metadata.grid_height, metadata.grid_width, 1
    )
    # Source content shifted one patch right in the reference image.
    homography = torch.tensor([[1.0, 0.0, 16.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    warped, validity = warp_patch_features(features, metadata, metadata, homography)
    assert torch.allclose(warped[:, 1:], features[:, :-1], atol=1e-5)
    assert torch.all(validity[:, 1:] > 0.99)
    assert torch.all(validity[:, 0] < 0.01)


def test_fusion_ignores_invalid_source_and_is_normalized() -> None:
    reference = torch.nn.functional.normalize(torch.randn(3, 4, 6), dim=-1)
    source = torch.nn.functional.normalize(torch.randn(3, 4, 6), dim=-1)
    validity = torch.zeros(3, 4)
    fused = fuse_aligned_features(reference, [(source, validity)])
    assert torch.allclose(fused, reference, atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(fused, dim=-1), torch.ones(3, 4), atol=1e-6)


def test_fusion_gradients_flow_through_valid_views() -> None:
    reference = torch.randn(2, 2, 4, requires_grad=True)
    source = torch.randn(2, 2, 4, requires_grad=True)
    fused = fuse_aligned_features(reference, [(source, torch.ones(2, 2))])
    fused.square().sum().backward()
    assert reference.grad is not None and torch.isfinite(reference.grad).all()
    assert source.grad is not None and torch.isfinite(source.grad).all()


def test_selective_fusion_exactly_falls_back_on_disagreement() -> None:
    reference = torch.nn.functional.normalize(torch.randn(5, 5, 8), dim=-1)
    source = -reference
    fused, diagnostics = fuse_selective_aligned_features(
        reference, [(source, torch.ones(5, 5))]
    )
    assert torch.equal(fused, reference)
    assert diagnostics["fallback_mask"].all()
    assert torch.count_nonzero(diagnostics["total_source_weight"]) == 0


def test_selective_fusion_caps_combined_source_weight() -> None:
    reference = torch.nn.functional.normalize(torch.randn(5, 5, 8), dim=-1)
    first = torch.nn.functional.normalize(reference + 0.01 * torch.randn_like(reference), dim=-1)
    second = torch.nn.functional.normalize(reference + 0.01 * torch.randn_like(reference), dim=-1)
    _, diagnostics = fuse_selective_aligned_features(
        reference,
        [(first, torch.ones(5, 5)), (second, torch.ones(5, 5))],
    )
    total = diagnostics["total_source_weight"]
    assert torch.all(total <= 0.500001)
    assert torch.all(total[1:-1, 1:-1] > 0)


def test_selective_fusion_rejects_isolated_false_agreement() -> None:
    reference = torch.zeros(5, 5, 4)
    reference[..., 0] = 1.0
    source = torch.zeros_like(reference)
    source[..., 1] = 1.0
    source[2, 2] = reference[2, 2]
    _, diagnostics = fuse_selective_aligned_features(
        reference, [(source, torch.ones(5, 5))]
    )
    assert diagnostics["total_source_weight"][2, 2] == 0
