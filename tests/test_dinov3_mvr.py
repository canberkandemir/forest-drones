from __future__ import annotations

import torch

from scripts.evaluate_dinov3_mvr_fcos import (
    comparison_report,
    flatten_sweep_rows,
    matched_budget_rows,
)
from scripts.train_dinov3_mvr_fcos import make_model
from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3MVRFCOSHead,
    DINOv3MVRNativeFCOSHead,
    DINOv3MVRNativeReadout,
    DINOv3MVRReadout,
    PreprocessInfo,
    load_mvr_feature_cache,
    save_mvr_feature_cache,
)


def normalized_layers(height: int = 5, width: int = 7) -> torch.Tensor:
    return torch.nn.functional.normalize(torch.randn(4, height, width, 384), dim=-1)


def test_mvr_readout_starts_as_exact_final_layer_fallback() -> None:
    layers = normalized_layers()
    readout = DINOv3MVRReadout()
    output = readout(layers)
    assert output.shape == layers[-1].shape
    assert torch.allclose(output, layers[-1], atol=1e-6)


def test_mvr_fcos_matches_baseline_at_initialization_and_has_gradients() -> None:
    torch.manual_seed(7)
    layers = normalized_layers()
    baseline = DINOv3FCOSHead()
    mvr = DINOv3MVRFCOSHead()
    mvr.fcos.load_state_dict(baseline.state_dict())
    baseline_outputs = baseline(layers[-1])
    mvr_outputs = mvr(layers)
    for expected, actual in zip(baseline_outputs, mvr_outputs):
        assert torch.allclose(actual, expected, atol=1e-6)
    sum(item.square().mean() for item in mvr_outputs).backward()
    assert mvr.readout.residual.weight.grad is not None
    assert torch.isfinite(mvr.readout.residual.weight.grad).all()
    assert mvr.fcos.classification.weight.grad is not None


def test_native_mvr_starts_as_exact_fallback_and_matches_baseline() -> None:
    torch.manual_seed(11)
    layers = normalized_layers()
    baseline = DINOv3FCOSHead()
    native = DINOv3MVRNativeFCOSHead()
    native.fcos.load_state_dict(baseline.state_dict())

    assert torch.allclose(native.readout(layers), layers[-1], atol=1e-6)
    for expected, actual in zip(baseline(layers[-1]), native(layers)):
        assert torch.allclose(actual, expected, atol=1e-6)


def test_native_mvr_projection_gradients_flow_after_residual_update() -> None:
    torch.manual_seed(13)
    layers = normalized_layers()
    model = DINOv3MVRNativeFCOSHead()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    first_loss = sum(output.square().mean() for output in model(layers))
    first_loss.backward()
    assert model.readout.residual.weight.grad is not None
    assert torch.isfinite(model.readout.residual.weight.grad).all()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second_loss = sum(output.square().mean() for output in model(layers))
    second_loss.backward()
    projection_grad = model.readout.layer_projections[0][0].weight.grad
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert torch.count_nonzero(projection_grad) > 0


def test_native_mvr_is_smaller_and_keeps_full_mvr_checkpoint_compatible() -> None:
    native = DINOv3MVRNativeReadout()
    full = DINOv3MVRReadout()
    assert sum(parameter.numel() for parameter in native.parameters()) == 197_248
    assert sum(parameter.numel() for parameter in native.parameters()) < sum(
        parameter.numel() for parameter in full.parameters()
    )
    assert not hasattr(native, "fine_context")
    assert not hasattr(native, "coarse_context")

    legacy_state = full.state_dict()
    reloaded = DINOv3MVRReadout()
    reloaded.load_state_dict(legacy_state, strict=True)
    assert tuple(reloaded.state_dict()) == tuple(legacy_state)


def test_model_factory_supports_all_three_arms_with_matched_fcos() -> None:
    torch.manual_seed(17)
    fcos = DINOv3FCOSHead()
    state = fcos.state_dict()
    baseline = make_model("baseline", state)
    native = make_model("mvr_native", state)
    full = make_model("mvr", state)

    assert isinstance(baseline, DINOv3FCOSHead)
    assert isinstance(native, DINOv3MVRNativeFCOSHead)
    assert isinstance(full, DINOv3MVRFCOSHead)
    for name, expected in state.items():
        assert torch.equal(native.fcos.state_dict()[name], expected)
        assert torch.equal(full.fcos.state_dict()[name], expected)


def test_mvr_cache_round_trip(tmp_path) -> None:
    features = torch.nn.functional.normalize(torch.randn(3, 4, 6, 384), dim=-1)
    info = PreprocessInfo(
        original_width=96,
        original_height=64,
        resized_width=96,
        resized_height=64,
        scale=1.0,
        grid_width=6,
        grid_height=4,
    )
    path = tmp_path / "features.npz"
    save_mvr_feature_cache(path, features, info, (3, 6, 9))
    loaded, loaded_info, layers = load_mvr_feature_cache(path)
    assert loaded_info == info
    assert layers == (3, 6, 9)
    assert loaded.shape == features.shape
    assert torch.allclose(loaded, features, atol=1e-3)


def test_mvr_readout_uses_conversion_friendly_trainable_modules() -> None:
    readout = DINOv3MVRReadout()
    allowed = (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.Sequential, torch.nn.ModuleList)
    assert all(isinstance(module, allowed) for module in readout.modules() if module is not readout)


def test_mvr_evaluation_tables_preserve_sweep_and_matched_budgets() -> None:
    metric = {
        "confidence_threshold": 0.5,
        "overall": {
            "precision": 0.75,
            "recall": 0.6,
            "true_positives": 3,
            "false_positives": 1,
            "false_negatives": 2,
        },
        "by_visibility": {
            "20": {"recall": 0.5},
            "40": {"recall": 0.6},
            "70": {"recall": 0.7},
            "100": {"recall": 0.8},
        },
        "false_positives_per_image": 0.25,
    }
    result = {
        "arm": "baseline",
        "AP": 0.3,
        "AP50": 0.6,
        "AP75": 0.2,
        "AR100": 0.4,
        "confidence_sweep": [metric],
        "matched_fp_budgets": {
            "0.15": None,
            "0.25": metric,
            "0.35": metric,
            "0.425": metric,
        },
    }

    sweep = flatten_sweep_rows([result])
    matched = matched_budget_rows([result])

    assert sweep[0]["R20"] == 0.5
    assert sweep[0]["R100"] == 0.8
    assert matched[0]["confidence_threshold"] is None
    assert matched[1]["recall"] == 0.6
    assert matched[-1]["R40"] == 0.6
    assert "exploratory" in comparison_report([result], matched)
