from __future__ import annotations

import torch

from scripts.evaluate_dinov3_mvr_fcos import (
    comparison_report,
    flatten_sweep_rows,
    matched_budget_rows,
)
from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3MVRFCOSHead,
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
        "by_visibility": {"20": {"recall": 0.5}},
        "false_positives_per_image": 0.25,
    }
    result = {
        "arm": "baseline",
        "AP": 0.3,
        "AP50": 0.6,
        "AP75": 0.2,
        "AR100": 0.4,
        "confidence_sweep": [metric],
        "matched_fp_budgets": {"0.15": None, "0.25": metric, "0.35": metric},
    }

    sweep = flatten_sweep_rows([result])
    matched = matched_budget_rows([result])

    assert sweep[0]["R20"] == 0.5
    assert matched[0]["confidence_threshold"] is None
    assert matched[1]["recall"] == 0.6
    assert "exploratory" in comparison_report([result], matched)
