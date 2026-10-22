from __future__ import annotations

import pytest
import torch

from forestpersons.dinov3_ann2snn import (
    ActivationPercentileCalibrator,
    MVR_SPIKE_SITES,
    SpikingMVRReadout,
    aggregate_spike_activity,
    ann_mvr_activations,
    integrate_and_fire_rate,
)
from forestpersons.dinov3_detector import DINOv3MVRReadout
from scripts.evaluate_dinov3_mvr_ann2snn import select_manifest_items


def test_integrate_and_fire_rate_has_valid_activity_and_improves_with_timesteps() -> None:
    activation = torch.tensor([0.0, 0.15, 0.35, 0.75])
    decoded4, stats4 = integrate_and_fire_rate(activation, threshold=1.0, timesteps=4)
    decoded16, stats16 = integrate_and_fire_rate(activation, threshold=1.0, timesteps=16)

    assert decoded4.shape == activation.shape
    assert torch.all(decoded4 >= 0.0)
    assert torch.mean(torch.abs(decoded16 - activation)) <= torch.mean(
        torch.abs(decoded4 - activation)
    )
    assert 0.0 <= stats4.activity <= 1.0
    assert 0.0 <= stats16.activity <= 1.0


def test_calibration_and_spiking_mvr_readout_are_finite() -> None:
    torch.manual_seed(4)
    readout = DINOv3MVRReadout()
    features = torch.nn.functional.normalize(torch.randn(4, 4, 5, 384), dim=-1)
    activations = ann_mvr_activations(readout, features)
    calibrator = ActivationPercentileCalibrator(samples_per_site_per_image=128)
    calibrator.update(activations)
    thresholds = calibrator.thresholds(percentile=99.0)

    assert set(thresholds) == set(MVR_SPIKE_SITES)
    assert all(value > 0.0 for value in thresholds.values())
    spiking = SpikingMVRReadout(readout, thresholds, timesteps=4)
    output, stats = spiking(features)

    assert output.shape == (4, 5, 384)
    assert torch.isfinite(output).all()
    activity = aggregate_spike_activity([stats])
    assert set(activity) == set(MVR_SPIKE_SITES) | {"overall"}
    assert all(0.0 <= value <= 1.0 for value in activity.values())


def test_spiking_readout_rejects_incomplete_thresholds() -> None:
    with pytest.raises(ValueError, match="every MVR spike site"):
        SpikingMVRReadout(DINOv3MVRReadout(), {"projection_0": 1.0}, timesteps=4)


def test_stratified_manifest_selection_is_deterministic_and_balanced() -> None:
    items = [
        {"image_id": level * 10 + index, "target_visibility": level}
        for level in (20, 40, 70, 100)
        for index in range(4)
    ]

    selected = select_manifest_items(items, limit=10, sampling="stratified")
    counts = {
        level: sum(int(item["target_visibility"]) == level for item in selected)
        for level in (20, 40, 70, 100)
    }

    assert counts == {20: 3, 40: 3, 70: 2, 100: 2}
    assert selected == select_manifest_items(items, limit=10, sampling="stratified")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_spiking_mvr_readout_mps() -> None:
    device = torch.device("mps")
    readout = DINOv3MVRReadout().to(device)
    features = torch.nn.functional.normalize(
        torch.randn(4, 3, 4, 384, device=device), dim=-1
    )
    thresholds = {name: 1.0 for name in MVR_SPIKE_SITES}
    output, stats = SpikingMVRReadout(readout, thresholds, timesteps=4).to(device)(features)
    assert output.device.type == "mps"
    assert torch.isfinite(output).all()
    assert len(stats) == len(MVR_SPIKE_SITES)
