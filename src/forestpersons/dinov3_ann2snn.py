"""Hybrid ANN2SNN conversion utilities for the frozen DINOv3 MVR detector.

Only the ReLU activations in the MVR readout are converted to deterministic
integrate-and-fire rate codes. Signed DINOv3 inputs, residual fusion,
normalization, the FCOS tower, box regression, and decoding remain analog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .dinov3_detector import DINOv3MVRReadout


MVR_SPIKE_SITES = (
    "projection_0",
    "projection_1",
    "projection_2",
    "projection_3",
    "fine_context",
    "coarse_context",
)


@dataclass(frozen=True)
class SpikeSiteStats:
    spikes: torch.Tensor
    neuron_steps: int

    @property
    def activity(self) -> float:
        if not self.neuron_steps:
            return 0.0
        return float(self.spikes.detach().cpu().item()) / self.neuron_steps


def integrate_and_fire_rate(
    activation: torch.Tensor,
    threshold: float | torch.Tensor,
    timesteps: int,
) -> tuple[torch.Tensor, SpikeSiteStats]:
    """Quantize a nonnegative ANN activation using soft-reset IF neurons.

    The same analog current is injected at every timestep. The decoded output
    is spike_count * threshold / T, which approaches the clipped ReLU response
    as T increases. At most one spike per neuron and timestep is emitted.
    """

    if timesteps <= 0:
        raise ValueError("timesteps must be positive")
    threshold_tensor = torch.as_tensor(
        threshold, dtype=activation.dtype, device=activation.device
    )
    if threshold_tensor.numel() != 1 or float(threshold_tensor) <= 0.0:
        raise ValueError("threshold must be a positive scalar")
    current = activation.clamp_min(0.0)
    membrane = torch.zeros_like(current)
    spike_count = torch.zeros_like(current)
    for _ in range(timesteps):
        membrane = membrane + current
        spikes = (membrane >= threshold_tensor).to(current.dtype)
        membrane = membrane - spikes * threshold_tensor
        spike_count = spike_count + spikes
    decoded = spike_count * threshold_tensor / float(timesteps)
    stats = SpikeSiteStats(
        spikes=spike_count.detach().sum(),
        neuron_steps=int(spike_count.numel() * timesteps),
    )
    return decoded, stats


def ann_mvr_activations(
    readout: DINOv3MVRReadout,
    layer_features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the six trained MVR ReLU activations used for calibration."""

    if layer_features.ndim != 4:
        raise ValueError("layer_features must have shape [layers, height, width, channels]")
    projected = []
    activations: dict[str, torch.Tensor] = {}
    for index, (projection, features) in enumerate(
        zip(readout.layer_projections, layer_features)
    ):
        preactivation = projection[0](features.permute(2, 0, 1).unsqueeze(0))
        value = F.relu(preactivation)
        activations[f"projection_{index}"] = value
        projected.append(value)
    native = torch.cat(projected, dim=1)
    fine_input = F.interpolate(native, scale_factor=2.0, mode="nearest")
    activations["fine_context"] = F.relu(readout.fine_context[0](fine_input))
    coarse_input = F.avg_pool2d(native, kernel_size=2, stride=2, ceil_mode=True)
    activations["coarse_context"] = F.relu(readout.coarse_context[0](coarse_input))
    return activations


class ActivationPercentileCalibrator:
    """Deterministic bounded sampling for scalar activation thresholds."""

    def __init__(self, samples_per_site_per_image: int = 2048) -> None:
        if samples_per_site_per_image <= 0:
            raise ValueError("samples_per_site_per_image must be positive")
        self.samples_per_site_per_image = samples_per_site_per_image
        self._samples: dict[str, list[torch.Tensor]] = {
            name: [] for name in MVR_SPIKE_SITES
        }

    def update(self, activations: dict[str, torch.Tensor]) -> None:
        if set(activations) != set(MVR_SPIKE_SITES):
            raise ValueError("activation sites do not match the MVR conversion boundary")
        for name, tensor in activations.items():
            values = tensor.detach().float().cpu().flatten()
            if values.numel() > self.samples_per_site_per_image:
                indices = torch.linspace(
                    0,
                    values.numel() - 1,
                    self.samples_per_site_per_image,
                ).long()
                values = values[indices]
            self._samples[name].append(values)

    def thresholds(self, percentile: float = 99.9, minimum: float = 1e-6) -> dict[str, float]:
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")
        result = {}
        quantile = percentile / 100.0
        for name in MVR_SPIKE_SITES:
            if not self._samples[name]:
                raise ValueError(f"No calibration samples collected for {name}")
            values = torch.cat(self._samples[name]).clamp_min(0.0)
            result[name] = max(float(torch.quantile(values, quantile).item()), minimum)
        return result


class SpikingMVRReadout(nn.Module):
    """Rate-coded IF replacement for the trained MVR readout ReLU sites."""

    def __init__(
        self,
        ann_readout: DINOv3MVRReadout,
        thresholds: dict[str, float],
        timesteps: int,
    ) -> None:
        super().__init__()
        if set(thresholds) != set(MVR_SPIKE_SITES):
            raise ValueError("thresholds must cover every MVR spike site")
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        self.readout = ann_readout
        self.thresholds = {name: float(value) for name, value in thresholds.items()}
        self.timesteps = int(timesteps)

    def _spike(
        self,
        name: str,
        preactivation: torch.Tensor,
        stats: dict[str, SpikeSiteStats],
    ) -> torch.Tensor:
        decoded, site_stats = integrate_and_fire_rate(
            F.relu(preactivation), self.thresholds[name], self.timesteps
        )
        stats[name] = site_stats
        return decoded

    def forward(
        self, layer_features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, SpikeSiteStats]]:
        if layer_features.ndim != 4:
            raise ValueError("layer_features must have shape [layers, height, width, channels]")
        layers, height, width, channels = layer_features.shape
        if layers != self.readout.layer_count or channels != self.readout.input_dim:
            raise ValueError("layer feature dimensions do not match the trained readout")

        stats: dict[str, SpikeSiteStats] = {}
        projected = []
        for index, (projection, features) in enumerate(
            zip(self.readout.layer_projections, layer_features)
        ):
            preactivation = projection[0](features.permute(2, 0, 1).unsqueeze(0))
            projected.append(self._spike(f"projection_{index}", preactivation, stats))
        native = torch.cat(projected, dim=1)

        fine_input = F.interpolate(native, scale_factor=2.0, mode="nearest")
        fine = self._spike(
            "fine_context", self.readout.fine_context[0](fine_input), stats
        )
        fine = F.avg_pool2d(fine, kernel_size=2, stride=2)

        coarse_input = F.avg_pool2d(native, kernel_size=2, stride=2, ceil_mode=True)
        coarse = self._spike(
            "coarse_context", self.readout.coarse_context[0](coarse_input), stats
        )
        coarse = F.interpolate(coarse, size=(height, width), mode="nearest")

        delta = self.readout.residual(torch.cat((native, fine, coarse), dim=1)).squeeze(0)
        reference = layer_features[-1].permute(2, 0, 1)
        fused = F.normalize(reference + delta, dim=0)
        return fused.permute(1, 2, 0), stats


def aggregate_spike_activity(stats: Iterable[dict[str, SpikeSiteStats]]) -> dict[str, float]:
    totals = {name: [0, 0] for name in MVR_SPIKE_SITES}
    for image_stats in stats:
        for name, item in image_stats.items():
            totals[name][0] += int(item.spikes.detach().cpu().item())
            totals[name][1] += item.neuron_steps
    result = {
        name: spikes / steps if steps else 0.0
        for name, (spikes, steps) in totals.items()
    }
    total_spikes = sum(value[0] for value in totals.values())
    total_steps = sum(value[1] for value in totals.values())
    result["overall"] = total_spikes / total_steps if total_steps else 0.0
    return result
