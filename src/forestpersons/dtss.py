"""Minimal Dynamic Time-Step Spiking layers.

This module implements the masking mechanism from Section III-A of
"Spiking Neural Networks with Dynamic Time Steps for Vision Transformers"
(arXiv:2311.16456). It is intentionally independent of the detector stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _DTSSMaskStep(torch.autograd.Function):
    """Binary threshold with the paper's rectangular surrogate derivative."""

    @staticmethod
    def forward(ctx, scores: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scores)
        return (scores >= 1.0).to(dtype=scores.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (scores,) = ctx.saved_tensors
        surrogate = ((scores > 0.0) & (scores < 2.0)).to(dtype=scores.dtype)
        return (gradient * surrogate,)


class _TriangularSpike(torch.autograd.Function):
    """Heaviside spike with the triangular LIF surrogate from the paper."""

    @staticmethod
    def forward(ctx, normalized_offset: torch.Tensor, gamma: float) -> torch.Tensor:
        ctx.save_for_backward(normalized_offset)
        ctx.gamma = float(gamma)
        return (normalized_offset >= 0.0).to(dtype=normalized_offset.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (normalized_offset,) = ctx.saved_tensors
        surrogate = ctx.gamma * (1.0 - normalized_offset.abs()).clamp(min=0.0)
        return gradient * surrogate, None


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def _initial_cumulative_scores(t_max: int, active_timesteps: int) -> torch.Tensor:
    if not 1 <= active_timesteps <= t_max:
        raise ValueError("active_timesteps must be in [1, t_max]")
    if active_timesteps == t_max:
        return torch.linspace(1.5, 1.1, t_max)

    # Keep initialization away from the discontinuity at exactly 1.0. Tiny
    # CPU/MPS rounding differences should not change the initial binary mask.
    active = torch.linspace(1.5, 1.1, active_timesteps)
    trailing = torch.linspace(0.75, 0.25, t_max - active_timesteps)
    return torch.cat((active, trailing))


@dataclass(frozen=True)
class DTSSDiagnostics:
    scores: torch.Tensor
    mask: torch.Tensor

    @property
    def active_timesteps(self) -> int:
        return int(self.mask.detach().sum().item())


class DynamicTimeStepMask(nn.Module):
    """Learn a prefix mask over at most ``t_max`` time steps.

    The paper computes scores by multiplying a non-negative trainable vector by
    a lower-triangular coefficient matrix. This is equivalent to a reverse
    cumulative sum. We parameterize the increments with ``softplus`` so they
    remain non-negative after optimization, guaranteeing a non-increasing score
    and therefore a consecutive prefix mask.
    """

    def __init__(self, t_max: int = 4, initial_timesteps: int | None = None) -> None:
        super().__init__()
        if t_max < 1:
            raise ValueError("t_max must be positive")
        initial_timesteps = initial_timesteps or max(1, math.ceil(t_max / 2))
        scores = _initial_cumulative_scores(t_max, initial_timesteps)
        increments = torch.cat((scores[:-1] - scores[1:], scores[-1:]))
        self.raw_timestep_parameters = nn.Parameter(_inverse_softplus(increments))
        self.t_max = int(t_max)

    @property
    def timestep_parameters(self) -> torch.Tensor:
        return F.softplus(self.raw_timestep_parameters)

    def cumulative_scores(self) -> torch.Tensor:
        return torch.flip(
            torch.cumsum(torch.flip(self.timestep_parameters, dims=(0,)), dim=0),
            dims=(0,),
        )

    def forward(self) -> DTSSDiagnostics:
        scores = self.cumulative_scores()
        return DTSSDiagnostics(scores=scores, mask=_DTSSMaskStep.apply(scores))

    def mask_loss(self) -> torch.Tensor:
        """Differentiable count of active time steps, as in the paper."""

        return self().mask.sum()


def _lif_sequence(
    currents: torch.Tensor,
    diagnostics: DTSSDiagnostics,
    *,
    leak: float,
    threshold: float,
    surrogate_gamma: float,
) -> torch.Tensor:
    membrane = torch.zeros_like(currents[0])
    outputs: list[torch.Tensor] = []
    for timestep in range(currents.shape[0]):
        potential = leak * membrane + currents[timestep]
        normalized_offset = potential / threshold - 1.0
        spikes = _TriangularSpike.apply(normalized_offset, surrogate_gamma)
        membrane = potential - spikes * threshold
        outputs.append(spikes * diagnostics.mask[timestep])
    return torch.stack(outputs)


class DTSSLIF(nn.Module):
    """LIF activation followed by the paper's timestep output mask."""

    def __init__(
        self,
        *,
        t_max: int = 4,
        initial_timesteps: int | None = None,
        leak: float = 0.5,
        threshold: float = 1.0,
        surrogate_gamma: float = 0.3,
    ) -> None:
        super().__init__()
        self.dtss = DynamicTimeStepMask(t_max=t_max, initial_timesteps=initial_timesteps)
        self.leak = float(leak)
        self.threshold = float(threshold)
        self.surrogate_gamma = float(surrogate_gamma)

    def forward(self, currents: torch.Tensor) -> tuple[torch.Tensor, DTSSDiagnostics]:
        if currents.ndim < 2:
            raise ValueError("currents must have shape [T, ..., features]")
        if currents.shape[0] != self.dtss.t_max:
            raise ValueError(f"expected T={self.dtss.t_max}, received T={currents.shape[0]}")
        diagnostics = self.dtss()
        return _lif_sequence(
            currents,
            diagnostics,
            leak=self.leak,
            threshold=self.threshold,
            surrogate_gamma=self.surrogate_gamma,
        ), diagnostics


class DTSSLIFLinear(nn.Module):
    """Linear transform followed by LIF dynamics and a DTSS output mask."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        t_max: int = 4,
        initial_timesteps: int | None = None,
        leak: float = 0.5,
        threshold: float = 1.0,
        surrogate_gamma: float = 0.3,
        bias: bool = True,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= leak <= 1.0:
            raise ValueError("leak must be in [0, 1]")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        self.dtss = DynamicTimeStepMask(t_max=t_max, initial_timesteps=initial_timesteps)
        self.leak = float(leak)
        self.threshold = float(threshold)
        self.surrogate_gamma = float(surrogate_gamma)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, DTSSDiagnostics]:
        if inputs.ndim < 2:
            raise ValueError("inputs must have shape [T, ..., features]")
        if inputs.shape[0] != self.dtss.t_max:
            raise ValueError(f"expected T={self.dtss.t_max}, received T={inputs.shape[0]}")

        diagnostics = self.dtss()
        currents = self.linear(inputs)
        shape = currents.shape
        currents = self.norm(currents.reshape(-1, shape[-1])).reshape(shape)
        return _lif_sequence(
            currents,
            diagnostics,
            leak=self.leak,
            threshold=self.threshold,
            surrogate_gamma=self.surrogate_gamma,
        ), diagnostics


class DTSSSpikingMLP(nn.Module):
    """Two small DTSS-LIF linear layers for isolated feasibility tests."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        t_max: int = 4,
        initial_timesteps: int = 2,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden = DTSSLIFLinear(
            in_features,
            hidden_features,
            t_max=t_max,
            initial_timesteps=initial_timesteps,
            batch_norm=batch_norm,
        )
        self.output = DTSSLIFLinear(
            hidden_features,
            out_features,
            t_max=t_max,
            initial_timesteps=initial_timesteps,
            batch_norm=batch_norm,
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[DTSSDiagnostics, DTSSDiagnostics]]:
        hidden, hidden_diagnostics = self.hidden(inputs)
        output, output_diagnostics = self.output(hidden)
        return output, (hidden_diagnostics, output_diagnostics)

    def mask_loss(self) -> torch.Tensor:
        return self.hidden.dtss.mask_loss() + self.output.dtss.mask_loss()


class DTSSSpikingSelfAttention(nn.Module):
    """Minimal multi-head spike self-attention from the paper's MSA block.

    Q, K, and V are binary DTSS-filtered spike tensors. Attention is the scaled
    ``(Q K^T) V`` product with no softmax, followed by a spiking attention
    activation and a spiking output projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        t_max: int = 4,
        initial_timesteps: int = 2,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        projection_args = {
            "t_max": t_max,
            "initial_timesteps": initial_timesteps,
            "batch_norm": True,
        }
        self.query = DTSSLIFLinear(embed_dim, embed_dim, **projection_args)
        self.key = DTSSLIFLinear(embed_dim, embed_dim, **projection_args)
        self.value = DTSSLIFLinear(embed_dim, embed_dim, **projection_args)
        self.attention_lif = DTSSLIF(t_max=t_max, initial_timesteps=initial_timesteps)
        self.projection = DTSSLIFLinear(embed_dim, embed_dim, **projection_args)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        time, batch, tokens, _ = tensor.shape
        return tensor.reshape(time, batch, tokens, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, DTSSDiagnostics]]:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [T, batch, tokens, embed_dim]")
        if inputs.shape[-1] != self.embed_dim:
            raise ValueError(f"expected embed_dim={self.embed_dim}, received {inputs.shape[-1]}")

        query, query_diagnostics = self.query(inputs)
        key, key_diagnostics = self.key(inputs)
        value, value_diagnostics = self.value(inputs)
        query_heads = self._split_heads(query)
        key_heads = self._split_heads(key)
        value_heads = self._split_heads(value)
        attention = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        context = torch.matmul(attention, value_heads) * self.scale
        context = context.permute(0, 1, 3, 2, 4).reshape_as(inputs)
        context_spikes, attention_diagnostics = self.attention_lif(context)
        output, projection_diagnostics = self.projection(context_spikes)
        return output, {
            "query": query_diagnostics,
            "key": key_diagnostics,
            "value": value_diagnostics,
            "attention": attention_diagnostics,
            "projection": projection_diagnostics,
        }

    def mask_loss(self) -> torch.Tensor:
        modules = (self.query, self.key, self.value, self.attention_lif, self.projection)
        return sum((module.dtss.mask_loss() for module in modules), start=torch.zeros((), device=self.query.linear.weight.device))


class DTSSSpikingTransformerBlock(nn.Module):
    """One isolated spiking Transformer block with two residual branches."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: int = 2,
        t_max: int = 4,
        initial_timesteps: int = 2,
    ) -> None:
        super().__init__()
        self.attention = DTSSSpikingSelfAttention(
            embed_dim,
            num_heads,
            t_max=t_max,
            initial_timesteps=initial_timesteps,
        )
        self.mlp = DTSSSpikingMLP(
            embed_dim,
            embed_dim * mlp_ratio,
            embed_dim,
            t_max=t_max,
            initial_timesteps=initial_timesteps,
            batch_norm=True,
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, DTSSDiagnostics]]:
        attention_output, diagnostics = self.attention(inputs)
        attention_residual = inputs + attention_output
        mlp_output, mlp_diagnostics = self.mlp(attention_residual)
        output = attention_residual + mlp_output
        diagnostics = {
            **diagnostics,
            "mlp_hidden": mlp_diagnostics[0],
            "mlp_output": mlp_diagnostics[1],
        }
        return output, diagnostics

    def mask_loss(self) -> torch.Tensor:
        return self.attention.mask_loss() + self.mlp.mask_loss()

