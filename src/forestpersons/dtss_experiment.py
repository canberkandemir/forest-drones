"""Deterministic synthetic classification utilities for DTSS validation."""

from __future__ import annotations

import copy
import random
import resource
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dtss import DTSSSpikingTransformerBlock, DynamicTimeStepMask


@dataclass(frozen=True)
class SyntheticExperimentConfig:
    seed: int = 20261020
    train_samples: int = 256
    test_samples: int = 128
    tokens: int = 6
    embed_dim: int = 16
    num_heads: int = 4
    classes: int = 2
    t_max: int = 4
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.002
    weight_decay: float = 0.0
    mask_loss_weight: float = 0.2


class DTSSSyntheticClassifier(nn.Module):
    """One spiking Transformer block plus temporal/token mean classifier."""

    def __init__(self, config: SyntheticExperimentConfig) -> None:
        super().__init__()
        self.t_max = config.t_max
        self.block = DTSSSpikingTransformerBlock(
            config.embed_dim,
            config.num_heads,
            t_max=config.t_max,
            initial_timesteps=config.t_max,
        )
        self.classifier = nn.Linear(config.embed_dim, config.classes)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        temporal_tokens = tokens.unsqueeze(0).expand(self.t_max, -1, -1, -1)
        encoded, diagnostics = self.block(temporal_tokens)
        pooled = encoded.mean(dim=(0, 2))
        return self.classifier(pooled), diagnostics


def make_synthetic_token_data(
    sample_count: int,
    *,
    tokens: int,
    embed_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create balanced token sequences with a deterministic distributed signal."""

    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(sample_count, dtype=torch.long) % 2
    permutation = torch.randperm(sample_count, generator=generator)
    labels = labels[permutation]
    features = 0.35 * torch.randn(sample_count, tokens, embed_dim, generator=generator)
    direction = labels.to(torch.float32).mul(2.0).sub(1.0)
    features[:, :2, :4] += 0.9 * direction[:, None, None]
    features[:, 2:, 4:8] += 0.35 * direction[:, None, None]
    return features, labels


def timestep_masks(model: nn.Module) -> dict[str, DynamicTimeStepMask]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, DynamicTimeStepMask)
    }


def set_timestep_learning(model: nn.Module, enabled: bool) -> None:
    for module in timestep_masks(model).values():
        module.raw_timestep_parameters.requires_grad_(enabled)


def active_timestep_summary(model: nn.Module) -> dict[str, int]:
    return {name: module().active_timesteps for name, module in timestep_masks(model).items()}


def _evaluate(
    model: DTSSSyntheticClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        logits, _ = model(features.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        accuracy = (logits.argmax(dim=1) == labels.to(device)).float().mean()
    return {"loss": float(loss.cpu()), "accuracy": float(accuracy.cpu())}


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _memory_snapshot(device: torch.device) -> dict[str, float | None]:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)
    if device.type != "mps":
        return {"process_high_water_rss_mib": rss, "peak_mps_driver_gib": None}
    return {
        "process_high_water_rss_mib": rss,
        "peak_mps_driver_gib": float(torch.mps.driver_allocated_memory()) / (1024.0**3),
    }


def _train_arm(
    name: str,
    initial_state: dict[str, torch.Tensor],
    config: SyntheticExperimentConfig,
    train_data: tuple[torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor],
    batch_orders: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    model = DTSSSyntheticClassifier(config).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    learned = name == "learned_dtss"
    set_timestep_learning(model, learned)
    initial_timestep_parameters = {
        key: value.raw_timestep_parameters.detach().cpu().clone()
        for key, value in timestep_masks(model).items()
    }
    # Warm identical kernels before timing, then restore parameters and buffers
    # so this does not consume training budget or alter initialization.
    warm_features = train_data[0][: config.batch_size].to(device)
    model.train()
    warm_logits, _ = model(warm_features)
    (warm_logits.square().mean() + 0.001 * model.block.mask_loss()).backward()
    _synchronize(device)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(copy.deepcopy(initial_state))
    set_timestep_learning(model, learned)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_features, train_labels = train_data
    history: list[dict[str, Any]] = []
    peak_mps_driver = 0
    start = time.perf_counter()

    for epoch, order in enumerate(batch_orders, start=1):
        model.train()
        task_loss_sum = 0.0
        correct = 0
        count = 0
        timestep_gradient_norms: list[float] = []
        for offset in range(0, len(order), config.batch_size):
            indices = order[offset : offset + config.batch_size]
            features = train_features[indices].to(device)
            labels = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            task_loss = F.cross_entropy(logits, labels)
            normalized_mask_loss = model.block.mask_loss() / (
                len(timestep_masks(model)) * config.t_max
            )
            objective = task_loss
            if learned:
                objective = objective + config.mask_loss_weight * normalized_mask_loss
            objective.backward()
            if learned:
                squared_norm = torch.zeros((), device=device)
                for module in timestep_masks(model).values():
                    gradient = module.raw_timestep_parameters.grad
                    if gradient is not None:
                        squared_norm = squared_norm + gradient.square().sum()
                timestep_gradient_norms.append(float(squared_norm.sqrt().detach().cpu()))
            optimizer.step()
            task_loss_sum += float(task_loss.detach().cpu()) * len(indices)
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            count += len(indices)
            if device.type == "mps":
                peak_mps_driver = max(peak_mps_driver, int(torch.mps.driver_allocated_memory()))

        test_metrics = _evaluate(model, *test_data, device)
        history.append(
            {
                "epoch": epoch,
                "train_task_loss": task_loss_sum / count,
                "train_accuracy": correct / count,
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "active_timesteps": active_timestep_summary(model),
                "mean_timestep_gradient_norm": (
                    sum(timestep_gradient_norms) / len(timestep_gradient_norms)
                    if timestep_gradient_norms
                    else 0.0
                ),
            }
        )

    _synchronize(device)
    runtime = time.perf_counter() - start
    final_timestep_parameters = timestep_masks(model)
    parameter_delta = {
        key: float(
            (
                final_timestep_parameters[key].raw_timestep_parameters.detach().cpu()
                - initial_timestep_parameters[key]
            )
            .norm()
            .item()
        )
        for key in initial_timestep_parameters
    }
    memory = _memory_snapshot(device)
    if device.type == "mps":
        memory["peak_mps_driver_gib"] = peak_mps_driver / (1024.0**3)
    return {
        "name": name,
        "device": str(device),
        "runtime_seconds": runtime,
        "memory": memory,
        "initial_active_timesteps": history[0]["active_timesteps"] if history else {},
        "final_active_timesteps": active_timestep_summary(model),
        "timestep_parameter_delta_norms": parameter_delta,
        "history": history,
        "final": history[-1],
    }


def run_controlled_experiment(
    config: SyntheticExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_data = make_synthetic_token_data(
        config.train_samples,
        tokens=config.tokens,
        embed_dim=config.embed_dim,
        seed=config.seed + 1,
    )
    test_data = make_synthetic_token_data(
        config.test_samples,
        tokens=config.tokens,
        embed_dim=config.embed_dim,
        seed=config.seed + 2,
    )
    initial_model = DTSSSyntheticClassifier(config)
    initial_state = copy.deepcopy(initial_model.state_dict())
    order_generator = torch.Generator().manual_seed(config.seed + 3)
    batch_orders = [
        torch.randperm(config.train_samples, generator=order_generator)
        for _ in range(config.epochs)
    ]
    arms = [
        _train_arm(
            name,
            initial_state,
            config,
            train_data,
            test_data,
            batch_orders,
            device,
        )
        for name in ("fixed_t4", "learned_dtss")
    ]
    return {
        "config": asdict(config),
        "direct_encoding": "Each static token tensor is repeated over T=4 steps.",
        "compute_caveat": (
            "Masks zero trailing outputs, but eager PyTorch still executes every timestep; "
            "runtime is not hardware timestep skipping."
        ),
        "arms": arms,
    }

