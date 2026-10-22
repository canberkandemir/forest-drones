"""LoRA utilities for DINOv3 + FCOS ForestPersons experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, AutoModel

from .dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3FCOSQualityHead,
    DINOv3FPNFCOSHead,
    DINOv3Stride8FCOSHead,
    MODEL_ID,
    PreprocessInfo,
    image_to_tensor,
    resize_to_patch_grid,
)


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for low-rank adapters inserted into DINOv3 attention."""

    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.05
    target_suffixes: tuple[str, ...] = ("attention.q_proj", "attention.v_proj")


class LoRALinear(nn.Module):
    """Wrap a Linear layer with a trainable low-rank residual update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        lora_hidden = F.linear(self.dropout(inputs), self.lora_a)
        lora_output = F.linear(lora_hidden, self.lora_b)
        return base_output + self.scaling * lora_output


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def find_lora_target_modules(
    model: nn.Module,
    target_suffixes: Iterable[str],
) -> list[str]:
    """Return Linear module names matching any configured suffix."""

    suffixes = tuple(target_suffixes)
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in suffixes)
    ]


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def apply_lora_adapters(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Freeze a model and replace selected Linear layers with LoRA wrappers."""

    freeze_module(model)
    target_names = find_lora_target_modules(model, config.target_suffixes)
    if not target_names:
        raise ValueError(f"No LoRA target modules found for suffixes: {config.target_suffixes}")
    for name in target_names:
        parent, child_name = _get_parent_module(model, name)
        base = getattr(parent, child_name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target is not Linear: {name}")
        setattr(
            parent,
            child_name,
            LoRALinear(base, rank=config.rank, alpha=config.alpha, dropout=config.dropout),
        )
    return target_names


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
    }


def load_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected_lora = [name for name in unexpected if "lora_" in name]
    missing_lora = [name for name in missing if "lora_" in name]
    if unexpected_lora or missing_lora:
        raise RuntimeError(
            f"LoRA state mismatch. Missing LoRA keys: {missing_lora}; "
            f"unexpected LoRA keys: {unexpected_lora}"
        )


def trainable_parameter_summary(backbone: nn.Module, head: nn.Module | None = None) -> dict[str, int]:
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    backbone_total = sum(p.numel() for p in backbone.parameters())
    head_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad) if head is not None else 0
    head_total = sum(p.numel() for p in head.parameters()) if head is not None else 0
    return {
        "backbone_trainable": backbone_trainable,
        "backbone_total": backbone_total,
        "head_trainable": head_trainable,
        "head_total": head_total,
        "total_trainable": backbone_trainable + head_trainable,
        "total_parameters": backbone_total + head_total,
    }


def load_dinov3_with_lora(
    device: torch.device,
    config: LoRAConfig,
    model_id: str = MODEL_ID,
) -> tuple[Any, nn.Module, list[str]]:
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    target_names = apply_lora_adapters(model, config)
    model.to(device)
    return processor, model, target_names


def extract_lora_patch_features(
    image: Image.Image,
    processor: Any,
    model: nn.Module,
    device: torch.device,
    short_side: int = 672,
) -> tuple[torch.Tensor, PreprocessInfo]:
    """Extract differentiable DINOv3 patch features for LoRA optimization."""

    resized, info = resize_to_patch_grid(image, short_side=short_side)
    pixel_values = image_to_tensor(resized, processor, device)
    outputs = model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state[0]
    num_patches = info.grid_height * info.grid_width
    patch_tokens = tokens[-num_patches:].reshape(info.grid_height, info.grid_width, -1)
    return F.normalize(patch_tokens.float(), dim=-1), info


def save_lora_fcos_checkpoint(
    path: Path,
    backbone: nn.Module,
    head: nn.Module,
    config: LoRAConfig,
    target_names: list[str],
    settings: dict[str, Any],
    history: list[dict[str, Any]],
    model_id: str = MODEL_ID,
    head_kind: str = "fcos",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone_model_id": model_id,
            "lora_config": asdict(config),
            "lora_target_modules": target_names,
            "lora_state": lora_state_dict(backbone),
            "head": head_kind,
            "head_input_dim": int(getattr(head.proj, "in_channels", 384)),
            "head_state": head.state_dict(),
            "settings": settings,
            "history": history,
            "parameter_summary": trainable_parameter_summary(backbone, head),
        },
        path,
    )


def load_lora_fcos_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, nn.Module, nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_data = checkpoint["lora_config"]
    config = LoRAConfig(
        rank=int(config_data["rank"]),
        alpha=float(config_data["alpha"]),
        dropout=float(config_data["dropout"]),
        target_suffixes=tuple(config_data["target_suffixes"]),
    )
    model_id = str(checkpoint.get("backbone_model_id", MODEL_ID))
    processor, backbone, target_names = load_dinov3_with_lora(device, config, model_id=model_id)
    load_lora_state_dict(backbone, checkpoint["lora_state"])
    head_input_dim = int(checkpoint.get("head_input_dim", getattr(backbone.config, "hidden_size", 384)))
    head_kind = str(checkpoint.get("head", "fcos"))
    if head_kind == "fpn_fcos":
        head = DINOv3FPNFCOSHead(input_dim=head_input_dim).to(device)
    elif head_kind == "stride8_fcos":
        head = DINOv3Stride8FCOSHead(input_dim=head_input_dim).to(device)
    elif head_kind == "fcos":
        head = DINOv3FCOSHead(input_dim=head_input_dim).to(device)
    elif head_kind == "fcos_quality":
        head = DINOv3FCOSQualityHead(input_dim=head_input_dim).to(device)
    else:
        raise ValueError(f"Unsupported LoRA detector head kind: {head_kind}")
    head.load_state_dict(checkpoint["head_state"])
    return processor, backbone, head, {
        "checkpoint": checkpoint,
        "target_names": target_names,
        "config": config,
        "model_id": model_id,
        "head_input_dim": head_input_dim,
        "head_kind": head_kind,
    }


def module_namespace(**kwargs: Any) -> Any:
    """Small helper for tests and dummy HF-like model outputs."""

    return SimpleNamespace(**kwargs)
