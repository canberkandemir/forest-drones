from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from forestpersons.dinov3_detector import DINOv3MVRFCOSHead
from forestpersons.dinov3_lora import (
    LoRAConfig,
    apply_lora_adapters,
    extract_lora_multilayer_patch_features,
)
from scripts.evaluate_dinov3_lora_mvr_fcos import validate_matched_checkpoints
from scripts.train_dinov3_lora_mvr_fcos import (
    gradient_group_summary,
    load_initial_mvr_head,
    optimizer_parameter_groups,
)


class DummyProcessor:
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]


class DummyHiddenDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 8)

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = False):
        patches = pixel_values.flatten(2).transpose(1, 2)
        patch_tokens = self.proj(patches)
        cls = patch_tokens.mean(dim=1, keepdim=True)
        tokens = torch.cat((cls, patch_tokens), dim=1)
        hidden = tuple(tokens * (1.0 + index / 20.0) for index in range(13))
        return SimpleNamespace(last_hidden_state=hidden[-1], hidden_states=hidden)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = TinyAttention()


def test_online_multilayer_features_have_expected_shape_and_gradients() -> None:
    model = DummyHiddenDINO()
    image = Image.new("RGB", (32, 16), color=(100, 80, 60))
    features, info = extract_lora_multilayer_patch_features(
        image, DummyProcessor(), model, torch.device("cpu"),
        short_side=16, layer_indices=(3, 6, 9, 12),
    )
    assert features.shape == (4, 1, 2, 8)
    assert (info.grid_height, info.grid_width) == (1, 2)
    assert torch.allclose(features.norm(dim=-1), torch.ones((4, 1, 2)), atol=1e-5)
    features.square().sum().backward()
    assert model.proj.weight.grad is not None
    assert torch.isfinite(model.proj.weight.grad).all()


def test_optimizer_groups_freeze_control_and_limit_adaptation_to_lora() -> None:
    frozen = TinyBackbone()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    head = nn.Linear(8, 2)
    frozen_groups = optimizer_parameter_groups("frozen", frozen, head, 1e-4, 3e-4)
    assert [group["name"] for group in frozen_groups] == ["mvr_fcos"]
    assert frozen_groups[0]["lr"] == 3e-4

    adapted = TinyBackbone()
    apply_lora_adapters(adapted, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))
    groups = optimizer_parameter_groups("lora", adapted, head, 1e-4, 3e-4)
    assert [group["name"] for group in groups] == ["lora", "mvr_fcos"]
    assert all(
        "lora_" in name
        for name, parameter in adapted.named_parameters()
        if parameter.requires_grad
    )


def test_gradient_group_summary_reports_finite_nonzero_flow() -> None:
    layer = nn.Linear(4, 2)
    layer(torch.ones(3, 4)).sum().backward()
    summary = gradient_group_summary(
        [{"name": "head", "params": list(layer.parameters()), "lr": 3e-4}]
    )["head"]
    assert summary["parameters_with_grad"] == 2
    assert summary["nonzero_gradient_tensors"] == 2
    assert summary["all_finite"] is True
    assert summary["l2_norm"] > 0
    assert summary["max_abs"] > 0


def test_full_mvr_initialization_strict_loads_existing_checkpoint(tmp_path: Path) -> None:
    original = DINOv3MVRFCOSHead()
    path = tmp_path / "mvr.pt"
    torch.save(
        {"arm": "mvr", "head": "mvr_fcos", "completed_epoch": 20,
         "model_state": original.state_dict()},
        path,
    )
    loaded, metadata = load_initial_mvr_head(path, torch.device("cpu"))
    assert metadata["completed_epoch"] == 20
    for name, expected in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], expected)


def matched_checkpoint(arm: str) -> dict:
    return {
        "experiment": "frozen_vs_lora_full_mvr",
        "arm": arm,
        "head": "mvr_fcos",
        "manifest_sha256": "manifest",
        "initial_head_sha256": "head",
        "completed_epoch": 2,
        "optimizer_steps": 2500,
        "training_config": {
            "epochs": 2, "gradient_accumulation": 4, "short_side": 672,
            "lr_head": 3e-4, "weight_decay": 1e-4, "box_loss_weight": 2.0,
            "focal_alpha": 0.75, "focal_gamma": 2.0,
            "gradient_clip_norm": 5.0, "seed": 20261012,
        },
        "history": [
            {"sample_order_sha256": "epoch1"},
            {"sample_order_sha256": "epoch2"},
        ],
    }


def test_evaluator_requires_matched_initialization_order_and_budget() -> None:
    frozen = matched_checkpoint("frozen")
    lora = matched_checkpoint("lora")
    validate_matched_checkpoints(frozen, lora)
    lora["history"][1]["sample_order_sha256"] = "different"
    with pytest.raises(ValueError, match="different image orders"):
        validate_matched_checkpoints(frozen, lora)
