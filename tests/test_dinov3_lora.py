from pathlib import Path

import torch
from PIL import Image
from torch import nn

from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    PreprocessInfo,
    build_fcos_targets,
    decode_fcos_predictions,
)
from forestpersons.dinov3_lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora_adapters,
    extract_lora_patch_features,
    load_lora_state_dict,
    lora_state_dict,
    module_namespace,
    save_lora_fcos_checkpoint,
    trainable_parameter_summary,
)
from forestpersons.evaluation import EvaluationPrediction, predictions_to_coco_results


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = TinyAttention()


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layer = nn.ModuleList([TinyLayer(), TinyLayer()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layer:
            x = layer.attention.v_proj(layer.attention.q_proj(x))
        return x


class DummyDINO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 384)

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.flatten(2).transpose(1, 2)
        tokens = self.proj(pooled)
        cls = tokens.mean(dim=1, keepdim=True)
        return module_namespace(last_hidden_state=torch.cat([cls, tokens], dim=1))


class DummyProcessor:
    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]


def test_lora_injection_freezes_base_and_targets_q_v_only() -> None:
    model = TinyBackbone()
    targets = apply_lora_adapters(model, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))

    assert set(targets) == {
        "model.layer.0.attention.q_proj",
        "model.layer.0.attention.v_proj",
        "model.layer.1.attention.q_proj",
        "model.layer.1.attention.v_proj",
    }
    assert isinstance(model.model.layer[0].attention.q_proj, LoRALinear)
    assert isinstance(model.model.layer[0].attention.v_proj, LoRALinear)
    assert not isinstance(model.model.layer[0].attention.k_proj, LoRALinear)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )


def test_lora_gradients_flow_only_to_adapters() -> None:
    model = TinyBackbone()
    apply_lora_adapters(model, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))
    output = model(torch.ones((1, 4, 8))).sum()
    output.backward()

    lora_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ]
    base_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    ]
    assert any(grad is not None and bool((grad.abs() > 0).any()) for grad in lora_grads)
    assert all(grad is None for grad in base_grads)


def test_extract_lora_patch_features_shape_and_scaling() -> None:
    image = Image.new("RGB", (32, 16), color=(128, 64, 32))
    features, info = extract_lora_patch_features(
        image,
        processor=DummyProcessor(),
        model=DummyDINO(),
        device=torch.device("cpu"),
        short_side=16,
    )

    assert features.shape == (1, 2, 384)
    assert info == PreprocessInfo(
        original_width=32,
        original_height=16,
        resized_width=32,
        resized_height=16,
        scale=1.0,
        grid_width=2,
        grid_height=1,
    )
    assert torch.allclose(features.norm(dim=-1), torch.ones((1, 2)), atol=1e-5)


def test_lora_state_save_and_reload_round_trip() -> None:
    model = TinyBackbone()
    apply_lora_adapters(model, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.fill_(0.25)
    state = lora_state_dict(model)

    reloaded = TinyBackbone()
    apply_lora_adapters(reloaded, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))
    load_lora_state_dict(reloaded, state)

    for name, parameter in reloaded.named_parameters():
        if "lora_" in name:
            assert torch.allclose(parameter, torch.full_like(parameter, 0.25))


def test_lora_checkpoint_saves_separate_adapter_and_head_state(tmp_path: Path) -> None:
    model = TinyBackbone()
    targets = apply_lora_adapters(model, LoRAConfig(rank=2, alpha=4.0, dropout=0.0))
    head = DINOv3FCOSHead(input_dim=384, hidden_dim=16)
    path = tmp_path / "lora.pt"

    save_lora_fcos_checkpoint(
        path,
        backbone=model,
        head=head,
        config=LoRAConfig(rank=2, alpha=4.0, dropout=0.0),
        target_names=targets,
        settings={"unit_test": True},
        history=[{"loss": 1.0}],
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert set(checkpoint["lora_state"])
    assert all("lora_" in key for key in checkpoint["lora_state"])
    assert checkpoint["head"] == "fcos"
    assert checkpoint["settings"]["unit_test"] is True
    assert checkpoint["parameter_summary"] == trainable_parameter_summary(model, head)


def test_lora_fcos_decode_is_evaluation_compatible() -> None:
    info = PreprocessInfo(
        original_width=64,
        original_height=64,
        resized_width=64,
        resized_height=64,
        scale=1.0,
        grid_width=2,
        grid_height=2,
    )
    logits = torch.full((2, 2), -10.0)
    centerness = torch.full((2, 2), -10.0)
    logits[0, 0] = 10.0
    centerness[0, 0] = 10.0
    raw_box = torch.zeros((2, 2, 4))

    predictions = decode_fcos_predictions(logits, raw_box, centerness, info, conf_threshold=0.5)
    coco = predictions_to_coco_results(
        [EvaluationPrediction(1, predictions[0]["box"], predictions[0]["confidence"])]
    )

    assert len(predictions) == 1
    assert coco[0]["image_id"] == 1
    assert coco[0]["category_id"] == 1
    assert coco[0]["score"] > 0.99


def test_fcos_assignment_still_uses_separate_xy_scales_for_lora_path() -> None:
    info = PreprocessInfo(
        original_width=200,
        original_height=100,
        resized_width=800,
        resized_height=800,
        scale=4.0,
        grid_width=50,
        grid_height=50,
    )
    gt = [{"box": [50.0, 25.0, 70.0, 45.0], "visibility_level": 20}]

    classification, box_targets, centerness, mask, stats = build_fcos_targets(gt, info)

    assert int(mask.sum()) > 0
    assert int(classification.sum()) == int(mask.sum())
    assert torch.all(box_targets[mask] > 0)
    assert torch.all((centerness[mask] > 0) & (centerness[mask] <= 1))
    assert stats["annotations_with_positive"] == 1
