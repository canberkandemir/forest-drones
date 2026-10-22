"""Adapter for the official DEIMv2 PyTorchModelHubMixin checkpoint.

The official Intellindust checkpoint is not packaged as a Transformers
``PreTrainedModel``.  This module deliberately mirrors the small wrapper from
the upstream ``hf_models.ipynb`` and keeps the upstream repository isolated
from the rest of the ForestPersons code.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .evaluation import EvaluationPrediction, clip_xyxy_box


DEIMV2_INPUT_SIZE = 640
DEIMV2_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEIMV2_IMAGE_STD = (0.229, 0.224, 0.225)
DEIMV2_COCO_PERSON_LABEL = 0


def mps_compatible_distribution_integral(
    logits: torch.Tensor,
    project: torch.Tensor,
    reg_max: int,
) -> torch.Tensor:
    """Evaluate DEIMv2's distribution integral without MPS 1-D linear.

    The upstream implementation uses ``F.linear(probabilities, project)``
    where ``project`` is one-dimensional. PyTorch 2.9's MPSGraph lowering
    treats that valid operation as a vector dot product and aborts when the
    leading dimension is not 33. This weighted sum is algebraically identical
    and keeps gradients through the distribution logits unchanged.
    """

    shape = logits.shape
    probabilities = torch.softmax(logits.reshape(-1, reg_max + 1), dim=1)
    weights = project.to(device=probabilities.device, dtype=probabilities.dtype).reshape(1, -1)
    integrated = (probabilities * weights).sum(dim=1).reshape(-1, 4)
    return integrated.reshape(list(shape[:-1]) + [-1])


def _install_mps_integral_workaround(model: nn.Module) -> bool:
    integral = getattr(getattr(model, "decoder", None), "integral", None)
    if integral is None or not hasattr(integral, "reg_max"):
        return False

    def forward(module: nn.Module, logits: torch.Tensor, project: torch.Tensor) -> torch.Tensor:
        return mps_compatible_distribution_integral(logits, project, int(module.reg_max))

    integral.forward = MethodType(forward, integral)
    return True


def build_deimv2_preprocess(input_size: int = DEIMV2_INPUT_SIZE) -> Compose:
    """Return the official ViT validation preprocessing pipeline."""

    return Compose(
        [
            Resize((input_size, input_size)),
            ToTensor(),
            Normalize(mean=DEIMV2_IMAGE_MEAN, std=DEIMV2_IMAGE_STD),
        ]
    )


def official_result_to_person_predictions(
    result: dict[str, torch.Tensor],
    *,
    image_id: int | str,
    width: int,
    height: int,
    min_score: float = 0.0,
) -> list[EvaluationPrediction]:
    """Convert official zero-based COCO outputs to one-class predictions.

    DEIMv2 is a query detector and the official postprocessor does not apply
    NMS.  This conversion therefore only selects COCO class 0 (person), clips
    boxes to the original image, and drops invalid or non-finite detections.
    """

    labels = result["labels"].detach().cpu().reshape(-1)
    boxes = result["boxes"].detach().cpu().reshape(-1, 4)
    scores = result["scores"].detach().cpu().reshape(-1)
    if not (len(labels) == len(boxes) == len(scores)):
        raise ValueError("DEIMv2 labels, boxes, and scores must have equal lengths")

    predictions: list[EvaluationPrediction] = []
    for label, box, score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        if int(label) != DEIMV2_COCO_PERSON_LABEL or float(score) < min_score:
            continue
        values = [float(value) for value in box]
        if not math.isfinite(float(score)) or not all(math.isfinite(value) for value in values):
            continue
        clipped = clip_xyxy_box(values, width, height)
        if clipped is None:
            continue
        predictions.append(
            EvaluationPrediction(
                image_id=image_id,
                box=clipped,
                confidence=float(score),
            )
        )
    return predictions


def load_official_deimv2(
    official_repo: str | Path,
    checkpoint_dir: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the official DEIMv2 model without modifying upstream source."""

    official_repo = Path(official_repo).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if not (official_repo / "engine").is_dir():
        raise FileNotFoundError(f"Missing official DEIMv2 engine: {official_repo / 'engine'}")
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint_dir / name).is_file():
            raise FileNotFoundError(f"Missing official DEIMv2 checkpoint file: {checkpoint_dir / name}")

    repo_text = str(official_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    from huggingface_hub import PyTorchModelHubMixin
    from engine.backbone import DINOv3STAs, HGNetv2
    from engine.deim import DEIMTransformer, DFINETransformer, HybridEncoder, LiteEncoder
    from engine.deim.postprocessor import PostProcessor

    class OfficialDEIMv2(nn.Module, PyTorchModelHubMixin):
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            self.backbone = (
                DINOv3STAs(**config["DINOv3STAs"])
                if "DINOv3STAs" in config
                else HGNetv2(**config["HGNetv2"])
            )
            self.encoder = (
                LiteEncoder(**config["LiteEncoder"])
                if "LiteEncoder" in config
                else HybridEncoder(**config["HybridEncoder"])
            )
            self.decoder = (
                DEIMTransformer(**config["DEIMTransformer"])
                if "DEIMTransformer" in config
                else DFINETransformer(**config["DFINETransformer"])
            )
            self.postprocessor = PostProcessor(**config["PostProcessor"])

        def forward(
            self,
            images: torch.Tensor,
            original_sizes: torch.Tensor,
        ) -> list[dict[str, torch.Tensor]]:
            features = self.backbone(images)
            features = self.encoder(features)
            outputs = self.decoder(features)
            return self.postprocessor(outputs, original_sizes)

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    model = OfficialDEIMv2.from_pretrained(str(checkpoint_dir)).eval().to(device)
    metadata = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "config": config,
        "official_repo": str(official_repo),
        "checkpoint_dir": str(checkpoint_dir),
    }
    return model, metadata


def load_official_deimv2_one_class_for_training(
    official_repo: str | Path,
    checkpoint_dir: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Create a one-class training model from the official COCO checkpoint.

    All shape-compatible weights are retained. The COCO person row (internal
    class index 0) initializes each one-class score head. The denoising class
    embedding likewise retains the person row and initializes its new padding
    row to zero. No detection loss or architecture component is replaced.
    """

    official_repo = Path(official_repo).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    for path in (official_repo / "engine", checkpoint_dir / "config.json", checkpoint_dir / "model.safetensors"):
        if not path.exists():
            raise FileNotFoundError(path)

    repo_text = str(official_repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    from huggingface_hub import PyTorchModelHubMixin
    from engine.backbone import DINOv3STAs, HGNetv2
    from engine.deim import DEIMTransformer, DFINETransformer, HybridEncoder, LiteEncoder
    from engine.deim.postprocessor import PostProcessor

    class OfficialDEIMv2OneClass(nn.Module, PyTorchModelHubMixin):
        def __init__(self, config: dict[str, Any]):
            super().__init__()
            self.backbone = (
                DINOv3STAs(**config["DINOv3STAs"])
                if "DINOv3STAs" in config
                else HGNetv2(**config["HGNetv2"])
            )
            self.encoder = (
                LiteEncoder(**config["LiteEncoder"])
                if "LiteEncoder" in config
                else HybridEncoder(**config["HybridEncoder"])
            )
            self.decoder = (
                DEIMTransformer(**config["DEIMTransformer"])
                if "DEIMTransformer" in config
                else DFINETransformer(**config["DFINETransformer"])
            )
            self.postprocessor = PostProcessor(**config["PostProcessor"])

        def forward_raw(
            self,
            images: torch.Tensor,
            targets: list[dict[str, torch.Tensor]] | None = None,
        ) -> dict[str, Any]:
            features = self.backbone(images)
            features = self.encoder(features)
            return self.decoder(features, targets=targets)

        def forward(
            self,
            images: torch.Tensor,
            original_sizes: torch.Tensor,
        ) -> list[dict[str, torch.Tensor]]:
            return self.postprocessor(self.forward_raw(images), original_sizes)

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    config["DEIMTransformer"] = dict(config["DEIMTransformer"])
    config["DEIMTransformer"]["num_classes"] = 1
    config["PostProcessor"] = dict(config["PostProcessor"])
    config["PostProcessor"]["num_classes"] = 1
    model = OfficialDEIMv2OneClass(config)

    source = load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    target = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    copied_person_rows: list[str] = []
    for name, target_tensor in target.items():
        source_tensor = source.get(name)
        if source_tensor is None:
            adapted[name] = target_tensor
        elif source_tensor.shape == target_tensor.shape:
            adapted[name] = source_tensor
        elif name == "decoder.denoising_class_embed.weight" and target_tensor.shape[0] == 2:
            value = target_tensor.clone()
            value[0].copy_(source_tensor[DEIMV2_COCO_PERSON_LABEL])
            value[1].zero_()
            adapted[name] = value
            copied_person_rows.append(name)
        elif (
            (name == "decoder.enc_score_head.weight" or name == "decoder.enc_score_head.bias"
             or name.startswith("decoder.dec_score_head."))
            and target_tensor.shape[0] == 1
            and source_tensor.shape[0] >= 1
            and source_tensor.shape[1:] == target_tensor.shape[1:]
        ):
            adapted[name] = source_tensor[DEIMV2_COCO_PERSON_LABEL : DEIMV2_COCO_PERSON_LABEL + 1]
            copied_person_rows.append(name)
        else:
            raise ValueError(
                f"Unsupported checkpoint shape change for {name}: "
                f"source={tuple(source_tensor.shape)}, target={tuple(target_tensor.shape)}"
            )
    incompatible = model.load_state_dict(adapted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint incompatibility: {incompatible}")
    model.to(device)
    mps_integral_workaround = False
    if device.type == "mps":
        mps_integral_workaround = _install_mps_integral_workaround(model)
        if not mps_integral_workaround:
            raise RuntimeError("Could not locate DEIMv2 distribution integral for the required MPS workaround")
    metadata = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_num_classes": 80,
        "target_num_classes": 1,
        "copied_person_row_keys": copied_person_rows,
        "copied_person_row_count": len(copied_person_rows),
        "category_mapping": "ForestPersons category_id 1 -> internal label 0",
        "mps_integral_workaround": mps_integral_workaround,
        "config": config,
    }
    return model, metadata


def load_official_deimv2_one_class_checkpoint(
    official_repo: str | Path,
    checkpoint_dir: str | Path,
    training_checkpoint: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a ForestPersons one-class checkpoint without changing its source weights."""

    from .deimv2_training import validate_training_checkpoint

    model, metadata = load_official_deimv2_one_class_for_training(
        official_repo, checkpoint_dir, device
    )
    checkpoint = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
    validate_training_checkpoint(checkpoint)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected fine-tuned checkpoint incompatibility: {incompatible}")
    model.eval()
    metadata = dict(metadata)
    metadata.update(
        {
            "fine_tuned_checkpoint": str(Path(training_checkpoint).resolve()),
            "training_config": checkpoint["training_config"],
            "completed_epochs": checkpoint.get("completed_epochs"),
            "optimizer_steps": checkpoint.get("optimizer_steps"),
        }
    )
    return model, metadata
