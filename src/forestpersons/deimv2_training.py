"""Training utilities for the official DEIMv2-S ForestPersons pilot."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .dataset import ForestPersonsDataset, ForestPersonsSample
from .deimv2_adapter import build_deimv2_preprocess


CHECKPOINT_ARCHITECTURE = "official_deimv2_s_oneclass_forestpersons"


def load_manifest_items(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["images"] if isinstance(payload, dict) and "images" in payload else payload
    if not isinstance(items, list):
        raise ValueError(f"Unsupported manifest format: {path}")
    return items


def samples_from_manifest(
    root: str | Path,
    manifest: str | Path,
    *,
    split: str,
    max_images: int | None = None,
) -> list[ForestPersonsSample]:
    """Resolve samples strictly by image ID and verify recorded filenames."""

    dataset = ForestPersonsDataset(root, split=split)
    by_id = {sample.image_id: sample for sample in dataset.samples}
    items = load_manifest_items(manifest)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        items = items[:max_images]

    seen_ids: set[int | str] = set()
    samples: list[ForestPersonsSample] = []
    for item in items:
        image_id = item.get("image_id")
        if image_id in seen_ids:
            raise ValueError(f"Duplicate manifest image_id: {image_id}")
        seen_ids.add(image_id)
        if image_id not in by_id:
            raise KeyError(f"Manifest image_id {image_id} is absent from {split}.json")
        sample = by_id[image_id]
        if sample.file_name != item.get("file_name"):
            raise ValueError(
                f"Filename mismatch for image_id {image_id}: "
                f"manifest={item.get('file_name')!r}, annotations={sample.file_name!r}"
            )
        if not sample.image_path.is_file():
            raise FileNotFoundError(sample.image_path)
        samples.append(sample)
    return samples


def normalized_person_boxes(
    annotations: list[dict[str, Any]],
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert clipped COCO xywh boxes to normalized cxcywh boxes."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    boxes: list[list[float]] = []
    for annotation in annotations:
        x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
        x1 = min(max(x, 0.0), float(width))
        y1 = min(max(y, 0.0), float(height))
        x2 = min(max(x + box_width, 0.0), float(width))
        y2 = min(max(y + box_height, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            [
                ((x1 + x2) * 0.5) / width,
                ((y1 + y2) * 0.5) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            ]
        )
    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def input_and_target(
    sample: ForestPersonsSample,
    device: torch.device,
    input_size: int,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB")
    boxes = normalized_person_boxes(sample.annotations, image.width, image.height)
    image_tensor = build_deimv2_preprocess(input_size)(image).unsqueeze(0).to(device)
    target = {
        "labels": torch.zeros(len(boxes), dtype=torch.int64, device=device),
        "boxes": boxes.to(device),
        "image_id": torch.tensor([sample.image_id], dtype=torch.int64, device=device),
        "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64, device=device),
        "size": torch.tensor([input_size, input_size], dtype=torch.int64, device=device),
    }
    return image_tensor, [target]


def make_official_criterion(official_repo: str | Path, device: torch.device) -> torch.nn.Module:
    import sys

    repo = str(Path(official_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from engine.deim import DEIMCriterion, HungarianMatcher

    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
        change_matcher=True,
        iou_order_alpha=4.0,
        matcher_change_epoch=100,
    )
    return DEIMCriterion(
        matcher=matcher,
        weight_dict={
            "loss_mal": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        },
        losses=["mal", "boxes", "local"],
        alpha=0.75,
        gamma=1.5,
        num_classes=1,
        reg_max=32,
    ).to(device)


def official_parameter_groups(
    model: torch.nn.Module,
    *,
    backbone_lr: float,
    detector_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Build the four parameter groups encoded by the official S config."""

    grouped: dict[str, list[torch.nn.Parameter]] = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "detector_decay": [],
        "detector_no_decay": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = "backbone.dinov3" in name
        is_no_decay = any(token in name.lower() for token in ("norm", "bn", "bias"))
        key = f"{'backbone' if is_backbone else 'detector'}_{'no_decay' if is_no_decay else 'decay'}"
        grouped[key].append(parameter)

    groups: list[dict[str, Any]] = []
    for name, parameters in grouped.items():
        if not parameters:
            continue
        lr = backbone_lr if name.startswith("backbone") else detector_lr
        decay = 0.0 if name.endswith("no_decay") else weight_decay
        groups.append(
            {
                "params": parameters,
                "lr": lr,
                "initial_lr": lr,
                "weight_decay": decay,
                "group_name": name,
            }
        )
    return groups


class PilotFlatCosineScheduler:
    """Optimizer-step version of DEIMv2's quadratic-warmup flat-cosine schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        flat_steps: int,
        min_lr_ratio: float,
    ) -> None:
        if not (0 <= warmup_steps <= flat_steps <= total_steps):
            raise ValueError("Expected 0 <= warmup_steps <= flat_steps <= total_steps")
        if not (0.0 < min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in (0, 1]")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.flat_steps = flat_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [float(group["initial_lr"]) for group in optimizer.param_groups]
        self.current_step = 0

    def factor(self, step: int) -> float:
        if self.warmup_steps and step <= self.warmup_steps:
            return (step / self.warmup_steps) ** 2
        if step <= self.flat_steps or self.flat_steps == self.total_steps:
            return 1.0
        progress = min(max((step - self.flat_steps) / (self.total_steps - self.flat_steps), 0.0), 1.0)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, step: int) -> list[float]:
        self.current_step = int(step)
        factor = self.factor(self.current_step)
        lrs = []
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor
            lrs.append(float(group["lr"]))
        return lrs

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "flat_steps": self.flat_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("total_steps", "warmup_steps", "flat_steps", "min_lr_ratio", "base_lrs"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Scheduler resume mismatch for {key}: {state[key]} != {getattr(self, key)}")
        self.step(int(state["current_step"]))


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state().cpu()
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps" and "mps" in state and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state["mps"])


def validate_training_checkpoint(checkpoint: dict[str, Any]) -> None:
    if checkpoint.get("architecture") != CHECKPOINT_ARCHITECTURE:
        raise ValueError(f"Unsupported DEIMv2 checkpoint architecture: {checkpoint.get('architecture')!r}")
    for key in ("model_state", "training_config", "class_mapping"):
        if key not in checkpoint:
            raise ValueError(f"DEIMv2 training checkpoint is missing {key}")


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)

