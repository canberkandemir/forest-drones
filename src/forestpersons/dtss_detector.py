"""Small patch-token detector for isolated DTSS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .dtss import DTSSSpikingTransformerBlock, DynamicTimeStepMask


@dataclass(frozen=True)
class DTSSDetectorConfig:
    image_size: int = 128
    patch_size: int = 16
    embed_dim: int = 32
    num_heads: int = 4
    t_max: int = 4
    initial_timesteps: int = 4

    @property
    def grid_size(self) -> int:
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        return self.image_size // self.patch_size


def scale_xywh_to_normalized_xyxy(
    box: list[float] | tuple[float, float, float, float],
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> torch.Tensor:
    """Scale COCO xywh independently in x/y, then normalize to [0, 1]."""

    x, y, width, height = (float(value) for value in box)
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    if min(original_width, original_height, resized_width, resized_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    scaled = torch.tensor(
        [x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y],
        dtype=torch.float32,
    )
    scaled[0::2].div_(resized_width)
    scaled[1::2].div_(resized_height)
    return scaled.clamp(0.0, 1.0)


def patch_centers(grid_size: int, *, device: torch.device | None = None) -> torch.Tensor:
    coordinates = (torch.arange(grid_size, device=device, dtype=torch.float32) + 0.5) / grid_size
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=1)


def assign_boxes_to_patches(
    boxes: torch.Tensor,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each GT to its nearest still-unused patch center."""

    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [objects, 4]")
    centers = patch_centers(grid_size, device=boxes.device)
    objectness = torch.zeros(len(centers), device=boxes.device)
    box_targets = torch.zeros(len(centers), 4, device=boxes.device)
    positive = torch.zeros(len(centers), dtype=torch.bool, device=boxes.device)
    for box in boxes:
        gt_center = torch.stack(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        distances = (centers - gt_center).square().sum(dim=1)
        for index in distances.argsort():
            if not positive[index]:
                positive[index] = True
                objectness[index] = 1.0
                box_targets[index] = box
                break
    return objectness, box_targets, positive


def decode_patch_boxes(raw_boxes: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Decode patch-relative center offsets and normalized width/height."""

    if raw_boxes.shape[-2:] != (grid_size * grid_size, 4):
        raise ValueError("raw_boxes do not match grid_size")
    centers = patch_centers(grid_size, device=raw_boxes.device)
    centers = centers.view(*([1] * (raw_boxes.ndim - 2)), -1, 2)
    center = centers + 0.5 * torch.tanh(raw_boxes[..., :2]) / grid_size
    size = torch.sigmoid(raw_boxes[..., 2:])
    return torch.cat((center - size / 2, center + size / 2), dim=-1).clamp(0.0, 1.0)


def box_iou_aligned(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(first[..., :2], second[..., :2])
    bottom_right = torch.minimum(first[..., 2:], second[..., 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    first_area = (first[..., 2:] - first[..., :2]).clamp(min=0).prod(dim=-1)
    second_area = (second[..., 2:] - second[..., :2]).clamp(min=0).prod(dim=-1)
    return intersection / (first_area + second_area - intersection).clamp(min=1e-7)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        repeated = boxes[current].expand_as(boxes[remaining])
        order = remaining[box_iou_aligned(repeated, boxes[remaining]) <= iou_threshold]
    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.stack(kept)


class DTSSPatchDetector(nn.Module):
    """Patch embedding, one spiking Transformer block, and a tiny dense head."""

    def __init__(self, config: DTSSDetectorConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            3,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position = nn.Parameter(
            torch.zeros(1, config.grid_size * config.grid_size, config.embed_dim)
        )
        self.block = DTSSSpikingTransformerBlock(
            config.embed_dim,
            config.num_heads,
            t_max=config.t_max,
            initial_timesteps=config.initial_timesteps,
        )
        self.objectness = nn.Linear(config.embed_dim, 1)
        self.box_regression = nn.Linear(config.embed_dim, 4)
        nn.init.constant_(self.objectness.bias, -4.0)
        nn.init.constant_(self.box_regression.bias[2:], -2.5)

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        if images.ndim != 4 or images.shape[-2:] != (
            self.config.image_size,
            self.config.image_size,
        ):
            raise ValueError("images must be square and match configured image_size")
        tokens = self.patch_embedding(images).flatten(2).transpose(1, 2) + self.position
        temporal = tokens.unsqueeze(0).expand(self.config.t_max, -1, -1, -1)
        encoded, diagnostics = self.block(temporal)
        output_mask = diagnostics["mlp_output"].mask.view(-1, 1, 1, 1)
        pooled = (encoded * output_mask).sum(dim=0) / output_mask.sum().clamp(min=1.0)
        raw_boxes = self.box_regression(pooled)
        return {
            "objectness_logits": self.objectness(pooled).squeeze(-1),
            "boxes": decode_patch_boxes(raw_boxes, self.config.grid_size),
            "raw_boxes": raw_boxes,
            "diagnostics": diagnostics,
        }

    def mask_loss(self) -> torch.Tensor:
        return self.block.mask_loss()

    def timestep_masks(self) -> dict[str, DynamicTimeStepMask]:
        return {
            name: module
            for name, module in self.named_modules()
            if isinstance(module, DynamicTimeStepMask)
        }

    def compute_loss(
        self,
        outputs: dict[str, Any],
        boxes_by_image: list[torch.Tensor],
        *,
        box_weight: float = 5.0,
    ) -> dict[str, torch.Tensor]:
        objectness_targets = []
        box_targets = []
        positive_masks = []
        for boxes in boxes_by_image:
            objectness, targets, positive = assign_boxes_to_patches(
                boxes.to(outputs["boxes"].device), self.config.grid_size
            )
            objectness_targets.append(objectness)
            box_targets.append(targets)
            positive_masks.append(positive)
        objectness_target = torch.stack(objectness_targets)
        box_target = torch.stack(box_targets)
        positive = torch.stack(positive_masks)
        positive_count = int(positive.sum().item())
        negative_count = positive.numel() - positive_count
        pos_weight = min(20.0, negative_count / max(positive_count, 1))
        classification = F.binary_cross_entropy_with_logits(
            outputs["objectness_logits"],
            objectness_target,
            pos_weight=torch.tensor(pos_weight, device=objectness_target.device),
        )
        if positive.any():
            regression = F.smooth_l1_loss(outputs["boxes"][positive], box_target[positive])
        else:
            regression = outputs["boxes"].sum() * 0.0
        return {
            "classification": classification,
            "regression": regression,
            "detection": classification + box_weight * regression,
            "positive_count": torch.tensor(float(positive_count), device=classification.device),
        }

    def predict(
        self,
        images: torch.Tensor,
        *,
        confidence_threshold: float = 0.01,
        nms_iou: float = 0.5,
        max_detections: int = 100,
    ) -> list[dict[str, torch.Tensor]]:
        outputs = self(images)
        probabilities = outputs["objectness_logits"].sigmoid()
        predictions = []
        for boxes, scores in zip(outputs["boxes"], probabilities):
            selected = scores >= confidence_threshold
            boxes = boxes[selected]
            scores = scores[selected]
            keep = nms(boxes, scores, nms_iou)[:max_detections]
            predictions.append({"boxes": boxes[keep], "scores": scores[keep]})
        return predictions


def set_dtss_learning(model: DTSSPatchDetector, enabled: bool) -> None:
    for module in model.timestep_masks().values():
        module.raw_timestep_parameters.requires_grad_(enabled)
        if not enabled:
            with torch.no_grad():
                # The initial state already has all four steps active; freezing
                # it creates the matched Fixed-T control.
                assert module().active_timesteps == module.t_max
