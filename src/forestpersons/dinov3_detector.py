"""Frozen-DINOv3 feature extraction and lightweight detection head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from torchvision.transforms import functional as TF
from transformers import AutoImageProcessor, AutoModel


MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
PATCH_SIZE = 16


@dataclass(frozen=True)
class PreprocessInfo:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    scale: float
    grid_width: int
    grid_height: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.original_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.original_height


def choose_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preferred == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if preferred == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)


def resize_to_patch_grid(image: Image.Image, short_side: int = 672) -> tuple[Image.Image, PreprocessInfo]:
    width, height = image.size
    scale = short_side / min(width, height)
    resized_width = max(PATCH_SIZE, int(round(width * scale / PATCH_SIZE)) * PATCH_SIZE)
    resized_height = max(PATCH_SIZE, int(round(height * scale / PATCH_SIZE)) * PATCH_SIZE)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    return resized, PreprocessInfo(
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        scale=resized_width / width,
        grid_width=resized_width // PATCH_SIZE,
        grid_height=resized_height // PATCH_SIZE,
    )


def image_to_tensor(image: Image.Image, processor: Any, device: torch.device) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    mean = torch.tensor(getattr(processor, "image_mean", [0.485, 0.456, 0.406])).view(3, 1, 1)
    std = torch.tensor(getattr(processor, "image_std", [0.229, 0.224, 0.225])).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


def load_dinov3(device: torch.device) -> tuple[Any, nn.Module]:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


@torch.inference_mode()
def extract_patch_features(
    image: Image.Image,
    processor: Any,
    model: nn.Module,
    device: torch.device,
    short_side: int = 672,
) -> tuple[torch.Tensor, PreprocessInfo]:
    resized, info = resize_to_patch_grid(image, short_side=short_side)
    pixel_values = image_to_tensor(resized, processor, device)
    outputs = model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state[0]
    num_patches = info.grid_height * info.grid_width
    patch_tokens = tokens[-num_patches:].reshape(info.grid_height, info.grid_width, -1)
    return torch.nn.functional.normalize(patch_tokens.float(), dim=-1), info


@torch.inference_mode()
def extract_intermediate_patch_features(
    image: Image.Image,
    processor: Any,
    model: nn.Module,
    device: torch.device,
    short_side: int = 672,
    layer_indices: tuple[int, ...] = (3, 6, 9),
) -> tuple[torch.Tensor, PreprocessInfo]:
    """Extract normalized patch grids after selected one-based transformer blocks."""

    if not layer_indices or any(index <= 0 for index in layer_indices):
        raise ValueError("layer_indices must contain positive one-based block indices")
    resized, info = resize_to_patch_grid(image, short_side=short_side)
    pixel_values = image_to_tensor(resized, processor, device)
    outputs = model(pixel_values=pixel_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None or max(layer_indices) >= len(hidden_states):
        raise ValueError(
            f"Requested layers {layer_indices}, but model returned {len(hidden_states or ())} hidden states"
        )
    num_patches = info.grid_height * info.grid_width
    grids = []
    for index in layer_indices:
        tokens = hidden_states[index][0]
        patches = tokens[-num_patches:].reshape(info.grid_height, info.grid_width, -1)
        grids.append(F.normalize(patches.float(), dim=-1))
    return torch.stack(grids), info


def cache_path_for_file(cache_root: Path, file_name: str) -> Path:
    return cache_root / Path(file_name).with_suffix(".npz")


def save_feature_cache(path: Path, features: torch.Tensor, info: PreprocessInfo) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.detach().cpu().numpy().astype(np.float16),
        original_width=info.original_width,
        original_height=info.original_height,
        resized_width=info.resized_width,
        resized_height=info.resized_height,
        scale=info.scale,
        grid_width=info.grid_width,
        grid_height=info.grid_height,
        patch_size=PATCH_SIZE,
    )


def load_feature_cache(path: Path) -> tuple[torch.Tensor, PreprocessInfo]:
    data = np.load(path)
    features = torch.from_numpy(data["features"].astype(np.float32))
    info = PreprocessInfo(
        original_width=int(data["original_width"]),
        original_height=int(data["original_height"]),
        resized_width=int(data["resized_width"]),
        resized_height=int(data["resized_height"]),
        scale=float(data["scale"]),
        grid_width=int(data["grid_width"]),
        grid_height=int(data["grid_height"]),
    )
    return features, info


def save_mvr_feature_cache(
    path: Path,
    features: torch.Tensor,
    info: PreprocessInfo,
    layer_indices: tuple[int, ...],
) -> None:
    """Save intermediate layers only; the canonical final-layer cache is reused."""

    if features.ndim != 4 or features.shape[0] != len(layer_indices):
        raise ValueError("MVR features must have shape [layers, height, width, channels]")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.detach().cpu().numpy().astype(np.float16),
        layer_indices=np.asarray(layer_indices, dtype=np.int16),
        original_width=info.original_width,
        original_height=info.original_height,
        resized_width=info.resized_width,
        resized_height=info.resized_height,
        scale=info.scale,
        grid_width=info.grid_width,
        grid_height=info.grid_height,
        patch_size=PATCH_SIZE,
    )


def load_mvr_feature_cache(path: Path) -> tuple[torch.Tensor, PreprocessInfo, tuple[int, ...]]:
    data = np.load(path)
    features = torch.from_numpy(data["features"].astype(np.float32))
    layer_indices = tuple(int(value) for value in data["layer_indices"].tolist())
    info = PreprocessInfo(
        original_width=int(data["original_width"]),
        original_height=int(data["original_height"]),
        resized_width=int(data["resized_width"]),
        resized_height=int(data["resized_height"]),
        scale=float(data["scale"]),
        grid_width=int(data["grid_width"]),
        grid_height=int(data["grid_height"]),
    )
    if features.shape[:3] != (len(layer_indices), info.grid_height, info.grid_width):
        raise ValueError(f"Corrupt MVR cache shape in {path}: {tuple(features.shape)}")
    return features, info, layer_indices


class DINOv3MVRReadout(nn.Module):
    """Residual multi-layer/multi-resolution readout over frozen patch grids.

    The trainable residual starts at zero, making the initial output exactly the
    normalized final-layer feature. Its Conv/ReLU/pooling/nearest-resize graph is
    intentionally simple so an ANN2SNN conversion can target the readout next.
    """

    def __init__(
        self,
        input_dim: int = 384,
        layer_count: int = 4,
        branch_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_count = layer_count
        self.branch_dim = branch_dim
        self.layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_dim, branch_dim, kernel_size=1),
                    nn.ReLU(inplace=False),
                )
                for _ in range(layer_count)
            ]
        )
        channels = layer_count * branch_dim
        self.fine_context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=False),
        )
        self.coarse_context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.ReLU(inplace=False),
        )
        self.residual = nn.Conv2d(channels * 3, input_dim, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        if layer_features.ndim != 4:
            raise ValueError("layer_features must have shape [layers, height, width, channels]")
        layers, height, width, channels = layer_features.shape
        if layers != self.layer_count or channels != self.input_dim:
            raise ValueError(
                f"Expected [{self.layer_count}, H, W, {self.input_dim}], got {tuple(layer_features.shape)}"
            )
        projected = []
        for projection, features in zip(self.layer_projections, layer_features):
            projected.append(projection(features.permute(2, 0, 1).unsqueeze(0)))
        native = torch.cat(projected, dim=1)
        fine = F.interpolate(native, scale_factor=2.0, mode="nearest")
        fine = self.fine_context(fine)
        fine = F.avg_pool2d(fine, kernel_size=2, stride=2)
        coarse = F.avg_pool2d(native, kernel_size=2, stride=2, ceil_mode=True)
        coarse = self.coarse_context(coarse)
        coarse = F.interpolate(coarse, size=(height, width), mode="nearest")
        delta = self.residual(torch.cat((native, fine, coarse), dim=1)).squeeze(0)
        reference = layer_features[-1].permute(2, 0, 1)
        fused = F.normalize(reference + delta, dim=0)
        return fused.permute(1, 2, 0)


class DINOv3MVRNativeReadout(nn.Module):
    """Residual multi-layer readout without fine/coarse context branches.

    This ablation preserves the final-layer feature as an exact initialization
    fallback while isolating the contribution of blocks 3/6/9/12 at their
    native patch-grid resolution. The existing full MVR module is intentionally
    left unchanged for checkpoint compatibility.
    """

    def __init__(
        self,
        input_dim: int = 384,
        layer_count: int = 4,
        branch_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_count = layer_count
        self.branch_dim = branch_dim
        self.layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_dim, branch_dim, kernel_size=1),
                    nn.ReLU(inplace=False),
                )
                for _ in range(layer_count)
            ]
        )
        channels = layer_count * branch_dim
        self.residual = nn.Conv2d(channels, input_dim, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        if layer_features.ndim != 4:
            raise ValueError("layer_features must have shape [layers, height, width, channels]")
        layers, _height, _width, channels = layer_features.shape
        if layers != self.layer_count or channels != self.input_dim:
            raise ValueError(
                f"Expected [{self.layer_count}, H, W, {self.input_dim}], got {tuple(layer_features.shape)}"
            )
        projected = [
            projection(features.permute(2, 0, 1).unsqueeze(0))
            for projection, features in zip(self.layer_projections, layer_features)
        ]
        native = torch.cat(projected, dim=1)
        delta = self.residual(native).squeeze(0)
        reference = layer_features[-1].permute(2, 0, 1)
        fused = F.normalize(reference + delta, dim=0)
        return fused.permute(1, 2, 0)


class DINOv3DetectionHead(nn.Module):
    """Small dense head over frozen DINO patch tokens."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.objectness = nn.Linear(hidden_dim, 1)
        self.box = nn.Linear(hidden_dim, 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(features)
        objectness_logits = self.objectness(hidden).squeeze(-1)
        raw_box = self.box(hidden)
        return objectness_logits, raw_box


class DINOv3LinearPatchHead(nn.Module):
    """Single linear patch decoder for frozen DINOv3 tokens.

    Each patch independently predicts one objectness logit and one box encoded
    as center offset plus log width/height relative to the patch center.
    """

    def __init__(self, input_dim: int = 384) -> None:
        super().__init__()
        self.prediction = nn.Linear(input_dim, 5)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.prediction(features)
        return raw[..., 0], raw[..., 1:5]


class DINOv3GridDetectionHead(nn.Module):
    """Convolutional grid head that uses neighboring DINO patch tokens."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 160) -> None:
        super().__init__()
        self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.context = nn.Sequential(
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        self.objectness = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.box = nn.Conv2d(hidden_dim, 4, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.permute(2, 0, 1).unsqueeze(0)
        hidden = self.context(self.proj(x))
        objectness_logits = self.objectness(hidden).squeeze(0).squeeze(0)
        raw_box = self.box(hidden).squeeze(0).permute(1, 2, 0)
        return objectness_logits, raw_box


class DINOv3FCOSHead(nn.Module):
    """Anchor-free FCOS-style grid head over DINO patch tokens."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 160) -> None:
        super().__init__()
        self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.shared = nn.Sequential(
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        self.classification = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.box = nn.Conv2d(hidden_dim, 4, kernel_size=1)
        self.centerness = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = features.permute(2, 0, 1).unsqueeze(0)
        hidden = self.shared(self.proj(x))
        class_logits = self.classification(hidden).squeeze(0).squeeze(0)
        box_raw = self.box(hidden).squeeze(0).permute(1, 2, 0)
        centerness_logits = self.centerness(hidden).squeeze(0).squeeze(0)
        return class_logits, box_raw, centerness_logits


class DINOv3MVRFCOSHead(nn.Module):
    """Multi-layer/multi-resolution readout followed by canonical FCOS."""

    def __init__(
        self,
        input_dim: int = 384,
        layer_count: int = 4,
        branch_dim: int = 64,
        hidden_dim: int = 160,
    ) -> None:
        super().__init__()
        self.readout = DINOv3MVRReadout(input_dim, layer_count, branch_dim)
        self.fcos = DINOv3FCOSHead(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, layer_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fcos(self.readout(layer_features))


class DINOv3MVRNativeFCOSHead(nn.Module):
    """Native-grid multi-layer MVR ablation followed by canonical FCOS."""

    def __init__(
        self,
        input_dim: int = 384,
        layer_count: int = 4,
        branch_dim: int = 64,
        hidden_dim: int = 160,
    ) -> None:
        super().__init__()
        self.readout = DINOv3MVRNativeReadout(input_dim, layer_count, branch_dim)
        self.fcos = DINOv3FCOSHead(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, layer_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.fcos(self.readout(layer_features))


class DINOv3FCOSQualityHead(DINOv3FCOSHead):
    """FCOS head with a learned localization-IoU prediction branch.

    All inherited module names match the canonical stride-16 head, so its
    trained tensors load unchanged while only the 1x1 quality predictor is new.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 160) -> None:
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim)
        self.quality = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.quality.weight)
        nn.init.zeros_(self.quality.bias)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = features.permute(2, 0, 1).unsqueeze(0)
        hidden = self.shared(self.proj(x))
        class_logits = self.classification(hidden).squeeze(0).squeeze(0)
        box_raw = self.box(hidden).squeeze(0).permute(1, 2, 0)
        centerness_logits = self.centerness(hidden).squeeze(0).squeeze(0)
        quality_logits = self.quality(hidden).squeeze(0).squeeze(0)
        return class_logits, box_raw, centerness_logits, quality_logits


class Stride8FeatureAdapter(nn.Module):
    """Learned 2x upsampling initialized as channel-preserving replication."""

    def __init__(self, channels: int = 160) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=2,
            stride=2,
            groups=channels,
            bias=False,
        )
        self.mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.upsample.weight.fill_(1.0)
            self.mix.weight.zero_()
            diagonal = torch.arange(self.mix.in_channels)
            self.mix.weight[diagonal, diagonal, 0, 0] = 1.0
            self.mix.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mix(self.upsample(features))


class DINOv3Stride8FCOSHead(nn.Module):
    """Single-level FCOS head on a learned stride-8 DINOv3 feature grid.

    Module names for the projection, shared tower, and predictors match
    :class:`DINOv3FCOSHead`, allowing their trained weights to be reused. Only
    ``stride8_adapter`` is new.
    """

    stride = 8

    def __init__(self, input_dim: int = 384, hidden_dim: int = 160) -> None:
        super().__init__()
        self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.stride8_adapter = Stride8FeatureAdapter(hidden_dim)
        self.shared = nn.Sequential(
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        self.classification = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.box = nn.Conv2d(hidden_dim, 4, kernel_size=1)
        self.centerness = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = features.permute(2, 0, 1).unsqueeze(0)
        hidden = self.shared(self.stride8_adapter(self.proj(x)))
        class_logits = self.classification(hidden).squeeze(0).squeeze(0)
        box_raw = self.box(hidden).squeeze(0).permute(1, 2, 0)
        centerness_logits = self.centerness(hidden).squeeze(0).squeeze(0)
        return class_logits, box_raw, centerness_logits


class DINOv3FPNFCOSHead(nn.Module):
    """Lightweight multi-scale FCOS head over DINO patch tokens.

    The fine level keeps the original DINO patch grid at stride 16. Medium and
    coarse levels are pooled from the projected patch grid at strides 32 and 64.
    The original single-scale FCOS `proj`, `shared`, and prediction heads keep
    their module names so compatible weights can be copied from existing FCOS
    checkpoints. Only the medium/coarse pyramid adapters are newly initialized.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 160,
        levels: tuple[tuple[str, int], ...] = (("fine", 16), ("medium", 32), ("coarse", 64)),
    ) -> None:
        super().__init__()
        self.levels = levels
        self.proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.pyramid_adapters = nn.ModuleDict(
            {
                "fine": nn.Identity(),
                "medium": nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(16, hidden_dim),
                    nn.GELU(),
                ),
                "coarse": nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GroupNorm(16, hidden_dim),
                    nn.GELU(),
                ),
            }
        )
        self.shared = nn.Sequential(
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(16, hidden_dim),
            nn.GELU(),
        )
        self.classification = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.box = nn.Conv2d(hidden_dim, 4, kernel_size=1)
        self.centerness = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> list[dict[str, Any]]:
        x = features.permute(2, 0, 1).unsqueeze(0)
        fine = self.proj(x)
        level_features: dict[str, torch.Tensor] = {"fine": fine}
        if any(name == "medium" for name, _stride in self.levels):
            level_features["medium"] = torch.nn.functional.avg_pool2d(
                fine,
                kernel_size=2,
                stride=2,
                ceil_mode=True,
            )
        if any(name == "coarse" for name, _stride in self.levels):
            medium_source = level_features.get("medium")
            if medium_source is None:
                medium_source = torch.nn.functional.avg_pool2d(fine, kernel_size=2, stride=2, ceil_mode=True)
            level_features["coarse"] = torch.nn.functional.avg_pool2d(
                medium_source,
                kernel_size=2,
                stride=2,
                ceil_mode=True,
            )

        outputs: list[dict[str, Any]] = []
        for name, stride in self.levels:
            adapted = self.pyramid_adapters[name](level_features[name])
            hidden = self.shared(adapted)
            outputs.append(
                {
                    "level": name,
                    "stride": stride,
                    "class_logits": self.classification(hidden).squeeze(0).squeeze(0),
                    "box_raw": self.box(hidden).squeeze(0).permute(1, 2, 0),
                    "centerness_logits": self.centerness(hidden).squeeze(0).squeeze(0),
                }
            )
        return outputs


def load_compatible_fcos_weights_into_fpn(
    fpn_head: DINOv3FPNFCOSHead,
    fcos_state: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Copy compatible single-scale FCOS weights into the FPN FCOS head."""

    fpn_state = fpn_head.state_dict()
    compatible = {
        name: tensor
        for name, tensor in fcos_state.items()
        if name in fpn_state and fpn_state[name].shape == tensor.shape
    }
    missing, unexpected = fpn_head.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "new_or_unloaded": sorted(missing),
        "unexpected": sorted(unexpected),
    }


def load_compatible_fcos_weights_into_stride8(
    stride8_head: DINOv3Stride8FCOSHead,
    fcos_state: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Load every shape-compatible stride-16 FCOS tensor into a stride-8 head."""

    target_state = stride8_head.state_dict()
    compatible = {
        name: tensor
        for name, tensor in fcos_state.items()
        if name in target_state and target_state[name].shape == tensor.shape
    }
    missing, unexpected = stride8_head.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "new_or_unloaded": sorted(missing),
        "unexpected": sorted(unexpected),
    }


def annotations_to_ground_truth(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ground_truth: list[dict[str, Any]] = []
    for annotation in annotations:
        metadata = annotation.get("metadata", {})
        visibility = metadata.get("visibility_level")
        if visibility is None:
            continue
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        ground_truth.append(
            {
                "annotation_id": annotation.get("id"),
                "box": [x, y, x + width, y + height],
                "visibility_level": int(visibility),
                "pose": metadata.get("pose", "unknown"),
            }
        )
    return ground_truth


def build_dense_targets(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    objectness = torch.zeros((info.grid_height, info.grid_width), dtype=torch.float32)
    box_targets = torch.zeros((info.grid_height, info.grid_width, 4), dtype=torch.float32)
    box_mask = torch.zeros((info.grid_height, info.grid_width), dtype=torch.bool)
    area_by_cell: dict[tuple[int, int], float] = {}
    collisions = 0

    for gt in ground_truth:
        x1 = gt["box"][0] * info.scale_x
        y1 = gt["box"][1] * info.scale_y
        x2 = gt["box"][2] * info.scale_x
        y2 = gt["box"][3] * info.scale_y
        x1 = min(max(x1, 0.0), info.resized_width - 1.0)
        x2 = min(max(x2, 0.0), info.resized_width - 1.0)
        y1 = min(max(y1, 0.0), info.resized_height - 1.0)
        y2 = min(max(y2, 0.0), info.resized_height - 1.0)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        gx = min(info.grid_width - 1, max(0, int(cx // PATCH_SIZE)))
        gy = min(info.grid_height - 1, max(0, int(cy // PATCH_SIZE)))
        key = (gy, gx)
        area = (x2 - x1) * (y2 - y1)
        if key in area_by_cell:
            collisions += 1
            if area >= area_by_cell[key]:
                continue
        area_by_cell[key] = area
        cell_x = (gx + 0.5) * PATCH_SIZE
        cell_y = (gy + 0.5) * PATCH_SIZE
        objectness[gy, gx] = 1.0
        box_mask[gy, gx] = True
        box_targets[gy, gx] = torch.tensor(
            [
                (cx - cell_x) / PATCH_SIZE,
                (cy - cell_y) / PATCH_SIZE,
                math.log(max((x2 - x1) / PATCH_SIZE, 1e-6)),
                math.log(max((y2 - y1) / PATCH_SIZE, 1e-6)),
            ],
            dtype=torch.float32,
        )
    return objectness, box_targets, box_mask, collisions


def build_fcos_targets(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
    center_sampling_radius: float = 1.5,
    stride: int = PATCH_SIZE,
    grid_height: int | None = None,
    grid_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    grid_height = grid_height if grid_height is not None else info.resized_height // stride
    grid_width = grid_width if grid_width is not None else info.resized_width // stride
    classification = torch.zeros((grid_height, grid_width), dtype=torch.float32)
    box_targets = torch.zeros((grid_height, grid_width, 4), dtype=torch.float32)
    centerness = torch.zeros((grid_height, grid_width), dtype=torch.float32)
    positive_mask = torch.zeros((grid_height, grid_width), dtype=torch.bool)
    assigned_area = torch.full((grid_height, grid_width), float("inf"), dtype=torch.float32)
    stats = {
        "positive_cells": 0,
        "center_fallbacks": 0,
        "multi_gt_overwrites": 0,
        "annotations_with_positive": 0,
    }

    grid_x = (torch.arange(grid_width, dtype=torch.float32) + 0.5) * stride
    grid_y = (torch.arange(grid_height, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    for gt in ground_truth:
        x1 = max(0.0, min(gt["box"][0] * info.scale_x, info.resized_width - 1.0))
        y1 = max(0.0, min(gt["box"][1] * info.scale_y, info.resized_height - 1.0))
        x2 = max(0.0, min(gt["box"][2] * info.scale_x, info.resized_width - 1.0))
        y2 = max(0.0, min(gt["box"][3] * info.scale_y, info.resized_height - 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        width = x2 - x1
        height = y2 - y1
        area = width * height
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        left = xx - x1
        top = yy - y1
        right = x2 - xx
        bottom = y2 - yy
        inside_box = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)

        radius = center_sampling_radius * stride
        center_x1 = max(x1, cx - radius)
        center_y1 = max(y1, cy - radius)
        center_x2 = min(x2, cx + radius)
        center_y2 = min(y2, cy + radius)
        inside_center = (xx >= center_x1) & (xx <= center_x2) & (yy >= center_y1) & (yy <= center_y2)
        candidates = inside_box & inside_center

        if not candidates.any():
            gx = min(grid_width - 1, max(0, int(cx // stride)))
            gy = min(grid_height - 1, max(0, int(cy // stride)))
            candidates[gy, gx] = True
            stats["center_fallbacks"] += 1

        assign = candidates & (area < assigned_area)
        overwritten = int((assign & positive_mask).sum().item())
        stats["multi_gt_overwrites"] += overwritten
        if not assign.any():
            continue

        l = (xx[assign] - x1).clamp(min=1e-3)
        t = (yy[assign] - y1).clamp(min=1e-3)
        r = (x2 - xx[assign]).clamp(min=1e-3)
        b = (y2 - yy[assign]).clamp(min=1e-3)
        lr = torch.minimum(l, r) / torch.maximum(l, r)
        tb = torch.minimum(t, b) / torch.maximum(t, b)
        center_targets = torch.sqrt((lr * tb).clamp(min=0.0, max=1.0))

        classification[assign] = 1.0
        positive_mask[assign] = True
        assigned_area[assign] = area
        box_targets[assign] = torch.stack([l, t, r, b], dim=1) / stride
        centerness[assign] = center_targets
        stats["annotations_with_positive"] += 1

    stats["positive_cells"] = int(positive_mask.sum().item())
    return classification, box_targets, centerness, positive_mask, stats


def analyze_fcos_positive_locations(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
    *,
    stride: int,
    center_sampling_radius: float = 1.5,
) -> list[dict[str, Any]]:
    """Report final assigned locations per GT using the production FCOS rules."""

    grid_height = info.resized_height // stride
    grid_width = info.resized_width // stride
    grid_x = (torch.arange(grid_width, dtype=torch.float32) + 0.5) * stride
    grid_y = (torch.arange(grid_height, dtype=torch.float32) + 0.5) * stride
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    assigned_area = torch.full((grid_height, grid_width), float("inf"), dtype=torch.float32)
    owner = torch.full((grid_height, grid_width), -1, dtype=torch.long)
    records: list[dict[str, Any]] = []

    for gt_index, gt in enumerate(ground_truth):
        x1 = max(0.0, min(gt["box"][0] * info.scale_x, info.resized_width - 1.0))
        y1 = max(0.0, min(gt["box"][1] * info.scale_y, info.resized_height - 1.0))
        x2 = max(0.0, min(gt["box"][2] * info.scale_x, info.resized_width - 1.0))
        y2 = max(0.0, min(gt["box"][3] * info.scale_y, info.resized_height - 1.0))
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        record = {
            "gt_index": gt_index,
            "annotation_id": gt.get("annotation_id"),
            "visibility_level": gt.get("visibility_level"),
            "resized_width": width,
            "resized_height": height,
            "resized_area": width * height,
            "positive_locations": 0,
            "used_center_fallback": False,
        }
        records.append(record)
        if width <= 0 or height <= 0:
            continue
        area = width * height
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        left = xx - x1
        top = yy - y1
        right = x2 - xx
        bottom = y2 - yy
        inside_box = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)
        radius = center_sampling_radius * stride
        inside_center = (
            (xx >= max(x1, cx - radius))
            & (xx <= min(x2, cx + radius))
            & (yy >= max(y1, cy - radius))
            & (yy <= min(y2, cy + radius))
        )
        candidates = inside_box & inside_center
        if not candidates.any():
            gx = min(grid_width - 1, max(0, int(cx // stride)))
            gy = min(grid_height - 1, max(0, int(cy // stride)))
            candidates[gy, gx] = True
            record["used_center_fallback"] = True
        assign = candidates & (area < assigned_area)
        assigned_area[assign] = area
        owner[assign] = gt_index

    for gt_index, record in enumerate(records):
        record["positive_locations"] = int((owner == gt_index).sum().item())
    return records


def _make_level_grid(height: int, width: int, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    grid_x = (torch.arange(width, dtype=torch.float32) + 0.5) * float(stride)
    grid_y = (torch.arange(height, dtype=torch.float32) + 0.5) * float(stride)
    return torch.meshgrid(grid_y, grid_x, indexing="ij")


def build_fpn_fcos_targets(
    ground_truth: list[dict[str, Any]],
    info: PreprocessInfo,
    level_shapes: list[tuple[str, int, int, int]],
    center_sampling_radius: float = 1.5,
    size_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build scale-aware FCOS targets for a lightweight feature pyramid.

    `level_shapes` entries are `(level_name, stride, height, width)`. Distances
    are encoded in units of the level stride, mirroring the single-scale FCOS
    target format where stride is the DINO patch size.
    """

    if size_ranges is None:
        size_ranges = {
            "fine": (0.0, 80.0),
            "medium": (48.0, 192.0),
            "coarse": (128.0, float("inf")),
        }
    targets: list[dict[str, Any]] = []
    assigned_area: dict[str, torch.Tensor] = {}
    stats = {
        "positive_cells": 0,
        "center_fallbacks": 0,
        "multi_gt_overwrites": 0,
        "annotations_with_positive": 0,
    }

    for level_name, stride, height, width in level_shapes:
        targets.append(
            {
                "level": level_name,
                "stride": stride,
                "classification": torch.zeros((height, width), dtype=torch.float32),
                "box_targets": torch.zeros((height, width, 4), dtype=torch.float32),
                "centerness": torch.zeros((height, width), dtype=torch.float32),
                "positive_mask": torch.zeros((height, width), dtype=torch.bool),
            }
        )
        assigned_area[level_name] = torch.full((height, width), float("inf"), dtype=torch.float32)

    for gt in ground_truth:
        x1 = max(0.0, min(gt["box"][0] * info.scale_x, info.resized_width - 1.0))
        y1 = max(0.0, min(gt["box"][1] * info.scale_y, info.resized_height - 1.0))
        x2 = max(0.0, min(gt["box"][2] * info.scale_x, info.resized_width - 1.0))
        y2 = max(0.0, min(gt["box"][3] * info.scale_y, info.resized_height - 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        width = x2 - x1
        height = y2 - y1
        max_side = max(width, height)
        area = width * height
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        annotation_has_positive = False

        for target in targets:
            level_name = str(target["level"])
            stride = int(target["stride"])
            min_size, max_size = size_ranges.get(level_name, (0.0, float("inf")))
            if not (min_size <= max_side <= max_size):
                continue
            yy, xx = _make_level_grid(
                int(target["classification"].shape[0]),
                int(target["classification"].shape[1]),
                stride,
            )
            left = xx - x1
            top = yy - y1
            right = x2 - xx
            bottom = y2 - yy
            inside_box = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)
            radius = center_sampling_radius * float(stride)
            center_x1 = max(x1, cx - radius)
            center_y1 = max(y1, cy - radius)
            center_x2 = min(x2, cx + radius)
            center_y2 = min(y2, cy + radius)
            inside_center = (xx >= center_x1) & (xx <= center_x2) & (yy >= center_y1) & (yy <= center_y2)
            candidates = inside_box & inside_center

            assign = candidates & (area < assigned_area[level_name])
            overwritten = int((assign & target["positive_mask"]).sum().item())
            stats["multi_gt_overwrites"] += overwritten
            if not assign.any():
                continue

            l = (xx[assign] - x1).clamp(min=1e-3)
            t = (yy[assign] - y1).clamp(min=1e-3)
            r = (x2 - xx[assign]).clamp(min=1e-3)
            b = (y2 - yy[assign]).clamp(min=1e-3)
            lr = torch.minimum(l, r) / torch.maximum(l, r)
            tb = torch.minimum(t, b) / torch.maximum(t, b)
            center_targets = torch.sqrt((lr * tb).clamp(min=0.0, max=1.0))

            target["classification"][assign] = 1.0
            target["positive_mask"][assign] = True
            assigned_area[level_name][assign] = area
            target["box_targets"][assign] = torch.stack([l, t, r, b], dim=1) / float(stride)
            target["centerness"][assign] = center_targets
            annotation_has_positive = True

        if not annotation_has_positive:
            finest = min(targets, key=lambda item: int(item["stride"]))
            stride = int(finest["stride"])
            gx = min(int(finest["classification"].shape[1]) - 1, max(0, int(cx // stride)))
            gy = min(int(finest["classification"].shape[0]) - 1, max(0, int(cy // stride)))
            key_area = assigned_area[str(finest["level"])][gy, gx]
            if area < float(key_area):
                center_x = (gx + 0.5) * stride
                center_y = (gy + 0.5) * stride
                l = max(center_x - x1, 1e-3)
                t = max(center_y - y1, 1e-3)
                r = max(x2 - center_x, 1e-3)
                b = max(y2 - center_y, 1e-3)
                lr = min(l, r) / max(l, r)
                tb = min(t, b) / max(t, b)
                finest["classification"][gy, gx] = 1.0
                finest["positive_mask"][gy, gx] = True
                assigned_area[str(finest["level"])][gy, gx] = area
                finest["box_targets"][gy, gx] = torch.tensor([l, t, r, b], dtype=torch.float32) / float(stride)
                finest["centerness"][gy, gx] = math.sqrt(max(0.0, min(1.0, lr * tb)))
                annotation_has_positive = True
                stats["center_fallbacks"] += 1

        if annotation_has_positive:
            stats["annotations_with_positive"] += 1

    stats["positive_cells"] = sum(int(target["positive_mask"].sum().item()) for target in targets)
    return targets, stats


def decode_predictions(
    objectness_logits: torch.Tensor,
    box_raw: torch.Tensor,
    info: PreprocessInfo,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.50,
    topk: int = 300,
) -> list[dict[str, Any]]:
    scores = objectness_logits.sigmoid().reshape(-1)
    boxes_raw = box_raw.reshape(-1, 4)
    keep = torch.nonzero(scores >= conf_threshold, as_tuple=False).flatten()
    if keep.numel() == 0:
        return []
    keep_scores = scores[keep]
    if keep.numel() > topk:
        top_scores, top_indices = torch.topk(keep_scores, k=topk)
        keep = keep[top_indices]
        keep_scores = top_scores

    grid_y = torch.div(keep, info.grid_width, rounding_mode="floor").float()
    grid_x = (keep % info.grid_width).float()
    raw = boxes_raw[keep]
    cx = (grid_x + 0.5 + raw[:, 0].clamp(-2.0, 2.0)) * PATCH_SIZE
    cy = (grid_y + 0.5 + raw[:, 1].clamp(-2.0, 2.0)) * PATCH_SIZE
    width = torch.exp(raw[:, 2].clamp(-4.0, 6.0)) * PATCH_SIZE
    height = torch.exp(raw[:, 3].clamp(-4.0, 6.0)) * PATCH_SIZE
    x1 = (cx - 0.5 * width).clamp(0, info.resized_width - 1)
    y1 = (cy - 0.5 * height).clamp(0, info.resized_height - 1)
    x2 = (cx + 0.5 * width).clamp(0, info.resized_width - 1)
    y2 = (cy + 0.5 * height).clamp(0, info.resized_height - 1)
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    keep_scores = keep_scores[valid]
    if boxes.numel() == 0:
        return []
    selected = nms(boxes, keep_scores, nms_threshold)
    boxes = boxes[selected]
    scale = torch.tensor(
        [info.scale_x, info.scale_y, info.scale_x, info.scale_y],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    boxes = boxes / scale
    keep_scores = keep_scores[selected]
    predictions = []
    for box, score in zip(boxes.detach().cpu(), keep_scores.detach().cpu()):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        predictions.append(
            {
                "box": [
                    max(0.0, min(x1, info.original_width - 1.0)),
                    max(0.0, min(y1, info.original_height - 1.0)),
                    max(0.0, min(x2, info.original_width - 1.0)),
                    max(0.0, min(y2, info.original_height - 1.0)),
                ],
                "confidence": float(score),
            }
        )
    return predictions


def decode_fcos_predictions(
    class_logits: torch.Tensor,
    box_raw: torch.Tensor,
    centerness_logits: torch.Tensor,
    info: PreprocessInfo,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.50,
    topk: int = 300,
    stride: int = PATCH_SIZE,
    quality_logits: torch.Tensor | None = None,
    class_score_power: float = 0.5,
    centerness_score_power: float = 0.5,
    quality_score_power: float = 0.0,
) -> list[dict[str, Any]]:
    class_scores = class_logits.sigmoid().reshape(-1)
    center_scores = centerness_logits.sigmoid().reshape(-1)
    if quality_score_power != 0.0 and quality_logits is None:
        raise ValueError("quality_logits are required when quality_score_power is nonzero")
    if (
        class_score_power == 0.5
        and centerness_score_power == 0.5
        and quality_score_power == 0.0
    ):
        scores = torch.sqrt((class_scores * center_scores).clamp(min=0.0, max=1.0))
    else:
        scores = class_scores.clamp(min=1e-12).pow(class_score_power)
        scores = scores * center_scores.clamp(min=1e-12).pow(centerness_score_power)
        if quality_score_power != 0.0:
            quality_scores = quality_logits.sigmoid().reshape(-1)
            scores = scores * quality_scores.clamp(min=1e-12).pow(quality_score_power)
    boxes_raw = box_raw.reshape(-1, 4)
    keep = torch.nonzero(scores >= conf_threshold, as_tuple=False).flatten()
    if keep.numel() == 0:
        return []
    keep_scores = scores[keep]
    if keep.numel() > topk:
        top_scores, top_indices = torch.topk(keep_scores, k=topk)
        keep = keep[top_indices]
        keep_scores = top_scores

    grid_height, grid_width = class_logits.shape
    if grid_height * grid_width != boxes_raw.shape[0]:
        raise ValueError("FCOS logits and box grid shapes do not match")
    grid_y = torch.div(keep, grid_width, rounding_mode="floor").float()
    grid_x = (keep % grid_width).float()
    centers_x = (grid_x + 0.5) * stride
    centers_y = (grid_y + 0.5) * stride
    distances = torch.nn.functional.softplus(boxes_raw[keep]).clamp(max=128.0) * stride
    left, top, right, bottom = distances.unbind(dim=1)
    boxes = torch.stack(
        [
            (centers_x - left).clamp(0, info.resized_width - 1),
            (centers_y - top).clamp(0, info.resized_height - 1),
            (centers_x + right).clamp(0, info.resized_width - 1),
            (centers_y + bottom).clamp(0, info.resized_height - 1),
        ],
        dim=1,
    )
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    keep_scores = keep_scores[valid]
    if boxes.numel() == 0:
        return []
    selected = nms(boxes, keep_scores, nms_threshold)
    boxes = boxes[selected]
    scale = torch.tensor(
        [info.scale_x, info.scale_y, info.scale_x, info.scale_y],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    boxes = boxes / scale
    keep_scores = keep_scores[selected]
    predictions = []
    for box, score in zip(boxes.detach().cpu(), keep_scores.detach().cpu()):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        predictions.append(
            {
                "box": [
                    max(0.0, min(x1, info.original_width - 1.0)),
                    max(0.0, min(y1, info.original_height - 1.0)),
                    max(0.0, min(x2, info.original_width - 1.0)),
                    max(0.0, min(y2, info.original_height - 1.0)),
                ],
                "confidence": float(score),
            }
        )
    return predictions


def decode_fpn_fcos_predictions(
    level_outputs: list[dict[str, Any]],
    info: PreprocessInfo,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.50,
    topk: int = 300,
) -> list[dict[str, Any]]:
    """Decode and NMS predictions from all FPN FCOS levels together."""

    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    for output in level_outputs:
        class_logits = output["class_logits"]
        box_raw = output["box_raw"]
        centerness_logits = output["centerness_logits"]
        stride = float(output["stride"])
        height, width = class_logits.shape

        class_scores = class_logits.sigmoid().reshape(-1)
        center_scores = centerness_logits.sigmoid().reshape(-1)
        scores = torch.sqrt((class_scores * center_scores).clamp(min=0.0, max=1.0))
        boxes_raw = box_raw.reshape(-1, 4)
        keep = torch.nonzero(scores >= conf_threshold, as_tuple=False).flatten()
        if keep.numel() == 0:
            continue
        keep_scores = scores[keep]
        grid_y = torch.div(keep, width, rounding_mode="floor").float()
        grid_x = (keep % width).float()
        centers_x = (grid_x + 0.5) * stride
        centers_y = (grid_y + 0.5) * stride
        distances = torch.nn.functional.softplus(boxes_raw[keep]).clamp(max=128.0) * stride
        left, top, right, bottom = distances.unbind(dim=1)
        boxes = torch.stack(
            [
                (centers_x - left).clamp(0, info.resized_width - 1),
                (centers_y - top).clamp(0, info.resized_height - 1),
                (centers_x + right).clamp(0, info.resized_width - 1),
                (centers_y + bottom).clamp(0, info.resized_height - 1),
            ],
            dim=1,
        )
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if valid.any():
            all_boxes.append(boxes[valid])
            all_scores.append(keep_scores[valid])

    if not all_boxes:
        return []
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    if scores.numel() > topk:
        scores, indices = torch.topk(scores, k=topk)
        boxes = boxes[indices]
    selected = nms(boxes, scores, nms_threshold)
    boxes = boxes[selected]
    scores = scores[selected]
    scale = torch.tensor(
        [info.scale_x, info.scale_y, info.scale_x, info.scale_y],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    boxes = boxes / scale

    predictions = []
    for box, score in zip(boxes.detach().cpu(), scores.detach().cpu()):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        predictions.append(
            {
                "box": [
                    max(0.0, min(x1, info.original_width - 1.0)),
                    max(0.0, min(y1, info.original_height - 1.0)),
                    max(0.0, min(x2, info.original_width - 1.0)),
                    max(0.0, min(y2, info.original_height - 1.0)),
                ],
                "confidence": float(score),
            }
        )
    return predictions
