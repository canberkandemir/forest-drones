"""Geometry helpers for exploratory DINOv3 multi-view feature fusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from .dinov3_detector import PreprocessInfo


PATCH_SIZE = 16


class TrainableReferencePreservingFusion(nn.Module):
    """Shared local gate for two aligned sources and a dominant reference.

    The gate receives reference/source features, their absolute difference,
    cosine agreement, and the geometric validity mask. Ground truth and
    detector outputs are never inputs. The same parameters are shared across
    sources, and combined source contribution is capped before convex fusion.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        hidden_dim: int = 32,
        maximum_total_source_weight: float = 0.50,
        initial_source_logit: float = -4.6,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        if not 0.0 <= maximum_total_source_weight <= 1.0:
            raise ValueError("maximum source weight must be in [0, 1]")
        gate_input_dim = 3 * feature_dim + 2
        self.gate = nn.Sequential(
            nn.Conv2d(gate_input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        self.maximum_total_source_weight = float(maximum_total_source_weight)
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gate[-1].bias, initial_source_logit)

    def forward(
        self,
        reference_features: torch.Tensor,
        aligned_sources: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
        if reference_features.ndim != 3:
            raise ValueError("reference features must have shape [H,W,C]")
        reference = F.normalize(reference_features, dim=-1)
        raw_weights: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for source_features, validity in aligned_sources:
            if source_features.shape != reference.shape:
                raise ValueError("aligned source features must match the reference shape")
            if validity.shape != reference.shape[:2]:
                raise ValueError("validity mask must match the reference grid")
            source = F.normalize(source_features, dim=-1)
            valid = validity.to(dtype=reference.dtype).clamp(0.0, 1.0)
            cosine = torch.sum(reference * source, dim=-1, keepdim=True).clamp(-1.0, 1.0)
            inputs = torch.cat(
                [reference, source, torch.abs(reference - source), cosine, valid.unsqueeze(-1)],
                dim=-1,
            ).permute(2, 0, 1).unsqueeze(0)
            source_logits = self.gate(inputs)[0, 0]
            reliable = valid >= 0.99
            weight = torch.sigmoid(source_logits) * valid * reliable.to(valid.dtype)
            logits.append(source_logits)
            raw_weights.append(weight)

        if not raw_weights:
            zeros = torch.zeros(reference.shape[:2], device=reference.device, dtype=reference.dtype)
            return reference_features, {
                "source_logits": [],
                "source_weights": [],
                "total_source_weight": zeros,
                "fallback_mask": torch.ones_like(zeros, dtype=torch.bool),
            }

        raw_total = torch.stack(raw_weights).sum(dim=0)
        if self.maximum_total_source_weight == 0.0:
            cap_scale = torch.zeros_like(raw_total)
        else:
            cap_scale = torch.minimum(
                torch.ones_like(raw_total),
                torch.full_like(raw_total, self.maximum_total_source_weight) / raw_total.clamp(min=1e-6),
            )
        weights = [weight * cap_scale for weight in raw_weights]
        total_weight = torch.stack(weights).sum(dim=0)
        numerator = reference.clone()
        for (source_features, _), weight in zip(aligned_sources, weights, strict=True):
            numerator = numerator + F.normalize(source_features, dim=-1) * weight.unsqueeze(-1)
        fused = F.normalize(numerator / (1.0 + total_weight).unsqueeze(-1), dim=-1)
        fallback = total_weight <= 1e-8
        fused = torch.where(fallback.unsqueeze(-1), reference_features, fused)
        return fused, {
            "source_logits": logits,
            "source_weights": weights,
            "total_source_weight": total_weight,
            "fallback_mask": fallback,
        }


DEFAULT_REGISTRATION_THRESHOLDS = {
    "minimum_ratio_test_matches": 40,
    "minimum_ransac_inliers": 30,
    "minimum_inlier_ratio": 0.50,
    "maximum_median_reprojection_error_registration_pixels": 2.0,
    "minimum_reference_overlap_fraction": 0.70,
    "minimum_source_inlier_spread_fraction": 0.05,
    "minimum_reference_inlier_spread_fraction": 0.05,
    "minimum_projected_area_ratio": 0.50,
    "maximum_projected_area_ratio": 1.50,
}


def registration_rejection_reasons(
    metrics: dict[str, float | int],
    thresholds: dict[str, float | int] | None = None,
) -> list[str]:
    """Return deterministic image-only registration rejection reasons."""

    limits = dict(DEFAULT_REGISTRATION_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    checks = (
        ("ratio_test_matches", "minimum_ratio_test_matches", "too_few_ratio_matches", False),
        ("ransac_inliers", "minimum_ransac_inliers", "too_few_ransac_inliers", False),
        ("inlier_ratio", "minimum_inlier_ratio", "low_inlier_ratio", False),
        (
            "median_reprojection_error_registration_pixels",
            "maximum_median_reprojection_error_registration_pixels",
            "high_reprojection_error",
            True,
        ),
        (
            "reference_overlap_fraction",
            "minimum_reference_overlap_fraction",
            "low_reference_overlap",
            False,
        ),
        (
            "source_inlier_spread_fraction",
            "minimum_source_inlier_spread_fraction",
            "localized_source_matches",
            False,
        ),
        (
            "reference_inlier_spread_fraction",
            "minimum_reference_inlier_spread_fraction",
            "localized_reference_matches",
            False,
        ),
    )
    reasons: list[str] = []
    for metric_key, threshold_key, reason, maximum in checks:
        value = float(metrics[metric_key])
        limit = float(limits[threshold_key])
        if (maximum and value > limit) or (not maximum and value < limit):
            reasons.append(reason)
    area_ratio = float(metrics["projected_area_ratio"])
    if area_ratio < float(limits["minimum_projected_area_ratio"]):
        reasons.append("projected_area_too_small")
    if area_ratio > float(limits["maximum_projected_area_ratio"]):
        reasons.append("projected_area_too_large")
    return reasons


def validate_unique_group_images(groups: list[dict]) -> None:
    """Reject manifests that reuse an image within or across view groups."""

    used: set[int | str] = set()
    references: set[int | str] = set()
    for group in groups:
        reference_id = group["reference"]["image_id"]
        if reference_id in references:
            raise ValueError(f"Duplicate reference image ID: {reference_id}")
        references.add(reference_id)
        image_ids = [reference_id, *[source["image_id"] for source in group["sources"]]]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"Group {group.get('group_id')} reuses a view")
        overlap = used.intersection(image_ids)
        if overlap:
            raise ValueError(f"Images reused across groups: {sorted(overlap, key=str)}")
        used.update(image_ids)


def resized_homography(
    source_to_reference: torch.Tensor,
    source_info: PreprocessInfo,
    reference_info: PreprocessInfo,
) -> torch.Tensor:
    """Convert an original-image homography to resized-image coordinates."""

    if source_to_reference.shape != (3, 3):
        raise ValueError("source_to_reference must be a 3x3 homography")
    device = source_to_reference.device
    dtype = source_to_reference.dtype
    source_scale = torch.tensor(
        [[source_info.scale_x, 0.0, 0.0], [0.0, source_info.scale_y, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    reference_scale = torch.tensor(
        [
            [reference_info.scale_x, 0.0, 0.0],
            [0.0, reference_info.scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return reference_scale @ source_to_reference @ torch.linalg.inv(source_scale)


def warp_patch_features(
    source_features: torch.Tensor,
    source_info: PreprocessInfo,
    reference_info: PreprocessInfo,
    source_to_reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp source [H,W,C] patch features onto a reference patch grid.

    Homographies are defined in original-image pixel coordinates. Sampling uses
    patch-center coordinates and returns both warped features and a soft validity
    mask so unsupported reference locations are never averaged with zeros.
    """

    if source_features.ndim != 3:
        raise ValueError("source_features must have shape [H,W,C]")
    source_height, source_width, _ = source_features.shape
    if source_height != source_info.grid_height or source_width != source_info.grid_width:
        raise ValueError("source feature shape does not match preprocessing metadata")
    device = source_features.device
    dtype = source_features.dtype
    homography = resized_homography(
        source_to_reference.to(device=device, dtype=torch.float32),
        source_info,
        reference_info,
    )
    inverse = torch.linalg.inv(homography)

    y = (torch.arange(reference_info.grid_height, device=device, dtype=torch.float32) + 0.5) * PATCH_SIZE
    x = (torch.arange(reference_info.grid_width, device=device, dtype=torch.float32) + 0.5) * PATCH_SIZE
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    reference_points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    source_points = reference_points @ inverse.T
    denominator = source_points[..., 2:3].clamp(min=1e-6)
    source_xy = source_points[..., :2] / denominator
    source_feature_x = source_xy[..., 0] / PATCH_SIZE - 0.5
    source_feature_y = source_xy[..., 1] / PATCH_SIZE - 0.5
    if source_width > 1:
        grid_x = 2.0 * source_feature_x / (source_width - 1) - 1.0
    else:
        grid_x = torch.zeros_like(source_feature_x)
    if source_height > 1:
        grid_y = 2.0 * source_feature_y / (source_height - 1) - 1.0
    else:
        grid_y = torch.zeros_like(source_feature_y)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    source = source_features.permute(2, 0, 1).unsqueeze(0)
    warped = F.grid_sample(
        source,
        grid.to(dtype=source.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0].permute(1, 2, 0)
    ones = torch.ones((1, 1, source_height, source_width), device=device, dtype=dtype)
    validity = F.grid_sample(
        ones,
        grid.to(dtype=dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    return warped, validity.clamp(0.0, 1.0)


def fuse_aligned_features(
    reference_features: torch.Tensor,
    aligned_sources: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Equal-weight average valid aligned views and L2-normalize channels."""

    numerator = reference_features.clone()
    denominator = torch.ones(reference_features.shape[:2], device=reference_features.device, dtype=reference_features.dtype)
    for features, validity in aligned_sources:
        if features.shape != reference_features.shape:
            raise ValueError("all aligned source features must match the reference shape")
        if validity.shape != reference_features.shape[:2]:
            raise ValueError("validity masks must match the reference spatial grid")
        weight = validity.to(dtype=reference_features.dtype)
        numerator = numerator + features * weight.unsqueeze(-1)
        denominator = denominator + weight
    return F.normalize(numerator / denominator.unsqueeze(-1).clamp(min=1e-6), dim=-1)


def fuse_selective_aligned_features(
    reference_features: torch.Tensor,
    aligned_sources: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    minimum_cosine: float = 0.90,
    full_trust_cosine: float = 0.98,
    local_kernel_size: int = 3,
    minimum_local_valid_fraction: float = 0.75,
    maximum_total_source_weight: float = 0.50,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | list[torch.Tensor]]]:
    """Fuse only locally consistent sources while preserving the reference.

    The rule is non-trainable and uses no detection or annotation information.
    A source patch contributes only when its feature cosine agrees with the
    reference both pointwise and over a valid local neighborhood. Combined
    source weight is capped so the reference remains dominant.
    """

    if reference_features.ndim != 3:
        raise ValueError("reference_features must have shape [H,W,C]")
    if not 0.0 <= minimum_cosine < full_trust_cosine <= 1.0:
        raise ValueError("cosine thresholds must satisfy 0 <= minimum < full <= 1")
    if local_kernel_size < 1 or local_kernel_size % 2 == 0:
        raise ValueError("local_kernel_size must be a positive odd integer")
    if maximum_total_source_weight < 0.0:
        raise ValueError("maximum_total_source_weight must be non-negative")

    reference = F.normalize(reference_features, dim=-1)
    source_weights: list[torch.Tensor] = []
    agreement_maps: list[torch.Tensor] = []
    padding = local_kernel_size // 2
    for features, validity in aligned_sources:
        if features.shape != reference.shape:
            raise ValueError("all aligned source features must match the reference shape")
        if validity.shape != reference.shape[:2]:
            raise ValueError("validity masks must match the reference spatial grid")
        source = F.normalize(features, dim=-1)
        valid = validity.to(dtype=reference.dtype).clamp(0.0, 1.0)
        point_cosine = torch.sum(reference * source, dim=-1).clamp(-1.0, 1.0)
        local_numerator = F.avg_pool2d(
            (point_cosine * valid).unsqueeze(0).unsqueeze(0),
            kernel_size=local_kernel_size,
            stride=1,
            padding=padding,
        )[0, 0]
        local_valid = F.avg_pool2d(
            valid.unsqueeze(0).unsqueeze(0),
            kernel_size=local_kernel_size,
            stride=1,
            padding=padding,
        )[0, 0]
        local_cosine = local_numerator / local_valid.clamp(min=1e-6)
        agreement = torch.minimum(point_cosine, local_cosine)
        weight = ((agreement - minimum_cosine) / (full_trust_cosine - minimum_cosine)).clamp(0.0, 1.0)
        reliable = (valid >= 0.99) & (local_valid >= minimum_local_valid_fraction)
        weight = weight * reliable.to(dtype=weight.dtype)
        source_weights.append(weight)
        agreement_maps.append(agreement)

    if not source_weights:
        zeros = torch.zeros(reference.shape[:2], device=reference.device, dtype=reference.dtype)
        return reference_features, {
            "source_weights": [],
            "agreement_maps": [],
            "total_source_weight": zeros,
            "fallback_mask": torch.ones_like(zeros, dtype=torch.bool),
        }

    raw_total = torch.stack(source_weights).sum(dim=0)
    if maximum_total_source_weight == 0.0:
        scale = torch.zeros_like(raw_total)
    else:
        scale = torch.minimum(
            torch.ones_like(raw_total),
            torch.full_like(raw_total, maximum_total_source_weight) / raw_total.clamp(min=1e-6),
        )
    capped_weights = [weight * scale for weight in source_weights]
    total_weight = torch.stack(capped_weights).sum(dim=0)
    numerator = reference.clone()
    for (features, _), weight in zip(aligned_sources, capped_weights, strict=True):
        numerator = numerator + F.normalize(features, dim=-1) * weight.unsqueeze(-1)
    fused = F.normalize(numerator / (1.0 + total_weight).unsqueeze(-1), dim=-1)
    fallback_mask = total_weight <= 1e-8
    fused = torch.where(fallback_mask.unsqueeze(-1), reference_features, fused)
    return fused, {
        "source_weights": capped_weights,
        "agreement_maps": agreement_maps,
        "total_source_weight": total_weight,
        "fallback_mask": fallback_mask,
    }
