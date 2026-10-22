"""Metrics for aligned AOS RGB/thermal DINOv3 patch features."""

from __future__ import annotations

import torch


def yolo_person_patch_mask(
    label_rows: list[list[float]], grid_height: int, grid_width: int
) -> torch.Tensor:
    """Mark patch centers inside any normalized YOLO box."""

    y = (torch.arange(grid_height, dtype=torch.float32) + 0.5) / grid_height
    x = (torch.arange(grid_width, dtype=torch.float32) + 0.5) / grid_width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mask = torch.zeros(grid_height, grid_width, dtype=torch.bool)
    for row in label_rows:
        if len(row) != 5 or int(row[0]) != 0:
            raise ValueError(f"Expected one-class normalized YOLO row, received {row}")
        _, xc, yc, width, height = row
        if not all(0.0 <= value <= 1.0 for value in (xc, yc, width, height)):
            raise ValueError(f"YOLO values must be normalized: {row}")
        mask |= (
            (xx >= xc - width / 2)
            & (xx <= xc + width / 2)
            & (yy >= yc - height / 2)
            & (yy <= yc + height / 2)
        )
    return mask


def local_cosine_correspondence(
    rgb_features: torch.Tensor,
    thermal_features: torch.Tensor,
    radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return same-location score, local best score, and local displacement."""

    if rgb_features.shape != thermal_features.shape or rgb_features.ndim != 3:
        raise ValueError("feature maps must share shape [height, width, channels]")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    height, width, _ = rgb_features.shape
    same = (rgb_features * thermal_features).sum(dim=-1)
    candidates = []
    offsets = []
    negative = torch.tensor(float("-inf"), device=rgb_features.device)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = torch.zeros_like(thermal_features)
            valid = torch.zeros(height, width, dtype=torch.bool, device=rgb_features.device)
            target_y = slice(max(0, -dy), min(height, height - dy))
            target_x = slice(max(0, -dx), min(width, width - dx))
            source_y = slice(max(0, dy), min(height, height + dy))
            source_x = slice(max(0, dx), min(width, width + dx))
            shifted[target_y, target_x] = thermal_features[source_y, source_x]
            valid[target_y, target_x] = True
            score = (rgb_features * shifted).sum(dim=-1)
            candidates.append(torch.where(valid, score, negative))
            offsets.append((dy, dx))
    stack = torch.stack(candidates)
    local_best, indices = stack.max(dim=0)
    offset_tensor = torch.tensor(offsets, device=indices.device, dtype=torch.long)
    displacement = offset_tensor[indices]
    return same, local_best, displacement


def global_patch_retrieval(
    rgb_features: torch.Tensor,
    thermal_features: torch.Tensor,
    *,
    batch_size: int = 256,
) -> dict[str, float]:
    """Measure whether each RGB patch retrieves its thermal spatial partner."""

    if rgb_features.shape != thermal_features.shape or rgb_features.ndim != 3:
        raise ValueError("feature maps must share shape [height, width, channels]")
    height, width, channels = rgb_features.shape
    rgb = rgb_features.reshape(-1, channels)
    thermal = thermal_features.reshape(-1, channels)
    query_indices = torch.arange(len(rgb), device=rgb.device)
    retrieved = []
    for start in range(0, len(rgb), batch_size):
        similarities = rgb[start : start + batch_size] @ thermal.T
        retrieved.append(similarities.argmax(dim=1))
    retrieved_indices = torch.cat(retrieved)
    query_y, query_x = torch.div(query_indices, width, rounding_mode="floor"), query_indices % width
    retrieved_y = torch.div(retrieved_indices, width, rounding_mode="floor")
    retrieved_x = retrieved_indices % width
    chebyshev = torch.maximum((query_y - retrieved_y).abs(), (query_x - retrieved_x).abs())
    return {
        "exact_patch_fraction": float((chebyshev == 0).float().mean().cpu()),
        "within_one_patch_fraction": float((chebyshev <= 1).float().mean().cpu()),
        "median_chebyshev_patch_error": float(chebyshev.float().median().cpu()),
    }
