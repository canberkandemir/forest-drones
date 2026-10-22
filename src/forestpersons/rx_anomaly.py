"""Classical RX anomaly scoring for ForestPersons RGB patches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .dinov3_detector import PATCH_SIZE, PreprocessInfo, resize_to_patch_grid


@dataclass(frozen=True)
class RXModel:
    """Gaussian background model used by the RX detector."""

    mean: np.ndarray
    covariance: np.ndarray
    inverse_covariance: np.ndarray
    regularization: float
    feature_names: tuple[str, ...]


def rgb_patch_features(image: Image.Image, short_side: int = 672) -> tuple[np.ndarray, PreprocessInfo]:
    """Return per-patch RGB mean/std features on the DINO-compatible grid.

    Feature vector per patch:
    ``[mean_r, mean_g, mean_b, std_r, std_g, std_b]`` in RGB space scaled to
    ``[0, 1]``. The representation is intentionally simple so RX remains a
    classical low-level anomaly baseline rather than another learned feature
    detector.
    """

    resized, info = resize_to_patch_grid(image.convert("RGB"), short_side=short_side)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    usable_h = info.grid_height * PATCH_SIZE
    usable_w = info.grid_width * PATCH_SIZE
    array = array[:usable_h, :usable_w]
    patches = array.reshape(
        info.grid_height,
        PATCH_SIZE,
        info.grid_width,
        PATCH_SIZE,
        3,
    ).transpose(0, 2, 1, 3, 4)
    means = patches.mean(axis=(2, 3))
    stds = patches.std(axis=(2, 3))
    features = np.concatenate([means, stds], axis=-1).astype(np.float32)
    return features, info


def fit_rx_model(
    features: np.ndarray,
    regularization: float = 1e-3,
) -> RXModel:
    """Fit RX background statistics from an ``N x D`` feature matrix."""

    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got {features.shape}")
    if features.shape[0] < 2:
        raise ValueError("At least two background features are required")
    mean = features.mean(axis=0, dtype=np.float64)
    centered = features.astype(np.float64) - mean
    covariance = (centered.T @ centered) / max(features.shape[0] - 1, 1)
    trace_scale = float(np.trace(covariance) / covariance.shape[0])
    ridge = regularization * trace_scale if trace_scale > 0 else regularization
    covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * ridge
    inverse_covariance = np.linalg.pinv(covariance)
    return RXModel(
        mean=mean.astype(np.float32),
        covariance=covariance.astype(np.float32),
        inverse_covariance=inverse_covariance.astype(np.float32),
        regularization=float(regularization),
        feature_names=("mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"),
    )


def rx_scores(features: np.ndarray, model: RXModel) -> np.ndarray:
    """Compute squared Mahalanobis RX scores for patch features."""

    if features.shape[-1] != model.mean.shape[0]:
        raise ValueError(
            f"Feature dimension {features.shape[-1]} does not match model dimension {model.mean.shape[0]}"
        )
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    centered = flat - model.mean.astype(np.float64)
    scores = np.einsum("nd,dd,nd->n", centered, model.inverse_covariance, centered)
    return scores.reshape(features.shape[:-1]).astype(np.float32)


def serialize_rx_model(model: RXModel) -> dict[str, Any]:
    return {
        "mean": model.mean.tolist(),
        "covariance": model.covariance.tolist(),
        "inverse_covariance": model.inverse_covariance.tolist(),
        "regularization": model.regularization,
        "feature_names": list(model.feature_names),
    }


def deserialize_rx_model(data: dict[str, Any]) -> RXModel:
    return RXModel(
        mean=np.asarray(data["mean"], dtype=np.float32),
        covariance=np.asarray(data["covariance"], dtype=np.float32),
        inverse_covariance=np.asarray(data["inverse_covariance"], dtype=np.float32),
        regularization=float(data["regularization"]),
        feature_names=tuple(data["feature_names"]),
    )
