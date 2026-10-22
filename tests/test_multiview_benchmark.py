from __future__ import annotations

import pytest

from forestpersons.multiview import registration_rejection_reasons, validate_unique_group_images
from scripts.evaluate_dinov3_multiview_benchmark import sequence_cluster_bootstrap


def valid_registration() -> dict[str, float | int]:
    return {
        "ratio_test_matches": 100,
        "ransac_inliers": 70,
        "inlier_ratio": 0.7,
        "median_reprojection_error_registration_pixels": 1.0,
        "reference_overlap_fraction": 0.9,
        "source_inlier_spread_fraction": 0.2,
        "reference_inlier_spread_fraction": 0.2,
        "projected_area_ratio": 1.0,
    }


def test_registration_quality_is_image_only_and_explicit() -> None:
    assert registration_rejection_reasons(valid_registration()) == []
    metrics = valid_registration()
    metrics["reference_overlap_fraction"] = 0.4
    metrics["projected_area_ratio"] = 2.0
    assert registration_rejection_reasons(metrics) == [
        "low_reference_overlap",
        "projected_area_too_large",
    ]


def test_manifest_rejects_reused_images() -> None:
    groups = [
        {"group_id": "a", "reference": {"image_id": 1}, "sources": [{"image_id": 2}, {"image_id": 3}]},
        {"group_id": "b", "reference": {"image_id": 4}, "sources": [{"image_id": 3}, {"image_id": 5}]},
    ]
    with pytest.raises(ValueError, match="reused across groups"):
        validate_unique_group_images(groups)


def test_sequence_bootstrap_identical_detection_has_zero_difference() -> None:
    rows = [
        {"sequence": "a", "visibility": 20, "single_detected_fixed": True, "fused_detected_fixed": True},
        {"sequence": "b", "visibility": 40, "single_detected_fixed": False, "fused_detected_fixed": False},
    ]
    result = sequence_cluster_bootstrap(rows, replicates=200, seed=3)
    assert result["overall"]["observed_difference"] == 0.0
    assert result["overall"]["ci95_percentile"] == [0.0, 0.0]
