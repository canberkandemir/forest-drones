from __future__ import annotations

import math

from forestpersons.selective_fusion import (
    MonotonicPrecisionCalibrator,
    calibrated_support_fusion,
    class_aware_nms,
    reliable_yolo_plus_dino,
    supported_yolo_rescue,
)


def prediction(image_id: int, box: list[float], score: float, source: str) -> dict:
    return {"image_id": image_id, "bbox": box, "score": score, "source": source, "category_id": 1}


def calibrator() -> MonotonicPrecisionCalibrator:
    return MonotonicPrecisionCalibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=2)


def test_monotonic_precision_calibrator_is_finite_and_monotonic() -> None:
    fitted = MonotonicPrecisionCalibrator.fit([0.1, 0.2, 0.3, 0.8, 0.9], [0, 1, 0, 1, 1], n_bins=3)
    values = [fitted.predict(score) for score in [0.0, 0.15, 0.4, 1.0]]
    assert all(math.isfinite(value) for value in values)
    assert values == sorted(values)


def test_class_aware_nms_keeps_highest_fused_score() -> None:
    boxes = [
        {**prediction(1, [0, 0, 10, 10], 0.9, "yolo"), "fused_score": 0.2},
        {**prediction(1, [0, 0, 10, 10], 0.4, "dino"), "fused_score": 0.8},
    ]
    kept = class_aware_nms(boxes, 0.5)
    assert len(kept) == 1
    assert kept[0]["source"] == "dino"


def test_reliable_yolo_plus_dino_rejects_spatial_duplicate() -> None:
    fitted = calibrator()
    yolo = [prediction(1, [0, 0, 10, 10], 0.8, "yolo")]
    dino = [
        prediction(1, [0, 0, 10, 10], 0.9, "dino"),
        prediction(1, [20, 20, 30, 30], 0.9, "dino"),
    ]
    fused = reliable_yolo_plus_dino(
        yolo, dino, fitted, fitted, yolo_keep_threshold=0.5, dino_add_threshold=0.5,
        duplicate_iou=0.5,
    )
    assert {item["fusion_method"] for item in fused} == {"reliable_yolo", "novel_dino"}
    assert len(fused) == 2


def test_supported_yolo_rescue_requires_cross_model_agreement() -> None:
    fitted = calibrator()
    yolo = [
        prediction(1, [0, 0, 10, 10], 0.8, "yolo"),
        prediction(1, [20, 20, 30, 30], 0.1, "yolo"),
        prediction(1, [40, 40, 50, 50], 0.1, "yolo"),
    ]
    dino = [prediction(1, [20, 20, 30, 30], 0.8, "dino")]
    fused = supported_yolo_rescue(
        yolo, dino, fitted, fitted, yolo_keep_threshold=0.5, yolo_rescue_floor=0.05,
        dino_support_threshold=0.5, support_iou=0.5,
    )
    assert len(fused) == 2
    assert sum(item["fusion_method"] == "supported_yolo_rescue" for item in fused) == 1


def test_calibrated_support_boosts_agreeing_candidate_without_ground_truth() -> None:
    fitted = calibrator()
    yolo = [prediction(1, [0, 0, 10, 10], 0.2, "yolo")]
    dino = [prediction(1, [0, 0, 10, 10], 0.8, "dino")]
    fused = calibrated_support_fusion(yolo, dino, fitted, fitted, nms_iou=0.5)
    assert len(fused) == 1
    assert fused[0]["fused_score"] >= fused[0]["base_calibrated_score"]
