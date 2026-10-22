"""Interpretable prediction-level fusion for ForestPersons person detectors.

The functions in this module operate on already-decoded, post-model-NMS boxes.
They never inspect ground truth at inference time. Ground truth is used only by
the experiment runner to fit score calibration and select an operating point on
a dedicated calibration split.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


Prediction = dict[str, Any]


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def area_consistency(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def predictions_by_image(predictions: Iterable[Prediction]) -> dict[int | str, list[Prediction]]:
    grouped: dict[int | str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["image_id"]].append(prediction)
    return dict(grouped)


def class_aware_nms(predictions: Iterable[Prediction], iou_threshold: float = 0.5) -> list[Prediction]:
    """Apply deterministic one-class NMS using each prediction's ``fused_score``."""

    kept: list[Prediction] = []
    for image_predictions in predictions_by_image(predictions).values():
        ordered = sorted(
            image_predictions,
            key=lambda item: (float(item["fused_score"]), float(item.get("score", 0.0))),
            reverse=True,
        )
        image_kept: list[Prediction] = []
        for candidate in ordered:
            if all(box_iou(candidate["bbox"], prior["bbox"]) < iou_threshold for prior in image_kept):
                image_kept.append(candidate)
        kept.extend(image_kept)
    return kept


@dataclass(frozen=True)
class MonotonicPrecisionCalibrator:
    """Small empirical score calibrator with quantile bins and isotonic pooling."""

    upper_edges: tuple[float, ...]
    probabilities: tuple[float, ...]
    counts: tuple[int, ...]

    @classmethod
    def fit(
        cls,
        scores: Iterable[float],
        labels: Iterable[int | bool],
        n_bins: int = 10,
    ) -> "MonotonicPrecisionCalibrator":
        score_array = np.asarray(list(scores), dtype=np.float64)
        label_array = np.asarray(list(labels), dtype=np.float64)
        if score_array.size == 0 or score_array.size != label_array.size:
            raise ValueError("Calibration scores and labels must be non-empty and aligned")
        if n_bins < 1:
            raise ValueError("n_bins must be positive")

        order = np.argsort(score_array, kind="stable")
        score_array = score_array[order]
        label_array = label_array[order]
        index_groups = [group for group in np.array_split(np.arange(score_array.size), min(n_bins, score_array.size)) if group.size]
        edges = [float(score_array[group[-1]]) for group in index_groups]
        counts = [int(group.size) for group in index_groups]
        # Jeffreys smoothing avoids exact zero/one estimates in small bins.
        values = [float((label_array[group].sum() + 0.5) / (group.size + 1.0)) for group in index_groups]

        blocks = [
            {"start": index, "end": index, "weight": float(count), "value": value}
            for index, (count, value) in enumerate(zip(counts, values, strict=True))
        ]
        index = 0
        while index < len(blocks) - 1:
            if blocks[index]["value"] <= blocks[index + 1]["value"]:
                index += 1
                continue
            left, right = blocks[index], blocks[index + 1]
            weight = left["weight"] + right["weight"]
            pooled = (left["value"] * left["weight"] + right["value"] * right["weight"]) / weight
            blocks[index : index + 2] = [{
                "start": left["start"],
                "end": right["end"],
                "weight": weight,
                "value": pooled,
            }]
            index = max(0, index - 1)

        isotonic = [0.0] * len(values)
        for block in blocks:
            for bin_index in range(int(block["start"]), int(block["end"]) + 1):
                isotonic[bin_index] = float(block["value"])
        return cls(tuple(edges), tuple(isotonic), tuple(counts))

    def predict(self, score: float) -> float:
        index = int(np.searchsorted(np.asarray(self.upper_edges), float(score), side="left"))
        index = min(index, len(self.probabilities) - 1)
        return float(self.probabilities[index])

    def as_dict(self) -> dict[str, Any]:
        return {
            "upper_edges": list(self.upper_edges),
            "probabilities": list(self.probabilities),
            "counts": list(self.counts),
        }


def best_cross_source_support(
    candidate: Prediction,
    other_predictions: Iterable[Prediction],
    calibrator: MonotonicPrecisionCalibrator,
) -> dict[str, float]:
    best = {"iou": 0.0, "score": 0.0, "calibrated_score": 0.0, "area_consistency": 0.0}
    for other in other_predictions:
        iou = box_iou(candidate["bbox"], other["bbox"])
        if iou <= best["iou"]:
            continue
        best = {
            "iou": iou,
            "score": float(other["score"]),
            "calibrated_score": calibrator.predict(float(other["score"])),
            "area_consistency": area_consistency(candidate["bbox"], other["bbox"]),
        }
    return best


def _copy_with_fused_score(prediction: Prediction, fused_score: float, method: str, **extra: Any) -> Prediction:
    return {
        **prediction,
        "fused_score": float(max(0.0, min(1.0, fused_score))),
        "fusion_method": method,
        **extra,
    }


def reliable_yolo_plus_dino(
    yolo: Iterable[Prediction],
    dino: Iterable[Prediction],
    yolo_calibrator: MonotonicPrecisionCalibrator,
    dino_calibrator: MonotonicPrecisionCalibrator,
    *,
    yolo_keep_threshold: float,
    dino_add_threshold: float,
    duplicate_iou: float,
    nms_iou: float = 0.5,
) -> list[Prediction]:
    """Keep reliable YOLO boxes and add only spatially novel DINO boxes."""

    yolo_by = predictions_by_image(yolo)
    dino_by = predictions_by_image(dino)
    selected: list[Prediction] = []
    for image_id in set(yolo_by) | set(dino_by):
        reliable_yolo = [p for p in yolo_by.get(image_id, []) if float(p["score"]) >= yolo_keep_threshold]
        selected.extend(
            _copy_with_fused_score(p, yolo_calibrator.predict(float(p["score"])), "reliable_yolo")
            for p in reliable_yolo
        )
        for prediction in dino_by.get(image_id, []):
            if float(prediction["score"]) < dino_add_threshold:
                continue
            overlap = max((box_iou(prediction["bbox"], p["bbox"]) for p in reliable_yolo), default=0.0)
            if overlap >= duplicate_iou:
                continue
            selected.append(
                _copy_with_fused_score(
                    prediction,
                    dino_calibrator.predict(float(prediction["score"])),
                    "novel_dino",
                    max_reliable_yolo_iou=overlap,
                )
            )
    return class_aware_nms(selected, nms_iou)


def supported_yolo_rescue(
    yolo: Iterable[Prediction],
    dino: Iterable[Prediction],
    yolo_calibrator: MonotonicPrecisionCalibrator,
    dino_calibrator: MonotonicPrecisionCalibrator,
    *,
    yolo_keep_threshold: float,
    yolo_rescue_floor: float,
    dino_support_threshold: float,
    support_iou: float,
    minimum_area_consistency: float = 0.25,
    nms_iou: float = 0.5,
) -> list[Prediction]:
    """Keep reliable YOLO boxes and rescue low-score YOLO boxes supported by DINO."""

    yolo_by = predictions_by_image(yolo)
    dino_by = predictions_by_image(dino)
    selected: list[Prediction] = []
    for image_id in set(yolo_by) | set(dino_by):
        supporting_dino = [p for p in dino_by.get(image_id, []) if float(p["score"]) >= dino_support_threshold]
        for prediction in yolo_by.get(image_id, []):
            score = float(prediction["score"])
            base = yolo_calibrator.predict(score)
            if score >= yolo_keep_threshold:
                selected.append(_copy_with_fused_score(prediction, base, "reliable_yolo"))
                continue
            if score < yolo_rescue_floor:
                continue
            support = best_cross_source_support(prediction, supporting_dino, dino_calibrator)
            if support["iou"] < support_iou or support["area_consistency"] < minimum_area_consistency:
                continue
            agreement = support["iou"] * support["area_consistency"] * support["calibrated_score"]
            fused_score = 1.0 - (1.0 - base) * (1.0 - agreement)
            selected.append(
                _copy_with_fused_score(prediction, fused_score, "supported_yolo_rescue", **support)
            )
    return class_aware_nms(selected, nms_iou)


def calibrated_support_fusion(
    yolo: Iterable[Prediction],
    dino: Iterable[Prediction],
    yolo_calibrator: MonotonicPrecisionCalibrator,
    dino_calibrator: MonotonicPrecisionCalibrator,
    *,
    nms_iou: float = 0.5,
) -> list[Prediction]:
    """Rank both sources with calibrated base scores plus cross-model agreement."""

    yolo_by = predictions_by_image(yolo)
    dino_by = predictions_by_image(dino)
    scored: list[Prediction] = []
    for image_id in set(yolo_by) | set(dino_by):
        sources = (("yolo", yolo_by.get(image_id, []), dino_by.get(image_id, []), yolo_calibrator, dino_calibrator),
                   ("dino", dino_by.get(image_id, []), yolo_by.get(image_id, []), dino_calibrator, yolo_calibrator))
        for source, own, other, own_calibrator, other_calibrator in sources:
            for prediction in own:
                base = own_calibrator.predict(float(prediction["score"]))
                support = best_cross_source_support(prediction, other, other_calibrator)
                agreement = support["iou"] * support["area_consistency"] * support["calibrated_score"]
                fused_score = 1.0 - (1.0 - base) * (1.0 - agreement)
                scored.append(
                    _copy_with_fused_score(
                        prediction,
                        fused_score,
                        "calibrated_support",
                        candidate_source=source,
                        base_calibrated_score=base,
                        **support,
                    )
                )
    return class_aware_nms(scored, nms_iou)
