from __future__ import annotations

import numpy as np

from scripts.evaluate_selective_fusion_holdout import paired_bootstrap


def counts(tp: list[int], gt: list[int]) -> dict[str, np.ndarray]:
    result = {"tp": np.asarray(tp, dtype=float), "gt": np.asarray(gt, dtype=float)}
    for visibility in (20, 40, 70, 100):
        result[f"tp_{visibility}"] = np.asarray(tp, dtype=float)
        result[f"gt_{visibility}"] = np.asarray(gt, dtype=float)
    return result


def test_paired_bootstrap_identical_methods_has_zero_interval() -> None:
    reference = counts([1, 0, 2], [1, 1, 2])
    result = paired_bootstrap(reference, reference, replicates=200, seed=7)
    assert result["overall"]["mean_difference"] == 0.0
    assert result["overall"]["ci95_percentile"] == [0.0, 0.0]


def test_paired_bootstrap_detects_consistent_recall_gain() -> None:
    reference = counts([0, 0, 0, 0], [1, 1, 1, 1])
    method = counts([1, 1, 1, 1], [1, 1, 1, 1])
    result = paired_bootstrap(reference, method, replicates=200, seed=9)
    assert result["overall"]["mean_difference"] == 1.0
    assert result["overall"]["ci95_percentile"] == [1.0, 1.0]
    assert result["overall"]["probability_difference_above_zero"] == 1.0
