import numpy as np
import torch

from forestpersons.dinov3_detector import PreprocessInfo
from forestpersons.patch_knn import (
    background_patch_mask,
    binary_classification_metrics,
    cosine_knn_anomaly_scores,
    labels_from_ground_truth,
)


def make_info() -> PreprocessInfo:
    return PreprocessInfo(
        original_width=64,
        original_height=64,
        resized_width=64,
        resized_height=64,
        scale=1.0,
        grid_width=4,
        grid_height=4,
    )


def test_patch_labels_mark_overlapping_person_patches() -> None:
    labels = labels_from_ground_truth(
        [{"box": [16.0, 16.0, 32.0, 32.0], "visibility_level": 40}],
        make_info(),
    )

    assert labels.person_mask.sum() == 1
    assert labels.person_mask[1, 1]
    assert labels.visibility[1, 1] == 40


def test_background_mask_uses_safety_margin() -> None:
    no_margin = background_patch_mask(
        [{"box": [16.0, 16.0, 32.0, 32.0], "visibility_level": 40}],
        make_info(),
        safety_margin=0.0,
    )
    with_margin = background_patch_mask(
        [{"box": [16.0, 16.0, 32.0, 32.0], "visibility_level": 40}],
        make_info(),
        safety_margin=16.0,
    )

    assert int((~no_margin).sum()) == 1
    assert int((~with_margin).sum()) == 9


def test_cosine_knn_anomaly_scores_are_low_for_reference_like_patch() -> None:
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    reference = torch.tensor([[1.0, 0.0]])

    scores = cosine_knn_anomaly_scores(features, reference, batch_size=1)

    assert torch.isclose(scores[0, 0], torch.tensor(0.0))
    assert torch.isclose(scores[0, 1], torch.tensor(1.0))


def test_binary_metrics_perfect_ranking() -> None:
    metrics = binary_classification_metrics(
        labels=np.array([0, 1, 0, 1], dtype=bool),
        scores=np.array([0.1, 0.9, 0.2, 0.8]),
    )

    assert metrics["positives"] == 2
    assert metrics["negatives"] == 2
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_binary_metrics_return_none_without_both_classes() -> None:
    metrics = binary_classification_metrics(
        labels=np.array([0, 0, 0], dtype=bool),
        scores=np.array([0.1, 0.2, 0.3]),
    )

    assert metrics["positives"] == 0
    assert metrics["auroc"] is None
    assert metrics["average_precision"] is None
