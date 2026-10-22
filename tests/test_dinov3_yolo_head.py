import torch

from forestpersons.dinov3_detector import PATCH_SIZE, PreprocessInfo
from forestpersons.dinov3_yolo import (
    DINOv3YOLOHead,
    build_task_aligned_yolo_targets,
    decode_yolo_predictions,
    dfl_expectation,
    distribution_focal_loss,
    iou_regression_loss,
    yolo_detection_loss,
    yolo_scores,
)


def test_yolo_head_output_shapes() -> None:
    head = DINOv3YOLOHead(input_dim=384, hidden_dim=16, reg_max=8)
    features = torch.randn(5, 7, 384)

    outputs = head(features)

    assert outputs["class_logits"].shape == (5, 7)
    assert outputs["distance_logits"].shape == (5, 7, 4, 9)


def test_yolo_quality_head_output_shapes() -> None:
    head = DINOv3YOLOHead(input_dim=384, hidden_dim=16, reg_max=8, include_quality=True)
    features = torch.randn(5, 7, 384)

    outputs = head(features)

    assert outputs["class_logits"].shape == (5, 7)
    assert outputs["distance_logits"].shape == (5, 7, 4, 9)
    assert outputs["quality_logits"].shape == (5, 7)


def test_dfl_expectation_decodes_argmax_bins() -> None:
    logits = torch.full((2, 4, 6), -20.0)
    logits[:, :, 3] = 20.0

    decoded = dfl_expectation(logits)

    assert torch.allclose(decoded, torch.full((2, 4), 3.0), atol=1e-4)


def test_distribution_focal_loss_has_finite_gradients() -> None:
    logits = torch.randn(3, 4, 9, requires_grad=True)
    targets = torch.tensor(
        [
            [1.5, 2.0, 3.25, 4.0],
            [0.0, 7.5, 5.0, 2.25],
            [8.5, 1.0, 1.0, 1.0],
        ]
    )

    loss = distribution_focal_loss(logits, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_iou_regression_loss_identical_partial_and_disjoint() -> None:
    identical = iou_regression_loss(
        torch.tensor([[2.0, 2.0, 2.0, 2.0]]),
        torch.tensor([[2.0, 2.0, 2.0, 2.0]]),
    )
    partial = iou_regression_loss(
        torch.tensor([[2.0, 2.0, 2.0, 2.0]]),
        torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
    )
    disjoint = iou_regression_loss(
        torch.tensor([[0.1, 0.1, 0.1, 0.1]]),
        torch.tensor([[3.0, 3.0, 3.0, 3.0]]),
    )

    assert identical < 1e-6
    assert partial > identical
    assert disjoint > partial


def test_task_aligned_assignment_selects_positive_cells_inside_box() -> None:
    info = PreprocessInfo(
        original_width=128,
        original_height=128,
        resized_width=128,
        resized_height=128,
        scale=1.0,
        grid_width=8,
        grid_height=8,
    )
    class_logits = torch.zeros((8, 8))
    predicted_distances = torch.ones((8, 8, 4)) * 2.0
    ground_truth = [{"box": [32.0, 32.0, 80.0, 80.0], "visibility_level": 20}]

    targets, stats = build_task_aligned_yolo_targets(
        ground_truth,
        info,
        class_logits=class_logits,
        predicted_distances=predicted_distances,
        topk=5,
        reg_max=8,
    )

    assert stats.annotations_with_positive == 1
    assert int(targets["positive_mask"].sum()) > 0
    positive_y, positive_x = torch.nonzero(targets["positive_mask"], as_tuple=True)
    centers_x = (positive_x.float() + 0.5) * PATCH_SIZE
    centers_y = (positive_y.float() + 0.5) * PATCH_SIZE
    assert torch.all((centers_x > 32.0) & (centers_x < 80.0))
    assert torch.all((centers_y > 32.0) & (centers_y < 80.0))


def test_tiny_box_uses_center_fallback() -> None:
    info = PreprocessInfo(
        original_width=128,
        original_height=128,
        resized_width=128,
        resized_height=128,
        scale=1.0,
        grid_width=8,
        grid_height=8,
    )
    targets, stats = build_task_aligned_yolo_targets(
        [{"box": [31.0, 31.0, 33.0, 33.0], "visibility_level": 20}],
        info,
        class_logits=torch.zeros((8, 8)),
        predicted_distances=torch.ones((8, 8, 4)),
        topk=3,
        reg_max=8,
    )

    assert stats.center_fallbacks == 1
    assert int(targets["positive_mask"].sum()) == 1


def test_yolo_loss_lambda_zero_positive_and_no_positive_cases() -> None:
    outputs = {
        "class_logits": torch.zeros((3, 3), requires_grad=True),
        "distance_logits": torch.randn(3, 3, 4, 9, requires_grad=True),
    }
    empty_targets = {
        "target_scores": torch.zeros((3, 3)),
        "target_distances": torch.zeros((3, 3, 4)),
        "positive_mask": torch.zeros((3, 3), dtype=torch.bool),
    }
    empty_loss, empty_parts = yolo_detection_loss(outputs, empty_targets)
    empty_loss.backward()

    assert torch.isfinite(empty_loss)
    assert empty_parts["iou_loss"] == 0.0
    assert empty_parts["dfl_loss"] == 0.0

    logits = torch.randn(1, 1, 4, 9, requires_grad=True)
    positive_outputs = {"class_logits": torch.zeros((1, 1), requires_grad=True), "distance_logits": logits}
    positive_targets = {
        "target_scores": torch.ones((1, 1)),
        "target_distances": torch.ones((1, 1, 4)) * 2.0,
        "positive_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    positive_loss, _parts = yolo_detection_loss(positive_outputs, positive_targets)
    positive_loss.backward()

    assert torch.isfinite(positive_loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_quality_loss_has_gradients_and_detached_iou_targets() -> None:
    distance_logits = torch.full((1, 1, 4, 9), -8.0, requires_grad=True)
    quality_logits = torch.zeros((1, 1), requires_grad=True)
    with torch.no_grad():
        distance_logits[..., 2] = 8.0
    outputs = {
        "class_logits": torch.zeros((1, 1), requires_grad=True),
        "distance_logits": distance_logits,
        "quality_logits": quality_logits,
    }
    targets = {
        "target_scores": torch.ones((1, 1)),
        "target_distances": torch.ones((1, 1, 4)) * 2.0,
        "positive_mask": torch.ones((1, 1), dtype=torch.bool),
    }

    loss, parts = yolo_detection_loss(outputs, targets, quality_weight=1.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["quality_loss"] > 0.0
    assert quality_logits.grad is not None
    assert torch.isfinite(quality_logits.grad).all()
    assert distance_logits.grad is not None
    assert torch.isfinite(distance_logits.grad).all()


def test_quality_loss_uses_balanced_positive_negative_normalization() -> None:
    quality_logits = torch.zeros((2, 2), requires_grad=True)
    outputs = {
        "class_logits": torch.zeros((2, 2), requires_grad=True),
        "distance_logits": torch.randn(2, 2, 4, 9, requires_grad=True),
        "quality_logits": quality_logits,
    }
    targets = {
        "target_scores": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "target_distances": torch.ones((2, 2, 4)) * 2.0,
        "positive_mask": torch.tensor([[True, False], [False, False]]),
    }

    _loss, parts = yolo_detection_loss(outputs, targets, quality_weight=1.0, quality_negative_weight=0.5)

    assert parts["quality_positive_count"] == 1.0
    assert parts["quality_negative_count"] == 3.0
    expected = parts["quality_positive_loss"] + 0.5 * parts["quality_negative_loss"]
    assert abs(parts["quality_loss"] - expected) < 1e-6


def test_quality_aware_scores_change_candidate_ordering() -> None:
    class_logits = torch.logit(torch.tensor([0.9, 0.8]), eps=1e-6)
    quality_logits = torch.logit(torch.tensor([0.1, 0.9]), eps=1e-6)

    class_scores = yolo_scores(class_logits, score_mode="class")
    quality_scores = yolo_scores(class_logits, quality_logits=quality_logits, score_mode="class_quality")

    assert int(class_scores.argmax()) == 0
    assert int(quality_scores.argmax()) == 1


def test_class_only_state_dict_stays_backward_compatible() -> None:
    old_head = DINOv3YOLOHead(input_dim=384, hidden_dim=16, reg_max=8, include_quality=False)
    new_class_only_head = DINOv3YOLOHead(input_dim=384, hidden_dim=16, reg_max=8, include_quality=False)

    new_class_only_head.load_state_dict(old_head.state_dict(), strict=True)

    assert new_class_only_head.include_quality is False


def test_decode_yolo_predictions_uses_separate_xy_scales() -> None:
    info = PreprocessInfo(
        original_width=200,
        original_height=100,
        resized_width=800,
        resized_height=800,
        scale=4.0,
        grid_width=50,
        grid_height=50,
    )
    class_logits = torch.full((50, 50), -20.0)
    distance_logits = torch.full((50, 50, 4, 9), -20.0)
    class_logits[10, 20] = 20.0
    distance_logits[10, 20, :, 2] = 20.0

    predictions = decode_yolo_predictions(class_logits, distance_logits, info, conf_threshold=0.5)

    assert len(predictions) == 1
    x1, y1, x2, y2 = predictions[0]["box"]
    assert 73.0 < x1 < 75.0
    assert 16.0 < y1 < 18.0
    assert 89.0 < x2 < 91.0
    assert 24.0 < y2 < 26.0
    assert predictions[0]["confidence"] > 0.99


def test_quality_aware_decode_changes_nms_survivor() -> None:
    info = PreprocessInfo(
        original_width=96,
        original_height=64,
        resized_width=96,
        resized_height=64,
        scale=1.0,
        grid_width=6,
        grid_height=4,
    )
    class_logits = torch.full((4, 6), -20.0)
    quality_logits = torch.full((4, 6), -20.0)
    distance_logits = torch.full((4, 6, 4, 9), -20.0)
    distance_logits[1, 2, :, 2] = 20.0
    distance_logits[1, 3, :, 2] = 20.0
    class_logits[1, 2] = torch.logit(torch.tensor(0.90))
    class_logits[1, 3] = torch.logit(torch.tensor(0.80))
    quality_logits[1, 2] = torch.logit(torch.tensor(0.10))
    quality_logits[1, 3] = torch.logit(torch.tensor(0.95))

    class_predictions = decode_yolo_predictions(
        class_logits,
        distance_logits,
        info,
        conf_threshold=0.01,
        nms_threshold=0.50,
        quality_logits=quality_logits,
        score_mode="class",
    )
    quality_predictions = decode_yolo_predictions(
        class_logits,
        distance_logits,
        info,
        conf_threshold=0.01,
        nms_threshold=0.50,
        quality_logits=quality_logits,
        score_mode="class_quality",
    )

    assert len(class_predictions) == 1
    assert len(quality_predictions) == 1
    assert class_predictions[0]["box"][0] < quality_predictions[0]["box"][0]
