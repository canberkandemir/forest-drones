from forestpersons.detection import box_iou, greedy_match
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_custom_metrics,
    evaluate_pycocotools,
    predictions_to_coco_results,
)


def test_box_iou_exact_partial_and_empty_overlap() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert round(box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 4) == 0.1429
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_greedy_match_counts_duplicate_detection_as_false_positive() -> None:
    ground_truth = [{"box": [0, 0, 10, 10], "visibility_level": 20}]
    predictions = [
        {"box": [0, 0, 10, 10], "confidence": 0.9},
        {"box": [0, 0, 10, 10], "confidence": 0.8},
    ]

    matches, matched_gt, matched_pred = greedy_match(ground_truth, predictions, 0.5)

    assert len(matches) == 1
    assert matched_gt == {0}
    assert len(matched_pred) == 1


def test_custom_metrics_handle_empty_predictions() -> None:
    ground_truth = {
        1: [
            EvaluationGroundTruth(
                image_id=1,
                annotation_id=1,
                box=[0, 0, 10, 10],
                visibility_level=40,
            )
        ]
    }

    metrics = evaluate_custom_metrics(ground_truth, {}, [1], confidence_threshold=0.5)

    assert metrics["overall"]["ground_truth"] == 1
    assert metrics["overall"]["true_positives"] == 0
    assert metrics["overall"]["false_negatives"] == 1
    assert metrics["overall"]["recall"] == 0.0
    assert metrics["by_visibility"]["40"]["false_negatives"] == 1


def test_custom_metrics_support_multiple_people_per_image() -> None:
    ground_truth = {
        7: [
            EvaluationGroundTruth(7, 1, [0, 0, 10, 10], 20),
            EvaluationGroundTruth(7, 2, [20, 20, 40, 40], 100),
        ]
    }
    predictions = {
        7: [
            EvaluationPrediction(7, [0, 0, 10, 10], 0.9),
            EvaluationPrediction(7, [20, 20, 40, 40], 0.8),
        ]
    }

    metrics = evaluate_custom_metrics(ground_truth, predictions, [7], confidence_threshold=0.5)

    assert metrics["overall"]["true_positives"] == 2
    assert metrics["overall"]["recall"] == 1.0
    assert metrics["by_visibility"]["20"]["recall"] == 1.0
    assert metrics["by_visibility"]["100"]["recall"] == 1.0


def test_coco_result_format_uses_xywh_and_person_category() -> None:
    results = predictions_to_coco_results(
        [EvaluationPrediction(image_id=3, box=[5, 7, 15, 27], confidence=0.6)]
    )

    assert results == [
        {"image_id": 3, "category_id": 1, "bbox": [5.0, 7.0, 10.0, 20.0], "score": 0.6}
    ]


def test_coco_style_ap_penalizes_duplicate_predictions() -> None:
    ground_truth = [EvaluationGroundTruth(1, 1, [0, 0, 10, 10], 70)]
    predictions = [
        EvaluationPrediction(1, [0, 0, 10, 10], 0.9),
        EvaluationPrediction(1, [0, 0, 10, 10], 0.8),
    ]

    metrics = evaluate_coco_style(ground_truth, predictions, [1], max_dets=(1, 100))

    assert metrics["AP50"] == 1.0
    assert metrics["AR_by_max_dets"]["1"] == 1.0
    assert metrics["AR_by_max_dets"]["100"] == 1.0


def test_confidence_sweep_reports_matched_fp_budget() -> None:
    ground_truth = {1: [EvaluationGroundTruth(1, 1, [0, 0, 10, 10], 20)]}
    predictions = {
        1: [
            EvaluationPrediction(1, [0, 0, 10, 10], 0.9),
            EvaluationPrediction(1, [20, 20, 30, 30], 0.1),
        ]
    }

    sweep = evaluate_confidence_sweep(
        ground_truth,
        predictions,
        [1],
        confidence_thresholds=[0.05, 0.5],
        fp_budgets=[0.0],
    )

    assert sweep["maximum_recall"]["overall"]["recall"] == 1.0
    assert sweep["matched_false_positive_budgets"]["0.0"]["confidence_threshold"] == 0.5


def test_local_coco_style_matches_pycocotools_simple_case() -> None:
    ground_truth = [
        EvaluationGroundTruth(1, 1, [0, 0, 10, 10], 20),
        EvaluationGroundTruth(2, 2, [20, 20, 40, 40], 100),
    ]
    predictions = [
        EvaluationPrediction(1, [0, 0, 10, 10], 0.95),
        EvaluationPrediction(1, [40, 40, 50, 50], 0.90),
        EvaluationPrediction(2, [20, 20, 40, 40], 0.80),
    ]

    local = evaluate_coco_style(ground_truth, predictions, [1, 2])
    official = evaluate_pycocotools(ground_truth, predictions, [1, 2])

    assert official["available"] is True
    assert abs(local["AP"] - official["AP"]) < 1e-9
    assert abs(local["AP50"] - official["AP50"]) < 1e-9
    assert abs(local["AP75"] - official["AP75"]) < 1e-9
    assert abs(local["AR_by_max_dets"]["100"] - official["AR@100"]) < 1e-9
