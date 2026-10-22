"""Calibrate and evaluate selective YOLO11s + DINOv3 FCOS fusion.

All detector predictions are loaded from COCO result JSON files. Fusion gates,
score calibration, and operating thresholds are selected on a dedicated
calibration subset, then applied once to the disjoint val600 pilot subset.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

from forestpersons import ForestPersonsDataset
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_confidence_sweep,
    evaluate_custom_metrics,
    evaluate_pycocotools,
)
from forestpersons.selective_fusion import (
    MonotonicPrecisionCalibrator,
    box_iou,
    calibrated_support_fusion,
    predictions_by_image,
    reliable_yolo_plus_dino,
    supported_yolo_rescue,
)


Prediction = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--calibration-yolo", type=Path, required=True)
    parser.add_argument("--calibration-dino", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-yolo", type=Path, required=True)
    parser.add_argument("--validation-dino", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp-budget", type=float, default=0.425)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["images"] if isinstance(data, dict) else data


def load_split(root: Path, manifest_path: Path) -> tuple[list[Any], list[int | str], dict[int | str, list[EvaluationGroundTruth]]]:
    dataset = ForestPersonsDataset(root, split="val")
    samples_by_id = {sample.image_id: sample for sample in dataset.samples}
    samples = []
    for item in load_manifest(manifest_path):
        sample = samples_by_id[item["image_id"]]
        if sample.file_name != item["file_name"]:
            raise ValueError(f"Image ID/filename mismatch for {item['image_id']}")
        if not sample.image_path.is_file():
            raise FileNotFoundError(sample.image_path)
        samples.append(sample)
    image_ids = [sample.image_id for sample in samples]
    ground_truth = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    return samples, image_ids, ground_truth


def load_predictions(path: Path, image_ids: set[int | str], source: str) -> list[Prediction]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    predictions = []
    for item in raw:
        if item["image_id"] not in image_ids or int(item.get("category_id", 1)) != 1:
            continue
        x, y, width, height = [float(value) for value in item["bbox"]]
        predictions.append({
            "image_id": item["image_id"],
            "category_id": 1,
            "bbox": [x, y, x + width, y + height],
            "score": float(item["score"]),
            "source": source,
        })
    return predictions


def to_eval_predictions(predictions: Iterable[Prediction], score_key: str) -> list[EvaluationPrediction]:
    return [
        EvaluationPrediction(
            image_id=item["image_id"],
            box=[float(value) for value in item["bbox"]],
            confidence=float(item[score_key]),
        )
        for item in predictions
    ]


def eval_by_image(predictions: Iterable[Prediction], image_ids: list[int | str], score_key: str) -> dict[int | str, list[EvaluationPrediction]]:
    result = {image_id: [] for image_id in image_ids}
    for prediction in to_eval_predictions(predictions, score_key):
        result[prediction.image_id].append(prediction)
    return result


def calibration_labels(
    predictions: list[Prediction],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    iou_threshold: float,
) -> list[int]:
    """Assign standard score-ordered one-to-one TP labels for calibration only."""

    labels = [0] * len(predictions)
    grouped: dict[int | str, list[int]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        grouped[prediction["image_id"]].append(index)
    for image_id, indices in grouped.items():
        used: set[int] = set()
        for index in sorted(indices, key=lambda value: float(predictions[value]["score"]), reverse=True):
            candidate = predictions[index]
            best_index = -1
            best_iou = 0.0
            for gt_index, gt in enumerate(ground_truth.get(image_id, [])):
                if gt_index in used:
                    continue
                overlap = box_iou(candidate["bbox"], gt.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index
            if best_index >= 0 and best_iou >= iou_threshold:
                labels[index] = 1
                used.add(best_index)
    return labels


def score_thresholds(step: float = 0.005) -> list[float]:
    return [float(value) for value in np.arange(0.0, 1.0 + step / 2.0, step)]


def sweep(
    predictions: list[Prediction],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    image_ids: list[int | str],
    score_key: str,
    fp_budget: float,
    iou_threshold: float,
) -> dict[str, Any]:
    return evaluate_confidence_sweep(
        ground_truth,
        eval_by_image(predictions, image_ids, score_key),
        image_ids,
        score_thresholds(),
        iou_threshold=iou_threshold,
        fp_budgets=[fp_budget],
    )


def point(
    predictions: list[Prediction],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    image_ids: list[int | str],
    score_key: str,
    threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    return evaluate_custom_metrics(
        ground_truth,
        eval_by_image(predictions, image_ids, score_key),
        image_ids,
        confidence_threshold=threshold,
        iou_threshold=iou_threshold,
    )


def choose_gate_configuration(
    configurations: Iterable[dict[str, float]],
    build: Callable[[dict[str, float]], list[Prediction]],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    image_ids: list[int | str],
    fp_budget: float,
    iou_threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for configuration in configurations:
        predictions = build(configuration)
        metrics = point(predictions, ground_truth, image_ids, "fused_score", 0.0, iou_threshold)
        rows.append({"configuration": configuration, "metrics": metrics, "prediction_count": len(predictions)})
    eligible = [row for row in rows if row["metrics"]["false_positives_per_image"] <= fp_budget]
    if not eligible:
        chosen = min(rows, key=lambda row: row["metrics"]["false_positives_per_image"])
    else:
        chosen = max(
            eligible,
            key=lambda row: (
                row["metrics"]["overall"]["recall"],
                row["metrics"]["overall"]["precision"],
                -row["metrics"]["false_positives_per_image"],
            ),
        )
    return dict(chosen["configuration"]), rows


def matched_annotation_ids(
    predictions: list[Prediction],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    score_key: str,
    threshold: float,
    iou_threshold: float,
) -> set[int | str | None]:
    matched: set[int | str | None] = set()
    for image_id, image_predictions in predictions_by_image(predictions).items():
        used: set[int | str | None] = set()
        for prediction in sorted(
            [item for item in image_predictions if float(item[score_key]) >= threshold],
            key=lambda item: float(item[score_key]),
            reverse=True,
        ):
            candidates = [gt for gt in ground_truth.get(image_id, []) if gt.annotation_id not in used]
            if not candidates:
                continue
            gt = max(candidates, key=lambda item: box_iou(prediction["bbox"], item.box))
            if box_iou(prediction["bbox"], gt.box) >= iou_threshold:
                used.add(gt.annotation_id)
                matched.add(gt.annotation_id)
    return matched


def full_metrics(
    predictions: list[Prediction],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    image_ids: list[int | str],
    score_key: str,
    locked_threshold: float,
    fp_budget: float,
    iou_threshold: float,
) -> dict[str, Any]:
    eval_predictions = to_eval_predictions(predictions, score_key)
    flat_ground_truth = [gt for values in ground_truth.values() for gt in values]
    confidence_sweep = sweep(predictions, ground_truth, image_ids, score_key, fp_budget, iou_threshold)
    return {
        "prediction_count": len(predictions),
        "locked_calibration_point": point(
            predictions, ground_truth, image_ids, score_key, locked_threshold, iou_threshold
        ),
        "exploratory_val600_matched_budget": confidence_sweep["matched_false_positive_budgets"][str(fp_budget)],
        "confidence_sweep": confidence_sweep,
        "pycocotools": evaluate_pycocotools(flat_ground_truth, eval_predictions, image_ids),
    }


def row_for_csv(method: str, point_name: str, metrics: dict[str, Any], coco: dict[str, Any]) -> dict[str, Any]:
    visibility = metrics["by_visibility"]
    return {
        "method": method,
        "point": point_name,
        "threshold": metrics["confidence_threshold"],
        "recall": metrics["overall"]["recall"],
        "precision": metrics["overall"]["precision"],
        "fp_per_image": metrics["false_positives_per_image"],
        "r20": visibility["20"]["recall"],
        "r40": visibility["40"]["recall"],
        "r70": visibility["70"]["recall"],
        "r100": visibility["100"]["recall"],
        "AP": coco["AP"],
        "AP50": coco["AP50"],
        "AP75": coco["AP75"],
        "AR@100": coco["AR@100"],
    }


def save_curve_plot(results: dict[str, Any], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, result in results.items():
        rows = result["confidence_sweep"]["rows"]
        points = sorted((row["false_positives_per_image"], row["overall"]["recall"]) for row in rows)
        ax.plot([x for x, _ in points], [y for _, y in points], label=name, linewidth=1.8)
    ax.axvline(0.425, color="black", linestyle="--", linewidth=1, label="0.425 FP/image")
    ax.set_xlim(0.0, 1.25)
    ax.set_ylim(0.6, 0.97)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Recall at IoU 0.50")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    _, calibration_ids, calibration_gt = load_split(args.root, args.calibration_manifest)
    _, validation_ids, validation_gt = load_split(args.root, args.validation_manifest)
    if set(calibration_ids) & set(validation_ids):
        raise ValueError("Calibration and validation image IDs overlap within the official val split")

    calibration_yolo = load_predictions(args.calibration_yolo, set(calibration_ids), "yolo")
    calibration_dino = load_predictions(args.calibration_dino, set(calibration_ids), "dino")
    validation_yolo = load_predictions(args.validation_yolo, set(validation_ids), "yolo")
    validation_dino = load_predictions(args.validation_dino, set(validation_ids), "dino")

    yolo_labels = calibration_labels(calibration_yolo, calibration_gt, args.iou_threshold)
    dino_labels = calibration_labels(calibration_dino, calibration_gt, args.iou_threshold)
    yolo_calibrator = MonotonicPrecisionCalibrator.fit(
        [item["score"] for item in calibration_yolo], yolo_labels, args.calibration_bins
    )
    dino_calibrator = MonotonicPrecisionCalibrator.fit(
        [item["score"] for item in calibration_dino], dino_labels, args.calibration_bins
    )

    calibration_baselines = {
        "yolo": [{**item, "fused_score": item["score"]} for item in calibration_yolo],
        "dino": [{**item, "fused_score": item["score"]} for item in calibration_dino],
    }
    baseline_thresholds = {}
    for name, predictions in calibration_baselines.items():
        selected = sweep(
            predictions, calibration_gt, calibration_ids, "fused_score", args.fp_budget, args.iou_threshold
        )["matched_false_positive_budgets"][str(args.fp_budget)]
        if selected is None:
            raise RuntimeError(f"No calibration operating point for {name}")
        baseline_thresholds[name] = float(selected["confidence_threshold"])

    a_configurations = [
        {"yolo_keep_threshold": yolo_keep, "dino_add_threshold": dino_add, "duplicate_iou": duplicate_iou}
        for yolo_keep, dino_add, duplicate_iou in itertools.product(
            (0.025, 0.05, 0.075, 0.10, 0.15),
            (0.50, 0.55, 0.60, 0.65, 0.70),
            (0.30, 0.50, 0.70),
        )
    ]
    selected_a, a_search = choose_gate_configuration(
        a_configurations,
        lambda config: reliable_yolo_plus_dino(
            calibration_yolo, calibration_dino, yolo_calibrator, dino_calibrator,
            nms_iou=args.nms_iou, **config,
        ),
        calibration_gt, calibration_ids, args.fp_budget, args.iou_threshold,
    )

    b_configurations = [
        {
            "yolo_keep_threshold": yolo_keep,
            "yolo_rescue_floor": rescue_floor,
            "dino_support_threshold": dino_support,
            "support_iou": support_iou,
        }
        for yolo_keep, rescue_floor, dino_support, support_iou in itertools.product(
            (0.025, 0.05, 0.075, 0.10),
            (0.001, 0.005, 0.01, 0.025),
            (0.45, 0.50, 0.55, 0.60),
            (0.30, 0.50, 0.70),
        )
        if rescue_floor < yolo_keep
    ]
    selected_b, b_search = choose_gate_configuration(
        b_configurations,
        lambda config: supported_yolo_rescue(
            calibration_yolo, calibration_dino, yolo_calibrator, dino_calibrator,
            nms_iou=args.nms_iou, **config,
        ),
        calibration_gt, calibration_ids, args.fp_budget, args.iou_threshold,
    )

    calibration_c = calibrated_support_fusion(
        calibration_yolo, calibration_dino, yolo_calibrator, dino_calibrator, nms_iou=args.nms_iou
    )
    selected_c_point = sweep(
        calibration_c, calibration_gt, calibration_ids, "fused_score", args.fp_budget, args.iou_threshold
    )["matched_false_positive_budgets"][str(args.fp_budget)]
    if selected_c_point is None:
        raise RuntimeError("No calibration operating point for calibrated support fusion")

    validation_methods = {
        "yolo": [{**item, "fused_score": item["score"]} for item in validation_yolo],
        "dino": [{**item, "fused_score": item["score"]} for item in validation_dino],
        "selective_A": reliable_yolo_plus_dino(
            validation_yolo, validation_dino, yolo_calibrator, dino_calibrator,
            nms_iou=args.nms_iou, **selected_a,
        ),
        "selective_B": supported_yolo_rescue(
            validation_yolo, validation_dino, yolo_calibrator, dino_calibrator,
            nms_iou=args.nms_iou, **selected_b,
        ),
        "calibrated_C": calibrated_support_fusion(
            validation_yolo, validation_dino, yolo_calibrator, dino_calibrator, nms_iou=args.nms_iou
        ),
    }
    locked_thresholds = {
        "yolo": baseline_thresholds["yolo"],
        "dino": baseline_thresholds["dino"],
        "selective_A": 0.0,
        "selective_B": 0.0,
        "calibrated_C": float(selected_c_point["confidence_threshold"]),
    }
    results = {
        name: full_metrics(
            predictions, validation_gt, validation_ids, "fused_score", locked_thresholds[name],
            args.fp_budget, args.iou_threshold,
        )
        for name, predictions in validation_methods.items()
    }

    reference_metrics = {
        "yolo_conf_0.05": point(
            validation_methods["yolo"], validation_gt, validation_ids, "fused_score", 0.05,
            args.iou_threshold,
        ),
        "dino_conf_0.55": point(
            validation_methods["dino"], validation_gt, validation_ids, "fused_score", 0.55,
            args.iou_threshold,
        ),
    }

    reference_yolo_ids = matched_annotation_ids(
        validation_methods["yolo"], validation_gt, "fused_score", 0.05, args.iou_threshold
    )
    reference_dino_ids = matched_annotation_ids(
        validation_methods["dino"], validation_gt, "fused_score", 0.55, args.iou_threshold
    )
    dino_only_reference = reference_dino_ids - reference_yolo_ids
    yolo_point_ids = {
        "locked_calibration": matched_annotation_ids(
            validation_methods["yolo"], validation_gt, "fused_score",
            float(results["yolo"]["locked_calibration_point"]["confidence_threshold"]),
            args.iou_threshold,
        ),
        "exploratory_val600_matched": matched_annotation_ids(
            validation_methods["yolo"], validation_gt, "fused_score",
            float(results["yolo"]["exploratory_val600_matched_budget"]["confidence_threshold"]),
            args.iou_threshold,
        ),
    }
    yolo_point_fp = {
        "locked_calibration": results["yolo"]["locked_calibration_point"]["overall"]["false_positives"],
        "exploratory_val600_matched": results["yolo"]["exploratory_val600_matched_budget"]["overall"]["false_positives"],
    }
    comparisons = {}
    for name, predictions in validation_methods.items():
        comparisons[name] = {}
        for point_name, metrics in (
            ("locked_calibration", results[name]["locked_calibration_point"]),
            ("exploratory_val600_matched", results[name]["exploratory_val600_matched_budget"]),
        ):
            if metrics is None:
                continue
            matched = matched_annotation_ids(
                predictions, validation_gt, "fused_score", float(metrics["confidence_threshold"]), args.iou_threshold
            )
            comparisons[name][point_name] = {
                "matched_gt": len(matched),
                "gained_vs_yolo_reference": len(matched - reference_yolo_ids),
                "lost_vs_yolo_reference": len(reference_yolo_ids - matched),
                "net_gt_vs_yolo_reference": len(matched) - len(reference_yolo_ids),
                "dino_only_reference_recovered": len(matched & dino_only_reference),
                "additional_false_positives_vs_yolo_reference": metrics["overall"]["false_positives"] - 208,
                "gained_vs_yolo_same_point": len(matched - yolo_point_ids[point_name]),
                "lost_vs_yolo_same_point": len(yolo_point_ids[point_name] - matched),
                "net_gt_vs_yolo_same_point": len(matched) - len(yolo_point_ids[point_name]),
                "additional_false_positives_vs_yolo_same_point": (
                    metrics["overall"]["false_positives"] - yolo_point_fp[point_name]
                ),
            }

    oracle_union = reference_yolo_ids | reference_dino_ids
    all_gt_by_id = {
        gt.annotation_id: gt for image_ground_truth in validation_gt.values() for gt in image_ground_truth
    }
    oracle_by_visibility = {}
    for visibility in (20, 40, 70, 100):
        ids = {key for key, gt in all_gt_by_id.items() if gt.visibility_level == visibility}
        oracle_by_visibility[str(visibility)] = len(ids & oracle_union) / len(ids)

    summary = {
        "settings": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "score_grid_step": 0.005,
            "selection_warning": "val600 has been inspected repeatedly; matched-budget val600 points are exploratory",
        },
        "split_integrity": {
            "calibration_images": len(calibration_ids),
            "validation_images": len(validation_ids),
            "image_id_overlap": len(set(calibration_ids) & set(validation_ids)),
            "calibration_source": "official validation split outside val600; not used for detector training",
        },
        "saved_predictions": {
            "calibration_yolo": len(calibration_yolo),
            "calibration_dino": len(calibration_dino),
            "validation_yolo": len(validation_yolo),
            "validation_dino": len(validation_dino),
            "validation_yolo_score_min": min(item["score"] for item in validation_yolo),
            "validation_dino_score_min": min(item["score"] for item in validation_dino),
        },
        "calibration": {
            "yolo": yolo_calibrator.as_dict(),
            "dino": dino_calibrator.as_dict(),
            "baseline_thresholds": baseline_thresholds,
            "selected_A": selected_a,
            "selected_B": selected_b,
            "selected_C_threshold": selected_c_point["confidence_threshold"],
            "A_search_count": len(a_search),
            "B_search_count": len(b_search),
        },
        "oracle_non_deployable": {
            "definition": "Union of GT IDs matched independently by YOLO at 0.05 and DINO at 0.55; GT-assisted, not an inference rule.",
            "matched_gt": len(oracle_union),
            "recall": len(oracle_union) / len(all_gt_by_id),
            "recall_by_visibility": oracle_by_visibility,
        },
        "dino_only_reference_count": len(dino_only_reference),
        "canonical_reference_metrics": reference_metrics,
        "results": results,
        "comparisons": comparisons,
    }
    (args.output_dir / "selective_fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.output_dir / "calibration_grid_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "configuration_json", "recall", "precision", "fp_per_image", "predictions"])
        for method_name, search_rows in (("selective_A", a_search), ("selective_B", b_search)):
            for search_row in search_rows:
                metrics = search_row["metrics"]
                writer.writerow([
                    method_name,
                    json.dumps(search_row["configuration"], sort_keys=True),
                    metrics["overall"]["recall"],
                    metrics["overall"]["precision"],
                    metrics["false_positives_per_image"],
                    search_row["prediction_count"],
                ])

    for name, predictions in validation_methods.items():
        coco = [
            {
                "image_id": item["image_id"],
                "category_id": 1,
                "bbox": [
                    item["bbox"][0], item["bbox"][1],
                    item["bbox"][2] - item["bbox"][0], item["bbox"][3] - item["bbox"][1],
                ],
                "score": item["fused_score"],
            }
            for item in predictions
        ]
        (args.output_dir / f"{name}_coco_results.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")

    csv_rows = []
    for name, result in results.items():
        for point_name, metrics in (
            ("locked_calibration", result["locked_calibration_point"]),
            ("exploratory_val600_matched", result["exploratory_val600_matched_budget"]),
        ):
            if metrics is not None:
                csv_rows.append(row_for_csv(name, point_name, metrics, result["pycocotools"]))
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    with (args.output_dir / "candidate_recovery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "point", "matched_gt", "gained_vs_yolo_ref", "lost_vs_yolo_ref",
            "net_gt_vs_yolo_ref", "dino_only_recovered", "additional_fp_vs_yolo_ref",
            "gained_vs_yolo_same_point", "lost_vs_yolo_same_point", "net_gt_vs_yolo_same_point",
            "additional_fp_vs_yolo_same_point",
        ])
        for name, points in comparisons.items():
            for point_name, values in points.items():
                writer.writerow([
                    name, point_name, values["matched_gt"], values["gained_vs_yolo_reference"],
                    values["lost_vs_yolo_reference"], values["net_gt_vs_yolo_reference"],
                    values["dino_only_reference_recovered"], values["additional_false_positives_vs_yolo_reference"],
                    values["gained_vs_yolo_same_point"], values["lost_vs_yolo_same_point"],
                    values["net_gt_vs_yolo_same_point"], values["additional_false_positives_vs_yolo_same_point"],
                ])

    save_curve_plot(results, args.output_dir / "recall_vs_fp_per_image.png")
    print(json.dumps({"calibration": summary["calibration"], "comparisons": comparisons}, indent=2))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
