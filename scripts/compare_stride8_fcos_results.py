"""Compare saved stride-16 and stride-8 FCOS val predictions without inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from forestpersons import ForestPersonsDataset
from forestpersons.detection import greedy_match
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_confidence_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stride16-dir", type=Path, required=True)
    parser.add_argument("--stride8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[int | str, list[EvaluationPrediction]]:
    grouped: dict[int | str, list[EvaluationPrediction]] = defaultdict(list)
    for row in json.loads(path.read_text(encoding="utf-8")):
        x, y, width, height = [float(value) for value in row["bbox"]]
        grouped[row["image_id"]].append(
            EvaluationPrediction(
                image_id=row["image_id"],
                box=[x, y, x + width, y + height],
                confidence=float(row["score"]),
            )
        )
    return grouped


def size_name(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area < 32**2:
        return "small_lt_32sq"
    if area < 96**2:
        return "medium_32_96sq"
    return "large_ge_96sq"


def matched_gt_keys(
    samples: list[Any],
    predictions: dict[int | str, list[EvaluationPrediction]],
    threshold: float,
) -> tuple[set[tuple[int | str, int | str]], dict[tuple[int | str, int | str], dict[str, Any]]]:
    matched: set[tuple[int | str, int | str]] = set()
    metadata: dict[tuple[int | str, int | str], dict[str, Any]] = {}
    for sample in samples:
        ground_truth = []
        for annotation in sample.annotations:
            visibility = annotation.get("metadata", {}).get("visibility_level")
            if visibility is None:
                continue
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            box = [x, y, x + width, y + height]
            ground_truth.append(
                {
                    "annotation_id": annotation.get("id"),
                    "box": box,
                    "visibility_level": int(visibility),
                }
            )
            key = (sample.image_id, annotation.get("id"))
            metadata[key] = {
                "image_id": sample.image_id,
                "file_name": sample.file_name,
                "annotation_id": annotation.get("id"),
                "visibility_level": int(visibility),
                "size_group": size_name(box),
                "bbox_area": width * height,
            }
        selected = [
            {"box": prediction.box, "confidence": prediction.confidence}
            for prediction in predictions.get(sample.image_id, [])
            if prediction.confidence >= threshold
        ]
        matches, _gt, _pred = greedy_match(ground_truth, selected, 0.50)
        for match in matches:
            gt = ground_truth[match["gt_index"]]
            matched.add((sample.image_id, gt["annotation_id"]))
    return matched, metadata


def outcome_summary(
    samples: list[Any],
    predictions16: dict[int | str, list[EvaluationPrediction]],
    predictions8: dict[int | str, list[EvaluationPrediction]],
    threshold16: float,
    threshold8: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matched16, metadata = matched_gt_keys(samples, predictions16, threshold16)
    matched8, _ = matched_gt_keys(samples, predictions8, threshold8)
    rows: list[dict[str, Any]] = []
    for key, row in metadata.items():
        found16 = key in matched16
        found8 = key in matched8
        if found8 and not found16:
            outcome = "recovered_by_stride8"
        elif found16 and not found8:
            outcome = "lost_by_stride8"
        elif found16 and found8:
            outcome = "both"
        else:
            outcome = "neither"
        rows.append({**row, "stride16_detected": found16, "stride8_detected": found8, "outcome": outcome})

    def grouped(field: str) -> dict[str, Any]:
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            values[str(row[field])].append(row)
        result = {}
        for key, group in sorted(values.items()):
            gt = len(group)
            found16 = sum(row["stride16_detected"] for row in group)
            found8 = sum(row["stride8_detected"] for row in group)
            result[key] = {
                "ground_truth": gt,
                "stride16_detected": found16,
                "stride8_detected": found8,
                "stride16_recall": found16 / gt,
                "stride8_recall": found8 / gt,
                "recovered_by_stride8": sum(row["outcome"] == "recovered_by_stride8" for row in group),
                "lost_by_stride8": sum(row["outcome"] == "lost_by_stride8" for row in group),
            }
        return result

    summary = {
        "threshold_stride16": threshold16,
        "threshold_stride8": threshold8,
        "ground_truth": len(rows),
        "recovered_by_stride8": sum(row["outcome"] == "recovered_by_stride8" for row in rows),
        "lost_by_stride8": sum(row["outcome"] == "lost_by_stride8" for row in rows),
        "both": sum(row["outcome"] == "both" for row in rows),
        "neither": sum(row["outcome"] == "neither" for row in rows),
        "by_visibility": grouped("visibility_level"),
        "by_size": grouped("size_group"),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data
    dataset = ForestPersonsDataset(args.root, split="val")
    by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [by_file[item["file_name"]] for item in items]
    image_ids = [sample.image_id for sample in samples]
    ground_truth = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    predictions16 = load_predictions(args.stride16_dir / "coco_results.json")
    predictions8 = load_predictions(args.stride8_dir / "coco_results.json")
    evaluation16 = json.loads((args.stride16_dir / "summary.json").read_text(encoding="utf-8"))
    evaluation8 = json.loads((args.stride8_dir / "summary.json").read_text(encoding="utf-8"))
    training16 = json.loads((args.stride16_dir.parent / "summary.json").read_text(encoding="utf-8"))
    training8 = json.loads((args.stride8_dir.parent / "summary.json").read_text(encoding="utf-8"))
    thresholds = [round(value / 1000, 3) for value in range(500, 751, 5)]
    budgets = [0.25, 0.425, 0.50, 1.0]
    sweep16 = evaluate_confidence_sweep(ground_truth, predictions16, image_ids, thresholds, 0.50, budgets)
    sweep8 = evaluate_confidence_sweep(ground_truth, predictions8, image_ids, thresholds, 0.50, budgets)
    selected16 = float(sweep16["matched_false_positive_budgets"]["0.425"]["confidence_threshold"])
    selected8 = float(sweep8["matched_false_positive_budgets"]["0.425"]["confidence_threshold"])
    matched_summary, matched_rows = outcome_summary(samples, predictions16, predictions8, selected16, selected8)
    fixed_summary, _fixed_rows = outcome_summary(samples, predictions16, predictions8, 0.60, 0.60)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    for model, sweep in (("stride16", sweep16), ("stride8", sweep8)):
        for row in sweep["rows"]:
            curve_rows.append(
                {
                    "model": model,
                    "confidence_threshold": row["confidence_threshold"],
                    "recall": row["overall"]["recall"],
                    "precision": row["overall"]["precision"],
                    "false_positives_per_image": row["false_positives_per_image"],
                    **{f"recall_visibility_{key}": value["recall"] for key, value in row["by_visibility"].items()},
                }
            )
    with (args.output_dir / "recall_fp_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    with (args.output_dir / "gt_outcomes_matched_fp0425.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matched_rows[0]))
        writer.writeheader()
        writer.writerows(matched_rows)

    metric_rows = []
    for model, evaluation, sweep in (
        ("stride16", evaluation16, sweep16),
        ("stride8", evaluation8, sweep8),
    ):
        fixed = next(row for row in sweep["rows"] if row["confidence_threshold"] == 0.60)
        matched = sweep["matched_false_positive_budgets"]["0.425"]
        maximum = evaluation["custom_confidence_sweep"]["maximum_recall"]
        metric_rows.append(
            {
                "model": model,
                "AP": evaluation["pycocotools"]["AP"],
                "AP50": evaluation["pycocotools"]["AP50"],
                "AP75": evaluation["pycocotools"]["AP75"],
                "AR100": evaluation["pycocotools"]["AR@100"],
                "maximum_recall_conf_min_0001": maximum["overall"]["recall"],
                "maximum_recall_fp_per_image": maximum["false_positives_per_image"],
                "fixed_confidence": 0.60,
                "fixed_recall": fixed["overall"]["recall"],
                "fixed_precision": fixed["overall"]["precision"],
                "fixed_fp_per_image": fixed["false_positives_per_image"],
                "fixed_R20": fixed["by_visibility"]["20"]["recall"],
                "fixed_R40": fixed["by_visibility"]["40"]["recall"],
                "fixed_R70": fixed["by_visibility"]["70"]["recall"],
                "fixed_R100": fixed["by_visibility"]["100"]["recall"],
                "exploratory_matched_threshold": matched["confidence_threshold"],
                "matched_recall": matched["overall"]["recall"],
                "matched_precision": matched["overall"]["precision"],
                "matched_fp_per_image": matched["false_positives_per_image"],
                "matched_R20": matched["by_visibility"]["20"]["recall"],
                "matched_R40": matched["by_visibility"]["40"]["recall"],
                "matched_R70": matched["by_visibility"]["70"]["recall"],
                "matched_R100": matched["by_visibility"]["100"]["recall"],
                "inference_seconds_per_image": evaluation["timing"]["seconds_per_image_end_to_end"],
            }
        )
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    training_rows = []
    for model, training in (("stride16", training16), ("stride8", training8)):
        for row in training["history"]:
            training_rows.append({"model": model, **row})
    scalar_fields = [
        "model", "epoch", "images", "optimizer_steps", "loss", "classification_loss",
        "box_loss", "centerness_loss", "positive_cells", "center_fallbacks",
        "multi_gt_overwrites", "image_order_sha256", "seconds",
    ]
    with (args.output_dir / "training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(training_rows)

    figure, axis = plt.subplots(figsize=(7, 5))
    for model, color in (("stride16", "#1877B8"), ("stride8", "#D1495B")):
        rows = [row for row in curve_rows if row["model"] == model]
        axis.plot(
            [row["false_positives_per_image"] for row in rows],
            [row["recall"] for row in rows],
            marker="o",
            markersize=3,
            label=model,
            color=color,
        )
    axis.axvline(0.425, color="black", linestyle="--", linewidth=1, label="FP/image 0.425")
    axis.set_xlim(0, 1.1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("False positives per image")
    axis.set_ylabel("Recall at IoU 0.50")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "recall_vs_fp.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    for model, color in (("stride16", "#1877B8"), ("stride8", "#D1495B")):
        rows = [row for row in training_rows if row["model"] == model]
        axis.plot([row["epoch"] for row in rows], [row["loss"] for row in rows], marker="o", label=model, color=color)
    axis.set_xticks([1, 2])
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean training loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "training_loss.png", dpi=180)
    plt.close(figure)

    summary = {
        "definition": {
            "dense_thresholds": thresholds,
            "matched_budget": 0.425,
            "selection_warning": "Thresholds were selected on val600 and are exploratory, not independently validated.",
            "size_groups": "COCO original-image box area: small < 32^2, medium < 96^2, large otherwise.",
        },
        "dense_sweep": {"stride16": sweep16, "stride8": sweep8},
        "matched_fp_0425_outcomes": matched_summary,
        "fixed_conf_060_outcomes": fixed_summary,
        "comparison_metrics": metric_rows,
        "training": {
            "stride16": {
                "seconds_total": training16["seconds_total"],
                "peak_observed_mps_driver_memory_bytes": max(row["mps_memory_bytes"]["driver_allocated"] for row in training16["history"]),
                "history": training16["history"],
            },
            "stride8": {
                "seconds_total": training8["seconds_total"],
                "peak_observed_mps_driver_memory_bytes": max(row["mps_memory_bytes"]["driver_allocated"] for row in training8["history"]),
                "history": training8["history"],
            },
        },
    }
    (args.output_dir / "comparison_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "stride16_at_0425": sweep16["matched_false_positive_budgets"]["0.425"],
        "stride8_at_0425": sweep8["matched_false_positive_budgets"]["0.425"],
        "matched_outcomes": matched_summary,
        "fixed_conf_060_outcomes": fixed_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
