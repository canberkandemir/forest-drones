"""Evaluate frozen selective-fusion rules on an independent ForestPersons holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou as evaluation_box_iou, evaluate_image_matches
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
    predictions_by_image,
    reliable_yolo_plus_dino,
    supported_yolo_rescue,
)


Prediction = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--yolo-predictions", type=Path, required=True)
    parser.add_argument("--dino-predictions", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261015)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["images"] if isinstance(data, dict) else data


def load_ground_truth(root: Path, manifest_path: Path) -> tuple[list[int | str], dict[int | str, list[EvaluationGroundTruth]]]:
    dataset = ForestPersonsDataset(root, split="val")
    samples_by_id = {sample.image_id: sample for sample in dataset.samples}
    samples = []
    for item in load_manifest(manifest_path):
        sample = samples_by_id[item["image_id"]]
        if sample.file_name != item["file_name"]:
            raise ValueError(f"Manifest mismatch for image ID {item['image_id']}")
        samples.append(sample)
    image_ids = [sample.image_id for sample in samples]
    ground_truth = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    return image_ids, ground_truth


def load_predictions(path: Path, image_ids: set[int | str], source: str) -> list[Prediction]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for item in raw:
        if item["image_id"] not in image_ids or int(item.get("category_id", 1)) != 1:
            continue
        x, y, width, height = [float(value) for value in item["bbox"]]
        result.append({
            "image_id": item["image_id"],
            "category_id": 1,
            "bbox": [x, y, x + width, y + height],
            "score": float(item["score"]),
            "source": source,
        })
    return result


def calibrator_from_dict(data: dict[str, Any]) -> MonotonicPrecisionCalibrator:
    return MonotonicPrecisionCalibrator(
        tuple(float(value) for value in data["upper_edges"]),
        tuple(float(value) for value in data["probabilities"]),
        tuple(int(value) for value in data["counts"]),
    )


def to_eval(predictions: Iterable[Prediction]) -> list[EvaluationPrediction]:
    return [
        EvaluationPrediction(
            image_id=item["image_id"], box=item["bbox"], confidence=float(item["fused_score"])
        )
        for item in predictions
    ]


def eval_by_image(predictions: Iterable[Prediction], image_ids: list[int | str]) -> dict[int | str, list[EvaluationPrediction]]:
    grouped = {image_id: [] for image_id in image_ids}
    for prediction in to_eval(predictions):
        grouped[prediction.image_id].append(prediction)
    return grouped


def matched_annotation_ids(
    predictions: list[Prediction],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    threshold: float,
    iou_threshold: float,
) -> set[int | str | None]:
    matched: set[int | str | None] = set()
    for image_id, image_predictions in predictions_by_image(predictions).items():
        used: set[int | str | None] = set()
        for prediction in sorted(
            [item for item in image_predictions if float(item["fused_score"]) >= threshold],
            key=lambda item: float(item["fused_score"]),
            reverse=True,
        ):
            candidates = [gt for gt in ground_truth.get(image_id, []) if gt.annotation_id not in used]
            if not candidates:
                continue
            gt = max(candidates, key=lambda item: evaluation_box_iou(item.box, prediction["bbox"]))
            if evaluation_box_iou(gt.box, prediction["bbox"]) >= iou_threshold:
                used.add(gt.annotation_id)
                matched.add(gt.annotation_id)
    return matched


def per_image_recall_counts(
    predictions: list[Prediction],
    ground_truth: dict[int | str, list[EvaluationGroundTruth]],
    image_ids: list[int | str],
    threshold: float,
    iou_threshold: float,
) -> dict[str, np.ndarray]:
    grouped = predictions_by_image(predictions)
    visibility_levels = (20, 40, 70, 100)
    output = {"gt": np.zeros(len(image_ids)), "tp": np.zeros(len(image_ids))}
    for visibility in visibility_levels:
        output[f"gt_{visibility}"] = np.zeros(len(image_ids))
        output[f"tp_{visibility}"] = np.zeros(len(image_ids))
    for index, image_id in enumerate(image_ids):
        gt = [
            {"box": item.box, "visibility_level": item.visibility_level, "annotation_id": item.annotation_id}
            for item in ground_truth.get(image_id, [])
        ]
        pred = [
            {"box": item["bbox"], "confidence": item["fused_score"]}
            for item in grouped.get(image_id, [])
            if float(item["fused_score"]) >= threshold
        ]
        _, _, _, overall, by_visibility = evaluate_image_matches(gt, pred, iou_threshold)
        output["gt"][index] = len(gt)
        output["tp"][index] = overall["true_positives"]
        for visibility in visibility_levels:
            stats = by_visibility[visibility]
            output[f"tp_{visibility}"][index] = stats["true_positives"]
            output[f"gt_{visibility}"][index] = stats["true_positives"] + stats["false_negatives"]
    return output


def paired_bootstrap(
    reference: dict[str, np.ndarray],
    method: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sample_count = reference["gt"].shape[0]
    results = {}
    for label, suffix in (("overall", ""), ("20", "_20"), ("40", "_40"), ("70", "_70"), ("100", "_100")):
        gt_key = f"gt{suffix}"
        tp_key = f"tp{suffix}"
        differences = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 250):
            end = min(replicates, start + 250)
            indices = rng.integers(0, sample_count, size=(end - start, sample_count))
            denominators = reference[gt_key][indices].sum(axis=1)
            ref_recall = np.divide(
                reference[tp_key][indices].sum(axis=1), denominators,
                out=np.zeros(end - start), where=denominators > 0,
            )
            method_recall = np.divide(
                method[tp_key][indices].sum(axis=1), denominators,
                out=np.zeros(end - start), where=denominators > 0,
            )
            differences[start:end] = method_recall - ref_recall
        results[label] = {
            "mean_difference": float(differences.mean()),
            "ci95_percentile": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))],
            "probability_difference_above_zero": float(np.mean(differences > 0.0)),
            "replicates": replicates,
            "resampling_unit": "image",
        }
    return results


def score_thresholds() -> list[float]:
    return [float(value) for value in np.arange(0.0, 1.0001, 0.005)]


def save_coco(path: Path, predictions: list[Prediction]) -> None:
    output = []
    for item in predictions:
        x1, y1, x2, y2 = item["bbox"]
        output.append({
            "image_id": item["image_id"], "category_id": 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": item["fused_score"],
        })
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def save_curve(results: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, result in results.items():
        points = sorted(
            (row["false_positives_per_image"], row["overall"]["recall"])
            for row in result["diagnostic_sweep"]["rows"]
        )
        ax.plot([x for x, _ in points], [y for _, y in points], label=name, linewidth=1.8)
    ax.axvline(0.425, color="black", linestyle="--", linewidth=1, label="0.425 FP/image")
    ax.set_xlim(0.0, 1.25)
    ax.set_ylim(0.55, 0.98)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Recall at IoU 0.50")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    if sha256_file(args.source_summary) != frozen["source_experiment_sha256"]:
        raise RuntimeError("Frozen source summary hash mismatch")
    for model in frozen["checkpoints"].values():
        if sha256_file(Path(model["path"])) != model["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {model['path']}")

    image_ids, ground_truth = load_ground_truth(args.root, args.manifest)
    yolo = load_predictions(args.yolo_predictions, set(image_ids), "yolo")
    dino = load_predictions(args.dino_predictions, set(image_ids), "dino")
    yolo_calibrator = calibrator_from_dict(source["calibration"]["yolo"])
    dino_calibrator = calibrator_from_dict(source["calibration"]["dino"])
    operating = frozen["operating_points"]
    iou_threshold = float(frozen["evaluation"]["match_iou"])
    fp_budget = float(frozen["evaluation"]["strict_fp_per_image_limit"])

    methods = {
        "yolo": [{**item, "fused_score": item["score"]} for item in yolo],
        "selective_A": reliable_yolo_plus_dino(
            yolo, dino, yolo_calibrator, dino_calibrator,
            yolo_keep_threshold=operating["selective_A"]["yolo_keep_threshold"],
            dino_add_threshold=operating["selective_A"]["dino_add_threshold"],
            duplicate_iou=operating["selective_A"]["duplicate_iou"],
            nms_iou=operating["selective_A"]["fusion_nms_iou"],
        ),
        "selective_B": supported_yolo_rescue(
            yolo, dino, yolo_calibrator, dino_calibrator,
            yolo_keep_threshold=operating["selective_B"]["yolo_keep_threshold"],
            yolo_rescue_floor=operating["selective_B"]["yolo_rescue_floor"],
            dino_support_threshold=operating["selective_B"]["dino_support_threshold"],
            support_iou=operating["selective_B"]["support_iou"],
            minimum_area_consistency=operating["selective_B"]["minimum_area_consistency"],
            nms_iou=operating["selective_B"]["fusion_nms_iou"],
        ),
    }
    thresholds = {
        "yolo": operating["yolo"]["confidence_threshold"],
        "selective_A": operating["selective_A"]["final_fused_score_threshold"],
        "selective_B": operating["selective_B"]["final_fused_score_threshold"],
    }

    flat_gt = [item for values in ground_truth.values() for item in values]
    results = {}
    matched_ids = {}
    image_counts = {}
    for name, predictions in methods.items():
        by_image = eval_by_image(predictions, image_ids)
        locked = evaluate_custom_metrics(
            ground_truth, by_image, image_ids, float(thresholds[name]), iou_threshold
        )
        diagnostic = evaluate_confidence_sweep(
            ground_truth, by_image, image_ids, score_thresholds(), iou_threshold, [fp_budget]
        )
        results[name] = {
            "locked": locked,
            "strict_fp_limit_satisfied": locked["false_positives_per_image"] <= fp_budget,
            "pycocotools": evaluate_pycocotools(flat_gt, to_eval(predictions), image_ids),
            "diagnostic_sweep": diagnostic,
            "diagnostic_matched_budget_is_test_tuned": diagnostic["matched_false_positive_budgets"][str(fp_budget)],
            "prediction_count": len(predictions),
        }
        matched_ids[name] = matched_annotation_ids(
            predictions, ground_truth, float(thresholds[name]), iou_threshold
        )
        image_counts[name] = per_image_recall_counts(
            predictions, ground_truth, image_ids, float(thresholds[name]), iou_threshold
        )
        save_coco(args.output_dir / f"{name}_coco_results.json", predictions)

    comparisons = {}
    yolo_ids = matched_ids["yolo"]
    for name in ("selective_A", "selective_B"):
        method_ids = matched_ids[name]
        comparisons[name] = {
            "rescued_gt": len(method_ids - yolo_ids),
            "yolo_true_detections_lost": len(yolo_ids - method_ids),
            "net_gt_change": len(method_ids) - len(yolo_ids),
            "recall_difference": (
                results[name]["locked"]["overall"]["recall"]
                - results["yolo"]["locked"]["overall"]["recall"]
            ),
            "r20_difference": (
                results[name]["locked"]["by_visibility"]["20"]["recall"]
                - results["yolo"]["locked"]["by_visibility"]["20"]["recall"]
            ),
            "additional_false_positives": (
                results[name]["locked"]["overall"]["false_positives"]
                - results["yolo"]["locked"]["overall"]["false_positives"]
            ),
            "paired_image_bootstrap": paired_bootstrap(
                image_counts["yolo"], image_counts[name], args.bootstrap_replicates,
                args.bootstrap_seed + (1 if name == "selective_A" else 2),
            ),
        }

    manifest_data = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = {
        "status": "independent_image-disjoint_holdout_evaluation",
        "settings": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "frozen_config_sha256": sha256_file(args.frozen_config),
            "manifest_sha256": sha256_file(args.manifest),
            "selective_fusion_code_sha256": sha256_file(Path("src/forestpersons/selective_fusion.py")),
            "no_holdout_parameter_selection": True,
        },
        "dataset_audit": manifest_data["summary"],
        "images": len(image_ids),
        "ground_truth": len(flat_gt),
        "raw_saved_predictions": {"yolo": len(yolo), "dino": len(dino)},
        "frozen_operating_points": operating,
        "results": results,
        "comparisons": comparisons,
        "limitations": [
            "Image identities and hashes are disjoint, but sequence folders overlap protected subsets.",
            "The optional matched-budget diagnostic uses holdout labels and is not the primary result.",
            "The DINO saved candidate floor is determined by its decoder/post-NMS output distribution.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method", "threshold", "recall", "precision", "fp_per_image", "tp", "fp", "fn",
            "r20", "r40", "r70", "r100", "AP", "AP50", "AP75", "AR@100",
            "strict_fp_limit_satisfied", "rescued_gt", "yolo_detections_lost", "net_gt_change",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in results.items():
            locked, coco = result["locked"], result["pycocotools"]
            comparison = comparisons.get(name, {})
            writer.writerow({
                "method": name, "threshold": locked["confidence_threshold"],
                "recall": locked["overall"]["recall"], "precision": locked["overall"]["precision"],
                "fp_per_image": locked["false_positives_per_image"],
                "tp": locked["overall"]["true_positives"], "fp": locked["overall"]["false_positives"],
                "fn": locked["overall"]["false_negatives"],
                "r20": locked["by_visibility"]["20"]["recall"],
                "r40": locked["by_visibility"]["40"]["recall"],
                "r70": locked["by_visibility"]["70"]["recall"],
                "r100": locked["by_visibility"]["100"]["recall"],
                "AP": coco["AP"], "AP50": coco["AP50"], "AP75": coco["AP75"],
                "AR@100": coco["AR@100"],
                "strict_fp_limit_satisfied": result["strict_fp_limit_satisfied"],
                "rescued_gt": comparison.get("rescued_gt", 0),
                "yolo_detections_lost": comparison.get("yolo_true_detections_lost", 0),
                "net_gt_change": comparison.get("net_gt_change", 0),
            })

    save_curve(results, args.output_dir / "recall_vs_fp_per_image.png")
    print(json.dumps({"results": {k: v["locked"] for k, v in results.items()}, "comparisons": comparisons}, indent=2))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
