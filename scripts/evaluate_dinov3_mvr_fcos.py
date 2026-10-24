#!/usr/bin/env python3
"""Evaluate matched baseline, native MVR, and full MVR FCOS checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3MVRFCOSHead,
    DINOv3MVRNativeFCOSHead,
    cache_path_for_file,
    decode_fcos_predictions,
    load_feature_cache,
    load_mvr_feature_cache,
)
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    predictions_to_coco_results,
)


THRESHOLDS = (0.001, 0.005) + tuple(value / 100 for value in range(1, 101))
FP_BUDGETS = (0.15, 0.25, 0.35, 0.425)


def collect_samples(root: Path, manifest_path: Path) -> list[Any]:
    dataset = ForestPersonsDataset(root, split="val")
    by_id = {int(sample.image_id): sample for sample in dataset.samples}
    items = json.loads(manifest_path.read_text(encoding="utf-8"))["images"]
    samples = []
    for item in items:
        sample = by_id[int(item["image_id"])]
        if sample.file_name != item["file_name"]:
            raise ValueError(f"Filename mismatch for image_id {item['image_id']}")
        samples.append(sample)
    if len(samples) != 600:
        raise ValueError(f"Expected val600, found {len(samples)}")
    return samples


def evaluate_arm(
    arm: str,
    checkpoint_path: Path,
    samples: list[Any],
    final_cache_root: Path,
    mvr_cache_root: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[EvaluationPrediction]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["arm"] != arm:
        raise ValueError(f"Checkpoint {checkpoint_path} is not the {arm} arm")
    model: torch.nn.Module
    if arm == "baseline":
        model = DINOv3FCOSHead()
    elif arm == "mvr_native":
        model = DINOv3MVRNativeFCOSHead()
    elif arm == "mvr":
        model = DINOv3MVRFCOSHead()
    else:
        raise ValueError(f"Unsupported arm: {arm}")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    predictions: list[EvaluationPrediction] = []
    ground_truth_by_image = {}
    predictions_by_image = {}
    image_ids = []
    seconds = 0.0
    with torch.inference_mode():
        for sample in samples:
            final, info = load_feature_cache(
                cache_path_for_file(final_cache_root, sample.file_name)
            )
            if arm == "baseline":
                features = final
            else:
                intermediate, mvr_info, layers = load_mvr_feature_cache(
                    cache_path_for_file(mvr_cache_root, sample.file_name)
                )
                if mvr_info != info or layers != (3, 6, 9):
                    raise ValueError(f"MVR cache mismatch for {sample.file_name}")
                features = torch.cat((intermediate, final.unsqueeze(0)), dim=0)
            started = time.perf_counter()
            logits, raw_box, centerness = model(features.to(device))
            decoded = decode_fcos_predictions(
                logits.detach().cpu(),
                raw_box.detach().cpu(),
                centerness.detach().cpu(),
                info,
                conf_threshold=0.001,
                nms_threshold=0.50,
                topk=300,
            )
            if device.type == "mps":
                torch.mps.synchronize()
            seconds += time.perf_counter() - started
            image_id = int(sample.image_id)
            image_ids.append(image_id)
            image_predictions = [
                EvaluationPrediction(
                    image_id=image_id,
                    box=[float(value) for value in item["box"]],
                    confidence=float(item["confidence"]),
                )
                for item in decoded
            ]
            predictions.extend(image_predictions)
            predictions_by_image[image_id] = image_predictions
            ground_truth_by_image[image_id] = annotations_to_eval_ground_truth(
                image_id, sample.annotations
            )
    ground_truth = [item for values in ground_truth_by_image.values() for item in values]
    coco = evaluate_coco_style(ground_truth, predictions, image_ids)
    sweep = evaluate_confidence_sweep(
        ground_truth_by_image,
        predictions_by_image,
        image_ids,
        THRESHOLDS,
        iou_threshold=0.50,
        fp_budgets=FP_BUDGETS,
    )
    return {
        "arm": arm,
        "checkpoint": str(checkpoint_path.resolve()),
        "images": len(image_ids),
        "ground_truth": len(ground_truth),
        "AP": coco["AP"],
        "AP50": coco["AP50"],
        "AP75": coco["AP75"],
        "AR100": coco["AR_by_max_dets"]["100"],
        "maximum_recall": sweep["maximum_recall"],
        "matched_fp_budgets": sweep["matched_false_positive_budgets"],
        "confidence_sweep": sweep["rows"],
        "evaluation_protocol": {
            "iou_threshold": 0.50,
            "matching": "one-to-one greedy matching in descending confidence order",
            "confidence_thresholds": list(THRESHOLDS),
            "fp_per_image_budgets": list(FP_BUDGETS),
            "selection": (
                "Each model independently selects the evaluated threshold with the "
                "highest recall that does not exceed the FP/image budget."
            ),
            "candidate_floor": 0.001,
            "nms_iou_threshold": 0.50,
            "topk": 300,
        },
        "seconds_cached_readout_and_head": seconds,
        "seconds_per_image": seconds / len(image_ids),
    }, predictions


def flatten_sweep_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for row in result["confidence_sweep"]:
            rows.append(
                {
                    "arm": result["arm"],
                    "confidence_threshold": row["confidence_threshold"],
                    "precision": row["overall"]["precision"],
                    "recall": row["overall"]["recall"],
                    **{
                        f"R{visibility}": row["by_visibility"]
                        .get(str(visibility), {})
                        .get("recall", 0.0)
                        for visibility in (20, 40, 70, 100)
                    },
                    "false_positives_per_image": row["false_positives_per_image"],
                    "true_positives": row["overall"]["true_positives"],
                    "false_positives": row["overall"]["false_positives"],
                    "false_negatives": row["overall"]["false_negatives"],
                }
            )
    return rows


def matched_budget_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for budget in FP_BUDGETS:
            matched = result["matched_fp_budgets"][str(budget)]
            rows.append(
                {
                    "arm": result["arm"],
                    "fp_per_image_budget": budget,
                    "confidence_threshold": (
                        matched["confidence_threshold"] if matched else None
                    ),
                    "precision": matched["overall"]["precision"] if matched else None,
                    "recall": matched["overall"]["recall"] if matched else None,
                    **{
                        f"R{visibility}": (
                            matched["by_visibility"]
                            .get(str(visibility), {})
                            .get("recall", 0.0)
                            if matched
                            else None
                        )
                        for visibility in (20, 40, 70, 100)
                    },
                    "false_positives_per_image": (
                        matched["false_positives_per_image"] if matched else None
                    ),
                    "AP": result["AP"],
                    "AP50": result["AP50"],
                    "AP75": result["AP75"],
                    "AR100": result["AR100"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_report(results: list[dict[str, Any]], matched: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen DINOv3 FCOS MVR readout ablation on ForestPersons val600",
        "",
        "All evaluated models use identical val600 images, IoU 0.50 one-to-one matching, "
        "candidate floor 0.001, NMS IoU 0.50, and top-300 decoding. Thresholds "
        "are selected separately per model from the same predeclared sweep. "
        "Because selection uses val600 labels, matched-budget points are exploratory.",
        "",
        "## COCO-style metrics",
        "",
        "| Model | AP | AP50 | AP75 | AR@100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['arm']} | {result['AP']:.4f} | {result['AP50']:.4f} | "
            f"{result['AP75']:.4f} | {result['AR100']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Matched FP/image budgets",
            "",
            "| Model | Budget | Threshold | Precision | Recall | R20 | R40 | R70 | R100 | FP/image |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matched:
        if row["confidence_threshold"] is None:
            lines.append(
                f"| {row['arm']} | {row['fp_per_image_budget']:.3f} | "
                "n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            lines.append(
                f"| {row['arm']} | {row['fp_per_image_budget']:.3f} | "
                f"{row['confidence_threshold']:.3f} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['R20']:.4f} | {row['R40']:.4f} | "
                f"{row['R70']:.4f} | {row['R100']:.4f} | "
                f"{row['false_positives_per_image']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final-cache-root", type=Path, required=True)
    parser.add_argument("--mvr-cache-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--mvr-native-checkpoint", type=Path)
    parser.add_argument("--mvr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    samples = collect_samples(args.root, args.manifest)
    results = []
    arms = [("baseline", args.baseline_checkpoint)]
    if args.mvr_native_checkpoint is not None:
        arms.append(("mvr_native", args.mvr_native_checkpoint))
    arms.append(("mvr", args.mvr_checkpoint))
    for arm, checkpoint in arms:
        metrics, predictions = evaluate_arm(
            arm,
            checkpoint,
            samples,
            args.final_cache_root,
            args.mvr_cache_root,
            device,
        )
        results.append(metrics)
        (args.output_dir / f"{arm}_predictions_coco.json").write_text(
            json.dumps(predictions_to_coco_results(predictions), indent=2) + "\n"
        )
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    sweep_rows = flatten_sweep_rows(results)
    matched_rows = matched_budget_rows(results)
    write_csv(args.output_dir / "confidence_sweep.csv", sweep_rows)
    write_csv(args.output_dir / "matched_fp_budgets.csv", matched_rows)
    (args.output_dir / "comparison_report.md").write_text(
        comparison_report(results, matched_rows), encoding="utf-8"
    )
    print(json.dumps(matched_rows, indent=2))


if __name__ == "__main__":
    main()
