#!/usr/bin/env python3
"""Evaluate frozen FCOS and two RoI-refined variants on ForestPersons val600."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.detection import greedy_match
from forestpersons.dinov3_detector import choose_device, decode_fcos_predictions
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.dinov3_roi_refiner import (
    decode_fcos_pre_nms_proposals,
    load_roi_refiner_checkpoint,
    refine_and_nms,
)
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_pycocotools,
    predictions_to_coco_results,
)


MODES = ("fcos", "refined_boxes", "refined_boxes_rescored")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"),
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path(
            "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/"
            "dinov3_lora_fcos.pt"
        ),
    )
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--pre-nms-topk", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--custom-iou", type=float, default=0.5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def load_samples(root: Path, split: str, manifest: Path, max_images: int | None = None) -> list[Any]:
    dataset = ForestPersonsDataset(root, split=split)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items = data["images"] if isinstance(data, dict) else data
    if max_images is not None:
        items = items[:max_images]
    by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [by_file[item["file_name"]] for item in items]
    missing = [sample.file_name for sample in samples if not sample.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images, e.g. {missing[0]}")
    return samples


def as_eval_predictions(image_id: int | str, decoded: list[dict[str, Any]]) -> list[EvaluationPrediction]:
    return [
        EvaluationPrediction(image_id=image_id, box=item["box"], confidence=float(item["confidence"]))
        for item in decoded
    ]


def matched_gt_keys(
    ground_truth_by_image: dict[int | str, list[EvaluationGroundTruth]],
    predictions_by_image: dict[int | str, list[EvaluationPrediction]],
    image_ids: list[int | str],
    threshold: float,
    iou_threshold: float,
) -> set[tuple[int | str, int | str | None]]:
    detected: set[tuple[int | str, int | str | None]] = set()
    for image_id in image_ids:
        gt_items = ground_truth_by_image[image_id]
        ground_truth = [
            {
                "box": gt.box,
                "visibility_level": gt.visibility_level,
                "annotation_id": gt.annotation_id,
            }
            for gt in gt_items
        ]
        predictions = [
            {"box": pred.box, "confidence": pred.confidence}
            for pred in predictions_by_image[image_id]
            if pred.confidence >= threshold
        ]
        _matches, matched_gt, _matched_pred = greedy_match(ground_truth, predictions, iou_threshold)
        for index in matched_gt:
            detected.add((image_id, gt_items[index].annotation_id))
    return detected


def visibility_counts(
    keys: set[tuple[int | str, int | str | None]],
    lookup: dict[tuple[int | str, int | str | None], EvaluationGroundTruth],
) -> dict[str, int]:
    return dict(sorted(Counter(str(lookup[key].visibility_level) for key in keys).items()))


def visibility_recall(row: dict[str, Any], visibility: int) -> float | None:
    group = row["by_visibility"].get(str(visibility))
    return float(group["recall"]) if group is not None else None


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.root, args.split, args.manifest, args.max_images)
    image_ids = [sample.image_id for sample in samples]
    ground_truth_by_image = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    all_ground_truth = [gt for values in ground_truth_by_image.values() for gt in values]
    gt_lookup = {(gt.image_id, gt.annotation_id): gt for gt in all_ground_truth}

    device = choose_device(args.device)
    processor, backbone, fcos_head, detector_meta = load_lora_fcos_checkpoint(
        args.detector_checkpoint, device
    )
    refiner, refiner_checkpoint = load_roi_refiner_checkpoint(args.refiner_checkpoint, device)
    if Path(refiner_checkpoint["source_detector_checkpoint"]) != args.detector_checkpoint:
        raise ValueError("Refiner source detector checkpoint does not match evaluation detector")
    backbone.eval()
    fcos_head.eval()
    refiner.eval()
    predictions: dict[str, dict[int | str, list[EvaluationPrediction]]] = {
        mode: {} for mode in MODES
    }
    started = time.perf_counter()
    peak_driver = 0
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            image = Image.open(sample.image_path).convert("RGB")
            features, info = extract_lora_patch_features(
                image,
                processor=processor,
                model=backbone,
                device=device,
                short_side=args.short_side,
            )
            class_logits, box_raw, centerness_logits = fcos_head(features)
            original = decode_fcos_predictions(
                class_logits.detach().cpu(),
                box_raw.detach().cpu(),
                centerness_logits.detach().cpu(),
                info,
                conf_threshold=args.prediction_conf_min,
                nms_threshold=args.nms_iou,
                topk=args.pre_nms_topk,
            )
            candidates = decode_fcos_pre_nms_proposals(
                class_logits,
                box_raw,
                centerness_logits,
                info,
                min_score=args.prediction_conf_min,
                pre_nms_topk=args.pre_nms_topk,
            )
            score_logits, box_deltas = refiner(features, candidates["boxes"])
            cpu_boxes = candidates["boxes"].detach().cpu()
            cpu_scores = candidates["scores"].detach().cpu()
            cpu_logits = score_logits.detach().cpu()
            cpu_deltas = box_deltas.detach().cpu()
            box_only = refine_and_nms(
                cpu_boxes,
                cpu_scores,
                cpu_logits,
                cpu_deltas,
                info,
                score_mode="proposal",
                confidence_threshold=0.0,
                nms_threshold=args.nms_iou,
                detections_per_image=args.max_detections,
            )
            rescored = refine_and_nms(
                cpu_boxes,
                cpu_scores,
                cpu_logits,
                cpu_deltas,
                info,
                score_mode="multiply",
                confidence_threshold=0.0,
                nms_threshold=args.nms_iou,
                detections_per_image=args.max_detections,
            )
            predictions["fcos"][sample.image_id] = as_eval_predictions(sample.image_id, original)
            predictions["refined_boxes"][sample.image_id] = as_eval_predictions(sample.image_id, box_only)
            predictions["refined_boxes_rescored"][sample.image_id] = as_eval_predictions(
                sample.image_id, rescored
            )
            if device.type == "mps":
                peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
            if index == 1 or index % 25 == 0 or index == len(samples):
                print(json.dumps({"images_processed": index, "total": len(samples)}), flush=True)
    if device.type == "mps":
        torch.mps.synchronize()
    runtime = time.perf_counter() - started

    thresholds = [0.0] + [round(index / 200.0, 3) for index in range(1, 201)]
    budgets = (0.25, 0.425, 0.5, 1.0)
    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for mode in MODES:
        flat = [pred for values in predictions[mode].values() for pred in values]
        coco = evaluate_coco_style(all_ground_truth, flat, image_ids)
        official = evaluate_pycocotools(all_ground_truth, flat, image_ids)
        sweep = evaluate_confidence_sweep(
            ground_truth_by_image,
            predictions[mode],
            image_ids,
            thresholds,
            iou_threshold=args.custom_iou,
            fp_budgets=budgets,
        )
        results[mode] = {
            "raw_predictions": len(flat),
            "coco_style": coco,
            "pycocotools": official,
            "confidence_sweep": sweep,
        }
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "coco_results.json").write_text(
            json.dumps(predictions_to_coco_results(flat), indent=2) + "\n", encoding="utf-8"
        )
        with (mode_dir / "confidence_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["threshold", "tp", "fp", "fn", "recall", "precision", "fp_per_image"]
            )
            for row in sweep["rows"]:
                overall = row["overall"]
                writer.writerow(
                    [
                        row["confidence_threshold"],
                        overall["true_positives"],
                        overall["false_positives"],
                        overall["false_negatives"],
                        overall["recall"],
                        overall["precision"],
                        row["false_positives_per_image"],
                    ]
                )
        matched = sweep["matched_false_positive_budgets"]["0.425"]
        comparison_rows.append(
            {
                "mode": mode,
                "AP": coco["AP"],
                "AP50": coco["AP50"],
                "AP75": coco["AP75"],
                "AR@100": coco["AR_by_max_dets"]["100"],
                "maximum_recall": sweep["maximum_recall"]["overall"]["recall"],
                "maximum_recall_fp_per_image": sweep["maximum_recall"]["false_positives_per_image"],
                "matched_threshold": matched["confidence_threshold"],
                "matched_recall": matched["overall"]["recall"],
                "matched_precision": matched["overall"]["precision"],
                "matched_fp_per_image": matched["false_positives_per_image"],
                "R20": visibility_recall(matched, 20),
                "R40": visibility_recall(matched, 40),
                "R70": visibility_recall(matched, 70),
                "R100": visibility_recall(matched, 100),
            }
        )

    matched_sets: dict[str, set[tuple[int | str, int | str | None]]] = {}
    for mode in MODES:
        row = results[mode]["confidence_sweep"]["matched_false_positive_budgets"]["0.425"]
        matched_sets[mode] = matched_gt_keys(
            ground_truth_by_image,
            predictions[mode],
            image_ids,
            float(row["confidence_threshold"]),
            args.custom_iou,
        )
    fcos_threshold = float(
        results["fcos"]["confidence_sweep"]["matched_false_positive_budgets"]["0.425"][
            "confidence_threshold"
        ]
    )
    common_threshold_sets = {
        mode: matched_gt_keys(
            ground_truth_by_image,
            predictions[mode],
            image_ids,
            fcos_threshold,
            args.custom_iou,
        )
        for mode in ("fcos", "refined_boxes")
    }
    recovery = {
        "matched_fp_budget_0.425": {},
        "box_refinement_at_common_fcos_threshold": {
            "threshold": fcos_threshold,
            "recovered": len(common_threshold_sets["refined_boxes"] - common_threshold_sets["fcos"]),
            "lost": len(common_threshold_sets["fcos"] - common_threshold_sets["refined_boxes"]),
            "recovered_by_visibility": visibility_counts(
                common_threshold_sets["refined_boxes"] - common_threshold_sets["fcos"], gt_lookup
            ),
            "lost_by_visibility": visibility_counts(
                common_threshold_sets["fcos"] - common_threshold_sets["refined_boxes"], gt_lookup
            ),
        },
    }
    for mode in ("refined_boxes", "refined_boxes_rescored"):
        recovered = matched_sets[mode] - matched_sets["fcos"]
        lost = matched_sets["fcos"] - matched_sets[mode]
        recovery["matched_fp_budget_0.425"][mode] = {
            "recovered": len(recovered),
            "lost": len(lost),
            "net": len(recovered) - len(lost),
            "recovered_by_visibility": visibility_counts(recovered, gt_lookup),
            "lost_by_visibility": visibility_counts(lost, gt_lookup),
        }

    summary = {
        "settings": {
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "confidence_thresholds": thresholds,
            "fp_budgets": list(budgets),
            "score_modes": {
                "fcos": "canonical sqrt(class * centerness)",
                "refined_boxes": "original FCOS proposal score",
                "refined_boxes_rescored": "FCOS proposal score * RoI person/quality probability",
            },
            "pilot_subset_warning": "Selected val600 pilot; thresholds selected here are exploratory.",
        },
        "images": len(samples),
        "ground_truth": len(all_ground_truth),
        "runtime_seconds": runtime,
        "seconds_per_image_three_modes": runtime / len(samples),
        "peak_mps_driver_memory_bytes": peak_driver if device.type == "mps" else None,
        "peak_mps_driver_memory_gib": peak_driver / (1024**3) if device.type == "mps" else None,
        "results": results,
        "recovery_analysis": recovery,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps({"comparison": comparison_rows, "recovery": recovery}, indent=2), flush=True)


if __name__ == "__main__":
    main()
