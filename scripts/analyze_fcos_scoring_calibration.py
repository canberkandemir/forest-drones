"""Evaluate FCOS class/centerness scoring rules without retraining.

The script exports decoded pre-NMS DINOv3 LoRA FCOS candidates with separate
classification and centerness probabilities, calibrates a fixed scoring rule on
a training-only subset, then evaluates that selected rule on val600.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torchvision.ops import nms

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import PATCH_SIZE, PreprocessInfo
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_pycocotools,
    predictions_to_coco_results,
)


FORMULAS = ("current_sqrt_cls_center", "class_alpha_0", "class_center_alpha_0.5", "class_center_alpha_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/dinov3_lora_fcos.pt"))
    parser.add_argument("--calibration-manifest", type=Path, default=Path("outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov3_lora_scoring_diagnostic/train400_calib_val600_20261013"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--custom-iou", type=float, default=0.50)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--fp-budget", type=float, default=0.425)
    parser.add_argument("--conf-thresholds", default="0.001,0.005,0.01,0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9")
    parser.add_argument("--max-calibration-images", type=int, default=None)
    parser.add_argument("--max-validation-images", type=int, default=None)
    parser.add_argument("--reuse-candidates", action="store_true")
    return parser.parse_args()


def parse_float_list(values: str) -> list[float]:
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def load_manifest(path: Path, max_images: int | None = None) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        items = data["images"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported manifest format: {path}")
    return list(items[:max_images] if max_images else items)


def samples_from_manifest(root: Path, split: str, manifest: list[dict[str, Any]]) -> list[Any]:
    dataset = ForestPersonsDataset(root, split=split)
    by_id = {sample.image_id: sample for sample in dataset.samples}
    samples = []
    for item in manifest:
        sample = by_id[item["image_id"]]
        if sample.file_name != item["file_name"]:
            raise ValueError(f"Manifest filename mismatch for image_id={item['image_id']}")
        samples.append(sample)
    return samples


def decode_pre_nms_candidates(
    class_logits: torch.Tensor,
    box_raw: torch.Tensor,
    centerness_logits: torch.Tensor,
    info: PreprocessInfo,
    quality_logits: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    class_scores = class_logits.sigmoid().reshape(-1)
    center_scores = centerness_logits.sigmoid().reshape(-1)
    quality_scores = quality_logits.sigmoid().reshape(-1) if quality_logits is not None else None
    boxes_raw = box_raw.reshape(-1, 4)

    keep = torch.arange(class_scores.numel())
    grid_y = torch.div(keep, info.grid_width, rounding_mode="floor").float()
    grid_x = (keep % info.grid_width).float()
    centers_x = (grid_x + 0.5) * PATCH_SIZE
    centers_y = (grid_y + 0.5) * PATCH_SIZE
    distances = torch.nn.functional.softplus(boxes_raw).clamp(max=128.0) * PATCH_SIZE
    left, top, right, bottom = distances.unbind(dim=1)
    boxes = torch.stack(
        [
            (centers_x - left).clamp(0, info.resized_width - 1),
            (centers_y - top).clamp(0, info.resized_height - 1),
            (centers_x + right).clamp(0, info.resized_width - 1),
            (centers_y + bottom).clamp(0, info.resized_height - 1),
        ],
        dim=1,
    )
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    keep = keep[valid]
    class_scores = class_scores[valid]
    center_scores = center_scores[valid]
    if quality_scores is not None:
        quality_scores = quality_scores[valid]
    scale = torch.tensor(
        [info.scale_x, info.scale_y, info.scale_x, info.scale_y],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    boxes = boxes / scale
    rows = []
    if quality_scores is None:
        quality_scores = torch.full_like(class_scores, float("nan"))
    for flat_index, box, cls, center, quality in zip(
        keep.cpu(), boxes.cpu(), class_scores.cpu(), center_scores.cpu(), quality_scores.cpu()
    ):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        rows.append(
            {
                "flat_index": int(flat_index),
                "box": [
                    max(0.0, min(x1, info.original_width - 1.0)),
                    max(0.0, min(y1, info.original_height - 1.0)),
                    max(0.0, min(x2, info.original_width - 1.0)),
                    max(0.0, min(y2, info.original_height - 1.0)),
                ],
                "class_score": float(cls),
                "centerness_score": float(center),
                "quality_score": float(quality),
            }
        )
    return rows


def candidate_score(candidate: dict[str, Any], formula: str) -> float:
    cls = float(candidate["class_score"])
    center = float(candidate["centerness_score"])
    if formula == "current_sqrt_cls_center":
        return math.sqrt(max(0.0, min(1.0, cls * center)))
    if formula == "class_alpha_0":
        return cls
    if formula == "class_center_alpha_0.5":
        return cls * math.sqrt(max(0.0, center))
    if formula == "class_center_alpha_1":
        return cls * center
    raise ValueError(f"Unknown formula: {formula}")


def export_candidates(
    samples: list[Any],
    split_name: str,
    output_path: Path,
    checkpoint: Path,
    device_name: str,
    short_side: int,
) -> dict[str, Any]:
    device = torch.device(device_name if device_name != "mps" or torch.backends.mps.is_available() else "cpu")
    processor, backbone, head, metadata = load_lora_fcos_checkpoint(checkpoint, device)
    if metadata.get("head_kind", "fcos") not in {"fcos", "fcos_quality"}:
        raise ValueError("This scoring diagnostic only supports single-scale FCOS checkpoints")
    backbone.eval()
    head.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    start = time.perf_counter()
    with gzip.open(output_path, "wt", encoding="utf-8") as handle, torch.inference_mode():
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            features, info = extract_lora_patch_features(
                image,
                processor=processor,
                model=backbone,
                device=device,
                short_side=short_side,
            )
            outputs = head(features)
            if metadata.get("head_kind") == "fcos_quality":
                class_logits, box_raw, centerness_logits, quality_logits = outputs
            else:
                class_logits, box_raw, centerness_logits = outputs
                quality_logits = None
            candidates = decode_pre_nms_candidates(
                class_logits.detach().cpu(),
                box_raw.detach().cpu(),
                centerness_logits.detach().cpu(),
                info,
                quality_logits.detach().cpu() if quality_logits is not None else None,
            )
            handle.write(
                json.dumps(
                    {
                        "image_id": sample.image_id,
                        "file_name": sample.file_name,
                        "width": sample.width,
                        "height": sample.height,
                        "candidates": candidates,
                    }
                )
                + "\n"
            )
            count += len(candidates)
    elapsed = time.perf_counter() - start
    return {
        "split": split_name,
        "images": len(samples),
        "candidates": count,
        "seconds_total": elapsed,
        "seconds_per_image": elapsed / len(samples) if samples else 0.0,
        "path": str(output_path),
    }


def load_candidate_file(path: Path) -> dict[int | str, list[dict[str, Any]]]:
    by_image: dict[int | str, list[dict[str, Any]]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_image[row["image_id"]] = row["candidates"]
    return by_image


def candidates_to_predictions(
    candidate_by_image: dict[int | str, list[dict[str, Any]]],
    image_ids: Iterable[int | str],
    formula: str,
    threshold: float,
    nms_iou: float,
    topk: int,
) -> dict[int | str, list[EvaluationPrediction]]:
    predictions: dict[int | str, list[EvaluationPrediction]] = {}
    for image_id in image_ids:
        rows = []
        for candidate in candidate_by_image.get(image_id, []):
            score = candidate_score(candidate, formula)
            if score >= threshold:
                rows.append((candidate, score))
        if not rows:
            predictions[image_id] = []
            continue
        rows.sort(key=lambda item: item[1], reverse=True)
        rows = rows[:topk]
        boxes = torch.tensor([row[0]["box"] for row in rows], dtype=torch.float32)
        scores = torch.tensor([row[1] for row in rows], dtype=torch.float32)
        selected = nms(boxes, scores, nms_iou)
        predictions[image_id] = [
            EvaluationPrediction(image_id=image_id, box=rows[int(index)][0]["box"], confidence=float(rows[int(index)][1]))
            for index in selected.tolist()
        ]
    return predictions


def flatten_predictions(predictions_by_image: dict[int | str, list[EvaluationPrediction]]) -> list[EvaluationPrediction]:
    return [prediction for predictions in predictions_by_image.values() for prediction in predictions]


def ground_truth_for_samples(samples: list[Any]) -> tuple[dict[int | str, list[Any]], list[Any], list[int | str]]:
    ground_truth_by_image = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    ground_truth = [gt for image_ground_truth in ground_truth_by_image.values() for gt in image_ground_truth]
    image_ids = [sample.image_id for sample in samples]
    return ground_truth_by_image, ground_truth, image_ids


def evaluate_formula_threshold(
    candidate_by_image: dict[int | str, list[dict[str, Any]]],
    samples: list[Any],
    formula: str,
    threshold: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gt_by_image, ground_truth, image_ids = ground_truth_for_samples(samples)
    predictions_by_image = candidates_to_predictions(
        candidate_by_image,
        image_ids,
        formula,
        threshold,
        args.nms_iou,
        args.topk,
    )
    predictions = flatten_predictions(predictions_by_image)
    sweep = evaluate_confidence_sweep(
        gt_by_image,
        predictions_by_image,
        image_ids,
        [threshold],
        iou_threshold=args.custom_iou,
        fp_budgets=[args.fp_budget],
    )
    row = sweep["rows"][0]
    return {
        "formula": formula,
        "threshold": threshold,
        "predictions": len(predictions),
        "custom": row,
        "coco_style": evaluate_coco_style(ground_truth, predictions, image_ids),
        "pycocotools": evaluate_pycocotools(ground_truth, predictions, image_ids),
        "coco_results": predictions_to_coco_results(predictions),
    }


def write_custom_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "formula",
            "threshold",
            "predictions",
            "recall",
            "precision",
            "fp_per_image",
            "r20",
            "r40",
            "r70",
            "r100",
            "thresholded_ap",
            "thresholded_ap50",
            "thresholded_ap75",
            "thresholded_ar100",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            custom = result["custom"]
            by_vis = custom["by_visibility"]
            writer.writerow(
                {
                    "formula": result["formula"],
                    "threshold": result["threshold"],
                    "predictions": result["predictions"],
                    "recall": custom["overall"]["recall"],
                    "precision": custom["overall"]["precision"],
                    "fp_per_image": custom["false_positives_per_image"],
                    "r20": by_vis.get("20", {}).get("recall", 0.0),
                    "r40": by_vis.get("40", {}).get("recall", 0.0),
                    "r70": by_vis.get("70", {}).get("recall", 0.0),
                    "r100": by_vis.get("100", {}).get("recall", 0.0),
                    "thresholded_ap": result["coco_style"]["AP"],
                    "thresholded_ap50": result["coco_style"]["AP50"],
                    "thresholded_ap75": result["coco_style"]["AP75"],
                    "thresholded_ar100": result["coco_style"]["AR"],
                }
            )


def select_calibrated_rule(results: list[dict[str, Any]], fp_budget: float) -> dict[str, Any]:
    eligible = [row for row in results if row["custom"]["false_positives_per_image"] <= fp_budget]
    if not eligible:
        raise RuntimeError(f"No calibration rule met FP/image <= {fp_budget}")
    return max(
        eligible,
        key=lambda row: (
            row["custom"]["overall"]["recall"],
            row["custom"]["by_visibility"].get("20", {}).get("recall", 0.0),
            row["custom"]["overall"]["precision"],
        ),
    )


def formula_coco_metrics(
    candidate_by_image: dict[int | str, list[dict[str, Any]]],
    samples: list[Any],
    formula: str,
    args: argparse.Namespace,
    min_threshold: float = 0.001,
) -> dict[str, Any]:
    _, ground_truth, image_ids = ground_truth_for_samples(samples)
    predictions_by_image = candidates_to_predictions(
        candidate_by_image,
        image_ids,
        formula,
        min_threshold,
        args.nms_iou,
        args.topk,
    )
    predictions = flatten_predictions(predictions_by_image)
    return {
        "min_threshold": min_threshold,
        "predictions": len(predictions),
        "coco_style": evaluate_coco_style(ground_truth, predictions, image_ids),
        "pycocotools": evaluate_pycocotools(ground_truth, predictions, image_ids),
        "coco_results": predictions_to_coco_results(predictions),
    }


def compact_result(result: dict[str, Any], ap_result: dict[str, Any] | None = None) -> dict[str, Any]:
    custom = result["custom"]
    ap_source = ap_result["coco_style"] if ap_result is not None else result["coco_style"]
    return {
        "formula": result["formula"],
        "threshold": result["threshold"],
        "AP": ap_source["AP"],
        "AP50": ap_source["AP50"],
        "AP75": ap_source["AP75"],
        "AR@100": ap_source["AR"],
        "AP_min_threshold": ap_result["min_threshold"] if ap_result is not None else result["threshold"],
        "AP_prediction_count": ap_result["predictions"] if ap_result is not None else result["predictions"],
        "predictions": result["predictions"],
        "recall": custom["overall"]["recall"],
        "precision": custom["overall"]["precision"],
        "false_positives_per_image": custom["false_positives_per_image"],
        "by_visibility": custom["by_visibility"],
    }


def main() -> None:
    args = parse_args()
    thresholds = parse_float_list(args.conf_thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = args.output_dir / "pre_nms_candidates"
    calibration_candidates_path = candidate_dir / "train400_candidates.jsonl.gz"
    validation_candidates_path = candidate_dir / "val600_candidates.jsonl.gz"

    calibration_manifest = load_manifest(args.calibration_manifest, args.max_calibration_images)
    validation_manifest = load_manifest(args.validation_manifest, args.max_validation_images)
    calibration_files = {item["file_name"] for item in calibration_manifest}
    validation_files = {item["file_name"] for item in validation_manifest}
    overlap = sorted(calibration_files & validation_files)
    if overlap:
        raise RuntimeError(f"Calibration/validation overlap detected: {overlap[:5]}")

    calibration_samples = samples_from_manifest(args.root, "train", calibration_manifest)
    validation_samples = samples_from_manifest(args.root, "val", validation_manifest)

    export_summaries = []
    if not args.reuse_candidates or not calibration_candidates_path.exists():
        export_summaries.append(
            export_candidates(
                calibration_samples,
                "train400_calibration",
                calibration_candidates_path,
                args.checkpoint,
                args.device,
                args.short_side,
            )
        )
    if not args.reuse_candidates or not validation_candidates_path.exists():
        export_summaries.append(
            export_candidates(
                validation_samples,
                "val600",
                validation_candidates_path,
                args.checkpoint,
                args.device,
                args.short_side,
            )
        )

    calibration_candidates = load_candidate_file(calibration_candidates_path)
    validation_candidates = load_candidate_file(validation_candidates_path)

    calibration_results = []
    for formula in FORMULAS:
        for threshold in thresholds:
            calibration_results.append(
                evaluate_formula_threshold(calibration_candidates, calibration_samples, formula, threshold, args)
            )
    write_custom_sweep_csv(args.output_dir / "calibration_scoring_sweep.csv", calibration_results)
    selected = select_calibrated_rule(calibration_results, args.fp_budget)

    original_calibration = next(
        row
        for row in calibration_results
        if row["formula"] == "current_sqrt_cls_center" and abs(row["threshold"] - 0.55) < 1e-9
    )

    selected_validation = evaluate_formula_threshold(
        validation_candidates,
        validation_samples,
        selected["formula"],
        selected["threshold"],
        args,
    )
    original_validation = evaluate_formula_threshold(
        validation_candidates,
        validation_samples,
        "current_sqrt_cls_center",
        0.55,
        args,
    )

    validation_comparison = [original_validation, selected_validation]
    write_custom_sweep_csv(args.output_dir / "val600_selected_rule_comparison.csv", validation_comparison)

    original_validation_ap = formula_coco_metrics(
        validation_candidates,
        validation_samples,
        "current_sqrt_cls_center",
        args,
        min_threshold=0.001,
    )
    selected_validation_ap = formula_coco_metrics(
        validation_candidates,
        validation_samples,
        selected["formula"],
        args,
        min_threshold=0.001,
    )
    (args.output_dir / "val600_original_coco_results.json").write_text(
        json.dumps(original_validation_ap["coco_results"], indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "val600_selected_coco_results.json").write_text(
        json.dumps(selected_validation_ap["coco_results"], indent=2),
        encoding="utf-8",
    )

    summary = {
        "settings": {
            "checkpoint": str(args.checkpoint),
            "calibration_manifest": str(args.calibration_manifest),
            "validation_manifest": str(args.validation_manifest),
            "short_side": args.short_side,
            "nms_iou": args.nms_iou,
            "topk": args.topk,
            "fp_budget": args.fp_budget,
            "formulas": list(FORMULAS),
            "thresholds": thresholds,
            "overlap_count": len(overlap),
        },
        "export_summaries": export_summaries,
        "calibration_original_rule": compact_result(original_calibration),
        "calibration_selected_rule": compact_result(selected),
        "validation_original_rule": compact_result(original_validation, original_validation_ap),
        "validation_selected_rule": compact_result(selected_validation, selected_validation_ap),
        "limitations": [
            "The selected scoring rule and threshold were chosen on train400 calibration only.",
            "AP/AP50/AP75/AR@100 use low-threshold ranked predictions for each scoring formula.",
            "Recall/precision/FP-image use the selected operating threshold.",
            "No model weights were modified.",
            "This script recomputes NMS from decoded pre-NMS grid candidates.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
