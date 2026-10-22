#!/usr/bin/env python3
"""Offline FCOS ranking oracle and candidate-recovery analysis on val600.

The script uses an existing decoded pre-NMS candidate dump. It performs no
model inference and does not modify checkpoints or canonical predictions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import nms

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, greedy_match
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    predictions_to_coco_results,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/error_analysis/dinov3_fcos_ranking_oracle_val600_20261015"
FORMULAS = ("class", "centerness", "oracle_iou", "oracle_iou_x_current")
MODES = ("topk_only", "nms_only", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/subsets/forestpersons_val_600_seed20261010_manifest.json",
    )
    parser.add_argument(
        "--candidate-dump",
        type=Path,
        default=ROOT
        / "outputs/dinov3_lora_scoring_diagnostic/train400_calib_val600_20261013"
        / "pre_nms_candidates/val600_candidates.jsonl.gz",
    )
    parser.add_argument(
        "--canonical-predictions",
        type=Path,
        default=ROOT
        / "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1"
        / "eval_val600/coco_results.json",
    )
    parser.add_argument(
        "--localization-diagnostics",
        type=Path,
        default=ROOT
        / "outputs/error_analysis/dinov3_only_localization_val600_20261015"
        / "dino_localization_failures.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-floor", type=float, default=0.001)
    parser.add_argument("--fixed-threshold", type=float, default=0.55)
    parser.add_argument("--fp-budget", type=float, default=0.425)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.50)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_ground_truth(
    root: Path, manifest_path: Path
) -> tuple[list[int], dict[int, list[Any]], list[Any], dict[int, list[dict[str, Any]]]]:
    dataset = ForestPersonsDataset(root, split="val")
    samples_by_id = {int(sample.image_id): sample for sample in dataset.samples}
    manifest = read_json(manifest_path)["images"]
    image_ids = [int(item["image_id"]) for item in manifest]
    eval_by_image: dict[int, list[Any]] = {}
    flat_eval: list[Any] = []
    raw_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for image_id in image_ids:
        sample = samples_by_id[image_id]
        eval_gt = annotations_to_eval_ground_truth(image_id, sample.annotations)
        eval_by_image[image_id] = eval_gt
        flat_eval.extend(eval_gt)
        for annotation in sample.annotations:
            metadata = annotation.get("metadata", {})
            visibility = metadata.get("visibility_level")
            if visibility is None:
                continue
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            raw_by_image[image_id].append(
                {
                    "annotation_id": int(annotation["id"]),
                    "box": [x, y, x + width, y + height],
                    "visibility": int(visibility),
                }
            )
    return image_ids, eval_by_image, flat_eval, raw_by_image


def pairwise_iou(boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((len(boxes), len(gt_boxes)), dtype=np.float64)
    xx1 = np.maximum(boxes[:, None, 0], gt_boxes[None, :, 0])
    yy1 = np.maximum(boxes[:, None, 1], gt_boxes[None, :, 1])
    xx2 = np.minimum(boxes[:, None, 2], gt_boxes[None, :, 2])
    yy2 = np.minimum(boxes[:, None, 3], gt_boxes[None, :, 3])
    intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    box_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    gt_area = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    return intersection / np.maximum(box_area[:, None] + gt_area[None, :] - intersection, 1e-12)


def strict_one_to_one_oracle_quality(boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Give nonzero quality to at most one candidate per GT and vice versa."""
    quality = np.zeros(len(boxes), dtype=np.float64)
    matrix = pairwise_iou(boxes, gt_boxes)
    if matrix.size == 0:
        return quality
    pairs = np.argwhere(matrix > 0.0)
    pairs = sorted(pairs.tolist(), key=lambda pair: float(matrix[pair[0], pair[1]]), reverse=True)
    used_candidates: set[int] = set()
    used_gt: set[int] = set()
    for candidate_index, gt_index in pairs:
        if candidate_index in used_candidates or gt_index in used_gt:
            continue
        quality[candidate_index] = float(matrix[candidate_index, gt_index])
        used_candidates.add(candidate_index)
        used_gt.add(gt_index)
        if len(used_gt) == len(gt_boxes):
            break
    return quality


def ranking_scores(
    formula: str,
    current: np.ndarray,
    class_score: np.ndarray,
    centerness: np.ndarray,
    boxes: np.ndarray,
    gt_boxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    oracle_quality = np.zeros(len(boxes), dtype=np.float64)
    if formula == "current":
        score = current
    elif formula == "class":
        score = class_score
    elif formula == "centerness":
        score = centerness
    elif formula in {"oracle_iou", "oracle_iou_x_current"}:
        oracle_quality = strict_one_to_one_oracle_quality(boxes, gt_boxes)
        if formula == "oracle_iou":
            score = oracle_quality
        else:
            score = oracle_quality * current
        score = score + current * 1e-7
    else:
        raise ValueError(f"Unknown formula: {formula}")
    return score, oracle_quality


def top_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    tensor = torch.as_tensor(scores, dtype=torch.float32)
    if len(scores) <= topk:
        return torch.argsort(tensor, descending=True).cpu().numpy()
    return torch.topk(tensor, k=topk).indices.cpu().numpy()


def process_variant(
    boxes: np.ndarray,
    current: np.ndarray,
    class_score: np.ndarray,
    centerness: np.ndarray,
    flat_indices: np.ndarray,
    gt_boxes: np.ndarray,
    formula: str,
    mode: str,
    topk: int,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alt_all, _ = ranking_scores(formula, current, class_score, centerness, boxes, gt_boxes)
    selection_score = alt_all if mode in {"topk_only", "combined"} else current
    selected = top_indices(selection_score, topk)
    if mode in {"nms_only", "combined"}:
        nms_score, _ = ranking_scores(
            formula,
            current[selected],
            class_score[selected],
            centerness[selected],
            boxes[selected],
            gt_boxes,
        )
    else:
        nms_score = current[selected]
    kept_local = nms(
        torch.as_tensor(boxes[selected], dtype=torch.float32),
        torch.as_tensor(nms_score, dtype=torch.float32),
        nms_iou,
    ).cpu().numpy()
    kept = selected[kept_local]
    final_order = np.argsort(-current[kept], kind="stable")
    kept = kept[final_order]
    return selected, kept, current[kept], flat_indices[kept]


def canonical_variant(
    boxes: np.ndarray,
    current: np.ndarray,
    flat_indices: np.ndarray,
    topk: int,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = top_indices(current, topk)
    kept_local = nms(
        torch.as_tensor(boxes[selected], dtype=torch.float32),
        torch.as_tensor(current[selected], dtype=torch.float32),
        nms_iou,
    ).cpu().numpy()
    kept = selected[kept_local]
    return selected, kept, current[kept], flat_indices[kept]


def append_predictions(
    target: dict[str, dict[int, list[EvaluationPrediction]]],
    variant: str,
    image_id: int,
    boxes: np.ndarray,
    scores: np.ndarray,
) -> None:
    target[variant][image_id] = [
        EvaluationPrediction(image_id=image_id, box=box.tolist(), confidence=float(score))
        for box, score in zip(boxes, scores)
    ]


def flatten_predictions(predictions: dict[int, list[EvaluationPrediction]], image_ids: list[int]) -> list[EvaluationPrediction]:
    return [prediction for image_id in image_ids for prediction in predictions.get(image_id, [])]


def matched_annotation_ids(
    gt_raw: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[EvaluationPrediction]],
    image_ids: list[int],
    threshold: float,
    iou_threshold: float,
) -> set[int]:
    detected: set[int] = set()
    for image_id in image_ids:
        gts = gt_raw[image_id]
        pred_rows = [
            {"box": prediction.box, "confidence": prediction.confidence}
            for prediction in predictions.get(image_id, [])
            if prediction.confidence >= threshold
        ]
        matches, _, _ = greedy_match(gts, pred_rows, iou_threshold)
        for match in matches:
            detected.add(int(gts[match["gt_index"]]["annotation_id"]))
    return detected


def compare_coco_results(generated: list[dict[str, Any]], canonical: list[dict[str, Any]]) -> dict[str, Any]:
    if len(generated) != len(canonical):
        return {"reproduced": False, "reason": "prediction_count_mismatch", "generated": len(generated), "canonical": len(canonical)}
    id_mismatches = 0
    max_box_error = 0.0
    max_score_error = 0.0
    for left, right in zip(generated, canonical):
        if (int(left["image_id"]), int(left["category_id"])) != (int(right["image_id"]), int(right["category_id"])):
            id_mismatches += 1
        max_box_error = max(max_box_error, max(abs(float(a) - float(b)) for a, b in zip(left["bbox"], right["bbox"])))
        max_score_error = max(max_score_error, abs(float(left["score"]) - float(right["score"])))
    reproduced = id_mismatches == 0 and max_box_error <= 1e-5 and max_score_error <= 1e-6
    return {
        "reproduced": reproduced,
        "generated_predictions": len(generated),
        "canonical_predictions": len(canonical),
        "image_or_category_order_mismatches": id_mismatches,
        "maximum_box_coordinate_error": max_box_error,
        "maximum_score_error": max_score_error,
        "tolerances": {"box": 1e-5, "score": 1e-6},
    }


def select_budget_row(rows: list[dict[str, Any]], fp_budget: float) -> dict[str, Any]:
    eligible = [row for row in rows if row["false_positives_per_image"] <= fp_budget]
    if not eligible:
        raise RuntimeError(f"No threshold satisfies FP/image <= {fp_budget}")
    return max(
        eligible,
        key=lambda row: (
            row["overall"]["recall"],
            row["by_visibility"].get("20", {}).get("recall", 0.0),
            row["overall"]["precision"],
        ),
    )


def metric_row(
    variant: str,
    coco: dict[str, Any],
    fixed: dict[str, Any],
    budget: dict[str, Any],
    maximum: dict[str, Any],
    fp_budget: float,
) -> dict[str, Any]:
    return {
        "variant": variant,
        "AP": coco["AP"],
        "AP50": coco["AP50"],
        "AP75": coco["AP75"],
        "AR@100": coco["AR"],
        "maximum_recall_at_floor": maximum["overall"]["recall"],
        "maximum_recall_fp_per_image": maximum["false_positives_per_image"],
        "maximum_r20": maximum["by_visibility"].get("20", {}).get("recall", 0.0),
        "maximum_r40": maximum["by_visibility"].get("40", {}).get("recall", 0.0),
        "maximum_r70": maximum["by_visibility"].get("70", {}).get("recall", 0.0),
        "maximum_r100": maximum["by_visibility"].get("100", {}).get("recall", 0.0),
        "fixed_threshold": fixed["confidence_threshold"],
        "fixed_recall": fixed["overall"]["recall"],
        "fixed_precision": fixed["overall"]["precision"],
        "fixed_fp_per_image": fixed["false_positives_per_image"],
        "fixed_r20": fixed["by_visibility"].get("20", {}).get("recall", 0.0),
        "fixed_r40": fixed["by_visibility"].get("40", {}).get("recall", 0.0),
        "fixed_r70": fixed["by_visibility"].get("70", {}).get("recall", 0.0),
        "fixed_r100": fixed["by_visibility"].get("100", {}).get("recall", 0.0),
        "matched_budget": fp_budget,
        "matched_threshold_exploratory": budget["confidence_threshold"],
        "matched_recall": budget["overall"]["recall"],
        "matched_precision": budget["overall"]["precision"],
        "matched_fp_per_image": budget["false_positives_per_image"],
        "matched_r20": budget["by_visibility"].get("20", {}).get("recall", 0.0),
        "matched_r40": budget["by_visibility"].get("40", {}).get("recall", 0.0),
        "matched_r70": budget["by_visibility"].get("70", {}).get("recall", 0.0),
        "matched_r100": budget["by_visibility"].get("100", {}).get("recall", 0.0),
    }


def make_plots(path: Path, sweeps: dict[str, list[dict[str, Any]]], selected_variants: list[str]) -> None:
    for metric, file_name, ylabel in (
        ("overall", "recall_vs_fp_per_image.png", "Recall @ IoU 0.50"),
        ("20", "r20_vs_fp_per_image.png", "Recall @ 20% visibility"),
    ):
        fig, ax = plt.subplots(figsize=(9, 6))
        for variant in selected_variants:
            rows = sorted(sweeps[variant], key=lambda row: row["false_positives_per_image"])
            x = [row["false_positives_per_image"] for row in rows]
            if metric == "overall":
                y = [row["overall"]["recall"] for row in rows]
            else:
                y = [row["by_visibility"].get(metric, {}).get("recall", 0.0) for row in rows]
            ax.plot(x, y, marker=".", linewidth=1.5, label=variant)
        ax.axvline(0.425, color="black", linestyle="--", linewidth=1, label="FP/image 0.425")
        ax.set_xlim(0, 1.25)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("False positives per image")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path / file_name, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_ids, gt_eval_by_image, flat_gt, gt_raw = load_ground_truth(args.root, args.manifest)
    image_id_set = set(image_ids)
    known_rows = [
        row for row in read_csv(args.localization_diagnostics)
        if float(row["pre_nms_best_iou"]) >= args.match_iou
    ]
    known_by_image: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in known_rows:
        known_by_image[int(row["image_id"])].append(row)

    variant_names = ["canonical_current"] + [f"{formula}_{mode}" for formula in FORMULAS for mode in MODES]
    predictions: dict[str, dict[int, list[EvaluationPrediction]]] = {
        name: {} for name in variant_names
    }
    ceiling_predictions: dict[str, dict[int, list[EvaluationPrediction]]] = {
        "strict_oracle_iou_ceiling": {},
        "strict_oracle_iou_x_current_ceiling": {},
    }
    recovery_rows: list[dict[str, Any]] = []
    candidate_count = 0

    with gzip.open(args.candidate_dump, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_id = int(row["image_id"])
            if image_id not in image_id_set:
                continue
            raw = row["candidates"]
            boxes = np.asarray([item["box"] for item in raw], dtype=np.float64)
            class_score = np.asarray([item["class_score"] for item in raw], dtype=np.float64)
            centerness = np.asarray([item["centerness_score"] for item in raw], dtype=np.float64)
            current = np.sqrt(class_score * centerness)
            flat_indices = np.asarray([item["flat_index"] for item in raw], dtype=np.int64)
            keep_floor = current >= args.candidate_floor
            boxes = boxes[keep_floor]
            class_score = class_score[keep_floor]
            centerness = centerness[keep_floor]
            current = current[keep_floor]
            flat_indices = flat_indices[keep_floor]
            candidate_count += len(boxes)
            gt_boxes = np.asarray([gt["box"] for gt in gt_raw[image_id]], dtype=np.float64)

            top, kept, scores, _ = canonical_variant(boxes, current, flat_indices, args.topk, args.nms_iou)
            append_predictions(predictions, "canonical_current", image_id, boxes[kept], scores)
            image_stage: dict[str, tuple[np.ndarray, np.ndarray]] = {"canonical_current": (top, kept)}

            for formula in FORMULAS:
                for mode in MODES:
                    variant = f"{formula}_{mode}"
                    selected, final, final_scores, _ = process_variant(
                        boxes, current, class_score, centerness, flat_indices,
                        gt_boxes, formula, mode, args.topk, args.nms_iou,
                    )
                    append_predictions(predictions, variant, image_id, boxes[final], final_scores)
                    image_stage[variant] = (selected, final)

            strict_quality = strict_one_to_one_oracle_quality(boxes, gt_boxes)
            assigned = np.flatnonzero(strict_quality > 0.0)
            for ceiling_name, ceiling_score in (
                ("strict_oracle_iou_ceiling", strict_quality),
                ("strict_oracle_iou_x_current_ceiling", strict_quality * current),
            ):
                ceiling_selected = assigned[top_indices(ceiling_score[assigned], args.topk)] if len(assigned) else assigned
                ceiling_kept_local = nms(
                    torch.as_tensor(boxes[ceiling_selected], dtype=torch.float32),
                    torch.as_tensor(ceiling_score[ceiling_selected], dtype=torch.float32),
                    args.nms_iou,
                ).cpu().numpy() if len(ceiling_selected) else np.asarray([], dtype=np.int64)
                ceiling_kept = ceiling_selected[ceiling_kept_local]
                append_predictions(
                    ceiling_predictions,
                    ceiling_name,
                    image_id,
                    boxes[ceiling_kept],
                    ceiling_score[ceiling_kept],
                )

            if image_id in known_by_image:
                matrix = pairwise_iou(boxes, gt_boxes)
                gt_index_by_ann = {int(gt["annotation_id"]): index for index, gt in enumerate(gt_raw[image_id])}
                for known in known_by_image[image_id]:
                    annotation_id = int(known["annotation_id"])
                    gt_index = gt_index_by_ann[annotation_id]
                    raw_best_index = int(np.argmax(matrix[:, gt_index]))
                    for variant, (selected, final) in image_stage.items():
                        final_best = float(matrix[final, gt_index].max()) if len(final) else 0.0
                        recovery_rows.append(
                            {
                                "image_id": image_id,
                                "annotation_id": annotation_id,
                                "visibility": int(known["visibility"]),
                                "variant": variant,
                                "best_raw_iou": float(matrix[raw_best_index, gt_index]),
                                "best_raw_in_top300": bool(np.any(selected == raw_best_index)),
                                "best_raw_survived_nms": bool(np.any(final == raw_best_index)),
                                "final_best_iou": final_best,
                                "final_has_iou_ge_0.50": final_best >= args.match_iou,
                            }
                        )

    canonical_generated = predictions_to_coco_results(flatten_predictions(predictions["canonical_current"], image_ids))
    reproducibility = compare_coco_results(canonical_generated, read_json(args.canonical_predictions))
    reproducibility.update(
        {
            "candidate_floor": args.candidate_floor,
            "topk_before_nms": args.topk,
            "nms_iou": args.nms_iou,
            "raw_candidates_after_floor": candidate_count,
            "stage_order": ["combined_score_floor", "topk", "nms", "evaluation_threshold"],
        }
    )
    (args.output_dir / "canonical_reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8"
    )
    if not reproducibility["reproduced"]:
        raise RuntimeError(f"Material canonical reproduction mismatch: {reproducibility}")

    thresholds = [0.001] + [round(value, 3) for value in np.arange(0.005, 0.951, 0.005)]
    summary_rows: list[dict[str, Any]] = []
    sweep_rows_csv: list[dict[str, Any]] = []
    sweeps: dict[str, list[dict[str, Any]]] = {}
    budget_detection_ids: dict[str, set[int]] = {}
    for variant in variant_names:
        flat_predictions = flatten_predictions(predictions[variant], image_ids)
        coco = evaluate_coco_style(flat_gt, flat_predictions, image_ids)
        sweep = evaluate_confidence_sweep(
            gt_eval_by_image, predictions[variant], image_ids, thresholds,
            iou_threshold=args.match_iou, fp_budgets=[args.fp_budget],
        )
        sweeps[variant] = sweep["rows"]
        budget = select_budget_row(sweep["rows"], args.fp_budget)
        fixed = min(sweep["rows"], key=lambda row: abs(row["confidence_threshold"] - args.fixed_threshold))
        maximum = sweep["rows"][0]
        summary_rows.append(metric_row(variant, coco, fixed, budget, maximum, args.fp_budget))
        budget_detection_ids[variant] = matched_annotation_ids(
            gt_raw, predictions[variant], image_ids,
            float(budget["confidence_threshold"]), args.match_iou,
        )
        for sweep_row in sweep["rows"]:
            sweep_rows_csv.append(
                {
                    "variant": variant,
                    "confidence_threshold": sweep_row["confidence_threshold"],
                    "recall": sweep_row["overall"]["recall"],
                    "precision": sweep_row["overall"]["precision"],
                    "fp_per_image": sweep_row["false_positives_per_image"],
                    "r20": sweep_row["by_visibility"].get("20", {}).get("recall", 0.0),
                    "r40": sweep_row["by_visibility"].get("40", {}).get("recall", 0.0),
                    "r70": sweep_row["by_visibility"].get("70", {}).get("recall", 0.0),
                    "r100": sweep_row["by_visibility"].get("100", {}).get("recall", 0.0),
                }
            )

    for row in recovery_rows:
        row["detected_at_matched_fp_budget"] = int(row["annotation_id"]) in budget_detection_ids[row["variant"]]
    write_csv(args.output_dir / "candidate_recovery_per_gt.csv", recovery_rows)

    recovery_summary: list[dict[str, Any]] = []
    for variant in variant_names:
        rows = [row for row in recovery_rows if row["variant"] == variant]
        recovery_summary.append(
            {
                "variant": variant,
                "known_recoverable_cases": len(rows),
                "best_raw_retained_by_top300": sum(bool(row["best_raw_in_top300"]) for row in rows),
                "best_raw_survived_nms": sum(bool(row["best_raw_survived_nms"]) for row in rows),
                "final_has_any_iou_ge_0.50_at_floor": sum(bool(row["final_has_iou_ge_0.50"]) for row in rows),
                "detected_at_matched_fp_budget": sum(bool(row["detected_at_matched_fp_budget"]) for row in rows),
            }
        )
    write_csv(args.output_dir / "candidate_recovery_summary.csv", recovery_summary)
    recovery_by_visibility: list[dict[str, Any]] = []
    for variant in variant_names:
        for visibility in (20, 40, 70, 100):
            rows = [
                row for row in recovery_rows
                if row["variant"] == variant and int(row["visibility"]) == visibility
            ]
            recovery_by_visibility.append(
                {
                    "variant": variant,
                    "visibility": visibility,
                    "known_recoverable_cases": len(rows),
                    "best_raw_retained_by_top300": sum(bool(row["best_raw_in_top300"]) for row in rows),
                    "best_raw_survived_nms": sum(bool(row["best_raw_survived_nms"]) for row in rows),
                    "final_has_any_iou_ge_0.50_at_floor": sum(bool(row["final_has_iou_ge_0.50"]) for row in rows),
                    "detected_at_matched_fp_budget": sum(bool(row["detected_at_matched_fp_budget"]) for row in rows),
                }
            )
    write_csv(args.output_dir / "candidate_recovery_by_visibility.csv", recovery_by_visibility)
    write_csv(args.output_dir / "ranking_stage_comparison.csv", summary_rows)
    write_csv(args.output_dir / "recall_fp_sweeps.csv", sweep_rows_csv)

    ceiling_rows: list[dict[str, Any]] = []
    for variant, by_image in ceiling_predictions.items():
        flat_predictions = flatten_predictions(by_image, image_ids)
        coco = evaluate_coco_style(flat_gt, flat_predictions, image_ids)
        ceiling_thresholds = [0.001] + [round(value, 3) for value in np.arange(0.05, 0.951, 0.05)]
        sweep = evaluate_confidence_sweep(
            gt_eval_by_image, by_image, image_ids, ceiling_thresholds,
            iou_threshold=args.match_iou, fp_budgets=[args.fp_budget],
        )
        budget = select_budget_row(sweep["rows"], args.fp_budget)
        maximum = sweep["rows"][0]
        ceiling_rows.append(
            {
                "variant": variant,
                "non_deployable_gt_informed": True,
                "assigned_predictions": len(flat_predictions),
                "AP": coco["AP"],
                "AP50": coco["AP50"],
                "AP75": coco["AP75"],
                "AR@100": coco["AR"],
                "maximum_recall_at_floor": maximum["overall"]["recall"],
                "maximum_fp_per_image": maximum["false_positives_per_image"],
                "matched_threshold_exploratory": budget["confidence_threshold"],
                "matched_recall": budget["overall"]["recall"],
                "matched_precision": budget["overall"]["precision"],
                "matched_fp_per_image": budget["false_positives_per_image"],
                "matched_r20": budget["by_visibility"].get("20", {}).get("recall", 0.0),
                "matched_r40": budget["by_visibility"].get("40", {}).get("recall", 0.0),
                "matched_r70": budget["by_visibility"].get("70", {}).get("recall", 0.0),
                "matched_r100": budget["by_visibility"].get("100", {}).get("recall", 0.0),
            }
        )
    write_csv(args.output_dir / "strict_oracle_ceiling.csv", ceiling_rows)

    selected_plot_variants = [
        "canonical_current",
        "class_combined",
        "centerness_combined",
        "oracle_iou_topk_only",
        "oracle_iou_nms_only",
        "oracle_iou_combined",
        "oracle_iou_x_current_combined",
    ]
    make_plots(args.output_dir, sweeps, selected_plot_variants)

    summary = {
        "scope": {
            "images": len(image_ids),
            "ground_truth": len(flat_gt),
            "known_recoverable_localization_cases": len(known_rows),
            "no_inference_or_training": True,
        },
        "postprocessing": reproducibility,
        "oracle_definition": {
            "assignment": "Greedy descending-IoU bipartite assignment; at most one candidate per GT and one GT per candidate.",
            "background": "Unassigned candidates receive oracle quality zero.",
            "stage_isolation": "Alternative quality changes top-k and/or NMS ordering; original sqrt(class*centerness) remains final confidence.",
            "strict_ceiling": "Only one-to-one assigned candidates are retained; GT-derived quality is also the final score. This is non-deployable.",
        },
        "ranking_stage_results": summary_rows,
        "candidate_recovery": recovery_summary,
        "strict_oracle_ceiling": ceiling_rows,
        "limitations": [
            "All oracle variants use val600 ground truth and are non-deployable upper bounds.",
            "Matched-FP thresholds selected on val600 are exploratory, not independently validated.",
            "Stage-isolated variants retain the original confidence for final filtering, isolating proposal selection and NMS ordering.",
            "The strict ceiling excludes arbitrary background by construction and is not a realistic precision estimate.",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
