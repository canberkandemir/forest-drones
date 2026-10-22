"""Fuse saved ForestPersons COCO detections without running model inference.

This script is intentionally prediction-level only. It consumes existing COCO
result JSON files, applies simple deterministic fusion baselines, and evaluates
them with the same ForestPersons evaluation utilities used elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from forestpersons import ForestPersonsDataset
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_pycocotools,
    predictions_to_coco_results,
)


def parse_float_list(values: str) -> list[float]:
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--yolo-predictions", type=Path, required=True)
    parser.add_argument("--lora-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--wbf-yolo-weight", type=float, default=1.0)
    parser.add_argument("--wbf-lora-weight", type=float, default=1.0)
    parser.add_argument(
        "--conf-thresholds",
        default="0.001,0.005,0.01,0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.55,0.6,0.62,0.64,0.66,0.68,0.7,0.75,0.8,0.9",
    )
    parser.add_argument("--fp-budgets", default="0.25,0.425,0.5,1.0")
    parser.add_argument("--custom-iou", type=float, default=0.50)
    parser.add_argument("--max-visualizations", type=int, default=8)
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def load_coco_predictions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected COCO result list: {path}")
    return data


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def box_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def convert_predictions(
    raw: list[dict[str, Any]],
    image_ids: set[int | str],
    source: str,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for item in raw:
        image_id = item.get("image_id")
        if image_id not in image_ids:
            continue
        if int(item.get("category_id", 1)) != 1:
            continue
        converted.append(
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": xywh_to_xyxy(item["bbox"]),
                "score": float(item["score"]),
                "source": source,
            }
        )
    return converted


def to_eval_prediction(item: dict[str, Any]) -> EvaluationPrediction:
    return EvaluationPrediction(
        image_id=item["image_id"],
        box=item["bbox"],
        confidence=float(item["score"]),
        category_id=1,
    )


def nms_predictions(predictions: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    by_image: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        by_image[item["image_id"]].append(item)
    kept: list[dict[str, Any]] = []
    for image_predictions in by_image.values():
        ordered = sorted(image_predictions, key=lambda item: item["score"], reverse=True)
        image_kept: list[dict[str, Any]] = []
        for candidate in ordered:
            if all(box_iou(candidate["bbox"], existing["bbox"]) < iou_threshold for existing in image_kept):
                image_kept.append(candidate)
        kept.extend(image_kept)
    return kept


def weighted_box_fusion(
    yolo: list[dict[str, Any]],
    lora: list[dict[str, Any]],
    iou_threshold: float,
    yolo_weight: float,
    lora_weight: float,
) -> list[dict[str, Any]]:
    """Minimal two-source weighted box fusion with source-weighted coordinates.

    Groups are formed greedily by descending score within each image. Fused score
    is the maximum member score to avoid artificially inflating confidence by
    summing two differently calibrated model scores.
    """

    source_weight = {"yolo": float(yolo_weight), "lora": float(lora_weight)}
    by_image: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for item in [*yolo, *lora]:
        by_image[item["image_id"]].append(item)
    fused: list[dict[str, Any]] = []
    for image_id, image_predictions in by_image.items():
        groups: list[list[dict[str, Any]]] = []
        for candidate in sorted(image_predictions, key=lambda item: item["score"], reverse=True):
            best_index = -1
            best_iou = 0.0
            for index, group in enumerate(groups):
                reference = group[0]["bbox"]
                iou = box_iou(candidate["bbox"], reference)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index >= 0 and best_iou >= iou_threshold:
                groups[best_index].append(candidate)
            else:
                groups.append([candidate])
        for group in groups:
            weighted = [0.0, 0.0, 0.0, 0.0]
            total_weight = 0.0
            sources = sorted({item["source"] for item in group})
            for item in group:
                weight = max(1e-8, item["score"]) * source_weight[item["source"]]
                total_weight += weight
                for index in range(4):
                    weighted[index] += item["bbox"][index] * weight
            box = [value / total_weight for value in weighted]
            fused.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": box,
                    "score": max(float(item["score"]) for item in group),
                    "source": "+".join(sources) if len(sources) > 1 else sources[0],
                    "members": len(group),
                }
            )
    return fused


def evaluate_method(
    name: str,
    predictions: list[dict[str, Any]],
    ground_truth_by_image: dict[int | str, list[Any]],
    ground_truth: list[Any],
    image_ids: list[int | str],
    thresholds: list[float],
    fp_budgets: list[float],
    iou_threshold: float,
) -> dict[str, Any]:
    eval_predictions = [to_eval_prediction(item) for item in predictions]
    return {
        "name": name,
        "raw_predictions": len(eval_predictions),
        "source_counts": dict(sorted({source: sum(1 for item in predictions if item.get("source") == source) for source in {item.get("source") for item in predictions}}.items())),
        "coco_style": evaluate_coco_style(ground_truth, eval_predictions, image_ids),
        "pycocotools": evaluate_pycocotools(ground_truth, eval_predictions, image_ids),
        "custom_confidence_sweep": evaluate_confidence_sweep(
            ground_truth_by_image,
            defaultdict(list, {image_id: [] for image_id in image_ids})
            | _predictions_by_image(eval_predictions, image_ids),
            image_ids,
            thresholds,
            iou_threshold=iou_threshold,
            fp_budgets=fp_budgets,
        ),
    }


def _predictions_by_image(
    predictions: list[EvaluationPrediction],
    image_ids: list[int | str],
) -> dict[int | str, list[EvaluationPrediction]]:
    by_image: dict[int | str, list[EvaluationPrediction]] = {image_id: [] for image_id in image_ids}
    for prediction in predictions:
        if prediction.image_id in by_image:
            by_image[prediction.image_id].append(prediction)
    return by_image


def write_coco_results(path: Path, predictions: list[dict[str, Any]]) -> None:
    eval_predictions = [to_eval_prediction(item) for item in predictions]
    path.write_text(json.dumps(predictions_to_coco_results(eval_predictions), indent=2), encoding="utf-8")


def summarize_budget(method: dict[str, Any], budget: str) -> dict[str, Any]:
    return method["custom_confidence_sweep"]["matched_false_positive_budgets"][budget]


def greedy_matched_gt(
    predictions: list[dict[str, Any]],
    ground_truth_by_image: dict[int | str, list[Any]],
    confidence: float,
    iou_threshold: float,
) -> set[int | str | None]:
    matched: set[int | str | None] = set()
    for image_id, image_predictions in _raw_predictions_by_image(predictions).items():
        used: set[int | str | None] = set()
        for prediction in sorted(
            [item for item in image_predictions if item["score"] >= confidence],
            key=lambda item: item["score"],
            reverse=True,
        ):
            best_iou = 0.0
            best_gt = None
            for gt in ground_truth_by_image.get(image_id, []):
                if gt.annotation_id in used:
                    continue
                iou = box_iou(prediction["bbox"], gt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt
            if best_gt is not None and best_iou >= iou_threshold:
                used.add(best_gt.annotation_id)
                matched.add(best_gt.annotation_id)
    return matched


def _raw_predictions_by_image(predictions: list[dict[str, Any]]) -> dict[int | str, list[dict[str, Any]]]:
    by_image: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        by_image[prediction["image_id"]].append(prediction)
    return by_image


def visualize_examples(
    output_dir: Path,
    samples_by_image: dict[int | str, Any],
    ground_truth_by_image: dict[int | str, list[Any]],
    yolo: list[dict[str, Any]],
    lora: list[dict[str, Any]],
    union: list[dict[str, Any]],
    example_ids: list[int | str | None],
    yolo_conf: float,
    lora_conf: float,
    union_conf: float,
    max_images: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_by = _raw_predictions_by_image(yolo)
    lora_by = _raw_predictions_by_image(lora)
    union_by = _raw_predictions_by_image(union)
    gt_to_image = {
        gt.annotation_id: image_id
        for image_id, gts in ground_truth_by_image.items()
        for gt in gts
    }
    written: list[str] = []
    for annotation_id in example_ids[:max_images]:
        image_id = gt_to_image.get(annotation_id)
        if image_id is None:
            continue
        sample = samples_by_image[image_id]
        image = Image.open(sample.image_path).convert("RGB")
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(f"Fusion example annotation {annotation_id}: green=GT, red=YOLO, blue=LoRA, magenta=fused", fontsize=9)
        for gt in ground_truth_by_image[image_id]:
            color = "lime" if gt.annotation_id == annotation_id else "white"
            width = 3 if gt.annotation_id == annotation_id else 1
            x1, y1, x2, y2 = gt.box
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=width, edgecolor=color, facecolor="none", alpha=0.9))
        for source, predictions, conf, color, y_offset in [
            ("Y", yolo_by.get(image_id, []), yolo_conf, "red", 12),
            ("L", lora_by.get(image_id, []), lora_conf, "deepskyblue", 24),
            ("F", union_by.get(image_id, []), union_conf, "magenta", 36),
        ]:
            for item in sorted([p for p in predictions if p["score"] >= conf], key=lambda p: p["score"], reverse=True)[:12]:
                x1, y1, x2, y2 = item["bbox"]
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.6, edgecolor=color, facecolor="none", alpha=0.8))
                ax.text(x1, y1 + y_offset, f"{source} {item['score']:.2f}", color=color, fontsize=7, bbox={"facecolor": "black", "alpha": 0.45, "pad": 1})
        path = output_dir / f"fusion_example_ann_{annotation_id}.jpg"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    return written


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(args.conf_thresholds)
    fp_budgets = parse_float_list(args.fp_budgets)

    dataset = ForestPersonsDataset(args.root, split=args.split)
    manifest = load_manifest(args.manifest)
    samples_by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [samples_by_file[item["file_name"]] for item in manifest]
    image_ids = [sample.image_id for sample in samples]
    image_id_set = set(image_ids)
    samples_by_image = {sample.image_id: sample for sample in samples}
    ground_truth_by_image = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    ground_truth = [
        gt for image_ground_truth in ground_truth_by_image.values() for gt in image_ground_truth
    ]

    yolo = convert_predictions(load_coco_predictions(args.yolo_predictions), image_id_set, "yolo")
    lora = convert_predictions(load_coco_predictions(args.lora_predictions), image_id_set, "lora")
    union_raw = [*yolo, *lora]
    union_nms = nms_predictions(union_raw, args.nms_iou)
    wbf = weighted_box_fusion(
        yolo,
        lora,
        iou_threshold=args.wbf_iou,
        yolo_weight=args.wbf_yolo_weight,
        lora_weight=args.wbf_lora_weight,
    )

    methods = {
        "yolo": yolo,
        "lora": lora,
        "union_nms": union_nms,
        "weighted_box_fusion": wbf,
    }
    summaries: dict[str, Any] = {}
    for method_name, predictions in methods.items():
        method_dir = args.output_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        write_coco_results(method_dir / "coco_results.json", predictions)
        summary = evaluate_method(
            method_name,
            predictions,
            ground_truth_by_image,
            ground_truth,
            image_ids,
            thresholds,
            fp_budgets,
            iou_threshold=args.custom_iou,
        )
        summaries[method_name] = summary
        (method_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (method_dir / "confidence_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["confidence", "recall", "precision", "fp_per_image", "r20", "r40", "r70", "r100"])
            for row in summary["custom_confidence_sweep"]["rows"]:
                byv = row["by_visibility"]
                writer.writerow([
                    row["confidence_threshold"],
                    row["overall"]["recall"],
                    row["overall"]["precision"],
                    row["false_positives_per_image"],
                    byv["20"]["recall"],
                    byv["40"]["recall"],
                    byv["70"]["recall"],
                    byv["100"]["recall"],
                ])

    # Recover LoRA-only instances from the earlier primary operating points.
    yolo_primary = greedy_matched_gt(yolo, ground_truth_by_image, confidence=0.05, iou_threshold=args.custom_iou)
    lora_primary = greedy_matched_gt(lora, ground_truth_by_image, confidence=0.60, iou_threshold=args.custom_iou)
    lora_only = sorted(lora_primary - yolo_primary)
    recovery: dict[str, Any] = {"lora_only_reference_count": len(lora_only), "by_method": {}}
    for method_name, predictions in methods.items():
        recovery["by_method"][method_name] = {}
        for budget in fp_budgets:
            row = summarize_budget(summaries[method_name], str(budget))
            matched = greedy_matched_gt(
                predictions,
                ground_truth_by_image,
                confidence=float(row["confidence_threshold"]),
                iou_threshold=args.custom_iou,
            )
            recovered = sorted(set(lora_only) & matched)
            recovery["by_method"][method_name][str(budget)] = {
                "confidence": row["confidence_threshold"],
                "recovered_lora_only": len(recovered),
                "recovered_annotation_ids": recovered,
                "false_positives": row["overall"]["false_positives"],
                "fp_per_image": row["false_positives_per_image"],
            }

    # Visualize examples where fusion at 0.425 recovers LoRA-only instances.
    union_budget = summarize_budget(summaries["union_nms"], "0.425")
    union_matched = greedy_matched_gt(
        union_nms,
        ground_truth_by_image,
        confidence=float(union_budget["confidence_threshold"]),
        iou_threshold=args.custom_iou,
    )
    example_ids = sorted(set(lora_only) & union_matched)
    visualizations = visualize_examples(
        args.output_dir / "visualizations",
        samples_by_image,
        ground_truth_by_image,
        yolo,
        lora,
        union_nms,
        example_ids,
        yolo_conf=0.05,
        lora_conf=0.60,
        union_conf=float(union_budget["confidence_threshold"]),
        max_images=args.max_visualizations,
    )

    comparison_csv = args.output_dir / "fusion_budget_comparison.csv"
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "budget", "confidence", "AP", "AP50", "AP75", "AR@100", "recall", "precision", "fp_per_image", "r20", "r40", "r70", "r100", "lora_only_recovered", "false_positives"])
        for method_name, summary in summaries.items():
            py = summary["pycocotools"]
            for budget in fp_budgets:
                row = summarize_budget(summary, str(budget))
                byv = row["by_visibility"]
                rec = recovery["by_method"][method_name][str(budget)]
                writer.writerow([
                    method_name,
                    budget,
                    row["confidence_threshold"],
                    py["AP"],
                    py["AP50"],
                    py["AP75"],
                    py["AR@100"],
                    row["overall"]["recall"],
                    row["overall"]["precision"],
                    row["false_positives_per_image"],
                    byv["20"]["recall"],
                    byv["40"]["recall"],
                    byv["70"]["recall"],
                    byv["100"]["recall"],
                    rec["recovered_lora_only"],
                    rec["false_positives"],
                ])

    report_lines = [
        "# YOLO11s + DINOv3 LoRA Saved-Prediction Fusion - ForestPersons val600",
        "",
        "This is an exploratory prediction-level fusion analysis using saved predictions only. No model inference, training, or dataset download was run.",
        "",
        "## Saved Prediction Availability",
        "",
        f"- YOLO predictions: {len(yolo)} boxes, score range {min(p['score'] for p in yolo):.4f} to {max(p['score'] for p in yolo):.4f}.",
        "- YOLO saved predictions include candidates below confidence 0.05, so low-confidence candidate recovery can be studied from existing artifacts.",
        f"- DINOv3 LoRA predictions: {len(lora)} boxes, score range {min(p['score'] for p in lora):.4f} to {max(p['score'] for p in lora):.4f}.",
        "",
        "## Fusion Methods",
        "",
        f"- `yolo`: saved YOLO11s predictions only.",
        f"- `lora`: saved DINOv3 LoRA predictions only.",
        f"- `union_nms`: raw union of YOLO and LoRA predictions followed by class-aware NMS at IoU {args.nms_iou}.",
        f"- `weighted_box_fusion`: greedy two-source weighted box fusion at IoU {args.wbf_iou}; fused score is max member score, not a learned calibrated score.",
        "",
        "No ground-truth labels were used to choose individual predictions. Ground truth is used only for evaluation and matched-FP threshold selection.",
        "",
        "## Matched FP/Image Budgets",
        "",
    ]
    for budget in fp_budgets:
        report_lines.extend([
            f"### FP/image <= {budget}",
            "",
            "| Method | Conf | AP | AP50 | AP75 | AR@100 | Recall | Precision | FP/img | R@20 | R@40 | R@70 | R@100 | LoRA-only recovered |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for method_name, summary in summaries.items():
            py = summary["pycocotools"]
            row = summarize_budget(summary, str(budget))
            byv = row["by_visibility"]
            rec = recovery["by_method"][method_name][str(budget)]
            report_lines.append(
                f"| {method_name} | {row['confidence_threshold']:.3g} | {py['AP']:.4f} | {py['AP50']:.4f} | {py['AP75']:.4f} | {py['AR@100']:.4f} | "
                f"{row['overall']['recall']:.4f} | {row['overall']['precision']:.4f} | {row['false_positives_per_image']:.4f} | "
                f"{byv['20']['recall']:.4f} | {byv['40']['recall']:.4f} | {byv['70']['recall']:.4f} | {byv['100']['recall']:.4f} | "
                f"{rec['recovered_lora_only']} |"
            )
        report_lines.append("")
    report_lines.extend([
        "## LoRA-Only Recovery",
        "",
        f"Reference LoRA-only set: {len(lora_only)} GT instances detected by LoRA at confidence 0.60 but missed by YOLO at confidence 0.05.",
        "",
        "See `fusion_budget_comparison.csv` for exact recovery counts and false positives at every budget.",
        "",
        "## Findings",
        "",
        "- The saved YOLO artifact is sufficient for low-confidence fusion analysis because it contains boxes below 0.05.",
        "- Simple union/NMS can recover some complementary DINOv3 detections, but score-scale mismatch is a major confounder: YOLO and DINOv3 confidences are not calibrated to the same probability scale.",
        "- Weighted box fusion is technically possible, but it should be treated as exploratory because it fuses boxes from differently calibrated detectors without a calibration set.",
        "- Because this analysis is performed on the same val600 subset used to inspect failures, any threshold or fusion parameter observations are exploratory and should not be claimed as unbiased generalization.",
        "",
        "## Outputs",
        "",
        f"- Comparison CSV: `{display_path(comparison_csv)}`",
        f"- Visualizations: `{display_path(args.output_dir / 'visualizations')}/`",
    ])
    report = args.output_dir / "fusion_report.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (args.output_dir / "fusion_summary.json").write_text(
        json.dumps(
            {
                "settings": vars(args) | {"root": str(args.root), "manifest": str(args.manifest), "yolo_predictions": str(args.yolo_predictions), "lora_predictions": str(args.lora_predictions), "output_dir": str(args.output_dir)},
                "lora_only_reference": lora_only,
                "recovery": recovery,
                "visualizations": visualizations,
                "methods": {
                    method_name: {
                        "raw_predictions": summary["raw_predictions"],
                        "pycocotools": summary["pycocotools"],
                        "matched_false_positive_budgets": summary["custom_confidence_sweep"]["matched_false_positive_budgets"],
                    }
                    for method_name, summary in summaries.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)
    print(comparison_csv)


if __name__ == "__main__":
    main()
