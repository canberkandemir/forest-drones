"""Evaluate pretrained YOLO11 on a local ForestPersons validation subset."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

from forestpersons import ForestPersonsDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolo11_baseline"))
    parser.add_argument("--max-visualizations", type=int, default=16)
    return parser.parse_args()


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def greedy_match(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            iou = box_iou(gt["box"], pred["box"])
            if iou >= iou_threshold:
                candidates.append((iou, gt_index, pred_index))
    candidates.sort(reverse=True, key=lambda item: item[0])

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, gt_index, pred_index in candidates:
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matches.append(
            {
                "gt_index": gt_index,
                "prediction_index": pred_index,
                "iou": iou,
                "visibility_level": ground_truth[gt_index]["visibility_level"],
                "confidence": predictions[pred_index]["confidence"],
            }
        )
    return matches, matched_gt, matched_pred


def safe_div(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_group(stats: dict[str, int]) -> dict[str, float | int]:
    true_positives = stats["true_positives"]
    false_positives = stats["false_positives"]
    false_negatives = stats["false_negatives"]
    ground_truth = true_positives + false_negatives
    return {
        "ground_truth": ground_truth,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "recall": safe_div(true_positives, ground_truth),
        "precision": safe_div(true_positives, true_positives + false_positives),
    }


def draw_comparison(
    image_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    output_path: Path,
) -> None:
    matched_gt = {match["gt_index"] for match in matches}
    matched_pred = {match["prediction_index"] for match in matches}

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(image)
    ax.set_title(image_path.name, fontsize=9)
    ax.axis("off")

    for index, gt in enumerate(ground_truth):
        x1, y1, x2, y2 = gt["box"]
        color = "lime" if index in matched_gt else "yellow"
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2.0,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 4),
            f"GT v={gt['visibility_level']} {gt['pose']}",
            color="black",
            fontsize=7,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 1},
        )

    for index, pred in enumerate(predictions):
        x1, y1, x2, y2 = pred["box"]
        color = "cyan" if index in matched_pred else "red"
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1.6,
            edgecolor=color,
            linestyle="--",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            min(image.height - 4, y2 + 4),
            f"YOLO {pred['confidence']:.2f}",
            color="black",
            fontsize=7,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 1},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = ForestPersonsDataset(args.root, split=args.split)
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest_data = json.load(handle)
    manifest = (
        manifest_data["images"]
        if isinstance(manifest_data, dict) and "images" in manifest_data
        else manifest_data
    )

    manifest_files = [item["file_name"] for item in manifest]
    samples_by_file = {sample.file_name: sample for sample in dataset.samples}
    selected_samples = [samples_by_file[file_name] for file_name in manifest_files]
    missing = [sample.file_name for sample in selected_samples if not sample.image_path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images, e.g. {missing[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_dir / "comparisons"
    model = YOLO(args.model)

    rows: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    overall_stats = defaultdict(int)
    visibility_stats: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prediction_total = 0
    inference_seconds_total = 0.0

    for image_index, sample in enumerate(selected_samples):
        start = time.perf_counter()
        result = model.predict(
            source=str(sample.image_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        elapsed = time.perf_counter() - start
        inference_seconds_total += elapsed

        predictions: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue
                predictions.append(
                    {
                        "box": [float(value) for value in box.xyxy[0].cpu().tolist()],
                        "confidence": float(box.conf[0]),
                        "class_id": cls,
                    }
                )

        ground_truth: list[dict[str, Any]] = []
        for annotation in sample.annotations:
            metadata = annotation.get("metadata", {})
            visibility = metadata.get("visibility_level")
            if visibility is None:
                continue
            ground_truth.append(
                {
                    "annotation_id": annotation.get("id"),
                    "box": xywh_to_xyxy([float(value) for value in annotation["bbox"]]),
                    "visibility_level": int(visibility),
                    "pose": metadata.get("pose", "unknown"),
                }
            )

        matches, matched_gt, matched_pred = greedy_match(
            ground_truth, predictions, args.iou
        )
        prediction_total += len(predictions)

        overall_stats["true_positives"] += len(matches)
        overall_stats["false_positives"] += len(predictions) - len(matched_pred)
        overall_stats["false_negatives"] += len(ground_truth) - len(matched_gt)

        for gt_index, gt in enumerate(ground_truth):
            stats = visibility_stats[gt["visibility_level"]]
            if gt_index in matched_gt:
                stats["true_positives"] += 1
            else:
                stats["false_negatives"] += 1

        for pred_index in range(len(predictions)):
            if pred_index in matched_pred:
                continue
            if ground_truth:
                best_gt = max(
                    ground_truth, key=lambda gt: box_iou(gt["box"], predictions[pred_index]["box"])
                )
                visibility_stats[best_gt["visibility_level"]]["false_positives"] += 1

        for match in matches:
            row = {
                "file_name": sample.file_name,
                "image_id": sample.image_id,
                "annotation_id": ground_truth[match["gt_index"]]["annotation_id"],
                "visibility_level": match["visibility_level"],
                "pose": ground_truth[match["gt_index"]]["pose"],
                "iou": match["iou"],
                "confidence": match["confidence"],
            }
            rows.append(row)

        image_summaries.append(
            {
                "file_name": sample.file_name,
                "image_id": sample.image_id,
                "ground_truth": len(ground_truth),
                "predictions": len(predictions),
                "true_positives": len(matches),
                "false_positives": len(predictions) - len(matched_pred),
                "false_negatives": len(ground_truth) - len(matched_gt),
                "inference_seconds": elapsed,
            }
        )

        if image_index < args.max_visualizations:
            output_name = sample.file_name.replace("/", "__")
            draw_comparison(
                sample.image_path,
                ground_truth,
                predictions,
                matches,
                comparison_dir / output_name,
            )

    metrics = {
        "settings": {
            "model": str(args.model),
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "split": args.split,
            "manifest": str(args.manifest),
        },
        "images_evaluated": len(selected_samples),
        "predictions_person_class": prediction_total,
        "overall": summarize_group(overall_stats),
        "false_positives_per_image": safe_div(
            overall_stats["false_positives"], len(selected_samples)
        ),
        "inference": {
            "seconds_total": inference_seconds_total,
            "seconds_per_image": inference_seconds_total / len(selected_samples)
            if selected_samples
            else 0.0,
            "images_per_second": len(selected_samples) / inference_seconds_total
            if inference_seconds_total > 0
            else 0.0,
        },
        "by_visibility": {
            str(visibility): summarize_group(visibility_stats[visibility])
            for visibility in sorted(visibility_stats)
        },
        "image_summaries": image_summaries,
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "matches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_name",
                "image_id",
                "annotation_id",
                "visibility_level",
                "pose",
                "iou",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved matches: {csv_path}")
    print(f"Saved comparisons: {comparison_dir}")


if __name__ == "__main__":
    main()
