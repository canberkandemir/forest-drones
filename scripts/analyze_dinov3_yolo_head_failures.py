"""Analyze localization failures for the DINOv3 LoRA YOLO-style head."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, xywh_to_xyxy


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def annotation_visibility(annotation: dict[str, Any]) -> int | None:
    metadata = annotation.get("metadata", {})
    visibility = metadata.get("visibility_level")
    return int(visibility) if visibility is not None else None


def coco_result_to_xyxy(result: dict[str, Any]) -> list[float]:
    x, y, w, h = [float(value) for value in result["bbox"]]
    return [x, y, x + w, y + h]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
    }


def draw_example(
    image_path: Path,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title, fontsize=9)
    for gt in ground_truth:
        x1, y1, x2, y2 = gt["box"]
        ax.add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.0, edgecolor="lime", facecolor="none")
        )
        ax.text(
            x1,
            max(0, y1 - 4),
            f"GT v={gt['visibility']}",
            color="black",
            fontsize=7,
            bbox={"facecolor": "lime", "alpha": 0.8, "pad": 1},
        )
    for pred in predictions[:20]:
        x1, y1, x2, y2 = pred["box"]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.3,
                edgecolor="red",
                linestyle="--",
                facecolor="none",
            )
        )
        ax.text(
            x1,
            min(image.height - 4, y2 + 4),
            f"{pred['score']:.2f}",
            color="black",
            fontsize=7,
            bbox={"facecolor": "red", "alpha": 0.65, "pad": 1},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/eval_val600/coco_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dinov3_lora_yolo_head/train400_seed20261013_r4_qv_from_lora5k_e2/error_analysis"),
    )
    parser.add_argument("--conf-threshold", type=float, default=0.60)
    parser.add_argument("--max-examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ForestPersonsDataset(args.root, split="val")
    selected = load_manifest(args.manifest)
    samples_by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [samples_by_file[item["file_name"]] for item in selected]
    predictions_raw = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions_by_image: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for result in predictions_raw:
        predictions_by_image[result["image_id"]].append(
            {
                "box": coco_result_to_xyxy(result),
                "score": float(result["score"]),
            }
        )
    for image_predictions in predictions_by_image.values():
        image_predictions.sort(key=lambda item: item["score"], reverse=True)

    rows: list[dict[str, Any]] = []
    examples: list[tuple[float, Any, list[dict[str, Any]], list[dict[str, Any]], str]] = []
    for sample in samples:
        predictions = predictions_by_image.get(sample.image_id, [])
        for annotation in sample.annotations:
            visibility = annotation_visibility(annotation)
            if visibility is None:
                continue
            gt_box = xywh_to_xyxy([float(value) for value in annotation["bbox"]])
            gt_w = max(0.0, gt_box[2] - gt_box[0])
            gt_h = max(0.0, gt_box[3] - gt_box[1])
            best_prediction = None
            best_iou = 0.0
            for prediction in predictions:
                iou = box_iou(gt_box, prediction["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_prediction = prediction
            if best_prediction is None:
                row = {
                    "image_id": sample.image_id,
                    "file_name": sample.file_name,
                    "annotation_id": annotation.get("id"),
                    "visibility": visibility,
                    "gt_width": gt_w,
                    "gt_height": gt_h,
                    "gt_area": gt_w * gt_h,
                    "best_iou": 0.0,
                    "best_score": 0.0,
                    "pred_width_ratio": 0.0,
                    "pred_height_ratio": 0.0,
                    "pred_area_ratio": 0.0,
                    "center_error_norm": 0.0,
                    "detected_at_conf": False,
                }
            else:
                pred = best_prediction["box"]
                pred_w = max(0.0, pred[2] - pred[0])
                pred_h = max(0.0, pred[3] - pred[1])
                gt_cx = 0.5 * (gt_box[0] + gt_box[2])
                gt_cy = 0.5 * (gt_box[1] + gt_box[3])
                pred_cx = 0.5 * (pred[0] + pred[2])
                pred_cy = 0.5 * (pred[1] + pred[3])
                diag = max((gt_w * gt_w + gt_h * gt_h) ** 0.5, 1e-6)
                row = {
                    "image_id": sample.image_id,
                    "file_name": sample.file_name,
                    "annotation_id": annotation.get("id"),
                    "visibility": visibility,
                    "gt_width": gt_w,
                    "gt_height": gt_h,
                    "gt_area": gt_w * gt_h,
                    "best_iou": best_iou,
                    "best_score": float(best_prediction["score"]),
                    "pred_width_ratio": pred_w / max(gt_w, 1e-6),
                    "pred_height_ratio": pred_h / max(gt_h, 1e-6),
                    "pred_area_ratio": (pred_w * pred_h) / max(gt_w * gt_h, 1e-6),
                    "center_error_norm": (((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5) / diag,
                    "detected_at_conf": bool(best_iou >= 0.50 and best_prediction["score"] >= args.conf_threshold),
                }
            rows.append(row)

        gt_for_image = [
            {
                "box": xywh_to_xyxy([float(value) for value in annotation["bbox"]]),
                "visibility": annotation_visibility(annotation),
            }
            for annotation in sample.annotations
            if annotation_visibility(annotation) is not None
        ]
        high_conf_predictions = [prediction for prediction in predictions if prediction["score"] >= args.conf_threshold]
        if gt_for_image:
            image_best_iou = max(
                (max((box_iou(gt["box"], pred["box"]) for pred in predictions), default=0.0) for gt in gt_for_image),
                default=0.0,
            )
            examples.append((image_best_iou, sample, gt_for_image, high_conf_predictions, "high_conf"))
            if high_conf_predictions:
                examples.append((1.0 - image_best_iou, sample, gt_for_image, high_conf_predictions, "localization"))

    by_visibility: dict[str, dict[str, Any]] = {}
    for visibility in sorted({int(row["visibility"]) for row in rows}):
        subset = [row for row in rows if int(row["visibility"]) == visibility]
        by_visibility[str(visibility)] = {
            "count": len(subset),
            "best_iou": summarize([float(row["best_iou"]) for row in subset]),
            "pct_iou_ge_050": sum(float(row["best_iou"]) >= 0.50 for row in subset) / len(subset),
            "pct_iou_ge_075": sum(float(row["best_iou"]) >= 0.75 for row in subset) / len(subset),
            "pct_score_ge_conf_given_iou_ge_050": (
                sum(float(row["best_score"]) >= args.conf_threshold and float(row["best_iou"]) >= 0.50 for row in subset)
                / max(1, sum(float(row["best_iou"]) >= 0.50 for row in subset))
            ),
            "width_ratio": summarize([float(row["pred_width_ratio"]) for row in subset if float(row["best_iou"]) > 0]),
            "height_ratio": summarize([float(row["pred_height_ratio"]) for row in subset if float(row["best_iou"]) > 0]),
            "area_ratio": summarize([float(row["pred_area_ratio"]) for row in subset if float(row["best_iou"]) > 0]),
            "center_error_norm": summarize([float(row["center_error_norm"]) for row in subset if float(row["best_iou"]) > 0]),
        }

    overall = {
        "gt_count": len(rows),
        "best_iou": summarize([float(row["best_iou"]) for row in rows]),
        "pct_iou_ge_050": sum(float(row["best_iou"]) >= 0.50 for row in rows) / len(rows),
        "pct_iou_ge_075": sum(float(row["best_iou"]) >= 0.75 for row in rows) / len(rows),
        "pct_score_ge_conf_given_iou_ge_050": (
            sum(float(row["best_score"]) >= args.conf_threshold and float(row["best_iou"]) >= 0.50 for row in rows)
            / max(1, sum(float(row["best_iou"]) >= 0.50 for row in rows))
        ),
        "width_ratio": summarize([float(row["pred_width_ratio"]) for row in rows if float(row["best_iou"]) > 0]),
        "height_ratio": summarize([float(row["pred_height_ratio"]) for row in rows if float(row["best_iou"]) > 0]),
        "area_ratio": summarize([float(row["pred_area_ratio"]) for row in rows if float(row["best_iou"]) > 0]),
        "center_error_norm": summarize([float(row["center_error_norm"]) for row in rows if float(row["best_iou"]) > 0]),
    }

    with (args.output_dir / "gt_best_candidate_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "settings": {
            "manifest": str(args.manifest),
            "predictions": str(args.predictions),
            "confidence_threshold": args.conf_threshold,
            "note": "Analysis uses saved post-NMS COCO predictions generated with prediction_conf_min=0.001.",
        },
        "overall": overall,
        "by_visibility": by_visibility,
        "limitations": [
            "Saved predictions are post-NMS; this analysis cannot directly measure pre-NMS suppression.",
            "Saved predictions used topk=300 per image after confidence filtering; candidates below 0.001 are unavailable.",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    examples_sorted = sorted(examples, key=lambda item: item[0])[: args.max_examples]
    for index, (_score, sample, gt_for_image, predictions, label) in enumerate(examples_sorted, start=1):
        draw_example(
            sample.image_path,
            gt_for_image,
            predictions,
            args.output_dir / "examples" / f"{index:02d}_{label}_{sample.file_name.replace('/', '__')}.jpg",
            title=f"{sample.file_name} | {label}",
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
