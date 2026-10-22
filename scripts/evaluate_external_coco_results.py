"""Evaluate external COCO-format detector predictions on ForestPersons.

Use this for predictions exported by MMDetection, detrex, Detectron2, or other
detector frameworks. The script expects COCO result dictionaries with
`image_id`, `bbox` in xywh format, `score`, and optionally `category_id`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from forestpersons import ForestPersonsDataset
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_pycocotools,
)


def parse_float_list(values: str) -> list[float]:
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--custom-iou", type=float, default=0.50)
    parser.add_argument("--conf-thresholds", default="0.001,0.005,0.01,0.025,0.05,0.10,0.25,0.50")
    parser.add_argument("--fp-budgets", default="0.425")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "annotations" in data:
        data = data["annotations"]
    if not isinstance(data, list):
        raise ValueError("Predictions must be a COCO result list or an object with an annotations list")
    return data


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = [float(value) for value in box]
    return [x, y, x + width, y + height]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ForestPersonsDataset(args.root, split=args.split)
    manifest = load_manifest(args.manifest)
    samples_by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [samples_by_file[item["file_name"]] for item in manifest]
    image_ids = [sample.image_id for sample in samples]
    image_id_set = set(image_ids)
    ground_truth_by_image = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    ground_truth = [
        gt for image_ground_truth in ground_truth_by_image.values() for gt in image_ground_truth
    ]

    predictions_by_image: dict[int | str, list[EvaluationPrediction]] = {image_id: [] for image_id in image_ids}
    skipped = 0
    for item in load_predictions(args.predictions):
        image_id = item.get("image_id")
        if image_id not in image_id_set:
            skipped += 1
            continue
        score = item.get("score", item.get("confidence"))
        if score is None:
            raise ValueError("Every prediction must include score or confidence")
        category_id = int(item.get("category_id", 1))
        if category_id != 1:
            skipped += 1
            continue
        predictions_by_image[image_id].append(
            EvaluationPrediction(
                image_id=image_id,
                box=xywh_to_xyxy(item["bbox"]),
                confidence=float(score),
                category_id=1,
            )
        )
    predictions = [
        prediction
        for image_predictions in predictions_by_image.values()
        for prediction in image_predictions
    ]

    coco_style = evaluate_coco_style(ground_truth, predictions, image_ids)
    pycocotools = evaluate_pycocotools(ground_truth, predictions, image_ids)
    confidence_sweep = evaluate_confidence_sweep(
        ground_truth_by_image,
        predictions_by_image,
        image_ids,
        parse_float_list(args.conf_thresholds),
        iou_threshold=args.custom_iou,
        fp_budgets=parse_float_list(args.fp_budgets),
    )
    summary = {
        "settings": {
            "root": str(args.root),
            "split": args.split,
            "manifest": str(args.manifest),
            "predictions": str(args.predictions),
            "custom_iou": args.custom_iou,
            "conf_thresholds": parse_float_list(args.conf_thresholds),
            "fp_budgets": parse_float_list(args.fp_budgets),
            "pilot_subset_warning": (
                "These are pilot-subset results unless the manifest contains the full official split."
            ),
        },
        "images_evaluated": len(samples),
        "ground_truth_annotations": len(ground_truth),
        "raw_predictions": len(predictions),
        "skipped_predictions": skipped,
        "coco_style": coco_style,
        "pycocotools": pycocotools,
        "custom_confidence_sweep": confidence_sweep,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "confidence_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "confidence_threshold",
                "ground_truth",
                "true_positives",
                "false_positives",
                "false_negatives",
                "recall",
                "precision",
                "false_positives_per_image",
            ],
        )
        writer.writeheader()
        for row in confidence_sweep["rows"]:
            writer.writerow(
                {
                    "confidence_threshold": row["confidence_threshold"],
                    "ground_truth": row["overall"]["ground_truth"],
                    "true_positives": row["overall"]["true_positives"],
                    "false_positives": row["overall"]["false_positives"],
                    "false_negatives": row["overall"]["false_negatives"],
                    "recall": row["overall"]["recall"],
                    "precision": row["overall"]["precision"],
                    "false_positives_per_image": row["false_positives_per_image"],
                }
            )
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
