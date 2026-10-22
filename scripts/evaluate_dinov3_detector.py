"""Evaluate a frozen-DINOv3 detector head on ForestPersons."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from forestpersons import ForestPersonsDataset
from forestpersons.detection import (
    draw_detection_comparison,
    evaluate_image_matches,
    safe_div,
    summarize_group,
)
from forestpersons.dinov3_detector import (
    DINOv3DetectionHead,
    DINOv3FCOSHead,
    DINOv3GridDetectionHead,
    DINOv3LinearPatchHead,
    annotations_to_ground_truth,
    cache_path_for_file,
    decode_fcos_predictions,
    decode_predictions,
    load_feature_cache,
)


def make_head(name: str) -> torch.nn.Module:
    if name == "linear":
        return DINOv3LinearPatchHead()
    if name == "mlp":
        return DINOv3DetectionHead()
    if name == "grid":
        return DINOv3GridDetectionHead()
    if name == "fcos":
        return DINOv3FCOSHead()
    raise ValueError(f"Unknown head: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dinov3_detector/eval"))
    parser.add_argument("--head", choices=["auto", "linear", "mlp", "grid", "fcos"], default="auto")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-visualizations", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = ForestPersonsDataset(args.root, split=args.split)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
    by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [by_file[item["file_name"]] for item in manifest]

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head_name = checkpoint.get("head") or checkpoint.get("settings", {}).get("head") or "mlp"
    if args.head != "auto":
        head_name = args.head
    head = make_head(head_name).to(device)
    head.load_state_dict(checkpoint["model_state"])
    head.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_dir / "comparisons"
    rows: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    overall_stats = defaultdict(int)
    visibility_stats: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    prediction_total = 0
    seconds_total = 0.0

    with torch.inference_mode():
        for image_index, sample in enumerate(samples):
            features, info = load_feature_cache(cache_path_for_file(args.cache_root, sample.file_name))
            features = features.to(device)
            start = time.perf_counter()
            if head_name == "fcos":
                logits, raw_box, centerness_logits = head(features)
                predictions = decode_fcos_predictions(
                    logits.detach().cpu(),
                    raw_box.detach().cpu(),
                    centerness_logits.detach().cpu(),
                    info,
                    conf_threshold=args.conf,
                    nms_threshold=args.nms_iou,
                )
            else:
                logits, raw_box = head(features)
                predictions = decode_predictions(
                    logits.detach().cpu(),
                    raw_box.detach().cpu(),
                    info,
                    conf_threshold=args.conf,
                    nms_threshold=args.nms_iou,
                )
            elapsed = time.perf_counter() - start
            seconds_total += elapsed
            ground_truth = annotations_to_ground_truth(sample.annotations)
            matches, matched_gt, matched_pred, overall, by_visibility = evaluate_image_matches(
                ground_truth,
                predictions,
                args.iou,
            )
            prediction_total += len(predictions)
            for key, value in overall.items():
                overall_stats[key] += value
            for visibility, stats in by_visibility.items():
                for key, value in stats.items():
                    visibility_stats[visibility][key] += value

            for match in matches:
                gt = ground_truth[match["gt_index"]]
                rows.append(
                    {
                        "file_name": sample.file_name,
                        "image_id": sample.image_id,
                        "annotation_id": gt["annotation_id"],
                        "visibility_level": match["visibility_level"],
                        "pose": gt["pose"],
                        "iou": match["iou"],
                        "confidence": match["confidence"],
                    }
                )

            image_summaries.append(
                {
                    "file_name": sample.file_name,
                    "image_id": sample.image_id,
                    "ground_truth": len(ground_truth),
                    "predictions": len(predictions),
                    "true_positives": len(matches),
                    "false_positives": len(predictions) - len(matched_pred),
                    "false_negatives": len(ground_truth) - len(matched_gt),
                    "inference_seconds_cached_head_only": elapsed,
                }
            )

            if image_index < args.max_visualizations:
                output_name = sample.file_name.replace("/", "__")
                draw_detection_comparison(
                    sample.image_path,
                    ground_truth,
                    predictions,
                    matches,
                    comparison_dir / output_name,
                    prediction_label="DINOv3",
                )

    metrics = {
        "settings": {
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "cache_root": str(args.cache_root),
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "nms_iou_threshold": args.nms_iou,
            "device": args.device,
            "split": args.split,
            "head": head_name,
        },
        "images_evaluated": len(samples),
        "predictions": prediction_total,
        "overall": summarize_group(overall_stats),
        "false_positives_per_image": safe_div(overall_stats["false_positives"], len(samples)),
        "inference": {
            "seconds_total_cached_head_only": seconds_total,
            "seconds_per_image_cached_head_only": seconds_total / len(samples) if samples else 0.0,
        },
        "by_visibility": {
            str(visibility): summarize_group(visibility_stats[visibility])
            for visibility in sorted(visibility_stats)
        },
        "image_summaries": image_summaries,
        "yolo11_reference": {
            "best_existing_baseline": "YOLO11s conf 0.05, IoU 0.50",
            "recall": 0.4673,
            "precision": 0.5785,
            "recall_visibility_20": 0.1782,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
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
    print(f"Saved metrics: {args.output_dir / 'metrics.json'}")
    print(f"Saved comparisons: {comparison_dir}")


if __name__ == "__main__":
    main()
