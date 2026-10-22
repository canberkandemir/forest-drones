"""Inspect false positives and false negatives for a DINOv3 detector run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch

from forestpersons import ForestPersonsDataset
from forestpersons.detection import evaluate_image_matches
from forestpersons.dinov3_detector import (
    DINOv3DetectionHead,
    DINOv3FCOSHead,
    DINOv3GridDetectionHead,
    annotations_to_ground_truth,
    cache_path_for_file,
    decode_fcos_predictions,
    decode_predictions,
    load_feature_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def make_head(name: str) -> torch.nn.Module:
    if name == "mlp":
        return DINOv3DetectionHead()
    if name == "grid":
        return DINOv3GridDetectionHead()
    if name == "fcos":
        return DINOv3FCOSHead()
    raise ValueError(f"Unknown head: {name}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head_name = checkpoint.get("head") or checkpoint.get("settings", {}).get("head") or "mlp"
    head = make_head(head_name).to(device)
    head.load_state_dict(checkpoint["model_state"])
    head.eval()

    dataset = ForestPersonsDataset(args.root, split=args.split)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
    by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [by_file[item["file_name"]] for item in manifest]

    false_negative_rows = []
    image_rows = []
    fp_nearest_visibility = Counter()

    with torch.inference_mode():
        for sample in samples:
            features, info = load_feature_cache(cache_path_for_file(args.cache_root, sample.file_name))
            features = features.to(device)
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
            ground_truth = annotations_to_ground_truth(sample.annotations)
            matches, matched_gt, matched_pred, _, _ = evaluate_image_matches(
                ground_truth,
                predictions,
                args.iou,
            )
            for gt_index, gt in enumerate(ground_truth):
                if gt_index in matched_gt:
                    continue
                x1, y1, x2, y2 = gt["box"]
                false_negative_rows.append(
                    {
                        "file_name": sample.file_name,
                        "annotation_id": gt["annotation_id"],
                        "visibility_level": gt["visibility_level"],
                        "pose": gt["pose"],
                        "bbox_width": x2 - x1,
                        "bbox_height": y2 - y1,
                        "bbox_area": (x2 - x1) * (y2 - y1),
                        "ground_truth_count": len(ground_truth),
                        "prediction_count": len(predictions),
                    }
                )
            for pred_index, pred in enumerate(predictions):
                if pred_index in matched_pred or not ground_truth:
                    continue
                best_gt = max(
                    ground_truth,
                    key=lambda gt: _iou_for_error(gt["box"], pred["box"]),
                )
                fp_nearest_visibility[int(best_gt["visibility_level"])] += 1
            image_rows.append(
                {
                    "file_name": sample.file_name,
                    "ground_truth": len(ground_truth),
                    "predictions": len(predictions),
                    "true_positives": len(matches),
                    "false_positives": len(predictions) - len(matched_pred),
                    "false_negatives": len(ground_truth) - len(matched_gt),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "false_negatives.csv", false_negative_rows)
    _write_csv(args.output_dir / "image_error_summary.csv", image_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "head": head_name,
        "conf": args.conf,
        "false_negatives_by_visibility": dict(
            sorted(Counter(row["visibility_level"] for row in false_negative_rows).items())
        ),
        "false_positives_by_nearest_gt_visibility": dict(sorted(fp_nearest_visibility.items())),
        "top_false_positive_images": sorted(
            image_rows,
            key=lambda row: row["false_positives"],
            reverse=True,
        )[:20],
        "top_false_negative_images": sorted(
            image_rows,
            key=lambda row: row["false_negatives"],
            reverse=True,
        )[:20],
    }
    (args.output_dir / "error_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _iou_for_error(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
