"""Analyze FCOS IoU-quality predictions and recovery of known val600 misses."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.ops import nms

from analyze_fcos_scoring_calibration import export_candidates, load_manifest, samples_from_manifest
from forestpersons.detection import box_iou, greedy_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-dump", type=Path, required=True)
    parser.add_argument("--known-failures", type=Path, required=True)
    parser.add_argument("--control-predictions", type=Path, required=True)
    parser.add_argument("--quality-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--quality-power", type=float, default=0.25)
    parser.add_argument("--baseline-threshold", type=float, default=0.555)
    parser.add_argument("--quality-threshold", type=float, default=0.520)
    return parser.parse_args()


def pairwise_iou(boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(boxes), len(gt_boxes)), dtype=np.float64)
    x1 = np.maximum(boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], gt_boxes[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    gt_area = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
    return inter / np.maximum(area[:, None] + gt_area[None, :] - inter, 1e-12)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def select_stages(
    boxes: np.ndarray,
    scores: np.ndarray,
    topk: int,
    nms_iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    floor = np.flatnonzero(scores >= 0.001)
    if len(floor) > topk:
        local = torch.topk(torch.as_tensor(scores[floor], dtype=torch.float32), k=topk).indices.numpy()
        selected = floor[local]
    else:
        selected = floor[np.argsort(-scores[floor], kind="stable")]
    kept_local = nms(
        torch.as_tensor(boxes[selected], dtype=torch.float32),
        torch.as_tensor(scores[selected], dtype=torch.float32),
        nms_iou,
    ).numpy() if len(selected) else np.asarray([], dtype=np.int64)
    return selected, selected[kept_local]


def load_gt(samples: list[Any]) -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, int], dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for sample in samples:
        for annotation in sample.annotations:
            metadata = annotation.get("metadata", {})
            if metadata.get("visibility_level") is None:
                continue
            x, y, width, height = map(float, annotation["bbox"])
            row = {
                "annotation_id": int(annotation["id"]),
                "box": [x, y, x + width, y + height],
                "visibility_level": int(metadata["visibility_level"]),
            }
            by_image[int(sample.image_id)].append(row)
            by_key[(int(sample.image_id), row["annotation_id"])] = row
    return by_image, by_key


def matched_gt_keys(
    prediction_path: Path,
    gt_by_image: dict[int, list[dict[str, Any]]],
    threshold: float,
) -> set[tuple[int, int]]:
    predictions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(prediction_path.read_text()):
        if float(row["score"]) < threshold:
            continue
        x, y, width, height = map(float, row["bbox"])
        predictions[int(row["image_id"])].append(
            {"box": [x, y, x + width, y + height], "confidence": float(row["score"])}
        )
    keys: set[tuple[int, int]] = set()
    for image_id, gt in gt_by_image.items():
        matches, _, _ = greedy_match(gt, predictions.get(image_id, []), 0.50)
        for match in matches:
            keys.add((image_id, int(gt[match["gt_index"]]["annotation_id"])))
    return keys


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    samples = samples_from_manifest(args.root, "val", manifest)
    gt_by_image, gt_by_key = load_gt(samples)
    if not args.candidate_dump.exists():
        export_candidates(
            samples,
            "val600_quality",
            args.candidate_dump,
            args.checkpoint,
            args.device,
            args.short_side,
        )

    with args.known_failures.open(newline="") as handle:
        known = {
            (int(row["image_id"]), int(row["annotation_id"])): row
            for row in csv.DictReader(handle)
            if float(row["pre_nms_best_iou"]) >= 0.50
        }

    quality_chunks: list[np.ndarray] = []
    iou_chunks: list[np.ndarray] = []
    top_quality_chunks: list[np.ndarray] = []
    top_iou_chunks: list[np.ndarray] = []
    recovery_rows: list[dict[str, Any]] = []
    with gzip.open(args.candidate_dump, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image_id = int(row["image_id"])
            raw = row["candidates"]
            boxes = np.asarray([candidate["box"] for candidate in raw], dtype=np.float64)
            cls = np.asarray([candidate["class_score"] for candidate in raw], dtype=np.float64)
            center = np.asarray([candidate["centerness_score"] for candidate in raw], dtype=np.float64)
            quality = np.asarray([candidate["quality_score"] for candidate in raw], dtype=np.float64)
            gt = gt_by_image[image_id]
            matrix = pairwise_iou(boxes, np.asarray([item["box"] for item in gt], dtype=np.float64))
            max_iou = matrix.max(axis=1) if matrix.shape[1] else np.zeros(len(boxes))
            overlap = max_iou > 0.0
            quality_chunks.append(quality[overlap])
            iou_chunks.append(max_iou[overlap])

            current_score = np.sqrt(cls * center)
            aware_score = current_score * np.power(np.maximum(quality, 1e-12), args.quality_power)
            stages = {}
            for name, scores in (("baseline", current_score), ("quality_aware", aware_score)):
                selected, kept = select_stages(boxes, scores, args.topk, args.nms_iou)
                stages[name] = (scores, selected, kept)
                if name == "baseline":
                    top_quality_chunks.append(quality[selected])
                    top_iou_chunks.append(max_iou[selected])

            gt_index = {int(item["annotation_id"]): index for index, item in enumerate(gt)}
            for key, known_row in known.items():
                if key[0] != image_id:
                    continue
                index = gt_index[key[1]]
                for name, (scores, selected, kept) in stages.items():
                    threshold = args.baseline_threshold if name == "baseline" else args.quality_threshold
                    good = matrix[:, index] >= 0.50
                    recovery_rows.append(
                        {
                            "image_id": image_id,
                            "annotation_id": key[1],
                            "visibility": int(known_row["visibility"]),
                            "scoring": name,
                            "raw_best_iou": float(matrix[:, index].max()),
                            "any_iou50_in_top300": bool(np.any(good[selected])),
                            "any_iou50_after_nms": bool(np.any(good[kept])),
                            "detected_at_threshold": bool(np.any(good[kept] & (scores[kept] >= threshold))),
                            "threshold": threshold,
                        }
                    )

    qualities = np.concatenate(quality_chunks)
    ious = np.concatenate(iou_chunks)
    top_qualities = np.concatenate(top_quality_chunks)
    top_ious = np.concatenate(top_iou_chunks)
    correlation = {
        "definition": "actual IoU is each candidate's maximum IoU with any GT in its image",
        "overlapping_raw_candidates": {
            "count": int(len(qualities)),
            "pearson": pearson(qualities, ious),
            "spearman": pearson(rankdata(qualities), rankdata(ious)),
        },
        "canonical_top300_candidates": {
            "count": int(len(top_qualities)),
            "pearson": pearson(top_qualities, top_ious),
            "spearman": pearson(rankdata(top_qualities), rankdata(top_ious)),
        },
    }

    control_keys = matched_gt_keys(args.control_predictions, gt_by_image, args.baseline_threshold)
    quality_keys = matched_gt_keys(args.quality_predictions, gt_by_image, args.quality_threshold)
    changed_rows = []
    for key in sorted(control_keys ^ quality_keys):
        gt = gt_by_key[key]
        changed_rows.append(
            {
                "image_id": key[0],
                "annotation_id": key[1],
                "visibility": gt["visibility_level"],
                "control_detected": key in control_keys,
                "quality_aware_detected": key in quality_keys,
            }
        )

    recovery_summary = {}
    for name in ("baseline", "quality_aware"):
        rows = [row for row in recovery_rows if row["scoring"] == name]
        recovery_summary[name] = {
            "known_cases": len(rows),
            "raw_iou50_available": sum(row["raw_best_iou"] >= 0.50 for row in rows),
            "top300_recovered": sum(row["any_iou50_in_top300"] for row in rows),
            "nms_recovered": sum(row["any_iou50_after_nms"] for row in rows),
            "detected_at_matched_budget_threshold": sum(row["detected_at_threshold"] for row in rows),
        }
    summary = {
        "quality_power": args.quality_power,
        "baseline_threshold": args.baseline_threshold,
        "quality_threshold": args.quality_threshold,
        "correlation": correlation,
        "known40_recovery": recovery_summary,
        "overall_gt_changes": {
            "control_detected": len(control_keys),
            "quality_aware_detected": len(quality_keys),
            "newly_recovered": len(quality_keys - control_keys),
            "lost": len(control_keys - quality_keys),
        },
    }
    write_csv(args.output_dir / "known40_recovery.csv", recovery_rows)
    write_csv(args.output_dir / "newly_recovered_and_lost.csv", changed_rows)
    (args.output_dir / "quality_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
