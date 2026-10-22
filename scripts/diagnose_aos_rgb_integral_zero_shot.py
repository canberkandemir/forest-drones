#!/usr/bin/env python3
"""Diagnose score, ranking, NMS, and localization failures for AOS zero-shot FCOS."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.ops import nms

from evaluate_aos_rgb_integral_zero_shot import discover_test_samples
from forestpersons.detection import box_iou
from forestpersons.dinov3_detector import choose_device, decode_fcos_predictions
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos-rgb-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--saved-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--prediction-floor", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    return parser.parse_args()


def decode_stages(class_logits: torch.Tensor, box_raw: torch.Tensor, centerness: torch.Tensor,
                  info: Any, prediction_floor: float, topk: int, nms_iou: float) -> dict[str, list[dict[str, Any]]]:
    class_probability = class_logits.sigmoid().reshape(-1).cpu()
    centerness_probability = centerness.sigmoid().reshape(-1).cpu()
    scores = torch.sqrt((class_probability * centerness_probability).clamp(0.0, 1.0))
    raw = box_raw.reshape(-1, 4).cpu()
    height, width = class_logits.shape
    indices = torch.arange(height * width)
    grid_y = torch.div(indices, width, rounding_mode="floor").float()
    grid_x = (indices % width).float()
    centers_x = (grid_x + 0.5) * 16
    centers_y = (grid_y + 0.5) * 16
    distances = torch.nn.functional.softplus(raw).clamp(max=128.0) * 16
    left, top, right, bottom = distances.unbind(1)
    boxes = torch.stack([
        (centers_x - left).clamp(0, info.resized_width - 1),
        (centers_y - top).clamp(0, info.resized_height - 1),
        (centers_x + right).clamp(0, info.resized_width - 1),
        (centers_y + bottom).clamp(0, info.resized_height - 1),
    ], dim=1)
    scale = torch.tensor([info.scale_x, info.scale_y, info.scale_x, info.scale_y])
    boxes = boxes / scale
    boxes[:, 0::2].clamp_(0, info.original_width - 1.0)
    boxes[:, 1::2].clamp_(0, info.original_height - 1.0)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    all_indices = indices[valid]
    floor_indices = all_indices[scores[all_indices] >= prediction_floor]
    if len(floor_indices) > topk:
        order = torch.topk(scores[floor_indices], topk).indices
        top_indices = floor_indices[order]
    else:
        top_indices = floor_indices
    selected = nms(boxes[top_indices], scores[top_indices], nms_iou)
    nms_indices = top_indices[selected]

    def records(stage_indices: torch.Tensor) -> list[dict[str, Any]]:
        return [
            {
                "grid_index": int(index),
                "box": [float(value) for value in boxes[index].tolist()],
                "score": float(scores[index]),
                "class_probability": float(class_probability[index]),
                "centerness_probability": float(centerness_probability[index]),
            }
            for index in stage_indices
        ]

    return {"raw": records(all_indices), "floor": records(floor_indices),
            "topk": records(top_indices), "nms": records(nms_indices)}


def best_for_gt(gt_box: list[float], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"best_iou": 0.0, "best_score": None, "best_class": None, "best_centerness": None,
                "has_iou50": False, "max_score_iou50": None}
    ious = [box_iou(gt_box, candidate["box"]) for candidate in candidates]
    index = max(range(len(ious)), key=ious.__getitem__)
    overlapping = [candidate["score"] for candidate, iou in zip(candidates, ious) if iou >= 0.50]
    return {
        "best_iou": float(ious[index]),
        "best_score": candidates[index]["score"],
        "best_class": candidates[index]["class_probability"],
        "best_centerness": candidates[index]["centerness_probability"],
        "has_iou50": bool(overlapping),
        "max_score_iou50": max(overlapping) if overlapping else None,
    }


def canonical_signature(predictions: list[dict[str, Any]]) -> list[tuple]:
    return sorted((int(p["image_id"]), round(float(p["score"]), 6),
                   *(round(float(v), 4) for v in p["bbox"])) for p in predictions)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_test_samples(args.aos_rgb_root)
    saved = json.loads(args.saved_predictions.read_text(encoding="utf-8"))
    device = choose_device(args.device)
    processor, backbone, head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    if metadata["head_kind"] != "fcos":
        raise ValueError(f"Expected FCOS checkpoint, got {metadata['head_kind']}")
    backbone.eval()
    head.eval()

    diagnostics = []
    regenerated = []
    stage_candidate_counts = {stage: 0 for stage in ("raw", "floor", "topk", "nms")}
    with torch.inference_mode():
        for sample in samples:
            image = Image.open(sample["image_path"]).convert("RGB")
            features, info = extract_lora_patch_features(
                image, processor, backbone, device, short_side=args.short_side
            )
            class_logits, box_raw, centerness = head(features)
            stages = decode_stages(class_logits, box_raw, centerness, info,
                                   args.prediction_floor, args.topk, args.nms_iou)
            canonical = decode_fcos_predictions(
                class_logits.cpu(), box_raw.cpu(), centerness.cpu(), info,
                conf_threshold=args.prediction_floor, nms_threshold=args.nms_iou,
                topk=args.topk,
            )
            for item in canonical:
                x1, y1, x2, y2 = item["box"]
                regenerated.append({"image_id": sample["image_id"], "category_id": 1,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "score": item["confidence"]})
            for stage, candidates in stages.items():
                stage_candidate_counts[stage] += len(candidates)
            for gt in sample["ground_truth"]:
                row = {
                    "image_id": sample["image_id"], "scene": sample["scene"],
                    "annotation_id": gt.annotation_id, "gt_box": gt.box,
                    "gt_width": gt.box[2] - gt.box[0], "gt_height": gt.box[3] - gt.box[1],
                }
                per_stage = {stage: best_for_gt(gt.box, candidates) for stage, candidates in stages.items()}
                for stage, values in per_stage.items():
                    for key, value in values.items():
                        row[f"{stage}_{key}"] = value
                if per_stage["nms"]["has_iou50"]:
                    if float(per_stage["nms"]["max_score_iou50"]) >= args.confidence:
                        category = "available_at_fixed_threshold"
                    else:
                        category = "confidence_below_0.55"
                elif per_stage["topk"]["has_iou50"]:
                    category = "suppressed_by_nms"
                elif per_stage["floor"]["has_iou50"]:
                    category = "excluded_by_top300"
                elif per_stage["raw"]["has_iou50"]:
                    category = "below_prediction_floor"
                else:
                    category = "poor_localization_no_iou50_candidate"
                row["failure_stage"] = category
                diagnostics.append(row)

    reproducible = canonical_signature(saved) == canonical_signature(regenerated)
    if not reproducible:
        raise RuntimeError("Regenerated canonical predictions do not match saved predictions")
    categories: dict[str, int] = {}
    for row in diagnostics:
        categories[row["failure_stage"]] = categories.get(row["failure_stage"], 0) + 1
    stage_coverage = {
        stage: {
            "gt_with_iou50_candidate": sum(bool(row[f"{stage}_has_iou50"]) for row in diagnostics),
            "gt_with_iou25_candidate": sum(float(row[f"{stage}_best_iou"]) >= 0.25 for row in diagnostics),
            "median_best_iou": float(torch.tensor([row[f"{stage}_best_iou"] for row in diagnostics]).median()),
        }
        for stage in ("raw", "floor", "topk", "nms")
    }
    gt_sizes = {
        "median_width": float(torch.tensor([row["gt_width"] for row in diagnostics]).median()),
        "median_height": float(torch.tensor([row["gt_height"] for row in diagnostics]).median()),
        "min_width": min(row["gt_width"] for row in diagnostics),
        "min_height": min(row["gt_height"] for row in diagnostics),
    }
    summary = {
        "saved_predictions_reproduced_exactly": reproducible,
        "images": len(samples), "ground_truth": len(diagnostics),
        "settings": {"short_side": args.short_side, "confidence": args.confidence,
                     "prediction_floor": args.prediction_floor, "topk": args.topk,
                     "nms_iou": args.nms_iou, "device": str(device)},
        "coordinate_validation": {
            "source_size": [512, 512], "resized_size": [672, 672],
            "scale_x": 1.3125, "scale_y": 1.3125,
            "labels": "normalized YOLO cxcywh decoded directly to 512x512 xyxy",
            "prediction_mapping": "decoded in 672x672 coordinates then divided independently by scale_x/scale_y",
        },
        "preprocessing": {
            "same_as_forestpersons": "same RGB conversion, bicubic patch-grid resize, DINOv3 processor normalization, and short side 672",
            "domain_difference": "AOS inputs are 512x512 synthetic-aperture integral composites; ForestPersons inputs are perspective UAV frames",
        },
        "stage_candidate_counts": stage_candidate_counts,
        "stage_coverage": stage_coverage,
        "failure_stage_counts": categories,
        "gt_box_sizes_pixels": gt_sizes,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = list(diagnostics[0].keys())
    with (args.output_dir / "per_gt_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
