"""Evaluate aligned DINOv3 feature averaging on a ForestPersons pilot."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, draw_detection_comparison, evaluate_image_matches
from forestpersons.dinov3_detector import choose_device, decode_fcos_predictions
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_custom_metrics,
    evaluate_pycocotools,
    predictions_to_coco_results,
)
from forestpersons.multiview import fuse_aligned_features, warp_patch_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def mps_memory() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


def predict(head: torch.nn.Module, features: torch.Tensor, info: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    logits, raw_box, centerness = head(features)
    return decode_fcos_predictions(
        logits.detach().cpu(),
        raw_box.detach().cpu(),
        centerness.detach().cpu(),
        info,
        conf_threshold=args.prediction_conf_min,
        nms_threshold=args.nms_iou,
    )


def combine_visuals(first: Path, second: Path, output: Path) -> None:
    left = Image.open(first).convert("RGB")
    right = Image.open(second).convert("RGB")
    height = min(left.height, right.height)
    left = left.resize((round(left.width * height / left.height), height))
    right = right.resize((round(right.width * height / right.height), height))
    header = 42
    canvas = Image.new("RGB", (left.width + right.width, height + header), "white")
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "Single reference view", fill="black")
    draw.text((left.width + 12, 12), "Aligned DINOv3 feature average", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    groups = manifest["groups"]
    dataset = ForestPersonsDataset(args.root, split=args.split)
    samples_by_id = {int(sample.image_id): sample for sample in dataset.samples}
    device = choose_device(args.device)
    processor, backbone, head, checkpoint_metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    backbone.eval()
    head.eval()

    single_by_image: dict[int, list[EvaluationPrediction]] = {}
    fused_by_image: dict[int, list[EvaluationPrediction]] = {}
    diagnostics: list[dict[str, Any]] = []
    single_seconds = 0.0
    multiview_seconds = 0.0
    peak_driver_memory = 0
    with torch.inference_mode():
        for group in groups:
            reference = group["reference"]
            sample = samples_by_id[int(reference["image_id"])]
            reference_image = Image.open(sample.image_path).convert("RGB")
            start = time.perf_counter()
            reference_features, reference_info = extract_lora_patch_features(
                reference_image, processor, backbone, device, short_side=args.short_side
            )
            single_decoded = predict(head, reference_features, reference_info, args)
            if device.type == "mps":
                torch.mps.synchronize()
            single_seconds += time.perf_counter() - start

            start = time.perf_counter()
            aligned_sources = []
            source_diagnostics = []
            for source in group["sources"]:
                source_sample = samples_by_id[int(source["image_id"])]
                source_image = Image.open(source_sample.image_path).convert("RGB")
                source_features, source_info = extract_lora_patch_features(
                    source_image, processor, backbone, device, short_side=args.short_side
                )
                homography = torch.tensor(
                    source["homography_source_to_reference"], dtype=torch.float32, device=device
                )
                aligned, validity = warp_patch_features(
                    source_features,
                    source_info,
                    reference_info,
                    homography,
                )
                valid = validity > 0.99
                cosine = torch.sum(reference_features * aligned, dim=-1)
                source_diagnostics.append(
                    {
                        "image_id": int(source["image_id"]),
                        "valid_grid_fraction": float(validity.mean().cpu()),
                        "mean_aligned_cosine_valid": float(cosine[valid].mean().cpu()) if valid.any() else None,
                    }
                )
                aligned_sources.append((aligned, validity))
            fused_features = fuse_aligned_features(reference_features, aligned_sources)
            fused_decoded = predict(head, fused_features, reference_info, args)
            if device.type == "mps":
                torch.mps.synchronize()
            multiview_seconds += time.perf_counter() - start
            memory = mps_memory()
            if memory:
                peak_driver_memory = max(peak_driver_memory, memory["driver_allocated"])

            image_id = int(reference["image_id"])
            single_by_image[image_id] = [
                EvaluationPrediction(image_id=image_id, box=item["box"], confidence=item["confidence"])
                for item in single_decoded
            ]
            fused_by_image[image_id] = [
                EvaluationPrediction(image_id=image_id, box=item["box"], confidence=item["confidence"])
                for item in fused_decoded
            ]
            ground_truth = annotations_to_eval_ground_truth(image_id, sample.annotations)
            gt_boxes = [item.box for item in ground_truth]
            best_single_iou = max((box_iou(gt_boxes[0], item.box) for item in single_by_image[image_id]), default=0.0)
            best_fused_iou = max((box_iou(gt_boxes[0], item.box) for item in fused_by_image[image_id]), default=0.0)
            diagnostics.append(
                {
                    "group_id": group["group_id"],
                    "sequence": group["sequence"],
                    "reference_image_id": image_id,
                    "reference_file": sample.file_name,
                    "visibility": ground_truth[0].visibility_level,
                    "views": 1 + len(group["sources"]),
                    "single_predictions": len(single_decoded),
                    "fused_predictions": len(fused_decoded),
                    "single_best_iou_all_scores": best_single_iou,
                    "fused_best_iou_all_scores": best_fused_iou,
                    "sources": source_diagnostics,
                }
            )

    image_ids = [int(group["reference"]["image_id"]) for group in groups]
    ground_truth_by_image = {
        image_id: annotations_to_eval_ground_truth(image_id, samples_by_id[image_id].annotations)
        for image_id in image_ids
    }
    ground_truth = [item for values in ground_truth_by_image.values() for item in values]
    single_predictions = [item for values in single_by_image.values() for item in values]
    fused_predictions = [item for values in fused_by_image.values() for item in values]

    def metrics(predictions_by_image: dict[int, list[EvaluationPrediction]], predictions: list[EvaluationPrediction]) -> dict[str, Any]:
        return {
            "fixed_threshold": evaluate_custom_metrics(
                ground_truth_by_image, predictions_by_image, image_ids, args.confidence, iou_threshold=0.5
            ),
            "coco_style": evaluate_coco_style(ground_truth, predictions, image_ids),
            "pycocotools": evaluate_pycocotools(ground_truth, predictions, image_ids),
        }

    single_metrics = metrics(single_by_image, single_predictions)
    fused_metrics = metrics(fused_by_image, fused_predictions)
    rescued = []
    lost = []
    per_group_rows = []
    for diagnostic in diagnostics:
        image_id = diagnostic["reference_image_id"]
        gt = [
            {
                "box": item.box,
                "visibility_level": item.visibility_level,
                "pose": item.pose,
                "annotation_id": item.annotation_id,
            }
            for item in ground_truth_by_image[image_id]
        ]
        single = [
            {"box": item.box, "confidence": item.confidence}
            for item in single_by_image[image_id]
            if item.confidence >= args.confidence
        ]
        fused = [
            {"box": item.box, "confidence": item.confidence}
            for item in fused_by_image[image_id]
            if item.confidence >= args.confidence
        ]
        single_detected = bool(evaluate_image_matches(gt, single, 0.5)[0])
        fused_detected = bool(evaluate_image_matches(gt, fused, 0.5)[0])
        if fused_detected and not single_detected:
            rescued.append(image_id)
        if single_detected and not fused_detected:
            lost.append(image_id)
        per_group_rows.append(
            {
                **{key: value for key, value in diagnostic.items() if key != "sources"},
                "single_detected_fixed": single_detected,
                "fused_detected_fixed": fused_detected,
            }
        )

        single_path = args.output_dir / "visualizations" / "single" / f"{diagnostic['group_id']}.jpg"
        fused_path = args.output_dir / "visualizations" / "fused" / f"{diagnostic['group_id']}.jpg"
        draw_detection_comparison(
            samples_by_id[image_id].image_path,
            gt,
            single,
            evaluate_image_matches(gt, single, 0.5)[0],
            single_path,
            prediction_label="single FCOS",
        )
        draw_detection_comparison(
            samples_by_id[image_id].image_path,
            gt,
            fused,
            evaluate_image_matches(gt, fused, 0.5)[0],
            fused_path,
            prediction_label="multi-view FCOS",
        )
        combine_visuals(
            single_path,
            fused_path,
            args.output_dir / "visualizations" / "comparisons" / f"{diagnostic['group_id']}.jpg",
        )

    settings = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "backbone": checkpoint_metadata["model_id"],
        "head": checkpoint_metadata["head_kind"],
        "device": str(device),
        "short_side": args.short_side,
        "fixed_confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "fusion": "equal-weight mean of reference and homography-warped valid source patch features, followed by L2 normalization",
        "shared_encoder": True,
        "training": False,
        "threshold_tuning": False,
        "selection_warning": "Groups were GT-qualified for correspondence; metrics are exploratory and selection-biased.",
    }
    result = {
        "settings": settings,
        "groups": len(groups),
        "views": sum(1 + len(group["sources"]) for group in groups),
        "ground_truth_people": len(ground_truth),
        "single_view": single_metrics,
        "multi_view": fused_metrics,
        "paired_fixed_threshold": {
            "rescued_image_ids": rescued,
            "lost_image_ids": lost,
            "net_detected_change": len(rescued) - len(lost),
        },
        "timing": {
            "single_reference_seconds_total": single_seconds,
            "single_reference_seconds_per_group": single_seconds / len(groups),
            "additional_multiview_seconds_total": multiview_seconds,
            "multiview_total_seconds_per_group": (single_seconds + multiview_seconds) / len(groups),
        },
        "peak_mps_driver_memory_bytes": peak_driver_memory,
        "diagnostics": diagnostics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "single_view_coco_results.json").write_text(
        json.dumps(predictions_to_coco_results(single_predictions), indent=2), encoding="utf-8"
    )
    (args.output_dir / "multi_view_coco_results.json").write_text(
        json.dumps(predictions_to_coco_results(fused_predictions), indent=2), encoding="utf-8"
    )
    with (args.output_dir / "per_group_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_group_rows[0]))
        writer.writeheader()
        writer.writerows(per_group_rows)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
