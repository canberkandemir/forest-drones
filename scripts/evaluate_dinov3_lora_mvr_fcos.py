#!/usr/bin/env python3
"""Evaluate matched online frozen and LoRA Full MVR + FCOS checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from forestpersons.dinov3_detector import (
    DINOv3MVRFCOSHead,
    decode_fcos_predictions,
    load_dinov3,
)
from forestpersons.dinov3_lora import (
    LoRAConfig,
    extract_lora_multilayer_patch_features,
    load_dinov3_with_lora,
    load_lora_state_dict,
)
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    predictions_to_coco_results,
)
from scripts.evaluate_dinov3_mvr_fcos import (
    FP_BUDGETS,
    THRESHOLDS,
    collect_samples,
    flatten_sweep_rows,
    matched_budget_rows,
    write_csv,
)
from scripts.train_dinov3_lora_mvr_fcos import MVR_LAYERS


def validate_matched_checkpoints(frozen: dict[str, Any], lora: dict[str, Any]) -> None:
    for expected_arm, checkpoint in (("frozen", frozen), ("lora", lora)):
        if checkpoint.get("experiment") != "frozen_vs_lora_full_mvr":
            raise ValueError(f"{expected_arm} checkpoint is from another experiment")
        if checkpoint.get("arm") != expected_arm or checkpoint.get("head") != "mvr_fcos":
            raise ValueError(f"Expected {expected_arm} Full MVR checkpoint")
    keys = ("manifest_sha256", "initial_head_sha256", "completed_epoch", "optimizer_steps")
    for key in keys:
        if frozen.get(key) != lora.get(key):
            raise ValueError(f"Matched comparison differs for {key}")
    frozen_config = frozen["training_config"]
    lora_config = lora["training_config"]
    for key in (
        "epochs", "gradient_accumulation", "short_side", "lr_head", "weight_decay",
        "box_loss_weight", "focal_alpha", "focal_gamma", "gradient_clip_norm", "seed",
    ):
        if frozen_config[key] != lora_config[key]:
            raise ValueError(f"Matched comparison training configuration differs for {key}")
    frozen_orders = [row["sample_order_sha256"] for row in frozen["history"]]
    lora_orders = [row["sample_order_sha256"] for row in lora["history"]]
    if frozen_orders != lora_orders:
        raise ValueError("Matched comparison used different image orders")


def load_arm(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[Any, torch.nn.Module, DINOv3MVRFCOSHead]:
    model_id = str(checkpoint["backbone_model_id"])
    if checkpoint["arm"] == "lora":
        data = checkpoint["lora_config"]
        config = LoRAConfig(
            rank=int(data["rank"]), alpha=float(data["alpha"]),
            dropout=float(data["dropout"]), target_suffixes=tuple(data["target_suffixes"]),
        )
        processor, backbone, _ = load_dinov3_with_lora(device, config, model_id=model_id)
        load_lora_state_dict(backbone, checkpoint["lora_state"])
    else:
        processor, backbone = load_dinov3(device)
    backbone.eval()
    head = DINOv3MVRFCOSHead().to(device)
    head.load_state_dict(checkpoint["head_state"], strict=True)
    head.eval()
    return processor, backbone, head


def evaluate_arm(
    checkpoint_path: Path,
    samples: list[Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[EvaluationPrediction]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    processor, backbone, head = load_arm(checkpoint, device)
    predictions: list[EvaluationPrediction] = []
    predictions_by_image: dict[int, list[EvaluationPrediction]] = {}
    ground_truth_by_image = {}
    image_ids: list[int] = []
    seconds = 0.0
    peak_mps = int(torch.mps.driver_allocated_memory()) if device.type == "mps" else 0
    short_side = int(checkpoint["training_config"]["short_side"])
    with torch.inference_mode():
        for sample in samples:
            with Image.open(sample.image_path) as opened:
                image = opened.convert("RGB")
            started = time.perf_counter()
            features, info = extract_lora_multilayer_patch_features(
                image, processor, backbone, device,
                short_side=short_side, layer_indices=MVR_LAYERS,
            )
            logits, raw_box, centerness = head(features)
            decoded = decode_fcos_predictions(
                logits.detach().cpu(), raw_box.detach().cpu(), centerness.detach().cpu(), info,
                conf_threshold=0.001, nms_threshold=0.50, topk=300,
            )
            if device.type == "mps":
                torch.mps.synchronize()
                peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))
            seconds += time.perf_counter() - started
            image_id = int(sample.image_id)
            image_ids.append(image_id)
            image_predictions = [
                EvaluationPrediction(
                    image_id=image_id,
                    box=[float(value) for value in item["box"]],
                    confidence=float(item["confidence"]),
                )
                for item in decoded
            ]
            predictions.extend(image_predictions)
            predictions_by_image[image_id] = image_predictions
            ground_truth_by_image[image_id] = annotations_to_eval_ground_truth(
                image_id, sample.annotations
            )
    ground_truth = [item for values in ground_truth_by_image.values() for item in values]
    coco = evaluate_coco_style(ground_truth, predictions, image_ids)
    sweep = evaluate_confidence_sweep(
        ground_truth_by_image, predictions_by_image, image_ids, THRESHOLDS,
        iou_threshold=0.50, fp_budgets=FP_BUDGETS,
    )
    return {
        "arm": checkpoint["arm"],
        "checkpoint": str(checkpoint_path.resolve()),
        "images": len(image_ids),
        "ground_truth": len(ground_truth),
        "AP": coco["AP"], "AP50": coco["AP50"], "AP75": coco["AP75"],
        "AR100": coco["AR_by_max_dets"]["100"],
        "maximum_recall": sweep["maximum_recall"],
        "matched_fp_budgets": sweep["matched_false_positive_budgets"],
        "confidence_sweep": sweep["rows"],
        "seconds_online_backbone_readout_head": seconds,
        "seconds_per_image": seconds / len(image_ids),
        "peak_mps_driver_bytes": peak_mps,
        "protocol": {
            "iou": 0.50, "matching": "one-to-one descending-confidence greedy",
            "candidate_floor": 0.001, "nms_iou": 0.50, "topk": 300,
            "fp_budgets": list(FP_BUDGETS),
            "threshold_selection": "exploratory, selected separately on reused val600",
        },
    }, predictions


def report(results: list[dict[str, Any]], matched: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen versus LoRA Full MVR + FCOS on ForestPersons val600", "",
        "Both arms started from the same trained Full MVR checkpoint and received matched online "
        "training passes. The backbone was frozen in the control; the experimental arm trained "
        "fresh rank-4 q/v LoRA plus the same MVR/FCOS head. Matched-budget thresholds selected "
        "on the reused val600 split are exploratory.", "",
        "| Arm | AP | AP50 | AP75 | AR@100 | sec/image |", "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['arm']} | {item['AP']:.4f} | {item['AP50']:.4f} | "
            f"{item['AP75']:.4f} | {item['AR100']:.4f} | {item['seconds_per_image']:.4f} |"
        )
    lines += ["", "| Arm | FP budget | Threshold | Recall | R20 | R40 | R70 | R100 | FP/image |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in matched:
        if row["confidence_threshold"] is None:
            continue
        lines.append(
            f"| {row['arm']} | {row['fp_per_image_budget']:.3f} | "
            f"{row['confidence_threshold']:.3f} | {row['recall']:.4f} | {row['R20']:.4f} | "
            f"{row['R40']:.4f} | {row['R70']:.4f} | {row['R100']:.4f} | "
            f"{row['false_positives_per_image']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    frozen = torch.load(args.frozen_checkpoint, map_location="cpu", weights_only=False)
    lora = torch.load(args.lora_checkpoint, map_location="cpu", weights_only=False)
    validate_matched_checkpoints(frozen, lora)
    args.output_dir.mkdir(parents=True)
    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    samples = collect_samples(args.root, args.manifest)
    results = []
    for path in (args.frozen_checkpoint, args.lora_checkpoint):
        metrics, predictions = evaluate_arm(path, samples, device)
        results.append(metrics)
        (args.output_dir / f"{metrics['arm']}_predictions_coco.json").write_text(
            json.dumps(predictions_to_coco_results(predictions), indent=2) + "\n"
        )
    matched = matched_budget_rows(results)
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    write_csv(args.output_dir / "confidence_sweep.csv", flatten_sweep_rows(results))
    write_csv(args.output_dir / "matched_fp_budgets.csv", matched)
    (args.output_dir / "comparison_report.md").write_text(report(results, matched))
    print(json.dumps(matched, indent=2))


if __name__ == "__main__":
    main()
