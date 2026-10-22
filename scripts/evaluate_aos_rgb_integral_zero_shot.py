#!/usr/bin/env python3
"""Zero-shot evaluation on canonical official AOS RGB integral test images."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from forestpersons.detection import draw_detection_comparison, evaluate_image_matches
from forestpersons.dinov3_detector import choose_device, decode_fcos_predictions
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    evaluate_coco_style,
    evaluate_custom_metrics,
    evaluate_pycocotools,
    predictions_to_coco_results,
)


TEST_SCENES = ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F9", "F11")
EXPECTED_PEOPLE = {"F0": 3, "F1": 10, "F2": 10, "F3": 6, "F4": 6,
                   "F5": 10, "F6": 10, "F7": 2, "F9": 0, "F11": 0}
CANONICAL_SUFFIX = "_rot000_fpH0.00_rx0.0_ry0.0.png"
VISUAL_SCENES = ("F0", "F3", "F5", "F9")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aos-rgb-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--prediction-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yolo_labels(path: Path, image_id: int, width: int, height: int) -> list[EvaluationGroundTruth]:
    results = []
    for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or int(fields[0]) != 0:
            raise ValueError(f"Invalid one-class YOLO row in {path}: {line}")
        xc, yc, box_width, box_height = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in (xc, yc, box_width, box_height)):
            raise ValueError(f"Out-of-range YOLO row in {path}: {line}")
        x1 = (xc - box_width / 2.0) * width
        y1 = (yc - box_height / 2.0) * height
        x2 = (xc + box_width / 2.0) * width
        y2 = (yc + box_height / 2.0) * height
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid decoded box in {path}: {[x1, y1, x2, y2]}")
        results.append(EvaluationGroundTruth(
            image_id=image_id,
            annotation_id=image_id * 100 + row_index,
            box=[x1, y1, x2, y2],
            visibility_level=100,
            pose="not_available",
        ))
    return results


def discover_test_samples(root: Path) -> list[dict[str, Any]]:
    if root.name != "TEST":
        raise ValueError(f"Expected official TEST directory, got: {root}")
    actual_scenes = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_scenes != set(TEST_SCENES):
        raise ValueError(f"Unexpected TEST scenes: expected {set(TEST_SCENES)}, found {actual_scenes}")
    samples = []
    for image_id, scene in enumerate(TEST_SCENES, start=1):
        scene_dir = root / scene
        images = sorted(scene_dir.glob(f"*{CANONICAL_SUFFIX}"))
        if len(images) != 1:
            raise ValueError(f"Expected exactly one canonical RGB integral for {scene}, found {len(images)}")
        image_path = images[0]
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing paired label: {label_path}")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (512, 512):
            raise ValueError(f"Unexpected AOS integral dimensions for {image_path}: {(width, height)}")
        ground_truth = read_yolo_labels(label_path, image_id, width, height)
        if len(ground_truth) != EXPECTED_PEOPLE[scene]:
            raise ValueError(
                f"Canonical {scene} label count mismatch: expected {EXPECTED_PEOPLE[scene]}, "
                f"found {len(ground_truth)}"
            )
        samples.append({
            "image_id": image_id,
            "scene": scene,
            "image_path": image_path,
            "label_path": label_path,
            "width": width,
            "height": height,
            "ground_truth": ground_truth,
        })
    return samples


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_test_samples(args.aos_rgb_root)
    image_ids = [sample["image_id"] for sample in samples]
    ground_truth_by_image = {sample["image_id"]: sample["ground_truth"] for sample in samples}
    ground_truth = [item for sample in samples for item in sample["ground_truth"]]

    device = choose_device(args.device)
    processor, backbone, head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    if metadata["head_kind"] != "fcos":
        raise ValueError(f"Expected canonical single-scale FCOS checkpoint, got {metadata['head_kind']}")
    backbone.eval()
    head.eval()

    predictions_by_image: dict[int, list[EvaluationPrediction]] = {}
    seconds = 0.0
    peak_mps_driver = 0
    with torch.inference_mode():
        for sample in samples:
            image = Image.open(sample["image_path"]).convert("RGB")
            start = time.perf_counter()
            features, info = extract_lora_patch_features(
                image, processor, backbone, device, short_side=args.short_side
            )
            class_logits, box_raw, centerness = head(features)
            decoded = decode_fcos_predictions(
                class_logits.detach().cpu(),
                box_raw.detach().cpu(),
                centerness.detach().cpu(),
                info,
                conf_threshold=args.prediction_floor,
                nms_threshold=args.nms_iou,
            )
            if device.type == "mps":
                torch.mps.synchronize()
                peak_mps_driver = max(peak_mps_driver, int(torch.mps.driver_allocated_memory()))
            seconds += time.perf_counter() - start
            predictions_by_image[sample["image_id"]] = [
                EvaluationPrediction(
                    image_id=sample["image_id"], box=item["box"], confidence=float(item["confidence"])
                )
                for item in decoded
            ]

    predictions = [item for values in predictions_by_image.values() for item in values]
    fixed_iou50 = evaluate_custom_metrics(
        ground_truth_by_image, predictions_by_image, image_ids, args.confidence, iou_threshold=0.50
    )
    fixed_iou25 = evaluate_custom_metrics(
        ground_truth_by_image, predictions_by_image, image_ids, args.confidence, iou_threshold=0.25
    )
    coco_local = evaluate_coco_style(ground_truth, predictions, image_ids)
    coco_official = evaluate_pycocotools(ground_truth, predictions, image_ids)
    if not coco_official.get("available"):
        raise RuntimeError(f"pycocotools unavailable: {coco_official.get('error')}")

    for sample in samples:
        if sample["scene"] not in VISUAL_SCENES:
            continue
        gt = [{"box": item.box, "visibility_level": None, "pose": "AOS integral"}
              for item in sample["ground_truth"]]
        pred = [{"box": item.box, "confidence": item.confidence}
                for item in predictions_by_image[sample["image_id"]]
                if item.confidence >= args.confidence]
        matches = evaluate_image_matches(
            [{**item, "visibility_level": 100} for item in gt], pred, 0.50
        )[0]
        draw_detection_comparison(
            sample["image_path"], gt, pred, matches,
            args.output_dir / "visualizations" / f"{sample['scene']}.png",
            prediction_label="ForestPersons DINOv3 FCOS",
        )

    manifest = {
        "dataset": "official AOS RGB integral images",
        "split": "TEST",
        "selection": "one predeclared canonical no-AHE product per official test scene",
        "canonical_suffix": CANONICAL_SUFFIX,
        "uses_train_or_validation_images": False,
        "images": [
            {
                "image_id": sample["image_id"], "scene": sample["scene"],
                "image_path": str(sample["image_path"]), "label_path": str(sample["label_path"]),
                "width": sample["width"], "height": sample["height"],
                "ground_truth": len(sample["ground_truth"]),
            }
            for sample in samples
        ],
    }
    settings = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "backbone": metadata["model_id"],
        "head": metadata["head_kind"],
        "device_requested": args.device,
        "device_resolved": str(device),
        "short_side": args.short_side,
        "fixed_confidence": args.confidence,
        "prediction_floor": args.prediction_floor,
        "nms_iou": args.nms_iou,
        "score_formula": "sqrt(sigmoid(classification) * sigmoid(centerness))",
        "box_match_iou_primary": 0.50,
        "box_match_iou_context": 0.25,
        "no_threshold_tuning": True,
    }
    summary = {
        "scope_warning": (
            "Zero-shot ForestPersons-trained detector on official AOS RGB integral images. "
            "Do not compare directly with the paper's thermal-integral detector protocol."
        ),
        "manifest": manifest,
        "settings": settings,
        "images": len(samples),
        "ground_truth_people": len(ground_truth),
        "raw_predictions_at_floor": len(predictions),
        "fixed_iou50": fixed_iou50,
        "fixed_iou25_context_only": fixed_iou25,
        "coco_local": coco_local,
        "pycocotools": coco_official,
        "ap50": coco_official["AP50"],
        "ar100_coco_iou_50_95": coco_official["AR@100"],
        "ar100_iou50": coco_local["per_max_det"]["100"]["per_iou"]["0.50"]["ar"],
        "timing": {
            "seconds_total": seconds,
            "seconds_per_image": seconds / len(samples),
            "peak_mps_driver_bytes": peak_mps_driver if device.type == "mps" else None,
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "coco_predictions.json").write_text(
        json.dumps(predictions_to_coco_results(predictions), indent=2) + "\n"
    )
    print(json.dumps({
        "images": summary["images"], "ground_truth_people": summary["ground_truth_people"],
        "fixed_iou50": fixed_iou50, "fixed_iou25_context_only": fixed_iou25,
        "AP50": summary["ap50"], "AR@100": summary["ar100_coco_iou_50_95"],
        "seconds_per_image": summary["timing"]["seconds_per_image"],
    }, indent=2))


if __name__ == "__main__":
    main()
