"""Unified ForestPersons detector evaluation framework.

This script evaluates already-trained detectors on locally available images.
It does not train models, download data, or touch existing experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.detection import draw_detection_comparison, evaluate_image_matches
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_pycocotools,
    predictions_to_coco_results,
)


def parse_float_list(values: str) -> list[float]:
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector", choices=["yolo", "dinov3", "dinov3_lora", "dinov3_lora_yolo"], required=True)
    parser.add_argument("--model", type=Path, help="YOLO checkpoint path")
    parser.add_argument("--checkpoint", type=Path, help="DINOv3 detector-head checkpoint path")
    parser.add_argument("--cache-root", type=Path, help="DINOv3 feature cache root")
    parser.add_argument("--head", choices=["auto", "linear", "mlp", "grid", "fcos"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--custom-iou", type=float, default=0.50)
    parser.add_argument("--yolo-score-mode", choices=["class", "class_quality"], default="class")
    parser.add_argument("--fcos-class-score-power", type=float, default=0.5)
    parser.add_argument("--fcos-centerness-score-power", type=float, default=0.5)
    parser.add_argument("--fcos-quality-score-power", type=float, default=0.0)
    parser.add_argument(
        "--conf-thresholds",
        default="0.001,0.005,0.01,0.025,0.05,0.10,0.25,0.50",
    )
    parser.add_argument("--fp-budgets", default="0.425")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-visualizations", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def make_head(name: str) -> torch.nn.Module:
    from forestpersons.dinov3_detector import (
        DINOv3DetectionHead,
        DINOv3FCOSHead,
        DINOv3GridDetectionHead,
        DINOv3LinearPatchHead,
    )

    if name == "linear":
        return DINOv3LinearPatchHead()
    if name == "mlp":
        return DINOv3DetectionHead()
    if name == "grid":
        return DINOv3GridDetectionHead()
    if name == "fcos":
        return DINOv3FCOSHead()
    raise ValueError(f"Unknown DINOv3 head: {name}")


def collect_samples(args: argparse.Namespace) -> list[Any]:
    dataset = ForestPersonsDataset(args.root, split=args.split)
    manifest = load_manifest(args.manifest)
    if args.max_images is not None:
        manifest = manifest[: args.max_images]
    samples_by_id = {sample.image_id: sample for sample in dataset.samples}
    samples = []
    for item in manifest:
        sample = samples_by_id[item["image_id"]]
        if sample.file_name != item["file_name"]:
            raise ValueError(
                f"Manifest image_id {item['image_id']!r} filename mismatch: "
                f"manifest={item['file_name']!r}, annotations={sample.file_name!r}"
            )
        samples.append(sample)
    missing = [sample.file_name for sample in samples if not sample.image_path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images, e.g. {missing[0]}")
    return samples


def predict_yolo(args: argparse.Namespace, samples: list[Any]) -> tuple[dict[int | str, list[EvaluationPrediction]], dict[str, float]]:
    if args.model is None:
        raise ValueError("--model is required for --detector yolo")
    from ultralytics import YOLO

    model = YOLO(args.model)
    predictions_by_image: dict[int | str, list[EvaluationPrediction]] = {}
    seconds_total = 0.0
    for sample in samples:
        start = time.perf_counter()
        result = model.predict(
            source=str(sample.image_path),
            conf=args.prediction_conf_min,
            iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        seconds_total += time.perf_counter() - start
        predictions: list[EvaluationPrediction] = []
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue
                xyxy = [float(value) for value in box.xyxy[0].cpu().tolist()]
                predictions.append(
                    EvaluationPrediction(
                        image_id=sample.image_id,
                        box=xyxy,
                        confidence=float(box.conf[0]),
                    )
                )
        predictions_by_image[sample.image_id] = predictions
    return predictions_by_image, {
        "seconds_total": seconds_total,
        "seconds_per_image": seconds_total / len(samples) if samples else 0.0,
        "images_per_second": len(samples) / seconds_total if seconds_total > 0 else 0.0,
    }


def predict_dinov3(args: argparse.Namespace, samples: list[Any]) -> tuple[dict[int | str, list[EvaluationPrediction]], dict[str, float]]:
    if args.checkpoint is None or args.cache_root is None:
        raise ValueError("--checkpoint and --cache-root are required for --detector dinov3")
    from forestpersons.dinov3_detector import (
        cache_path_for_file,
        decode_fcos_predictions,
        decode_predictions,
        load_feature_cache,
    )

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    head_name = checkpoint.get("head") or checkpoint.get("settings", {}).get("head") or "mlp"
    if args.head != "auto":
        head_name = args.head
    head = make_head(head_name).to(device)
    head.load_state_dict(checkpoint["model_state"])
    head.eval()

    predictions_by_image: dict[int | str, list[EvaluationPrediction]] = {}
    seconds_total = 0.0
    with torch.inference_mode():
        for sample in samples:
            features, info = load_feature_cache(cache_path_for_file(args.cache_root, sample.file_name))
            features = features.to(device)
            start = time.perf_counter()
            if head_name == "fcos":
                logits, raw_box, centerness_logits = head(features)
                decoded = decode_fcos_predictions(
                    logits.detach().cpu(),
                    raw_box.detach().cpu(),
                    centerness_logits.detach().cpu(),
                    info,
                    conf_threshold=args.prediction_conf_min,
                    nms_threshold=args.nms_iou,
                )
            else:
                logits, raw_box = head(features)
                decoded = decode_predictions(
                    logits.detach().cpu(),
                    raw_box.detach().cpu(),
                    info,
                    conf_threshold=args.prediction_conf_min,
                    nms_threshold=args.nms_iou,
                )
            seconds_total += time.perf_counter() - start
            predictions_by_image[sample.image_id] = [
                EvaluationPrediction(
                    image_id=sample.image_id,
                    box=prediction["box"],
                    confidence=float(prediction["confidence"]),
                )
                for prediction in decoded
            ]
    return predictions_by_image, {
        "seconds_total_cached_head_only": seconds_total,
        "seconds_per_image_cached_head_only": seconds_total / len(samples) if samples else 0.0,
        "images_per_second_cached_head_only": len(samples) / seconds_total if seconds_total > 0 else 0.0,
    }


def predict_dinov3_lora(
    args: argparse.Namespace,
    samples: list[Any],
) -> tuple[dict[int | str, list[EvaluationPrediction]], dict[str, Any]]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --detector dinov3_lora")
    from forestpersons.dinov3_detector import choose_device, decode_fcos_predictions, decode_fpn_fcos_predictions
    from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint

    device = choose_device(args.device)
    processor, backbone, head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    head_kind = metadata.get("head_kind", "fcos")
    backbone.eval()
    head.eval()

    predictions_by_image: dict[int | str, list[EvaluationPrediction]] = {}
    seconds_total = 0.0
    score_bins = 20
    score_histograms = {
        name: {"counts": torch.zeros(score_bins, dtype=torch.int64), "sum": 0.0, "sum_sq": 0.0, "count": 0}
        for name in ("classification", "centerness", "quality", "combined")
    }

    def update_score_histogram(name: str, values: torch.Tensor) -> None:
        values = values.detach().float().cpu().clamp(0.0, 1.0)
        state = score_histograms[name]
        state["counts"] += torch.histc(values, bins=score_bins, min=0.0, max=1.0).to(torch.int64)
        state["sum"] += float(values.sum())
        state["sum_sq"] += float(values.square().sum())
        state["count"] += int(values.numel())

    with torch.inference_mode():
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            start = time.perf_counter()
            features, info = extract_lora_patch_features(
                image,
                processor=processor,
                model=backbone,
                device=device,
                short_side=args.short_side,
            )
            if head_kind == "fpn_fcos":
                outputs = head(features)
                decoded = decode_fpn_fcos_predictions(
                    [
                        {
                            key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in output.items()
                        }
                        for output in outputs
                    ],
                    info,
                    conf_threshold=args.prediction_conf_min,
                    nms_threshold=args.nms_iou,
                )
            else:
                head_outputs = head(features)
                if head_kind == "fcos_quality":
                    logits, raw_box, centerness_logits, quality_logits = head_outputs
                else:
                    logits, raw_box, centerness_logits = head_outputs
                    quality_logits = None
                class_scores = logits.sigmoid()
                center_scores = centerness_logits.sigmoid()
                update_score_histogram("classification", class_scores)
                update_score_histogram("centerness", center_scores)
                if quality_logits is not None:
                    quality_scores = quality_logits.sigmoid()
                    update_score_histogram("quality", quality_scores)
                else:
                    quality_scores = None
                combined_scores = (
                    class_scores.clamp(min=1e-12).pow(args.fcos_class_score_power)
                    * center_scores.clamp(min=1e-12).pow(args.fcos_centerness_score_power)
                )
                if args.fcos_quality_score_power != 0.0:
                    if quality_scores is None:
                        raise ValueError("Nonzero FCOS quality power requires an fcos_quality checkpoint")
                    combined_scores = combined_scores * quality_scores.clamp(min=1e-12).pow(
                        args.fcos_quality_score_power
                    )
                update_score_histogram("combined", combined_scores)
                feature_stride = 8 if head_kind == "stride8_fcos" else 16
                decoded = decode_fcos_predictions(
                    logits.detach().cpu(),
                    raw_box.detach().cpu(),
                    centerness_logits.detach().cpu(),
                    info,
                    conf_threshold=args.prediction_conf_min,
                    nms_threshold=args.nms_iou,
                    stride=feature_stride,
                    quality_logits=quality_logits.detach().cpu() if quality_logits is not None else None,
                    class_score_power=args.fcos_class_score_power,
                    centerness_score_power=args.fcos_centerness_score_power,
                    quality_score_power=args.fcos_quality_score_power,
                )
            seconds_total += time.perf_counter() - start
            predictions_by_image[sample.image_id] = [
                EvaluationPrediction(
                    image_id=sample.image_id,
                    box=prediction["box"],
                    confidence=float(prediction["confidence"]),
                )
                for prediction in decoded
            ]
    score_distribution_summary = {}
    edges = [index / score_bins for index in range(score_bins + 1)]
    for name, state in score_histograms.items():
        count = int(state["count"])
        mean = float(state["sum"]) / count if count else 0.0
        variance = max(0.0, float(state["sum_sq"]) / count - mean * mean) if count else 0.0
        score_distribution_summary[name] = {
            "count": count,
            "mean": mean,
            "std": variance ** 0.5,
            "bin_edges": edges,
            "counts": state["counts"].tolist(),
        }
    return predictions_by_image, {
        "seconds_total_end_to_end": seconds_total,
        "seconds_per_image_end_to_end": seconds_total / len(samples) if samples else 0.0,
        "images_per_second_end_to_end": len(samples) / seconds_total if seconds_total > 0 else 0.0,
        "pre_nms_score_distributions": score_distribution_summary,
    }


def predict_dinov3_lora_yolo(
    args: argparse.Namespace,
    samples: list[Any],
) -> tuple[dict[int | str, list[EvaluationPrediction]], dict[str, float]]:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --detector dinov3_lora_yolo")
    from forestpersons.dinov3_detector import choose_device
    from forestpersons.dinov3_lora import extract_lora_patch_features
    from forestpersons.dinov3_yolo import decode_yolo_predictions, load_lora_yolo_checkpoint

    device = choose_device(args.device)
    processor, backbone, head, _metadata = load_lora_yolo_checkpoint(args.checkpoint, device)
    backbone.eval()
    head.eval()

    predictions_by_image: dict[int | str, list[EvaluationPrediction]] = {}
    seconds_total = 0.0
    with torch.inference_mode():
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            start = time.perf_counter()
            features, info = extract_lora_patch_features(
                image,
                processor=processor,
                model=backbone,
                device=device,
                short_side=args.short_side,
            )
            outputs = head(features)
            decoded = decode_yolo_predictions(
                outputs["class_logits"].detach().cpu(),
                outputs["distance_logits"].detach().cpu(),
                info,
                conf_threshold=args.prediction_conf_min,
                nms_threshold=args.nms_iou,
                quality_logits=outputs.get("quality_logits", None).detach().cpu()
                if "quality_logits" in outputs
                else None,
                score_mode=args.yolo_score_mode,
            )
            seconds_total += time.perf_counter() - start
            predictions_by_image[sample.image_id] = [
                EvaluationPrediction(
                    image_id=sample.image_id,
                    box=prediction["box"],
                    confidence=float(prediction["confidence"]),
                )
                for prediction in decoded
            ]
    return predictions_by_image, {
        "seconds_total_end_to_end": seconds_total,
        "seconds_per_image_end_to_end": seconds_total / len(samples) if samples else 0.0,
        "images_per_second_end_to_end": len(samples) / seconds_total if seconds_total > 0 else 0.0,
    }


def save_visualizations(
    samples: list[Any],
    ground_truth_by_image: dict[int | str, list[EvaluationGroundTruth]],
    predictions_by_image: dict[int | str, list[EvaluationPrediction]],
    confidence_threshold: float,
    iou_threshold: float,
    max_visualizations: int,
    output_dir: Path,
    prediction_label: str,
) -> None:
    for sample in samples[:max_visualizations]:
        ground_truth = [
            {
                "annotation_id": gt.annotation_id,
                "box": gt.box,
                "visibility_level": gt.visibility_level,
                "pose": gt.pose,
            }
            for gt in ground_truth_by_image.get(sample.image_id, [])
        ]
        predictions = [
            {"box": pred.box, "confidence": pred.confidence}
            for pred in predictions_by_image.get(sample.image_id, [])
            if pred.confidence >= confidence_threshold
        ]
        matches, _, _ = evaluate_image_matches(ground_truth, predictions, iou_threshold)[:3]
        output_name = sample.file_name.replace("/", "__")
        draw_detection_comparison(
            sample.image_path,
            ground_truth,
            predictions,
            matches,
            output_dir / output_name,
            prediction_label=prediction_label,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(args)
    image_ids = [sample.image_id for sample in samples]
    ground_truth_by_image = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    ground_truth = [
        gt for image_ground_truth in ground_truth_by_image.values() for gt in image_ground_truth
    ]

    if args.detector == "yolo":
        predictions_by_image, timing = predict_yolo(args, samples)
        model_label = args.model.name if args.model else "yolo"
    elif args.detector == "dinov3":
        predictions_by_image, timing = predict_dinov3(args, samples)
        model_label = args.checkpoint.parent.name if args.checkpoint else "dinov3"
    elif args.detector == "dinov3_lora":
        predictions_by_image, timing = predict_dinov3_lora(args, samples)
        model_label = args.checkpoint.parent.name if args.checkpoint else "dinov3_lora"
    else:
        predictions_by_image, timing = predict_dinov3_lora_yolo(args, samples)
        model_label = args.checkpoint.parent.name if args.checkpoint else "dinov3_lora_yolo"
    predictions = [
        prediction
        for image_predictions in predictions_by_image.values()
        for prediction in image_predictions
    ]

    coco_metrics = evaluate_coco_style(ground_truth, predictions, image_ids)
    pycocotools_metrics = evaluate_pycocotools(ground_truth, predictions, image_ids)
    confidence_thresholds = parse_float_list(args.conf_thresholds)
    fp_budgets = parse_float_list(args.fp_budgets)
    custom_sweep = evaluate_confidence_sweep(
        ground_truth_by_image,
        predictions_by_image,
        image_ids,
        confidence_thresholds,
        iou_threshold=args.custom_iou,
        fp_budgets=fp_budgets,
    )

    best_visual_row = custom_sweep["matched_false_positive_budgets"].get(str(fp_budgets[0]))
    if best_visual_row is None:
        best_visual_row = custom_sweep["maximum_recall"]
    visual_conf = float(best_visual_row["confidence_threshold"]) if best_visual_row else confidence_thresholds[0]
    save_visualizations(
        samples,
        ground_truth_by_image,
        predictions_by_image,
        confidence_threshold=visual_conf,
        iou_threshold=args.custom_iou,
        max_visualizations=args.max_visualizations,
        output_dir=args.output_dir / "comparisons",
        prediction_label=model_label,
    )

    result = {
        "settings": {
            "detector": args.detector,
            "model": str(args.model) if args.model else None,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "cache_root": str(args.cache_root) if args.cache_root else None,
            "root": str(args.root),
            "split": args.split,
            "manifest": str(args.manifest),
            "max_images": args.max_images,
            "prediction_conf_min": args.prediction_conf_min,
            "nms_iou": args.nms_iou,
            "custom_iou": args.custom_iou,
            "yolo_score_mode": args.yolo_score_mode,
            "fcos_score_formula": (
                "class_probability^class_power * centerness_probability^centerness_power "
                "* quality_probability^quality_power"
            ),
            "fcos_class_score_power": args.fcos_class_score_power,
            "fcos_centerness_score_power": args.fcos_centerness_score_power,
            "fcos_quality_score_power": args.fcos_quality_score_power,
            "conf_thresholds": confidence_thresholds,
            "fp_budgets": fp_budgets,
            "imgsz": args.imgsz,
            "short_side": args.short_side,
            "device": args.device,
            "pilot_subset_warning": (
                "These are local pilot-subset results unless the manifest contains the full "
                "official ForestPersons split."
            ),
        },
        "images_evaluated": len(samples),
        "ground_truth_annotations": len(ground_truth),
        "raw_predictions": len(predictions),
        "timing": timing,
        "coco_style": coco_metrics,
        "pycocotools": pycocotools_metrics,
        "custom_confidence_sweep": custom_sweep,
    }

    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "coco_results.json").write_text(
        json.dumps(predictions_to_coco_results(predictions), indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "confidence_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "confidence_threshold",
            "ground_truth",
            "true_positives",
            "false_positives",
            "false_negatives",
            "recall",
            "precision",
            "false_positives_per_image",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in custom_sweep["rows"]:
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

    print(json.dumps(result, indent=2))
    print(f"Saved summary: {args.output_dir / 'summary.json'}")
    print(f"Saved COCO-format predictions: {args.output_dir / 'coco_results.json'}")
    print(f"Saved confidence sweep: {args.output_dir / 'confidence_sweep.csv'}")


if __name__ == "__main__":
    main()
