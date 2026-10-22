#!/usr/bin/env python3
"""Run a bounded hybrid ANN2SNN pilot for the trained DINOv3 MVR detector."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_ann2snn import (
    ActivationPercentileCalibrator,
    SpikingMVRReadout,
    aggregate_spike_activity,
    ann_mvr_activations,
)
from forestpersons.dinov3_detector import (
    DINOv3MVRFCOSHead,
    cache_path_for_file,
    decode_fcos_predictions,
    load_feature_cache,
    load_mvr_feature_cache,
)
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_custom_metrics,
    predictions_to_coco_results,
)


def collect_samples(
    root: Path,
    split: str,
    manifest: Path,
    limit: int,
    sampling: str = "prefix",
) -> list[Any]:
    dataset = ForestPersonsDataset(root, split=split)
    by_id = {int(sample.image_id): sample for sample in dataset.samples}
    items = json.loads(manifest.read_text(encoding="utf-8"))["images"]
    if limit <= 0 or limit > len(items):
        raise ValueError(f"limit must be in [1, {len(items)}]")
    items = select_manifest_items(items, limit, sampling)
    samples = []
    for item in items:
        sample = by_id[int(item["image_id"])]
        if sample.file_name != item["file_name"]:
            raise ValueError(f"Filename mismatch for image_id {item['image_id']}")
        samples.append(sample)
    return samples


def select_manifest_items(
    items: list[dict[str, Any]],
    limit: int,
    sampling: str,
) -> list[dict[str, Any]]:
    """Select a deterministic prefix or visibility-stratified manifest subset."""

    if sampling == "prefix":
        return items[:limit]
    if sampling != "stratified":
        raise ValueError(f"Unknown sampling strategy: {sampling}")
    levels = (20, 40, 70, 100)
    base, remainder = divmod(limit, len(levels))
    selected = []
    for index, level in enumerate(levels):
        requested = base + (1 if index < remainder else 0)
        candidates = [
            item for item in items if int(item.get("target_visibility", -1)) == level
        ]
        if len(candidates) < requested:
            raise ValueError(
                f"Need {requested} target-visibility-{level} images, found {len(candidates)}"
            )
        selected.extend(candidates[:requested])
    return selected


def sample_visibility_distribution(samples: list[Any]) -> dict[str, Any]:
    annotation_counts = {str(level): 0 for level in (20, 40, 70, 100)}
    image_presence = {str(level): 0 for level in (20, 40, 70, 100)}
    for sample in samples:
        present = set()
        for annotation in sample.annotations:
            visibility = annotation.get("metadata", {}).get("visibility_level")
            if visibility is None:
                continue
            key = str(int(visibility))
            if key in annotation_counts:
                annotation_counts[key] += 1
                present.add(key)
        for key in present:
            image_presence[key] += 1
    return {
        "annotation_counts": annotation_counts,
        "image_presence": image_presence,
    }


def load_layers(
    sample: Any,
    final_cache_root: Path,
    mvr_cache_root: Path,
) -> tuple[torch.Tensor, Any]:
    final, info = load_feature_cache(cache_path_for_file(final_cache_root, sample.file_name))
    intermediate, mvr_info, layers = load_mvr_feature_cache(
        cache_path_for_file(mvr_cache_root, sample.file_name)
    )
    if info != mvr_info or layers != (3, 6, 9):
        raise ValueError(f"Feature cache mismatch for {sample.file_name}")
    return torch.cat((intermediate, final.unsqueeze(0)), dim=0), info


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def calibrate(
    model: DINOv3MVRFCOSHead,
    samples: list[Any],
    final_cache_root: Path,
    mvr_cache_root: Path,
    device: torch.device,
    percentile: float,
) -> dict[str, float]:
    calibrator = ActivationPercentileCalibrator()
    with torch.inference_mode():
        for sample in samples:
            features, _ = load_layers(sample, final_cache_root, mvr_cache_root)
            activations = ann_mvr_activations(model.readout, features.to(device))
            calibrator.update(activations)
    return calibrator.thresholds(percentile)


def evaluate_mode(
    name: str,
    model: DINOv3MVRFCOSHead,
    samples: list[Any],
    final_cache_root: Path,
    mvr_cache_root: Path,
    device: torch.device,
    confidence_threshold: float,
    spiking_readout: SpikingMVRReadout | None,
) -> tuple[dict[str, Any], list[EvaluationPrediction]]:
    ground_truth_by_image = {}
    predictions_by_image = {}
    all_predictions: list[EvaluationPrediction] = []
    image_ids = []
    image_spike_stats = []
    elapsed = 0.0
    peak_mps = 0
    with torch.inference_mode():
        for sample in samples:
            features, info = load_layers(sample, final_cache_root, mvr_cache_root)
            features = features.to(device)
            synchronize(device)
            started = time.perf_counter()
            if spiking_readout is None:
                outputs = model(features)
            else:
                fused, spike_stats = spiking_readout(features)
                image_spike_stats.append(spike_stats)
                outputs = model.fcos(fused)
            synchronize(device)
            elapsed += time.perf_counter() - started
            if device.type == "mps":
                peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))
            class_logits, raw_box, centerness = outputs
            decoded = decode_fcos_predictions(
                class_logits.detach().cpu(),
                raw_box.detach().cpu(),
                centerness.detach().cpu(),
                info,
                conf_threshold=0.001,
                nms_threshold=0.50,
                topk=300,
            )
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
            predictions_by_image[image_id] = image_predictions
            all_predictions.extend(image_predictions)
            ground_truth_by_image[image_id] = annotations_to_eval_ground_truth(
                image_id, sample.annotations
            )
    ground_truth = [item for values in ground_truth_by_image.values() for item in values]
    coco = evaluate_coco_style(ground_truth, all_predictions, image_ids)
    custom = evaluate_custom_metrics(
        ground_truth_by_image,
        predictions_by_image,
        image_ids,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.50,
    )
    result = {
        "mode": name,
        "images": len(image_ids),
        "ground_truth": len(ground_truth),
        "AP50": coco["AP50"],
        "recall": custom["overall"]["recall"],
        "precision": custom["overall"]["precision"],
        "R20": custom["by_visibility"].get("20", {}).get("recall", 0.0),
        "R40": custom["by_visibility"].get("40", {}).get("recall", 0.0),
        "R70": custom["by_visibility"].get("70", {}).get("recall", 0.0),
        "R100": custom["by_visibility"].get("100", {}).get("recall", 0.0),
        "false_positives_per_image": custom["false_positives_per_image"],
        "confidence_threshold": confidence_threshold,
        "latency_seconds": elapsed,
        "latency_seconds_per_image": elapsed / len(image_ids),
        "peak_mps_driver_bytes": peak_mps if device.type == "mps" else None,
        "spike_activity": (
            aggregate_spike_activity(image_spike_stats) if image_spike_stats else None
        ),
    }
    return result, all_predictions


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# DINOv3 MVR hybrid ANN2SNN pilot",
        "",
        "Frozen signed DINOv3 features remain analog. Only the six ReLU sites in "
        "the MVR readout use calibrated deterministic IF rate coding. Residual "
        "fusion, feature normalization, the GroupNorm/GELU FCOS tower, analog box "
        "regression, NMS, and decoding are unchanged.",
        "",
        "Spike activity is an operation proxy only; it is not an energy measurement.",
        "",
        "| Mode | AP50 | Recall | R20 | R40 | R70 | R100 | Precision | FP/image | s/image | Spike activity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        activity = result["spike_activity"]
        activity_text = f"{activity['overall']:.4f}" if activity else "n/a"
        lines.append(
            f"| {result['mode']} | {result['AP50']:.4f} | {result['recall']:.4f} | "
            f"{result['R20']:.4f} | {result['R40']:.4f} | {result['R70']:.4f} | "
            f"{result['R100']:.4f} | {result['precision']:.4f} | "
            f"{result['false_positives_per_image']:.4f} | "
            f"{result['latency_seconds_per_image']:.4f} | {activity_text} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is a hybrid layerwise rate-conversion study, not deployment on "
            "neuromorphic hardware. Analog convolutions are still executed densely, "
            "and latency on MPS does not represent event-driven hardware latency.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--train-final-cache-root", type=Path)
    parser.add_argument("--train-mvr-cache-root", type=Path)
    parser.add_argument("--val-final-cache-root", type=Path, required=True)
    parser.add_argument("--val-mvr-cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=100)
    parser.add_argument("--eval-images", type=int, default=100)
    parser.add_argument("--calibration-percentile", type=float, default=99.9)
    parser.add_argument("--reuse-calibration-metrics", type=Path)
    parser.add_argument("--timesteps", default="4,8,16")
    parser.add_argument(
        "--eval-sampling", choices=("prefix", "stratified"), default="prefix"
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.63)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    timesteps = tuple(int(value) for value in args.timesteps.split(","))
    if not timesteps or any(value <= 0 for value in timesteps):
        raise ValueError("--timesteps must contain positive integers")
    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("arm") != "mvr":
        raise ValueError("Checkpoint is not a trained MVR arm")
    model = DINOv3MVRFCOSHead().to(device).eval()
    model.load_state_dict(checkpoint["model_state"])

    evaluation_samples = collect_samples(
        args.root,
        "val",
        args.val_manifest,
        args.eval_images,
        sampling=args.eval_sampling,
    )
    if args.reuse_calibration_metrics:
        source = json.loads(args.reuse_calibration_metrics.read_text(encoding="utf-8"))
        if Path(source["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError("Reused calibration was produced from a different checkpoint")
        thresholds = {
            str(name): float(value)
            for name, value in source["calibration"]["thresholds"].items()
        }
        calibration_record = {
            **source["calibration"],
            "reused_from": str(args.reuse_calibration_metrics.resolve()),
        }
    else:
        required = {
            "--train-manifest": args.train_manifest,
            "--train-final-cache-root": args.train_final_cache_root,
            "--train-mvr-cache-root": args.train_mvr_cache_root,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"Fresh calibration requires: {', '.join(missing)}"
            )
        calibration_samples = collect_samples(
            args.root, "train", args.train_manifest, args.calibration_images
        )
        thresholds = calibrate(
            model,
            calibration_samples,
            args.train_final_cache_root,
            args.train_mvr_cache_root,
            device,
            args.calibration_percentile,
        )
        calibration_record = {
            "split": "ForestPersons train manifest only",
            "manifest": str(args.train_manifest.resolve()),
            "images": args.calibration_images,
            "image_ids": [int(sample.image_id) for sample in calibration_samples],
            "percentile": args.calibration_percentile,
            "thresholds": thresholds,
        }

    results = []
    ann_metrics, ann_predictions = evaluate_mode(
        "ANN",
        model,
        evaluation_samples,
        args.val_final_cache_root,
        args.val_mvr_cache_root,
        device,
        args.confidence_threshold,
        None,
    )
    results.append(ann_metrics)
    (args.output_dir / "ann_predictions_coco.json").write_text(
        json.dumps(predictions_to_coco_results(ann_predictions), indent=2) + "\n"
    )
    for timestep in timesteps:
        spiking = SpikingMVRReadout(model.readout, thresholds, timestep).to(device).eval()
        metrics, predictions = evaluate_mode(
            f"SNN_T{timestep}",
            model,
            evaluation_samples,
            args.val_final_cache_root,
            args.val_mvr_cache_root,
            device,
            args.confidence_threshold,
            spiking,
        )
        results.append(metrics)
        (args.output_dir / f"snn_t{timestep}_predictions_coco.json").write_text(
            json.dumps(predictions_to_coco_results(predictions), indent=2) + "\n"
        )

    payload = {
        "experiment": "hybrid_mvr_relu_ann2snn",
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "calibration": calibration_record,
        "evaluation": {
            "split": "ForestPersons val600 manifest; no threshold tuning",
            "manifest": str(args.val_manifest.resolve()),
            "images": args.eval_images,
            "sampling": args.eval_sampling,
            "image_ids": [int(sample.image_id) for sample in evaluation_samples],
            "visibility_distribution": sample_visibility_distribution(
                evaluation_samples
            ),
            "iou_threshold": 0.50,
            "confidence_threshold": args.confidence_threshold,
            "candidate_floor": 0.001,
            "nms_iou_threshold": 0.50,
            "topk": 300,
        },
        "conversion_boundary": {
            "spiking": [
                "MVR layer projection ReLUs (4)",
                "MVR fine-context ReLU",
                "MVR coarse-context ReLU",
            ],
            "analog": [
                "cached frozen signed DINOv3 features",
                "MVR residual projection and L2 normalization",
                "FCOS projection, GroupNorm/GELU tower, classification and centerness",
                "box regression, decoding and NMS",
            ],
        },
        "results": results,
        "energy_claim": "None; spike activity is not a hardware energy measurement.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "mode",
            "AP50",
            "recall",
            "R20",
            "R40",
            "R70",
            "R100",
            "precision",
            "false_positives_per_image",
            "latency_seconds_per_image",
            "spike_activity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **{key: result[key] for key in fieldnames[:-1]},
                    "spike_activity": (
                        result["spike_activity"]["overall"]
                        if result["spike_activity"]
                        else None
                    ),
                }
            )
    write_report(args.output_dir / "report.md", payload)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
