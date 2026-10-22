"""Evaluate the official COCO-pretrained DEIMv2-S on ForestPersons subsets."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.deimv2_adapter import (
    DEIMV2_INPUT_SIZE,
    build_deimv2_preprocess,
    load_official_deimv2,
    load_official_deimv2_one_class_checkpoint,
    official_result_to_person_predictions,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fine-tuned-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--input-size", type=int, default=DEIMV2_INPUT_SIZE)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--fp-budget", type=float, default=0.425)
    parser.add_argument(
        "--operating-threshold",
        type=float,
        help="Calibration-locked threshold to report separately from the exploratory subset sweep.",
    )
    parser.add_argument("--max-visualizations", type=int, default=8)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload["images"] if isinstance(payload, dict) and "images" in payload else payload
    if not isinstance(images, list):
        raise ValueError(f"Unsupported manifest format: {path}")
    return images


def collect_samples(
    root: Path,
    manifest_path: Path,
    max_images: int | None,
    split: str = "val",
) -> list[Any]:
    dataset = ForestPersonsDataset(root, split=split)
    by_id = {sample.image_id: sample for sample in dataset.samples}
    items = load_manifest(manifest_path)
    if max_images is not None:
        items = items[:max_images]
    samples = []
    for item in items:
        sample = by_id[item["image_id"]]
        if sample.file_name != item["file_name"]:
            raise ValueError(
                f"Manifest mismatch for image_id={sample.image_id}: "
                f"{item['file_name']} != {sample.file_name}"
            )
        if not sample.image_path.is_file():
            raise FileNotFoundError(sample.image_path)
        samples.append(sample)
    return samples


def threshold_grid(predictions: list[EvaluationPrediction]) -> list[float]:
    scores = np.asarray([prediction.confidence for prediction in predictions], dtype=np.float64)
    fixed = np.concatenate(
        [
            np.asarray([0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.005, 0.01, 0.025, 0.05]),
            np.linspace(0.075, 1.0, 38),
        ]
    )
    if scores.size:
        quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 101))
        fixed = np.concatenate([fixed, quantiles, np.nextafter(quantiles, np.inf)])
    return sorted({float(value) for value in fixed if 0.0 <= value <= 1.0})


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def predict(
    model: torch.nn.Module,
    samples: list[Any],
    device: torch.device,
    input_size: int,
    min_score: float,
) -> tuple[dict[int | str, list[EvaluationPrediction]], dict[str, Any]]:
    preprocess = build_deimv2_preprocess(input_size)
    latencies: list[float] = []
    predictions_by_image: dict[int | str, list[EvaluationPrediction]] = {}
    peak_mps_allocated = 0
    peak_mps_driver = 0
    if device.type == "mps":
        torch.mps.empty_cache()

    with torch.inference_mode():
        for index, sample in enumerate(samples):
            image = Image.open(sample.image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            original_size = torch.tensor([[image.width, image.height]], device=device)
            synchronize(device)
            started = time.perf_counter()
            result = model(image_tensor, original_size)[0]
            synchronize(device)
            latencies.append(time.perf_counter() - started)
            predictions_by_image[sample.image_id] = official_result_to_person_predictions(
                result,
                image_id=sample.image_id,
                width=image.width,
                height=image.height,
                min_score=min_score,
            )
            if device.type == "mps":
                peak_mps_allocated = max(peak_mps_allocated, int(torch.mps.current_allocated_memory()))
                peak_mps_driver = max(peak_mps_driver, int(torch.mps.driver_allocated_memory()))
            if (index + 1) % 25 == 0 or index + 1 == len(samples):
                print(f"processed {index + 1}/{len(samples)}", flush=True)

    steady = latencies[1:] if len(latencies) > 1 else latencies
    return predictions_by_image, {
        "images": len(samples),
        "seconds_total": float(sum(latencies)),
        "first_image_seconds": float(latencies[0]) if latencies else None,
        "steady_mean_seconds": float(statistics.mean(steady)) if steady else None,
        "steady_median_seconds": float(statistics.median(steady)) if steady else None,
        "steady_p95_seconds": float(np.quantile(steady, 0.95)) if steady else None,
        "peak_mps_allocated_bytes_observed": peak_mps_allocated,
        "peak_mps_driver_bytes_observed": peak_mps_driver,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def save_sweep(
    rows: list[dict[str, Any]],
    output_dir: Path,
    title: str = "DEIMv2-S recall versus false positives",
) -> None:
    with (output_dir / "confidence_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["threshold", "recall", "precision", "fp_per_image", "r20", "r40", "r70", "r100"]
        )
        for row in rows:
            by_visibility = row["by_visibility"]
            writer.writerow(
                [
                    row["confidence_threshold"],
                    row["overall"]["recall"],
                    row["overall"]["precision"],
                    row["false_positives_per_image"],
                    by_visibility.get("20", {}).get("recall", 0.0),
                    by_visibility.get("40", {}).get("recall", 0.0),
                    by_visibility.get("70", {}).get("recall", 0.0),
                    by_visibility.get("100", {}).get("recall", 0.0),
                ]
            )

    ordered = sorted(rows, key=lambda row: row["false_positives_per_image"])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        [row["false_positives_per_image"] for row in ordered],
        [row["overall"]["recall"] for row in ordered],
        marker=".",
        linewidth=1.2,
    )
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Recall at IoU 0.50")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_vs_fp_per_image.png", dpi=160)
    plt.close(fig)


def save_visualizations(
    samples: list[Any],
    ground_truth_by_image: dict[int | str, list[EvaluationGroundTruth]],
    predictions_by_image: dict[int | str, list[EvaluationPrediction]],
    threshold: float,
    count: int,
    output_dir: Path,
    prediction_label: str,
) -> None:
    for sample in samples[:count]:
        ground_truth = [
            {
                "annotation_id": gt.annotation_id,
                "box": gt.box,
                "visibility_level": gt.visibility_level,
                "pose": gt.pose,
            }
            for gt in ground_truth_by_image[sample.image_id]
        ]
        predictions = [
            {"box": prediction.box, "confidence": prediction.confidence}
            for prediction in predictions_by_image[sample.image_id]
            if prediction.confidence >= threshold
        ]
        matches = evaluate_image_matches(ground_truth, predictions, 0.50)[0]
        draw_detection_comparison(
            sample.image_path,
            ground_truth,
            predictions,
            matches,
            output_dir / sample.file_name.replace("/", "__"),
            prediction_label=prediction_label,
        )


def run(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = collect_samples(args.root, args.manifest, args.max_images, args.split)
    image_ids = [sample.image_id for sample in samples]
    ground_truth_by_image = {
        sample.image_id: annotations_to_eval_ground_truth(sample.image_id, sample.annotations)
        for sample in samples
    }
    ground_truth = [gt for values in ground_truth_by_image.values() for gt in values]

    load_started = time.perf_counter()
    if args.fine_tuned_checkpoint is None:
        model, model_metadata = load_official_deimv2(
            args.official_repo, args.checkpoint_dir, device
        )
        experiment_label = "zero-shot COCO-pretrained pilot, not ForestPersons-trained"
        prediction_label = "DEIMv2-S zero-shot"
    else:
        model, model_metadata = load_official_deimv2_one_class_checkpoint(
            args.official_repo,
            args.checkpoint_dir,
            args.fine_tuned_checkpoint,
            device,
        )
        experiment_label = "ForestPersons one-class fine-tuned DEIMv2-S pilot"
        prediction_label = "DEIMv2-S fine-tuned"
    model_load_seconds = time.perf_counter() - load_started
    predictions_by_image, timing = predict(
        model,
        samples,
        device,
        args.input_size,
        args.min_score,
    )
    predictions = [prediction for values in predictions_by_image.values() for prediction in values]
    thresholds = threshold_grid(predictions)
    if args.operating_threshold is not None:
        if not 0.0 <= args.operating_threshold <= 1.0:
            raise ValueError("operating threshold must be in [0, 1]")
        thresholds = sorted(set(thresholds + [float(args.operating_threshold)]))
    custom_sweep = evaluate_confidence_sweep(
        ground_truth_by_image,
        predictions_by_image,
        image_ids,
        thresholds,
        iou_threshold=0.50,
        fp_budgets=[args.fp_budget],
    )
    coco = evaluate_coco_style(ground_truth, predictions, image_ids)
    official_coco = evaluate_pycocotools(ground_truth, predictions, image_ids)
    matched = custom_sweep["matched_false_positive_budgets"][str(args.fp_budget)]
    locked = None
    if args.operating_threshold is not None:
        locked = next(
            row
            for row in custom_sweep["rows"]
            if abs(float(row["confidence_threshold"]) - args.operating_threshold) < 1e-12
        )
    visual_threshold = (
        args.operating_threshold
        if args.operating_threshold is not None
        else matched["confidence_threshold"] if matched else 0.0
    )

    save_sweep(custom_sweep["rows"], args.output_dir, f"{prediction_label} recall versus false positives")
    save_visualizations(
        samples,
        ground_truth_by_image,
        predictions_by_image,
        visual_threshold,
        args.max_visualizations,
        args.output_dir / "comparisons",
        prediction_label,
    )
    (args.output_dir / "coco_results.json").write_text(
        json.dumps(predictions_to_coco_results(predictions), indent=2), encoding="utf-8"
    )
    result = {
        "label": experiment_label,
        "settings": {
            "model_id": "Intellindust/DEIMv2_DINOv3_S_COCO",
            "fine_tuned_checkpoint": str(args.fine_tuned_checkpoint) if args.fine_tuned_checkpoint else None,
            "manifest": str(args.manifest),
            "split": args.split,
            "images": len(samples),
            "ground_truth": len(ground_truth),
            "device": str(device),
            "input_size": [args.input_size, args.input_size],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "person_mapping": "official zero-based COCO label 0 -> ForestPersons category_id 1",
            "nms": "none (official query-detector postprocessing)",
            "official_num_top_queries": 300,
            "minimum_saved_score": args.min_score,
            "fp_budget": args.fp_budget,
            "threshold_selection": "exploratory selection on the evaluated subset",
            "calibration_locked_threshold": args.operating_threshold,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": torch.backends.mps.is_available(),
        },
        "model": model_metadata,
        "model_load_seconds": model_load_seconds,
        "timing_memory": timing,
        "prediction_count": len(predictions),
        "score_range": {
            "minimum": min((p.confidence for p in predictions), default=None),
            "maximum": max((p.confidence for p in predictions), default=None),
        },
        "coco_style_local": coco,
        "coco_official_pycocotools": official_coco,
        "custom_sweep": custom_sweep,
        "calibration_locked_operating_point": locked,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
    requested = torch.device(args.device)
    if requested.type == "mps" and not torch.backends.mps.is_available():
        if not args.allow_cpu_fallback:
            raise RuntimeError("MPS requested but unavailable")
        requested = torch.device("cpu")
    try:
        result = run(args, requested)
    except RuntimeError:
        if requested.type != "mps" or not args.allow_cpu_fallback:
            raise
        print("MPS execution failed; retrying the complete run on CPU", flush=True)
        result = run(args, torch.device("cpu"))
    matched = result["custom_sweep"]["matched_false_positive_budgets"][str(args.fp_budget)]
    print(json.dumps({"coco": result["coco_style_local"], "matched_fp_budget": matched}, indent=2))


if __name__ == "__main__":
    main()
