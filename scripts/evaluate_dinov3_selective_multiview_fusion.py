"""Evaluate conservative alignment-aware feature fusion on benchmark49.

The single-view and equal-mean arms are loaded from the frozen benchmark49
prediction dumps. Only the selective arm performs model inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, draw_detection_comparison, evaluate_image_matches, xywh_to_xyxy
from forestpersons.dinov3_detector import choose_device, decode_fcos_predictions
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_custom_metrics,
    evaluate_pycocotools,
    predictions_to_coco_results,
)
from forestpersons.multiview import (
    fuse_selective_aligned_features,
    validate_unique_group_images,
    warp_patch_features,
)


VISIBILITIES = (20, 40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--prediction-conf-min", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261017)
    parser.add_argument("--visual-examples", type=int, default=12)
    return parser.parse_args()


def load_coco_predictions(path: Path) -> dict[int, list[EvaluationPrediction]]:
    output: dict[int, list[EvaluationPrediction]] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        image_id = int(item["image_id"])
        output.setdefault(image_id, []).append(
            EvaluationPrediction(
                image_id=image_id,
                box=xywh_to_xyxy([float(value) for value in item["bbox"]]),
                confidence=float(item["score"]),
            )
        )
    return output


def decode(head: torch.nn.Module, features: torch.Tensor, info: Any, args: argparse.Namespace) -> list[EvaluationPrediction]:
    logits, raw_box, centerness = head(features)
    items = decode_fcos_predictions(
        logits.detach().cpu(),
        raw_box.detach().cpu(),
        centerness.detach().cpu(),
        info,
        conf_threshold=args.prediction_conf_min,
        nms_threshold=args.nms_iou,
    )
    return [
        EvaluationPrediction(image_id=-1, box=item["box"], confidence=float(item["confidence"]))
        for item in items
    ]


def metrics(
    gt_by_image: dict[int, list[EvaluationGroundTruth]],
    predictions: dict[int, list[EvaluationPrediction]],
    image_ids: list[int],
    confidence: float,
) -> dict[str, Any]:
    flat_gt = [item for image_id in image_ids for item in gt_by_image[image_id]]
    flat_predictions = [item for image_id in image_ids for item in predictions.get(image_id, [])]
    return {
        "fixed_threshold": evaluate_custom_metrics(gt_by_image, predictions, image_ids, confidence, 0.50),
        "coco_style": evaluate_coco_style(flat_gt, flat_predictions, image_ids),
        "pycocotools": evaluate_pycocotools(flat_gt, flat_predictions, image_ids),
    }


def matched_annotation_ids(
    ground_truth: list[EvaluationGroundTruth],
    predictions: list[EvaluationPrediction],
    confidence: float,
) -> tuple[set[int | str | None], int]:
    gt = [{"box": item.box, "visibility_level": item.visibility_level, "annotation_id": item.annotation_id} for item in ground_truth]
    pred = [{"box": item.box, "confidence": item.confidence} for item in predictions if item.confidence >= confidence]
    _, matched_gt, matched_pred, _, _ = evaluate_image_matches(gt, pred, 0.50)
    return {ground_truth[index].annotation_id for index in matched_gt}, len(pred) - len(matched_pred)


def best_iou(predictions: list[EvaluationPrediction], box: list[float]) -> float:
    return max((box_iou(box, item.box) for item in predictions), default=0.0)


def sequence_bootstrap(rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    sequences = sorted({str(row["sequence"]) for row in rows})
    by_sequence = {sequence: [row for row in rows if str(row["sequence"]) == sequence] for sequence in sequences}
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for visibility in (None, *VISIBILITIES):
        label = "overall" if visibility is None else str(visibility)
        differences = []
        for _ in range(replicates):
            sampled = rng.choice(sequences, size=len(sequences), replace=True)
            selected = [row for sequence in sampled for row in by_sequence[str(sequence)] if visibility is None or row["visibility"] == visibility]
            if not selected:
                continue
            differences.append(np.mean([row["selective_detected"] for row in selected]) - np.mean([row["single_detected"] for row in selected]))
        values = np.asarray(differences, dtype=np.float64)
        observed_rows = [row for row in rows if visibility is None or row["visibility"] == visibility]
        output[label] = {
            "observed_difference": float(np.mean([row["selective_detected"] for row in observed_rows]) - np.mean([row["single_detected"] for row in observed_rows])),
            "ci95_percentile": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
            "probability_difference_above_zero": float(np.mean(values > 0.0)),
            "valid_replicates": int(values.size),
            "resampling_unit": "ForestPersons sequence",
        }
    return output


def assert_reproduced(name: str, measured: dict[str, Any], saved: dict[str, Any]) -> None:
    checks = (
        ("recall", measured["fixed_threshold"]["overall"]["recall"], saved["fixed_threshold"]["overall"]["recall"]),
        ("FP/image", measured["fixed_threshold"]["false_positives_per_image"], saved["fixed_threshold"]["false_positives_per_image"]),
        ("AP", measured["pycocotools"]["AP"], saved["pycocotools"]["AP"]),
        ("AR@100", measured["pycocotools"]["AR@100"], saved["pycocotools"]["AR@100"]),
    )
    mismatches = [f"{metric}: {actual} != {expected}" for metric, actual, expected in checks if abs(float(actual) - float(expected)) > 1e-10]
    if mismatches:
        raise RuntimeError(f"Frozen {name} reproduction mismatch: " + "; ".join(mismatches))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detection_panel(path: Path, gt: list[EvaluationGroundTruth], predictions: list[EvaluationPrediction], confidence: float, title: str, output: Path) -> None:
    gt_items = [{"box": item.box, "visibility_level": item.visibility_level, "annotation_id": item.annotation_id} for item in gt]
    pred_items = [{"box": item.box, "confidence": item.confidence} for item in predictions if item.confidence >= confidence]
    matches = evaluate_image_matches(gt_items, pred_items, 0.50)[0]
    draw_detection_comparison(path, gt_items, pred_items, matches, output, title)


def make_gallery(alignment: Path, panels: list[Path], output: Path, title: str) -> None:
    images = [Image.open(alignment).convert("RGB"), *[Image.open(path).convert("RGB") for path in panels]]
    width = 1400
    resized = [image.resize((width, round(image.height * width / image.width))) for image in images]
    header = 46
    canvas = Image.new("RGB", (width, header + sum(image.height for image in resized)), "white")
    ImageDraw.Draw(canvas).text((14, 14), title, fill="black")
    y = header
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("selection_uses_annotations") is not False or manifest.get("selection_uses_model_predictions") is not False:
        raise ValueError("Benchmark selection must be independent of annotations and model predictions")
    groups = manifest["groups"]
    validate_unique_group_images(groups)

    dataset = ForestPersonsDataset(args.root, split="val")
    samples = {int(sample.image_id): sample for sample in dataset.samples}
    image_ids = [int(group["reference"]["image_id"]) for group in groups]
    gt_by_image = {image_id: annotations_to_eval_ground_truth(image_id, samples[image_id].annotations) for image_id in image_ids}

    baseline_summary = json.loads((args.baseline_dir / "summary.json").read_text(encoding="utf-8"))
    single = load_coco_predictions(args.baseline_dir / "single_view_coco_results.json")
    mean = load_coco_predictions(args.baseline_dir / "multi_view_coco_results.json")
    single_metrics = metrics(gt_by_image, single, image_ids, args.confidence)
    mean_metrics = metrics(gt_by_image, mean, image_ids, args.confidence)
    assert_reproduced("single-view", single_metrics, baseline_summary["single_view"])
    assert_reproduced("equal-mean", mean_metrics, baseline_summary["multi_view"])

    device = choose_device(args.device)
    processor, backbone, head, _ = load_lora_fcos_checkpoint(args.checkpoint, device)
    backbone.eval()
    head.eval()
    selective: dict[int, list[EvaluationPrediction]] = {}
    group_rows: list[dict[str, Any]] = []
    gate_maps: dict[str, torch.Tensor] = {}
    reference_infos: dict[str, Any] = {}
    total_seconds = 0.0
    peak_memory = 0

    with torch.inference_mode():
        for index, group in enumerate(groups, start=1):
            started = time.perf_counter()
            reference_id = int(group["reference"]["image_id"])
            reference_image = Image.open(samples[reference_id].image_path).convert("RGB")
            reference_features, reference_info = extract_lora_patch_features(reference_image, processor, backbone, device, short_side=args.short_side)
            aligned_sources = []
            for source in group["sources"]:
                source_id = int(source["image_id"])
                source_image = Image.open(samples[source_id].image_path).convert("RGB")
                source_features, source_info = extract_lora_patch_features(source_image, processor, backbone, device, short_side=args.short_side)
                homography = torch.tensor(source["homography_source_to_reference"], dtype=torch.float32, device=device)
                aligned_sources.append(warp_patch_features(source_features, source_info, reference_info, homography))
            fused, gate = fuse_selective_aligned_features(reference_features, aligned_sources)
            predictions = decode(head, fused, reference_info, args)
            selective[reference_id] = [EvaluationPrediction(reference_id, item.box, item.confidence) for item in predictions]
            if device.type == "mps":
                torch.mps.synchronize()
                peak_memory = max(peak_memory, int(torch.mps.driver_allocated_memory()))
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            total_weight = gate["total_source_weight"].detach().cpu()
            gate_maps[group["group_id"]] = total_weight
            reference_infos[group["group_id"]] = reference_info
            group_rows.append({
                "group_id": group["group_id"],
                "sequence": group["sequence"],
                "reference_image_id": reference_id,
                "ground_truth_people": len(gt_by_image[reference_id]),
                "active_patch_fraction": float((total_weight > 0).float().mean()),
                "fallback_patch_fraction": float((total_weight <= 1e-8).float().mean()),
                "mean_total_source_weight": float(total_weight.mean()),
                "maximum_total_source_weight": float(total_weight.max()),
                "seconds": elapsed,
            })
            print(f"[{index:02d}/{len(groups)}] {group['group_id']} active={group_rows[-1]['active_patch_fraction']:.3f}", flush=True)

    selective_metrics = metrics(gt_by_image, selective, image_ids, args.confidence)
    per_gt_rows: list[dict[str, Any]] = []
    for group in groups:
        group_id = group["group_id"]
        sequence = group["sequence"]
        image_id = int(group["reference"]["image_id"])
        gt_items = gt_by_image[image_id]
        single_ids, _ = matched_annotation_ids(gt_items, single.get(image_id, []), args.confidence)
        mean_ids, _ = matched_annotation_ids(gt_items, mean.get(image_id, []), args.confidence)
        selective_ids, _ = matched_annotation_ids(gt_items, selective.get(image_id, []), args.confidence)
        sample = samples[image_id]
        weight_map = gate_maps[group_id]
        info = reference_infos[group_id]
        for gt in gt_items:
            x1, y1, x2, y2 = gt.box
            center_x = int(np.clip(((x1 + x2) * 0.5 * info.scale_x / 16.0) - 0.5, 0, weight_map.shape[1] - 1))
            center_y = int(np.clip(((y1 + y2) * 0.5 * info.scale_y / 16.0) - 0.5, 0, weight_map.shape[0] - 1))
            per_gt_rows.append({
                "group_id": group_id,
                "sequence": sequence,
                "reference_image_id": image_id,
                "annotation_id": gt.annotation_id,
                "visibility": gt.visibility_level,
                "single_detected": gt.annotation_id in single_ids,
                "mean_detected": gt.annotation_id in mean_ids,
                "selective_detected": gt.annotation_id in selective_ids,
                "single_best_iou": best_iou(single.get(image_id, []), gt.box),
                "mean_best_iou": best_iou(mean.get(image_id, []), gt.box),
                "selective_best_iou": best_iou(selective.get(image_id, []), gt.box),
                "posthoc_gate_weight_at_gt_center": float(weight_map[center_y, center_x]),
                "reference_width": sample.width,
                "reference_height": sample.height,
            })

    rescued = [row for row in per_gt_rows if row["selective_detected"] and not row["single_detected"]]
    lost = [row for row in per_gt_rows if row["single_detected"] and not row["selective_detected"]]
    restored_from_mean = [row for row in per_gt_rows if row["single_detected"] and not row["mean_detected"] and row["selective_detected"]]
    bootstrap = sequence_bootstrap(per_gt_rows, args.bootstrap_replicates, args.bootstrap_seed)
    write_csv(args.output_dir / "per_group_diagnostics.csv", group_rows)
    write_csv(args.output_dir / "per_gt_diagnostics.csv", per_gt_rows)
    (args.output_dir / "sequence_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    flat_selective = [item for image_id in image_ids for item in selective.get(image_id, [])]
    (args.output_dir / "selective_coco_results.json").write_text(json.dumps(predictions_to_coco_results(flat_selective), indent=2), encoding="utf-8")

    priority_ids = list(dict.fromkeys(
        [row["group_id"] for row in rescued]
        + [row["group_id"] for row in lost]
        + [row["group_id"] for row in restored_from_mean]
        + [row["group_id"] for row in sorted(per_gt_rows, key=lambda row: abs(row["selective_best_iou"] - row["single_best_iou"]), reverse=True)]
    ))[: args.visual_examples]
    groups_by_id = {group["group_id"]: group for group in groups}
    visual_index = []
    for group_id in priority_ids:
        group = groups_by_id[group_id]
        image_id = int(group["reference"]["image_id"])
        parts = args.output_dir / "visual_examples" / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        panel_paths = []
        for label, prediction_map in (("A_single", single), ("B_equal_mean", mean), ("C_selective", selective)):
            panel = parts / f"{group_id}_{label}.jpg"
            detection_panel(samples[image_id].image_path, gt_by_image[image_id], prediction_map.get(image_id, []), args.confidence, label, panel)
            panel_paths.append(panel)
        reason = "rescued" if any(row["group_id"] == group_id for row in rescued) else "lost" if any(row["group_id"] == group_id for row in lost) else "restored_or_changed"
        output = args.output_dir / "visual_examples" / f"{group_id}_{reason}.jpg"
        make_gallery(args.manifest.parent / "alignment_gallery" / f"{group_id}.jpg", panel_paths, output, f"{group_id} | {group['sequence']} | {reason}")
        visual_index.append({"group_id": group_id, "reason": reason, "path": str(output)})
    (args.output_dir / "visual_examples" / "index.json").write_text(json.dumps(visual_index, indent=2), encoding="utf-8")

    result = {
        "settings": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "short_side": args.short_side,
            "fixed_confidence": args.confidence,
            "prediction_confidence_floor": args.prediction_conf_min,
            "nms_iou": args.nms_iou,
            "fusion_rule": {
                "minimum_cosine": 0.90,
                "full_trust_cosine": 0.98,
                "local_kernel_size": 3,
                "minimum_local_valid_fraction": 0.75,
                "maximum_total_source_weight": 0.50,
                "reference_fallback": True,
                "gt_or_detection_assisted": False,
            },
            "benchmark49_is_exploratory": True,
            "training": False,
        },
        "frozen_baseline_reproduction": "exact for recall, FP/image, AP, and AR@100",
        "single_view": single_metrics,
        "equal_mean": mean_metrics,
        "selective": selective_metrics,
        "paired_fixed_threshold": {
            "selective_rescued_vs_single": len(rescued),
            "selective_lost_vs_single": len(lost),
            "net_change": len(rescued) - len(lost),
            "single_losses_from_equal_mean_restored_by_selective": len(restored_from_mean),
        },
        "gate_diagnostics": {
            "mean_active_patch_fraction": float(np.mean([row["active_patch_fraction"] for row in group_rows])),
            "mean_fallback_patch_fraction": float(np.mean([row["fallback_patch_fraction"] for row in group_rows])),
            "mean_total_source_weight": float(np.mean([row["mean_total_source_weight"] for row in group_rows])),
            "note": "GT-center weights are post-hoc diagnostics only and never influence fusion.",
        },
        "sequence_cluster_bootstrap": bootstrap,
        "timing": {"selective_total_seconds": total_seconds, "selective_seconds_per_group": total_seconds / len(groups)},
        "peak_mps_driver_memory_bytes": peak_memory,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "single": single_metrics["fixed_threshold"],
        "equal_mean": mean_metrics["fixed_threshold"],
        "selective": selective_metrics["fixed_threshold"],
        "selective_AP": selective_metrics["pycocotools"]["AP"],
        "selective_AP50": selective_metrics["pycocotools"]["AP50"],
        "selective_AP75": selective_metrics["pycocotools"]["AP75"],
        "selective_AR100": selective_metrics["pycocotools"]["AR@100"],
        "rescued": len(rescued),
        "lost": len(lost),
        "restored_from_mean": len(restored_from_mean),
        "timing": result["timing"],
        "peak_mps_driver_memory_bytes": peak_memory,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
