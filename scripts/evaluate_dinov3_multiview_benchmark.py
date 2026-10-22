"""Evaluate a frozen registration-only DINOv3 multi-view benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, draw_detection_comparison, evaluate_image_matches
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
from forestpersons.multiview import fuse_aligned_features, validate_unique_group_images, warp_patch_features


VISIBILITIES = (20, 40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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


def mps_driver_memory() -> int:
    if not torch.backends.mps.is_available():
        return 0
    return int(torch.mps.driver_allocated_memory())


def eval_predictions(items: list[dict[str, Any]], image_id: int) -> list[EvaluationPrediction]:
    return [
        EvaluationPrediction(image_id=image_id, box=item["box"], confidence=float(item["confidence"]))
        for item in items
    ]


def fixed_matches(
    ground_truth: list[EvaluationGroundTruth],
    predictions: list[EvaluationPrediction],
    confidence: float,
) -> tuple[set[int | str | None], int, int, list[dict[str, Any]]]:
    gt = [
        {
            "box": item.box,
            "visibility_level": item.visibility_level,
            "annotation_id": item.annotation_id,
            "pose": item.pose,
        }
        for item in ground_truth
    ]
    pred = [
        {"box": item.box, "confidence": item.confidence}
        for item in predictions
        if item.confidence >= confidence
    ]
    matches, matched_gt, matched_pred, _, _ = evaluate_image_matches(gt, pred, 0.50)
    annotation_ids = {ground_truth[index].annotation_id for index in matched_gt}
    return annotation_ids, len(pred) - len(matched_pred), len(ground_truth) - len(matched_gt), matches


def best_iou(predictions: list[EvaluationPrediction], box: list[float]) -> float:
    return max((box_iou(box, prediction.box) for prediction in predictions), default=0.0)


def sequence_cluster_bootstrap(
    rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired recall-difference bootstrap with sequence as the resampling unit."""

    sequences = sorted({str(row["sequence"]) for row in rows})
    grouped: dict[str, dict[str, float]] = {
        sequence: defaultdict(float) for sequence in sequences
    }
    for row in rows:
        bucket = grouped[str(row["sequence"])]
        visibility = int(row["visibility"])
        bucket["gt"] += 1
        bucket["single_tp"] += int(bool(row["single_detected_fixed"]))
        bucket["fused_tp"] += int(bool(row["fused_detected_fixed"]))
        bucket[f"gt_{visibility}"] += 1
        bucket[f"single_tp_{visibility}"] += int(bool(row["single_detected_fixed"]))
        bucket[f"fused_tp_{visibility}"] += int(bool(row["fused_detected_fixed"]))
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for label in ("overall", *[str(value) for value in VISIBILITIES]):
        suffix = "" if label == "overall" else f"_{label}"
        differences: list[float] = []
        for _ in range(replicates):
            sampled = rng.integers(0, len(sequences), size=len(sequences))
            denominator = sum(grouped[sequences[index]][f"gt{suffix}"] for index in sampled)
            if denominator <= 0:
                continue
            single = sum(grouped[sequences[index]][f"single_tp{suffix}"] for index in sampled) / denominator
            fused = sum(grouped[sequences[index]][f"fused_tp{suffix}"] for index in sampled) / denominator
            differences.append(fused - single)
        values = np.asarray(differences, dtype=np.float64)
        output[label] = {
            "observed_difference": (
                sum(grouped[item][f"fused_tp{suffix}"] for item in sequences)
                - sum(grouped[item][f"single_tp{suffix}"] for item in sequences)
            ) / max(sum(grouped[item][f"gt{suffix}"] for item in sequences), 1.0),
            "bootstrap_mean_difference": float(values.mean()) if values.size else None,
            "ci95_percentile": (
                [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                if values.size else None
            ),
            "probability_difference_above_zero": float(np.mean(values > 0.0)) if values.size else None,
            "valid_replicates": int(values.size),
            "resampling_unit": "ForestPersons sequence",
        }
    return output


def combine_gallery(alignment: Path, comparison: Path, output: Path, title: str) -> None:
    first = Image.open(alignment).convert("RGB")
    second = Image.open(comparison).convert("RGB")
    target_width = 1200
    first = first.resize((target_width, round(first.height * target_width / first.width)))
    second = second.resize((target_width, round(second.height * target_width / second.width)))
    header = 52
    canvas = Image.new("RGB", (target_width, first.height + second.height + header), "white")
    canvas.paste(first, (0, header))
    canvas.paste(second, (0, header + first.height))
    ImageDraw.Draw(canvas).text((16, 16), title, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("selection_uses_annotations") is not False:
        raise ValueError("Benchmark manifest must explicitly exclude annotation-assisted selection")
    if manifest.get("selection_uses_model_predictions") is not False:
        raise ValueError("Benchmark manifest must explicitly exclude prediction-assisted selection")
    groups = manifest["groups"]
    validate_unique_group_images(groups)

    dataset = ForestPersonsDataset(args.root, split="val")
    samples_by_id = {int(sample.image_id): sample for sample in dataset.samples}
    for group in groups:
        for view in [group["reference"], *group["sources"]]:
            sample = samples_by_id[int(view["image_id"])]
            if sample.file_name != view["file_name"]:
                raise ValueError(f"Manifest filename mismatch for image {view['image_id']}")
            if not sample.image_path.exists():
                raise FileNotFoundError(sample.image_path)

    device = choose_device(args.device)
    processor, backbone, head, checkpoint_metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    backbone.eval()
    head.eval()
    single_by_image: dict[int, list[EvaluationPrediction]] = {}
    fused_by_image: dict[int, list[EvaluationPrediction]] = {}
    diagnostics: list[dict[str, Any]] = []
    single_seconds = 0.0
    source_fusion_seconds = 0.0
    peak_memory = 0

    with torch.inference_mode():
        for group_index, group in enumerate(groups, start=1):
            reference = group["reference"]
            image_id = int(reference["image_id"])
            sample = samples_by_id[image_id]
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
                    source_features, source_info, reference_info, homography
                )
                valid = validity > 0.99
                cosine = torch.sum(reference_features * aligned, dim=-1)
                source_diagnostics.append({
                    "image_id": int(source["image_id"]),
                    "valid_grid_fraction": float(validity.mean().cpu()),
                    "mean_aligned_cosine_valid": float(cosine[valid].mean().cpu()) if valid.any() else None,
                })
                aligned_sources.append((aligned, validity))
            fused_features = fuse_aligned_features(reference_features, aligned_sources)
            fused_decoded = predict(head, fused_features, reference_info, args)
            if device.type == "mps":
                torch.mps.synchronize()
            source_fusion_seconds += time.perf_counter() - start
            peak_memory = max(peak_memory, mps_driver_memory())

            single_predictions = eval_predictions(single_decoded, image_id)
            fused_predictions = eval_predictions(fused_decoded, image_id)
            single_by_image[image_id] = single_predictions
            fused_by_image[image_id] = fused_predictions
            ground_truth = annotations_to_eval_ground_truth(image_id, sample.annotations)
            single_ids, single_fp, single_fn, _ = fixed_matches(ground_truth, single_predictions, args.confidence)
            fused_ids, fused_fp, fused_fn, _ = fixed_matches(ground_truth, fused_predictions, args.confidence)
            per_gt = []
            for gt in ground_truth:
                per_gt.append({
                    "annotation_id": gt.annotation_id,
                    "visibility": gt.visibility_level,
                    "single_best_iou_all_scores": best_iou(single_predictions, gt.box),
                    "fused_best_iou_all_scores": best_iou(fused_predictions, gt.box),
                    "single_detected_fixed": gt.annotation_id in single_ids,
                    "fused_detected_fixed": gt.annotation_id in fused_ids,
                })
            diagnostics.append({
                "group_id": group["group_id"],
                "sequence": group["sequence"],
                "sequence_seen_in_pilot8": bool(group["sequence_seen_in_pilot8"]),
                "reference_image_id": image_id,
                "reference_file": sample.file_name,
                "ground_truth_people": len(ground_truth),
                "single_predictions_all_scores": len(single_predictions),
                "fused_predictions_all_scores": len(fused_predictions),
                "single_fp_fixed": single_fp,
                "fused_fp_fixed": fused_fp,
                "single_fn_fixed": single_fn,
                "fused_fn_fixed": fused_fn,
                "mean_single_best_iou": float(np.mean([item["single_best_iou_all_scores"] for item in per_gt])),
                "mean_fused_best_iou": float(np.mean([item["fused_best_iou_all_scores"] for item in per_gt])),
                "source_diagnostics": source_diagnostics,
                "per_gt": per_gt,
            })
            print(f"[{group_index:02d}/{len(groups)}] {group['group_id']} {group['sequence']}", flush=True)

    image_ids = [int(group["reference"]["image_id"]) for group in groups]
    ground_truth_by_image = {
        image_id: annotations_to_eval_ground_truth(image_id, samples_by_id[image_id].annotations)
        for image_id in image_ids
    }
    flat_gt = [item for values in ground_truth_by_image.values() for item in values]
    flat_single = [item for values in single_by_image.values() for item in values]
    flat_fused = [item for values in fused_by_image.values() for item in values]

    def metrics(by_image: dict[int, list[EvaluationPrediction]], flat: list[EvaluationPrediction]) -> dict[str, Any]:
        return {
            "fixed_threshold": evaluate_custom_metrics(
                ground_truth_by_image, by_image, image_ids, args.confidence, iou_threshold=0.50
            ),
            "coco_style": evaluate_coco_style(flat_gt, flat, image_ids),
            "pycocotools": evaluate_pycocotools(flat_gt, flat, image_ids),
        }

    single_metrics = metrics(single_by_image, flat_single)
    fused_metrics = metrics(fused_by_image, flat_fused)
    per_gt_rows = []
    group_by_annotation: dict[int | str | None, str] = {}
    for diagnostic in diagnostics:
        for item in diagnostic["per_gt"]:
            group_by_annotation[item["annotation_id"]] = diagnostic["group_id"]
            per_gt_rows.append({
                "group_id": diagnostic["group_id"],
                "sequence": diagnostic["sequence"],
                "reference_image_id": diagnostic["reference_image_id"],
                **item,
            })
    rescued = [row for row in per_gt_rows if row["fused_detected_fixed"] and not row["single_detected_fixed"]]
    lost = [row for row in per_gt_rows if row["single_detected_fixed"] and not row["fused_detected_fixed"]]
    bootstrap = sequence_cluster_bootstrap(per_gt_rows, args.bootstrap_replicates, args.bootstrap_seed)

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "per_gt_diagnostics.csv", per_gt_rows)
    group_rows = [
        {key: value for key, value in row.items() if key not in {"per_gt", "source_diagnostics"}}
        for row in diagnostics
    ]
    write_csv(args.output_dir / "per_group_diagnostics.csv", group_rows)
    (args.output_dir / "sequence_bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    (args.output_dir / "single_view_coco_results.json").write_text(
        json.dumps(predictions_to_coco_results(flat_single), indent=2), encoding="utf-8"
    )
    (args.output_dir / "multi_view_coco_results.json").write_text(
        json.dumps(predictions_to_coco_results(flat_fused), indent=2), encoding="utf-8"
    )

    rescued_groups = {row["group_id"] for row in rescued}
    lost_groups = {row["group_id"] for row in lost}
    ranked = sorted(
        diagnostics,
        key=lambda row: abs(row["mean_fused_best_iou"] - row["mean_single_best_iou"]),
        reverse=True,
    )
    selected_ids = list(dict.fromkeys([*sorted(rescued_groups), *sorted(lost_groups), *[row["group_id"] for row in ranked]]))[: args.visual_examples]
    visual_index = []
    groups_by_id = {group["group_id"]: group for group in groups}
    diagnostics_by_id = {row["group_id"]: row for row in diagnostics}
    for group_id in selected_ids:
        group = groups_by_id[group_id]
        diagnostic = diagnostics_by_id[group_id]
        image_id = diagnostic["reference_image_id"]
        gt_items = ground_truth_by_image[image_id]
        gt = [{"box": item.box, "visibility_level": item.visibility_level, "pose": item.pose, "annotation_id": item.annotation_id} for item in gt_items]
        single = [{"box": item.box, "confidence": item.confidence} for item in single_by_image[image_id] if item.confidence >= args.confidence]
        fused = [{"box": item.box, "confidence": item.confidence} for item in fused_by_image[image_id] if item.confidence >= args.confidence]
        single_path = args.output_dir / "visual_examples" / "parts" / f"{group_id}_single.jpg"
        fused_path = args.output_dir / "visual_examples" / "parts" / f"{group_id}_fused.jpg"
        draw_detection_comparison(samples_by_id[image_id].image_path, gt, single, evaluate_image_matches(gt, single, 0.50)[0], single_path, "single FCOS")
        draw_detection_comparison(samples_by_id[image_id].image_path, gt, fused, evaluate_image_matches(gt, fused, 0.50)[0], fused_path, "multi-view FCOS")
        left = Image.open(single_path).convert("RGB")
        right = Image.open(fused_path).convert("RGB")
        height = min(left.height, right.height)
        left = left.resize((round(left.width * height / left.height), height))
        right = right.resize((round(right.width * height / right.height), height))
        comparison = Image.new("RGB", (left.width + right.width, height), "white")
        comparison.paste(left, (0, 0))
        comparison.paste(right, (left.width, 0))
        comparison_path = args.output_dir / "visual_examples" / "parts" / f"{group_id}_comparison.jpg"
        comparison.save(comparison_path, quality=90)
        reason = "rescued" if group_id in rescued_groups else "lost" if group_id in lost_groups else "large_iou_change"
        final_path = args.output_dir / "visual_examples" / f"{group_id}_{reason}.jpg"
        combine_gallery(
            args.manifest.parent / "alignment_gallery" / f"{group_id}.jpg",
            comparison_path,
            final_path,
            f"{group_id} | {group['sequence']} | {reason}",
        )
        visual_index.append({"group_id": group_id, "reason": reason, "path": str(final_path)})
    (args.output_dir / "visual_examples" / "index.json").write_text(json.dumps(visual_index, indent=2), encoding="utf-8")

    annotation_integrity = {
        "reference_images": len(image_ids),
        "ground_truth_people": len(flat_gt),
        "unique_annotation_ids": len({item.annotation_id for item in flat_gt}),
        "images_with_multiple_people": sum(len(items) > 1 for items in ground_truth_by_image.values()),
        "visibility_counts": {
            str(value): sum(item.visibility_level == value for item in flat_gt) for value in VISIBILITIES
        },
        "iscrowd_nonzero": sum(
            int(annotation.get("iscrowd", 0)) != 0
            for image_id in image_ids for annotation in samples_by_id[image_id].annotations
        ),
        "evaluation_unit": "reference-frame annotation; each reference frame and annotation appears once",
    }
    result = {
        "settings": {
            "manifest": str(args.manifest),
            "checkpoint": str(args.checkpoint),
            "backbone": checkpoint_metadata["model_id"],
            "head": checkpoint_metadata["head_kind"],
            "device": str(device),
            "short_side": args.short_side,
            "fixed_confidence": args.confidence,
            "prediction_confidence_floor": args.prediction_conf_min,
            "nms_iou": args.nms_iou,
            "match_iou": 0.50,
            "threshold_tuning": False,
            "fusion": "equal-weight mean of reference and homography-warped valid source patch features followed by L2 normalization",
            "shared_encoder": True,
            "training": False,
        },
        "manifest_summary": {
            "groups": len(groups),
            "views": 3 * len(groups),
            "unique_sequences": len({group["sequence"] for group in groups}),
            "sequences_seen_in_pilot8": sum(bool(group["sequence_seen_in_pilot8"]) for group in groups),
            "selection_uses_annotations": manifest["selection_uses_annotations"],
            "selection_uses_model_predictions": manifest["selection_uses_model_predictions"],
        },
        "annotation_integrity": annotation_integrity,
        "single_view": single_metrics,
        "multi_view": fused_metrics,
        "paired_fixed_threshold": {
            "rescued_annotation_ids": [row["annotation_id"] for row in rescued],
            "lost_annotation_ids": [row["annotation_id"] for row in lost],
            "rescued_count": len(rescued),
            "lost_count": len(lost),
            "net_detected_change": len(rescued) - len(lost),
        },
        "sequence_cluster_bootstrap": bootstrap,
        "timing": {
            "single_reference_seconds_total": single_seconds,
            "single_reference_seconds_per_group": single_seconds / len(groups),
            "additional_source_and_fusion_seconds_total": source_fusion_seconds,
            "multi_view_total_seconds_per_group": (single_seconds + source_fusion_seconds) / len(groups),
        },
        "peak_mps_driver_memory_bytes": peak_memory,
        "diagnostics": diagnostics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "groups": len(groups),
        "ground_truth_people": len(flat_gt),
        "single_fixed": single_metrics["fixed_threshold"],
        "fused_fixed": fused_metrics["fixed_threshold"],
        "single_AP": single_metrics["pycocotools"].get("AP"),
        "fused_AP": fused_metrics["pycocotools"].get("AP"),
        "single_AR100": single_metrics["pycocotools"].get("AR@100"),
        "fused_AR100": fused_metrics["pycocotools"].get("AR@100"),
        "rescued": len(rescued),
        "lost": len(lost),
        "runtime": result["timing"],
        "peak_mps_driver_memory_bytes": peak_memory,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
