"""Train one lightweight reference-preserving DINOv3 multi-view fusion gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.detection import evaluate_image_matches
from forestpersons.dinov3_detector import (
    PreprocessInfo,
    annotations_to_ground_truth,
    build_fcos_targets,
    choose_device,
    decode_fcos_predictions,
)
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.evaluation import (
    EvaluationGroundTruth,
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_custom_metrics,
    evaluate_pycocotools,
    predictions_to_coco_results,
)
from forestpersons.multiview import (
    TrainableReferencePreservingFusion,
    fuse_aligned_features,
    fuse_selective_aligned_features,
    validate_unique_group_images,
    warp_patch_features,
)
from scripts.train_dinov3_lora_fcos import compute_fcos_losses


VISIBILITIES = (20, 40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(
            "outputs/multiview/forestpersons_train5000_multiview_audit_seed20261017/"
            "proposed_visibility_stratified_sequence_split.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/"
            "dinov3_lora_fcos.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--prediction-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def info_from_dict(data: dict[str, Any]) -> PreprocessInfo:
    return PreprocessInfo(**{key: data[key] for key in PreprocessInfo.__dataclass_fields__})


def safe_group_name(group_id: str) -> str:
    return hashlib.sha1(group_id.encode()).hexdigest()[:16]


def ensure_group_ids(groups: list[dict[str, Any]]) -> None:
    """Add stable IDs when an audited split stores only reference identities."""

    for group in groups:
        group.setdefault("group_id", str(group["reference"]["identity"]).replace("::", "__"))


def load_cached_group(cache_dir: Path, group: dict[str, Any], device: torch.device) -> dict[str, Any]:
    item = torch.load(cache_dir / f"{safe_group_name(group['group_id'])}.pt", map_location="cpu", weights_only=False)
    return {
        "reference": item["reference"].to(device),
        "sources": [(source.to(device), valid.to(device)) for source, valid in item["sources"]],
        "info": info_from_dict(item["info"]),
    }


def prepare_feature_cache(
    groups: list[dict[str, Any]],
    samples: dict[int, Any],
    cache_dir: Path,
    checkpoint: Path,
    device: torch.device,
    short_side: int,
    rebuild: bool,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = [cache_dir / f"{safe_group_name(group['group_id'])}.pt" for group in groups]
    if not rebuild and all(path.exists() for path in expected):
        return json.loads((cache_dir / "cache_summary.json").read_text(encoding="utf-8"))

    processor, backbone, head, _ = load_lora_fcos_checkpoint(checkpoint, device)
    backbone.eval()
    head.eval()
    extraction_seconds: list[float] = []
    alignment_seconds: list[float] = []
    peak_memory = 0
    with torch.inference_mode():
        for index, (group, path) in enumerate(zip(groups, expected, strict=True), start=1):
            if path.exists() and not rebuild:
                continue
            reference_id = int(group["reference"]["image_id"])
            started = time.perf_counter()
            reference, reference_info = extract_lora_patch_features(
                Image.open(samples[reference_id].image_path).convert("RGB"),
                processor,
                backbone,
                device,
                short_side=short_side,
            )
            extraction_seconds.append(time.perf_counter() - started)
            aligned_sources = []
            for source in group["sources"]:
                source_id = int(source["image_id"])
                started = time.perf_counter()
                source_features, source_info = extract_lora_patch_features(
                    Image.open(samples[source_id].image_path).convert("RGB"),
                    processor,
                    backbone,
                    device,
                    short_side=short_side,
                )
                extraction_seconds.append(time.perf_counter() - started)
                started = time.perf_counter()
                aligned_sources.append(
                    warp_patch_features(
                        source_features,
                        source_info,
                        reference_info,
                        torch.tensor(source["homography_source_to_reference"], device=device),
                    )
                )
                alignment_seconds.append(time.perf_counter() - started)
            torch.save(
                {
                    "group_id": group["group_id"],
                    "reference": reference.detach().cpu(),
                    "sources": [(features.detach().cpu(), valid.detach().cpu()) for features, valid in aligned_sources],
                    "info": asdict(reference_info),
                    "dtype": "float32",
                },
                path,
            )
            if device.type == "mps":
                torch.mps.synchronize()
                peak_memory = max(peak_memory, int(torch.mps.driver_allocated_memory()))
            print(f"cache [{index:03d}/{len(groups)}] {group['group_id']}", flush=True)
    summary = {
        "groups": len(groups),
        "images": 3 * len(groups),
        "dtype": "float32",
        "checkpoint_sha256": sha256(checkpoint),
        "short_side": short_side,
        "mean_feature_extraction_seconds_per_image": float(np.mean(extraction_seconds)) if extraction_seconds else None,
        "mean_alignment_seconds_per_source": float(np.mean(alignment_seconds)) if alignment_seconds else None,
        "peak_mps_driver_memory_bytes": peak_memory,
    }
    (cache_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del processor, backbone, head
    if device.type == "mps":
        torch.mps.empty_cache()
    return summary


def loss_arguments() -> SimpleNamespace:
    return SimpleNamespace(
        focal_alpha=0.75,
        focal_gamma=2.0,
        focal_normalization="mean",
        giou_loss_weight=0.0,
        box_loss_weight=2.0,
        quality_loss_weight=0.0,
    )


def group_loss(
    fusion: TrainableReferencePreservingFusion,
    head: torch.nn.Module,
    cached: dict[str, Any],
    annotations: list[Any],
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    fused, gate = fusion(cached["reference"], cached["sources"])
    logits, raw_box, centerness_logits = head(fused)
    ground_truth = annotations_to_ground_truth(annotations)
    classification, boxes, centerness, positive, target_stats = build_fcos_targets(ground_truth, cached["info"])
    loss, parts = compute_fcos_losses(
        logits,
        raw_box,
        centerness_logits,
        classification.to(logits.device),
        boxes.to(logits.device),
        centerness.to(logits.device),
        positive.to(logits.device),
        loss_arguments(),
    )
    total_weight = gate["total_source_weight"]
    gate_stats = {
        "mean_source_weight": float(total_weight.detach().mean().cpu()),
        "active_fraction": float((total_weight.detach() > 1e-8).float().mean().cpu()),
        "positive_cells": float(target_stats["positive_cells"]),
    }
    return loss, parts, gate_stats


def decode(
    head: torch.nn.Module,
    features: torch.Tensor,
    info: PreprocessInfo,
    image_id: int,
    args: argparse.Namespace,
) -> list[EvaluationPrediction]:
    logits, raw_box, centerness = head(features)
    predictions = decode_fcos_predictions(
        logits.detach().cpu(),
        raw_box.detach().cpu(),
        centerness.detach().cpu(),
        info,
        conf_threshold=args.prediction_floor,
        nms_threshold=args.nms_iou,
    )
    return [EvaluationPrediction(image_id, item["box"], float(item["confidence"])) for item in predictions]


def evaluate_method(
    method: str,
    groups: list[dict[str, Any]],
    samples: dict[int, Any],
    cache_dir: Path,
    device: torch.device,
    head: torch.nn.Module,
    fusion: TrainableReferencePreservingFusion,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[int, list[EvaluationPrediction]], dict[str, float]]:
    predictions: dict[int, list[EvaluationPrediction]] = {}
    elapsed: list[float] = []
    weight_means: list[float] = []
    active: list[float] = []
    with torch.inference_mode():
        for group in groups:
            cached = load_cached_group(cache_dir, group, device)
            started = time.perf_counter()
            if method == "single":
                features = cached["reference"]
            elif method == "equal_mean":
                features = fuse_aligned_features(cached["reference"], cached["sources"])
            elif method == "fixed_selective":
                features, diagnostics = fuse_selective_aligned_features(cached["reference"], cached["sources"])
                weight = diagnostics["total_source_weight"]
                weight_means.append(float(weight.mean().cpu()))
                active.append(float((weight > 1e-8).float().mean().cpu()))
            elif method == "learned":
                features, diagnostics = fusion(cached["reference"], cached["sources"])
                weight = diagnostics["total_source_weight"]
                weight_means.append(float(weight.mean().cpu()))
                active.append(float((weight > 1e-8).float().mean().cpu()))
            else:
                raise ValueError(method)
            image_id = int(group["reference"]["image_id"])
            predictions[image_id] = decode(head, features, cached["info"], image_id, args)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed.append(time.perf_counter() - started)

    image_ids = [int(group["reference"]["image_id"]) for group in groups]
    gt_by_image = {
        image_id: annotations_to_eval_ground_truth(image_id, samples[image_id].annotations)
        for image_id in image_ids
    }
    flat_gt = [item for image_id in image_ids for item in gt_by_image[image_id]]
    flat_predictions = [item for image_id in image_ids for item in predictions[image_id]]
    metrics = {
        "fixed_threshold": evaluate_custom_metrics(gt_by_image, predictions, image_ids, args.confidence, 0.50),
        "pycocotools": evaluate_pycocotools(flat_gt, flat_predictions, image_ids),
    }
    diagnostics = {
        "cached_feature_fusion_and_head_seconds_per_group": float(np.mean(elapsed)),
        "mean_total_source_weight": float(np.mean(weight_means)) if weight_means else 0.0,
        "mean_active_patch_fraction": float(np.mean(active)) if active else 0.0,
    }
    return metrics, predictions, diagnostics


def train_epoch(
    fusion: TrainableReferencePreservingFusion,
    head: torch.nn.Module,
    groups: list[dict[str, Any]],
    samples: dict[int, Any],
    cache_dir: Path,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, Any]:
    fusion.train()
    order = list(range(len(groups)))
    random.Random(args.seed + epoch).shuffle(order)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    part_rows: list[dict[str, float]] = []
    gate_rows: list[dict[str, float]] = []
    optimizer_steps = 0
    peak_memory = 0
    started = time.perf_counter()
    for position, group_index in enumerate(order, start=1):
        group = groups[group_index]
        image_id = int(group["reference"]["image_id"])
        cached = load_cached_group(cache_dir, group, device)
        loss, parts, gate_stats = group_loss(fusion, head, cached, samples[image_id].annotations)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss for {group['group_id']}: {loss}")
        (loss / args.gradient_accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        part_rows.append(parts)
        gate_rows.append(gate_stats)
        if position % args.gradient_accumulation == 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(fusion.parameters(), args.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite fusion gradient norm: {gradient_norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if device.type == "mps":
            peak_memory = max(peak_memory, int(torch.mps.driver_allocated_memory()))
    if len(order) % args.gradient_accumulation:
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), args.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    return {
        "loss": float(np.mean(losses)),
        "loss_components": {key: float(np.mean([row[key] for row in part_rows])) for key in part_rows[0]},
        "mean_total_source_weight": float(np.mean([row["mean_source_weight"] for row in gate_rows])),
        "mean_active_patch_fraction": float(np.mean([row["active_fraction"] for row in gate_rows])),
        "mean_positive_cells": float(np.mean([row["positive_cells"] for row in gate_rows])),
        "optimizer_steps": optimizer_steps,
        "seconds": time.perf_counter() - started,
        "peak_mps_driver_memory_bytes": peak_memory,
    }


def validation_loss(
    fusion: TrainableReferencePreservingFusion,
    head: torch.nn.Module,
    groups: list[dict[str, Any]],
    samples: dict[int, Any],
    cache_dir: Path,
    device: torch.device,
) -> float:
    fusion.eval()
    values = []
    with torch.inference_mode():
        for group in groups:
            image_id = int(group["reference"]["image_id"])
            cached = load_cached_group(cache_dir, group, device)
            loss, _, _ = group_loss(fusion, head, cached, samples[image_id].annotations)
            values.append(float(loss.cpu()))
    return float(np.mean(values))


def matched_ids(
    gt: list[EvaluationGroundTruth], predictions: list[EvaluationPrediction], confidence: float
) -> set[int | str | None]:
    gt_rows = [
        {
            "box": item.box,
            "annotation_id": item.annotation_id,
            "visibility_level": item.visibility_level,
        }
        for item in gt
    ]
    pred_rows = [{"box": item.box, "confidence": item.confidence} for item in predictions if item.confidence >= confidence]
    _, matched_gt, _, _, _ = evaluate_image_matches(gt_rows, pred_rows, 0.50)
    return {gt[index].annotation_id for index in matched_gt}


def save_checkpoint(path: Path, fusion: TrainableReferencePreservingFusion, epoch: int, criterion: tuple[float, ...], args: argparse.Namespace) -> None:
    torch.save(
        {
            "fusion_state": fusion.state_dict(),
            "epoch": epoch,
            "criterion": criterion,
            "architecture": "TrainableReferencePreservingFusion",
            "feature_dim": 384,
            "hidden_dim": 32,
            "maximum_total_source_weight": 0.50,
            "base_checkpoint": str(args.checkpoint),
            "base_checkpoint_sha256": sha256(args.checkpoint),
        },
        path,
    )


def smoke_test(
    group: dict[str, Any],
    sample: Any,
    cache_dir: Path,
    head: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    fusion = TrainableReferencePreservingFusion().to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    head_before = tensor_state_digest(head)
    fusion_before = tensor_state_digest(fusion)
    started = time.perf_counter()
    cached = load_cached_group(cache_dir, group, device)
    loss, parts, gate_stats = group_loss(fusion, head, cached, sample.annotations)
    loss.backward()
    gradients = [parameter.grad for parameter in fusion.parameters()]
    gate_gradients_finite = all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    invalid_fused, invalid_diagnostics = fusion(
        cached["reference"],
        [(features, torch.zeros_like(validity)) for features, validity in cached["sources"]],
    )
    result = {
        "loss": float(loss.detach().cpu()),
        "loss_components": parts,
        "gate_statistics": gate_stats,
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gate_gradients": gate_gradients_finite,
        "fusion_parameters_changed": fusion_before != tensor_state_digest(fusion),
        "frozen_fcos_unchanged": head_before == tensor_state_digest(head),
        "fcos_parameters_with_gradients": sum(parameter.grad is not None for parameter in head.parameters()),
        "invalid_mask_exact_reference_fallback": bool(torch.equal(invalid_fused, cached["reference"])),
        "invalid_mask_all_fallback": bool(invalid_diagnostics["fallback_mask"].all()),
        "seconds": time.perf_counter() - started,
        "mps_driver_memory_bytes": int(torch.mps.driver_allocated_memory()) if device.type == "mps" else None,
    }
    if not all(
        result[key]
        for key in (
            "finite_loss",
            "finite_gate_gradients",
            "fusion_parameters_changed",
            "frozen_fcos_unchanged",
            "invalid_mask_exact_reference_fallback",
            "invalid_mask_all_fallback",
        )
    ) or result["fcos_parameters_with_gradients"] != 0:
        raise RuntimeError(f"Smoke test failed: {result}")
    return result


def write_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    rows = []
    for method, result in comparison.items():
        fixed = result["metrics"]["fixed_threshold"]
        coco = result["metrics"]["pycocotools"]
        rows.append(
            {
                "method": method,
                "TP": fixed["overall"]["true_positives"],
                "FP": fixed["overall"]["false_positives"],
                "FN": fixed["overall"]["false_negatives"],
                "recall": fixed["overall"]["recall"],
                "precision": fixed["overall"]["precision"],
                "FP_per_image": fixed["false_positives_per_image"],
                **{f"R{visibility}": fixed["by_visibility"][str(visibility)]["recall"] for visibility in VISIBILITIES},
                "AP": coco["AP"],
                "AP50": coco["AP50"],
                "AP75": coco["AP75"],
                "AR100": coco["AR@100"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "summary.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed experiment: {result_path}")

    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    train_groups = split["training_groups"]
    validation_groups = split["validation_groups"]
    ensure_group_ids(train_groups)
    ensure_group_ids(validation_groups)
    validate_unique_group_images(train_groups + validation_groups)
    if set(split["training_sequences"]) & set(split["validation_sequences"]):
        raise ValueError("Training and validation sequences overlap")

    dataset = ForestPersonsDataset(args.root, split="train")
    samples = {int(sample.image_id): sample for sample in dataset.samples}
    all_groups = train_groups + validation_groups
    missing = [int(view["image_id"]) for group in all_groups for view in [group["reference"], *group["sources"]] if int(view["image_id"]) not in samples]
    if missing:
        raise KeyError(f"Missing dataset image IDs: {missing[:10]}")

    device = choose_device(args.device)
    cache_dir = args.output_dir / "feature_cache"
    cache_summary = prepare_feature_cache(
        all_groups, samples, cache_dir, args.checkpoint, device, args.short_side, args.rebuild_cache
    )
    processor, backbone, head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    if metadata["head_kind"] != "fcos":
        raise ValueError(f"Expected the canonical FCOS checkpoint, got {metadata['head_kind']}")
    del processor, backbone
    if device.type == "mps":
        torch.mps.empty_cache()

    fusion = TrainableReferencePreservingFusion().to(device)
    finalization_started = time.perf_counter()
    if args.finalize_only:
        smoke = json.loads((args.output_dir / "smoke_test.json").read_text(encoding="utf-8"))
        history = json.loads((args.output_dir / "history.json").read_text(encoding="utf-8"))
    else:
        smoke = smoke_test(train_groups[0], samples[int(train_groups[0]["reference"]["image_id"])], cache_dir, head, device, args)
        (args.output_dir / "smoke_test.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
        print("SMOKE", json.dumps(smoke, sort_keys=True), flush=True)
        if args.smoke_only:
            return

        torch.manual_seed(args.seed)
        fusion = TrainableReferencePreservingFusion().to(device)
        optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history: list[dict[str, Any]] = []
        best_criterion: tuple[float, ...] | None = None
        for epoch in range(1, args.epochs + 1):
            train = train_epoch(fusion, head, train_groups, samples, cache_dir, optimizer, device, args, epoch)
            val_loss = validation_loss(fusion, head, validation_groups, samples, cache_dir, device)
            val_metrics, _, gate_diagnostics = evaluate_method(
                "learned", validation_groups, samples, cache_dir, device, head, fusion, args
            )
            fixed = val_metrics["fixed_threshold"]
            coco = val_metrics["pycocotools"]
            criterion = (float(coco["AR@100"]), float(fixed["overall"]["recall"]), -float(fixed["false_positives_per_image"]))
            row = {
                "epoch": epoch,
                "train": train,
                "validation_loss": val_loss,
                "validation_fixed_threshold": fixed,
                "validation_coco": coco,
                "validation_gate": gate_diagnostics,
                "selection_criterion": criterion,
                "gate_collapse_reference_only": gate_diagnostics["mean_total_source_weight"] < 0.005,
                "gate_collapse_excessive_source": gate_diagnostics["mean_total_source_weight"] > 0.45,
            }
            history.append(row)
            save_checkpoint(args.output_dir / "latest_fusion.pt", fusion, epoch, criterion, args)
            if best_criterion is None or criterion > best_criterion:
                best_criterion = criterion
                save_checkpoint(args.output_dir / "best_fusion.pt", fusion, epoch, criterion, args)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                f"epoch={epoch} train_loss={train['loss']:.6f} val_loss={val_loss:.6f} "
                f"recall={fixed['overall']['recall']:.4f} AR100={coco['AR@100']:.4f} "
                f"FP/image={fixed['false_positives_per_image']:.4f} gate={gate_diagnostics['mean_total_source_weight']:.4f}",
                flush=True,
            )

    best = torch.load(args.output_dir / "best_fusion.pt", map_location=device, weights_only=False)
    fusion.load_state_dict(best["fusion_state"])
    fusion.eval()
    comparison: dict[str, Any] = {}
    prediction_maps: dict[str, dict[int, list[EvaluationPrediction]]] = {}
    for method in ("single", "equal_mean", "fixed_selective", "learned"):
        method_metrics, method_predictions, diagnostics = evaluate_method(
            method, validation_groups, samples, cache_dir, device, head, fusion, args
        )
        comparison[method] = {"metrics": method_metrics, "diagnostics": diagnostics}
        prediction_maps[method] = method_predictions
        flat = [prediction for values in method_predictions.values() for prediction in values]
        (args.output_dir / f"{method}_coco_results.json").write_text(
            json.dumps(predictions_to_coco_results(flat), indent=2), encoding="utf-8"
        )

    gt_by_image = {
        int(group["reference"]["image_id"]): annotations_to_eval_ground_truth(
            int(group["reference"]["image_id"]), samples[int(group["reference"]["image_id"])].annotations
        )
        for group in validation_groups
    }
    single_detected: set[int | str | None] = set()
    learned_detected: set[int | str | None] = set()
    for image_id, gt in gt_by_image.items():
        single_detected |= matched_ids(gt, prediction_maps["single"][image_id], args.confidence)
        learned_detected |= matched_ids(gt, prediction_maps["learned"][image_id], args.confidence)
    paired = {
        "learned_rescued_vs_single": len(learned_detected - single_detected),
        "learned_lost_vs_single": len(single_detected - learned_detected),
        "net_change": len(learned_detected) - len(single_detected),
        "rescued_annotation_ids": sorted(learned_detected - single_detected, key=str),
        "lost_annotation_ids": sorted(single_detected - learned_detected, key=str),
    }
    write_comparison_csv(args.output_dir / "comparison.csv", comparison)

    settings = {
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": 1,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_clip": args.gradient_clip,
        "short_side": args.short_side,
        "fixed_confidence": args.confidence,
        "prediction_floor": args.prediction_floor,
        "nms_iou": args.nms_iou,
        "losses_unchanged": {
            "focal_alpha": 0.75,
            "focal_gamma": 2.0,
            "focal_normalization": "mean_all_grid_locations",
            "box": "SmoothL1(log1p distances), beta=0.25, weight=2",
            "centerness": "positive-only BCE",
        },
        "best_checkpoint_criterion": "lexicographic max AR@100, then recall@0.55, then lower FP/image",
        "trainable_parameters": sum(parameter.numel() for parameter in fusion.parameters()),
        "base_checkpoint": str(args.checkpoint),
        "base_checkpoint_sha256": sha256(args.checkpoint),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "fusion_gate_validation_only": True,
        "detector_pretraining_leakage_limitation": (
            "The fusion validation sequences are held out from gate training, but the original detector was trained on train5000 containing these images."
        ),
    }
    summary = {
        "settings": settings,
        "cache": cache_summary,
        "smoke": smoke,
        "best_epoch": int(best["epoch"]),
        "best_criterion": best["criterion"],
        "history": history,
        "comparison": comparison,
        "paired": paired,
        "total_training_seconds": float(sum(row["train"]["seconds"] for row in history)),
        "final_comparison_seconds": time.perf_counter() - finalization_started,
    }
    (args.output_dir / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": summary["best_epoch"], "paired": paired, "comparison": comparison}, indent=2), flush=True)


if __name__ == "__main__":
    main()
