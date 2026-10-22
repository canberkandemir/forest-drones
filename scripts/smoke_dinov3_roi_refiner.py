#!/usr/bin/env python3
"""Bounded real-image smoke test for the frozen FCOS RoI refinement head."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import annotations_to_ground_truth, choose_device
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.dinov3_roi_refiner import (
    DINOv3RoIRefinementHead,
    decode_fcos_pre_nms_proposals,
    ground_truth_boxes_resized,
    match_and_sample_proposals,
    refine_and_nms,
    roi_refinement_loss,
    save_roi_refiner_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json"),
    )
    parser.add_argument(
        "--detector-checkpoint",
        type=Path,
        default=Path(
            "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/"
            "dinov3_lora_fcos.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dinov3_roi_refiner/smoke_train4_seed20261014"),
    )
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--min-proposal-score", type=float, default=0.001)
    parser.add_argument("--pre-nms-topk", type=int, default=1000)
    parser.add_argument("--samples-per-image", type=int, default=128)
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument("--sanity-inference-topk", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def load_samples(root: Path, split: str, manifest: Path, max_images: int) -> list[Any]:
    dataset = ForestPersonsDataset(root, split=split)
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    items = manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data
    by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [by_file[item["file_name"]] for item in items[:max_images]]
    missing = [str(sample.image_path) for sample in samples if not sample.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing smoke image(s): {missing[:3]}")
    return samples


def mps_memory() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise RuntimeError("MPS was requested but is not available")
    samples = load_samples(args.root, args.split, args.manifest, args.max_images)
    processor, backbone, fcos_head, detector_meta = load_lora_fcos_checkpoint(
        args.detector_checkpoint,
        device,
    )
    if detector_meta["head_kind"] != "fcos":
        raise ValueError(f"Expected single-scale FCOS checkpoint, got {detector_meta['head_kind']}")
    backbone.eval()
    fcos_head.eval()
    for module in (backbone, fcos_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    refiner = DINOv3RoIRefinementHead(input_dim=detector_meta["head_input_dim"]).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows: list[dict[str, Any]] = []
    peak_driver = 0
    started = time.perf_counter()

    for sample in samples:
        image_started = time.perf_counter()
        image = Image.open(sample.image_path).convert("RGB")
        ground_truth = annotations_to_ground_truth(sample.annotations)
        with torch.no_grad():
            features, info = extract_lora_patch_features(
                image,
                processor=processor,
                model=backbone,
                device=device,
                short_side=args.short_side,
            )
            class_logits, box_raw, centerness_logits = fcos_head(features)
            candidates = decode_fcos_pre_nms_proposals(
                class_logits,
                box_raw,
                centerness_logits,
                info,
                min_score=args.min_proposal_score,
                pre_nms_topk=args.pre_nms_topk,
            )

        gt_boxes = ground_truth_boxes_resized(ground_truth, info, device=device)
        sample_match = match_and_sample_proposals(
            candidates["boxes"],
            gt_boxes,
            samples_per_image=args.samples_per_image,
            positive_fraction=args.positive_fraction,
            generator=generator,
        )
        indices = sample_match["indices"]
        if indices.numel() == 0:
            raise RuntimeError(f"No trainable proposals for {sample.file_name}")
        sampled_proposals = candidates["boxes"][indices]
        labels = sample_match["labels"]
        if gt_boxes.numel() > 0:
            matched_gt = gt_boxes[sample_match["matched_gt_indices"]]
        else:
            matched_gt = torch.zeros_like(sampled_proposals)

        optimizer.zero_grad(set_to_none=True)
        score_logits, deltas = refiner(features.detach(), sampled_proposals)
        loss, parts = roi_refinement_loss(
            score_logits,
            deltas,
            sampled_proposals,
            labels,
            matched_gt,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss for {sample.file_name}: {float(loss)}")
        loss.backward()
        gradient_tensors = [p.grad for p in refiner.parameters() if p.grad is not None]
        nonzero_gradients = sum(int(bool((grad.detach().abs() > 0).any().item())) for grad in gradient_tensors)
        optimizer.step()

        with torch.no_grad():
            inference_count = min(args.sanity_inference_topk, int(candidates["boxes"].shape[0]))
            inference_order = torch.topk(candidates["scores"], k=inference_count).indices
            inference_boxes = candidates["boxes"][inference_order]
            inference_scores = candidates["scores"][inference_order]
            all_score_logits, all_deltas = refiner(features, inference_boxes)
            detections = refine_and_nms(
                inference_boxes,
                inference_scores,
                all_score_logits,
                all_deltas,
                info,
                confidence_threshold=0.05,
            )
        if device.type == "mps":
            torch.mps.synchronize()
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        rows.append(
            {
                "file_name": sample.file_name,
                "ground_truth": len(ground_truth),
                "pre_nms_proposals": int(candidates["boxes"].shape[0]),
                "sampled_positive": parts["positive"],
                "sampled_negative": parts["negative"],
                "loss": float(loss.detach().cpu()),
                "classification_loss": parts["classification"],
                "box_loss": parts["box"],
                "gradient_tensors": len(gradient_tensors),
                "nonzero_gradient_tensors": nonzero_gradients,
                "sanity_detections": len(detections),
                "runtime_seconds": time.perf_counter() - image_started,
            }
        )
        print(json.dumps(rows[-1]), flush=True)

    elapsed = time.perf_counter() - started
    trainable = sum(parameter.numel() for parameter in refiner.parameters() if parameter.requires_grad)
    frozen_trainable = sum(
        parameter.numel()
        for module in (backbone, fcos_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    summary = {
        "status": "completed",
        "device": str(device),
        "torch_version": torch.__version__,
        "detector_checkpoint": str(args.detector_checkpoint),
        "manifest": str(args.manifest),
        "images": len(rows),
        "short_side": args.short_side,
        "runtime_seconds": elapsed,
        "seconds_per_image": elapsed / max(1, len(rows)),
        "peak_mps_driver_memory_bytes": peak_driver if device.type == "mps" else None,
        "peak_mps_driver_memory_gib": peak_driver / (1024**3) if device.type == "mps" else None,
        "refiner_trainable_parameters": trainable,
        "frozen_detector_trainable_parameters": frozen_trainable,
        "roi_align": {
            "output_size": refiner.output_size,
            "spatial_scale": refiner.spatial_scale,
            "sampling_ratio": 2,
            "aligned": True,
        },
        "proposal_matching": {
            "positive_iou": 0.5,
            "negative_iou": 0.3,
            "samples_per_image": args.samples_per_image,
            "positive_fraction": args.positive_fraction,
        },
        "memory_after_run": mps_memory(),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_roi_refiner_checkpoint(
        args.output_dir / "roi_refiner_smoke.pt",
        refiner,
        source_detector_checkpoint=args.detector_checkpoint,
        settings=vars(args),
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
