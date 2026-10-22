#!/usr/bin/env python3
"""Train only the lightweight RoI head over frozen DINOv3-LoRA FCOS proposals."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import annotations_to_ground_truth, choose_device
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint
from forestpersons.dinov3_roi_refiner import (
    DINOv3RoIRefinementHead,
    decode_fcos_pre_nms_proposals,
    ground_truth_boxes_resized,
    match_and_sample_proposals,
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
        default=Path("outputs/dinov3_roi_refiner/train400_seed20261014_e2_frozen_lora5k"),
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--min-proposal-score", type=float, default=0.001)
    parser.add_argument("--pre-nms-topk", type=int, default=1000)
    parser.add_argument("--samples-per-image", type=int, default=128)
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument("--positive-iou", type=float, default=0.5)
    parser.add_argument("--negative-iou", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def load_samples(root: Path, split: str, manifest: Path) -> list[Any]:
    dataset = ForestPersonsDataset(root, split=split)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    items = data["images"] if isinstance(data, dict) else data
    by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [by_file[item["file_name"]] for item in items]
    missing = [sample.file_name for sample in samples if not sample.image_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected images, e.g. {missing[0]}")
    return samples


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise RuntimeError("MPS was requested but is unavailable")
    samples = load_samples(args.root, args.split, args.manifest)
    processor, backbone, fcos_head, detector_meta = load_lora_fcos_checkpoint(
        args.detector_checkpoint,
        device,
    )
    if detector_meta["head_kind"] != "fcos":
        raise ValueError(f"Expected single-scale FCOS, got {detector_meta['head_kind']}")
    backbone.eval()
    fcos_head.eval()
    for module in (backbone, fcos_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    refiner = DINOv3RoIRefinementHead(input_dim=detector_meta["head_input_dim"]).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history: list[dict[str, Any]] = []
    peak_driver = 0
    total_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        refiner.train()
        order = list(range(len(samples)))
        random.Random(args.seed + epoch).shuffle(order)
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "box": 0.0,
            "positive": 0,
            "negative": 0,
            "proposals": 0,
            "images_without_positive": 0,
        }
        epoch_started = time.perf_counter()
        for step, sample_index in enumerate(order, start=1):
            sample = samples[sample_index]
            image = Image.open(sample.image_path).convert("RGB")
            ground_truth = annotations_to_ground_truth(sample.annotations)
            with torch.inference_mode():
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
            sampled = match_and_sample_proposals(
                candidates["boxes"],
                gt_boxes,
                positive_iou=args.positive_iou,
                negative_iou=args.negative_iou,
                samples_per_image=args.samples_per_image,
                positive_fraction=args.positive_fraction,
                generator=generator,
            )
            indices = sampled["indices"]
            if indices.numel() == 0:
                raise RuntimeError(f"No sampled proposals for {sample.file_name}")
            proposals = candidates["boxes"][indices]
            labels = sampled["labels"]
            matched_gt = (
                gt_boxes[sampled["matched_gt_indices"]]
                if gt_boxes.numel()
                else torch.zeros_like(proposals)
            )
            optimizer.zero_grad(set_to_none=True)
            score_logits, box_deltas = refiner(features.detach(), proposals)
            loss, parts = roi_refinement_loss(
                score_logits,
                box_deltas,
                proposals,
                labels,
                matched_gt,
                box_loss_weight=args.box_loss_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}")
            loss.backward()
            nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=5.0)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["classification"] += parts["classification"]
            totals["box"] += parts["box"]
            totals["positive"] += parts["positive"]
            totals["negative"] += parts["negative"]
            totals["proposals"] += int(candidates["boxes"].shape[0])
            totals["images_without_positive"] += int(parts["positive"] == 0)
            if device.type == "mps":
                peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
            if step == 1 or step % 25 == 0 or step == len(order):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "images": len(order),
                            "loss": float(loss.detach().cpu()),
                            "positive": parts["positive"],
                            "negative": parts["negative"],
                        }
                    ),
                    flush=True,
                )
        sync(device)
        count = len(order)
        epoch_row = {
            "epoch": epoch,
            "images": count,
            "optimizer_steps": count,
            "loss": totals["loss"] / count,
            "classification_loss": totals["classification"] / count,
            "box_loss": totals["box"] / count,
            "positive_proposals": totals["positive"],
            "negative_proposals": totals["negative"],
            "mean_positive_per_image": totals["positive"] / count,
            "mean_negative_per_image": totals["negative"] / count,
            "mean_pre_nms_proposals": totals["proposals"] / count,
            "images_without_positive": totals["images_without_positive"],
            "runtime_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row), flush=True)
        save_roi_refiner_checkpoint(
            args.output_dir / f"epoch_{epoch}.pt",
            refiner,
            source_detector_checkpoint=args.detector_checkpoint,
            settings=vars(args),
            history=history,
        )

    total_runtime = time.perf_counter() - total_started
    final_path = args.output_dir / "roi_refiner.pt"
    save_roi_refiner_checkpoint(
        final_path,
        refiner,
        source_detector_checkpoint=args.detector_checkpoint,
        settings=vars(args),
        history=history,
    )
    summary = {
        "status": "completed",
        "settings": vars(args),
        "device_resolved": str(device),
        "images": len(samples),
        "epochs": args.epochs,
        "optimizer_steps": len(samples) * args.epochs,
        "trainable_parameters": sum(p.numel() for p in refiner.parameters() if p.requires_grad),
        "frozen_detector_trainable_parameters": sum(
            p.numel() for module in (backbone, fcos_head) for p in module.parameters() if p.requires_grad
        ),
        "runtime_seconds": total_runtime,
        "peak_mps_driver_memory_bytes": peak_driver if device.type == "mps" else None,
        "peak_mps_driver_memory_gib": peak_driver / (1024**3) if device.type == "mps" else None,
        "history": history,
        "checkpoint": str(final_path),
    }
    (args.output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
