#!/usr/bin/env python3
"""Matched online training for frozen versus fresh-LoRA Full MVR + FCOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from forestpersons.dinov3_detector import (
    DINOv3MVRFCOSHead,
    MODEL_ID,
    build_fcos_targets,
    load_dinov3,
)
from forestpersons.dinov3_lora import (
    LoRAConfig,
    extract_lora_multilayer_patch_features,
    load_dinov3_with_lora,
    load_lora_state_dict,
    lora_state_dict,
    trainable_parameter_summary,
)
from scripts.train_dinov3_detector_head import fcos_box_loss, sigmoid_focal_loss
from scripts.train_dinov3_lora_fcos import OnlineForestPersonsDataset
from scripts.train_dinov3_mvr_fcos import epoch_order, file_digest, state_digest


MVR_LAYERS = (3, 6, 9, 12)


@dataclass(frozen=True)
class OnlineMVRConfig:
    epochs: int = 2
    gradient_accumulation: int = 4
    short_side: int = 672
    lr_lora: float = 1e-4
    lr_head: float = 3e-4
    weight_decay: float = 1e-4
    box_loss_weight: float = 2.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    gradient_clip_norm: float = 5.0
    seed: int = 20261012


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("frozen", "lora"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--init-mvr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--lr-lora", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-loss-weight", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--expected-samples", type=int, default=5000)
    parser.add_argument(
        "--smoke-optimizer-steps",
        type=int,
        help="Stop after this many real optimizer steps and write smoke-only artifacts.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_initial_mvr_head(path: Path, device: torch.device) -> tuple[DINOv3MVRFCOSHead, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("arm") != "mvr" or checkpoint.get("head") != "mvr_fcos":
        raise ValueError("Initialization checkpoint must be a trained Full MVR FCOS checkpoint")
    head = DINOv3MVRFCOSHead().to(device)
    head.load_state_dict(checkpoint["model_state"], strict=True)
    return head, checkpoint


def optimizer_parameter_groups(
    arm: str,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    lr_lora: float,
    lr_head: float,
) -> list[dict[str, Any]]:
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if arm == "frozen":
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("Frozen arm contains trainable backbone parameters")
        return [{"params": head_parameters, "lr": lr_head, "name": "mvr_fcos"}]
    lora_parameters = [parameter for parameter in backbone.parameters() if parameter.requires_grad]
    if not lora_parameters or any(
        parameter.requires_grad and "lora_" not in name
        for name, parameter in backbone.named_parameters()
    ):
        raise ValueError("LoRA arm must train adapters and no original backbone parameters")
    return [
        {"params": lora_parameters, "lr": lr_lora, "name": "lora"},
        {"params": head_parameters, "lr": lr_head, "name": "mvr_fcos"},
    ]


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def gradient_group_summary(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize finite/nonzero gradients before global clipping."""

    result: dict[str, dict[str, Any]] = {}
    for group in groups:
        gradients = [
            parameter.grad.detach()
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        squared_norm = sum(float(gradient.float().square().sum().cpu()) for gradient in gradients)
        result[str(group["name"])] = {
            "parameters_with_grad": len(gradients),
            "nonzero_gradient_tensors": sum(
                int(bool(torch.count_nonzero(gradient))) for gradient in gradients
            ),
            "all_finite": finite,
            "l2_norm": squared_norm**0.5,
            "max_abs": max(
                (float(gradient.abs().max().cpu()) for gradient in gradients), default=0.0
            ),
        }
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def mps_driver_memory() -> int:
    return int(torch.mps.driver_allocated_memory()) if torch.backends.mps.is_available() else 0


def main() -> None:
    args = parse_args()
    if args.smoke_optimizer_steps is not None and args.smoke_optimizer_steps <= 0:
        raise ValueError("smoke_optimizer_steps must be positive")
    if args.resume and args.smoke_optimizer_steps is not None:
        raise ValueError("Smoke runs are intentionally non-resumable")
    config = OnlineMVRConfig(
        epochs=args.epochs,
        gradient_accumulation=args.gradient_accumulation,
        short_side=args.short_side,
        lr_lora=args.lr_lora,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
        box_loss_weight=args.box_loss_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        gradient_clip_norm=args.gradient_clip,
        seed=args.seed,
    )
    if config.gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if args.arm == "lora":
        processor, backbone, target_names = load_dinov3_with_lora(
            device, lora_config, model_id=args.model_id
        )
        backbone.train()
    else:
        processor, backbone = load_dinov3(device)
        target_names = []
        backbone.eval()
    head, init_checkpoint = load_initial_mvr_head(args.init_mvr_checkpoint, device)
    head.train()
    initial_head_sha = state_digest(head.state_dict())

    dataset = OnlineForestPersonsDataset(args.root, "train", args.manifest)
    if len(dataset) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} samples, found {len(dataset)}")
    manifest_sha = file_digest(args.manifest)
    groups = optimizer_parameter_groups(
        args.arm, backbone, head, config.lr_lora, config.lr_head
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)

    output = args.output_dir.resolve()
    latest = output / "latest.pt"
    history: list[dict[str, Any]]
    completed_epoch = 0
    global_step = 0
    runtime = 0.0
    peak_mps = mps_driver_memory() if device.type == "mps" else 0
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(latest)
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["arm"] != args.arm or saved["manifest_sha256"] != manifest_sha:
            raise ValueError("Resume arm or manifest differs")
        if saved["initial_head_sha256"] != initial_head_sha:
            raise ValueError("Resume Full MVR initialization differs")
        if saved["training_config"] != asdict(config):
            raise ValueError("Resume training configuration differs")
        if saved["backbone_model_id"] != args.model_id:
            raise ValueError("Resume backbone model differs")
        expected_lora = asdict(lora_config) if args.arm == "lora" else None
        if saved["lora_config"] != expected_lora:
            raise ValueError("Resume LoRA configuration differs")
        head.load_state_dict(saved["head_state"], strict=True)
        if args.arm == "lora":
            load_lora_state_dict(backbone, saved["lora_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        history = list(saved["history"])
        completed_epoch = int(saved["completed_epoch"])
        global_step = int(saved["optimizer_steps"])
        runtime = float(saved["runtime_seconds"])
        peak_mps = max(peak_mps, int(saved.get("peak_mps_driver_bytes", 0)))
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True)
        history = []
        run_config = {
            "experiment": "frozen_vs_lora_full_mvr",
            "arm": args.arm,
            "model_id": args.model_id,
            "mvr_layers": list(MVR_LAYERS),
            "training_config": asdict(config),
            "lora_config": asdict(lora_config) if args.arm == "lora" else None,
            "lora_target_modules": target_names,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha,
            "training_samples": len(dataset),
            "init_mvr_checkpoint": str(args.init_mvr_checkpoint.resolve()),
            "init_mvr_checkpoint_sha256": file_digest(args.init_mvr_checkpoint),
            "initial_head_sha256": initial_head_sha,
            "initial_checkpoint_epoch": int(init_checkpoint["completed_epoch"]),
            "parameter_summary": trainable_parameter_summary(backbone, head),
            "device": str(device),
            "losses": "unchanged mean focal + 2*SmoothL1(log LTRB) + positive-only BCE centerness",
            "scheduler": None,
            "smoke_optimizer_steps": args.smoke_optimizer_steps,
        }
        (output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        totals = {"loss": 0.0, "classification": 0.0, "box": 0.0, "centerness": 0.0}
        positives = fallbacks = overwrites = 0
        sample_keys: list[str] = []
        gradient_snapshots: list[dict[str, Any]] = []
        processed_images = 0
        epoch_steps = 0
        max_grad_norm = 0.0
        clipped_steps = 0
        started = time.perf_counter()
        order = epoch_order(len(dataset), config.seed, epoch)
        for position, index in enumerate(order, start=1):
            item = dataset[int(index)]
            processed_images += 1
            sample_keys.append(f"{item['image_id']}\t{item['file_name']}")
            context = torch.enable_grad() if args.arm == "lora" else torch.no_grad()
            with context:
                features, info = extract_lora_multilayer_patch_features(
                    item["image"], processor, backbone, device,
                    short_side=config.short_side, layer_indices=MVR_LAYERS,
                )
            classification, box_targets, centerness, positive, stats = build_fcos_targets(
                item["ground_truth"], info
            )
            classification = classification.to(device)
            box_targets = box_targets.to(device)
            centerness = centerness.to(device)
            positive = positive.to(device)
            logits, raw_box, center_logits = head(features)
            classification_loss = sigmoid_focal_loss(
                logits.reshape(-1), classification.reshape(-1),
                alpha=config.focal_alpha, gamma=config.focal_gamma,
            )
            box_loss = fcos_box_loss(raw_box, box_targets, positive)
            center_loss = (
                F.binary_cross_entropy_with_logits(center_logits[positive], centerness[positive])
                if positive.any() else center_logits.sum() * 0.0
            )
            loss = classification_loss + config.box_loss_weight * box_loss + center_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, item {position}")
            (loss / config.gradient_accumulation).backward()
            totals["loss"] += float(loss.detach().cpu())
            totals["classification"] += float(classification_loss.detach().cpu())
            totals["box"] += float(box_loss.detach().cpu())
            totals["centerness"] += float(center_loss.detach().cpu())
            positives += int(positive.sum())
            fallbacks += int(stats["center_fallbacks"])
            overwrites += int(stats["multi_gt_overwrites"])
            if position % config.gradient_accumulation == 0 or position == len(dataset):
                parameters = [p for group in groups for p in group["params"]]
                group_gradients = gradient_group_summary(groups)
                if not all(item["all_finite"] for item in group_gradients.values()):
                    raise FloatingPointError("Non-finite parameter gradient")
                gradient_snapshots.append(
                    {"optimizer_step": global_step + 1, "groups": group_gradients}
                )
                norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
                norm_value = float(norm.detach().cpu())
                if not math.isfinite(norm_value):
                    raise FloatingPointError("Non-finite gradient norm")
                max_grad_norm = max(max_grad_norm, norm_value)
                clipped_steps += int(norm_value > config.gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                epoch_steps += 1
                if (
                    args.smoke_optimizer_steps is not None
                    and global_step >= args.smoke_optimizer_steps
                ):
                    break
            if device.type == "mps":
                peak_mps = max(peak_mps, mps_driver_memory())
        synchronize(device)
        elapsed = time.perf_counter() - started
        runtime += elapsed
        count = processed_images
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / count,
            "classification_loss": totals["classification"] / count,
            "box_loss": totals["box"] / count,
            "centerness_loss": totals["centerness"] / count,
            "positive_cells": positives,
            "center_fallbacks": fallbacks,
            "multi_gt_overwrites": overwrites,
            "optimizer_steps_epoch": epoch_steps,
            "optimizer_steps_total": global_step,
            "processed_images": processed_images,
            "epoch_complete": processed_images == len(dataset),
            "max_pre_clip_gradient_norm": max_grad_norm,
            "clipped_steps": clipped_steps,
            "sample_order_sha256": hashlib.sha256("\n".join(sample_keys).encode()).hexdigest(),
            "runtime_seconds": elapsed,
            "peak_mps_driver_bytes": peak_mps,
            "gradient_snapshots": gradient_snapshots,
        }
        history.append(row)
        payload = {
            "format_version": 1,
            "experiment": "frozen_vs_lora_full_mvr",
            "arm": args.arm,
            "backbone_model_id": args.model_id,
            "head": "mvr_fcos",
            "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "lora_config": asdict(lora_config) if args.arm == "lora" else None,
            "lora_target_modules": target_names,
            "lora_state": lora_state_dict(backbone) if args.arm == "lora" else {},
            "optimizer_state": optimizer.state_dict(),
            "completed_epoch": epoch,
            "optimizer_steps": global_step,
            "training_config": asdict(config),
            "manifest_sha256": manifest_sha,
            "initial_head_sha256": initial_head_sha,
            "history": history,
            "runtime_seconds": runtime,
            "peak_mps_driver_bytes": peak_mps,
            "parameter_summary": trainable_parameter_summary(backbone, head),
            "smoke_test": args.smoke_optimizer_steps is not None,
            "processed_images": processed_images,
            "gradient_snapshots": gradient_snapshots,
        }
        if args.smoke_optimizer_steps is not None:
            atomic_torch_save(payload, output / "smoke.pt")
            summary = {
                "status": "completed_smoke_test",
                "arm": args.arm,
                "requested_optimizer_steps": args.smoke_optimizer_steps,
                "training_config": asdict(config),
                "parameter_summary": trainable_parameter_summary(backbone, head),
                "result": row,
                "artifact": str((output / "smoke.pt").resolve()),
            }
            (output / "smoke_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(json.dumps(summary, indent=2), flush=True)
            return
        atomic_torch_save(payload, latest)
        atomic_torch_save(payload, output / f"epoch_{epoch:03d}.pt")
        (output / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if not history:
        raise RuntimeError("No epoch was executed; increase --epochs or use the existing final checkpoint")
    atomic_torch_save(torch.load(latest, map_location="cpu", weights_only=False), output / "final.pt")
    print(f"Saved completed {args.arm} checkpoint to {output / 'final.pt'}")


if __name__ == "__main__":
    main()
