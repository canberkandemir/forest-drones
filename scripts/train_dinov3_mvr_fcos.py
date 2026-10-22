#!/usr/bin/env python3
"""Train resumable matched frozen-DINOv3 baseline or MVR FCOS readout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    DINOv3FCOSHead,
    DINOv3MVRFCOSHead,
    annotations_to_ground_truth,
    build_fcos_targets,
    cache_path_for_file,
    load_feature_cache,
    load_mvr_feature_cache,
)
from scripts.train_dinov3_detector_head import fcos_box_loss, sigmoid_focal_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    box_loss_weight: float = 2.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    gradient_clip_norm: float = 5.0
    seed: int = 20261012


class CachedReadoutDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        manifest: Path,
        final_cache_root: Path,
        mvr_cache_root: Path | None,
        arm: str,
    ) -> None:
        dataset = ForestPersonsDataset(root, split=split)
        by_id = {int(sample.image_id): sample for sample in dataset.samples}
        items = json.loads(manifest.read_text(encoding="utf-8"))["images"]
        self.samples = []
        for item in items:
            image_id = int(item["image_id"])
            sample = by_id[image_id]
            if sample.file_name != item["file_name"]:
                raise ValueError(f"Filename mismatch for image_id {image_id}")
            self.samples.append(sample)
        self.final_cache_root = final_cache_root
        self.mvr_cache_root = mvr_cache_root
        self.arm = arm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        final, info = load_feature_cache(
            cache_path_for_file(self.final_cache_root, sample.file_name)
        )
        if self.arm == "baseline":
            features = final
        else:
            if self.mvr_cache_root is None:
                raise ValueError("MVR cache root is required for the MVR arm")
            intermediate, mvr_info, layers = load_mvr_feature_cache(
                cache_path_for_file(self.mvr_cache_root, sample.file_name)
            )
            if layers != (3, 6, 9) or mvr_info != info:
                raise ValueError(f"MVR cache mismatch for {sample.file_name}")
            features = torch.cat((intermediate, final.unsqueeze(0)), dim=0)
        ground_truth = annotations_to_ground_truth(sample.annotations)
        classification, boxes, centerness, positive, stats = build_fcos_targets(
            ground_truth, info
        )
        return {
            "features": features,
            "classification": classification,
            "boxes": boxes,
            "centerness": centerness,
            "positive": positive,
            "stats": stats,
        }


def epoch_order(length: int, seed: int, epoch: int) -> torch.Tensor:
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch))


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_model(arm: str, fcos_state: dict[str, torch.Tensor]) -> torch.nn.Module:
    if arm == "baseline":
        model = DINOv3FCOSHead()
        model.load_state_dict(fcos_state)
        return model
    model = DINOv3MVRFCOSHead()
    model.fcos.load_state_dict(fcos_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "mvr"), required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final-cache-root", type=Path, required=True)
    parser.add_argument("--mvr-cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--box-loss-weight", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.arm == "mvr" and args.mvr_cache_root is None:
        raise ValueError("--mvr-cache-root is required for --arm mvr")
    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        box_loss_weight=args.box_loss_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        gradient_clip_norm=args.gradient_clip,
        seed=args.seed,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    canonical_fcos = DINOv3FCOSHead()
    fcos_state = copy.deepcopy(canonical_fcos.state_dict())
    fcos_sha = state_digest(fcos_state)
    model = make_model(args.arm, fcos_state).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    dataset = CachedReadoutDataset(
        args.root,
        "train",
        args.manifest,
        args.final_cache_root,
        args.mvr_cache_root,
        args.arm,
    )
    if len(dataset) != 5000:
        raise ValueError(f"Expected train5000, found {len(dataset)} samples")

    output = args.output_dir.resolve()
    latest = output / "last.pt"
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(latest)
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["arm"] != args.arm or checkpoint["fcos_initial_sha256"] != fcos_sha:
            raise ValueError("Resume checkpoint arm or matched FCOS initialization differs")
        saved = checkpoint["training_config"]
        for key, value in asdict(config).items():
            if key != "epochs" and saved[key] != value:
                raise ValueError(f"Resume configuration differs for {key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        history = checkpoint["history"]
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        runtime = float(checkpoint.get("runtime_seconds", 0.0))
        peak_mps = int(checkpoint.get("peak_mps_driver_bytes", 0))
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.mkdir(parents=True)
        history = []
        start_epoch = 1
        runtime = 0.0
        peak_mps = 0
        (output / "run_config.json").write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "training_config": asdict(config),
                    "manifest": str(args.manifest.resolve()),
                    "final_cache_root": str(args.final_cache_root.resolve()),
                    "mvr_cache_root": str(args.mvr_cache_root.resolve()) if args.mvr_cache_root else None,
                    "fcos_initial_sha256": fcos_sha,
                    "mvr_layers": [3, 6, 9, 12] if args.arm == "mvr" else [12],
                },
                indent=2,
            )
            + "\n"
        )

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "classification": 0.0,
            "box": 0.0,
            "centerness": 0.0,
        }
        positives = center_fallbacks = overwrites = 0
        started = time.perf_counter()
        for index in epoch_order(len(dataset), config.seed, epoch):
            item = dataset[int(index)]
            features = item["features"].to(device)
            classification = item["classification"].to(device)
            box_targets = item["boxes"].to(device)
            centerness = item["centerness"].to(device)
            positive = item["positive"].to(device)
            class_logits, raw_box, center_logits = model(features)
            classification_loss = sigmoid_focal_loss(
                class_logits.reshape(-1),
                classification.reshape(-1),
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
            )
            box_loss = fcos_box_loss(raw_box, box_targets, positive)
            center_loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(
                    center_logits[positive], centerness[positive]
                )
                if positive.any()
                else center_logits.sum() * 0.0
            )
            loss = classification_loss + config.box_loss_weight * box_loss + center_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["classification"] += float(classification_loss.detach().cpu())
            totals["box"] += float(box_loss.detach().cpu())
            totals["centerness"] += float(center_loss.detach().cpu())
            positives += int(positive.sum())
            center_fallbacks += int(item["stats"]["center_fallbacks"])
            overwrites += int(item["stats"]["multi_gt_overwrites"])
            if device.type == "mps":
                peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        runtime += elapsed
        row = {
            "epoch": epoch,
            **{key: value / len(dataset) for key, value in totals.items()},
            "positive_cells": positives,
            "center_fallbacks": center_fallbacks,
            "multi_gt_overwrites": overwrites,
            "runtime_seconds": elapsed,
        }
        history.append(row)
        payload = {
            "format_version": 1,
            "arm": args.arm,
            "head": "fcos" if args.arm == "baseline" else "mvr_fcos",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "completed_epoch": epoch,
            "training_config": asdict(config),
            "history": history,
            "fcos_initial_sha256": fcos_sha,
            "runtime_seconds": runtime,
            "peak_mps_driver_bytes": peak_mps,
        }
        save_checkpoint(output / f"epoch_{epoch:03d}.pt", payload)
        save_checkpoint(latest, payload)
        (output / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if start_epoch > config.epochs:
        print(f"Checkpoint already completed {config.epochs} epochs.")
    final = output / "dinov3_head.pt"
    save_checkpoint(
        final,
        torch.load(latest, map_location="cpu", weights_only=False),
    )
    print(f"Saved checkpoint: {final}")


if __name__ == "__main__":
    main()
