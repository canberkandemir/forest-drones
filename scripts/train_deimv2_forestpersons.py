"""Guarded one-class DEIMv2-S fine-tuning on ForestPersons train400.

The full two-epoch run requires explicit non-commercial research and supervisor
confirmation flags. A separate technical-validation mode is capped at 20
images and five optimizer steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forestpersons.deimv2_adapter import load_official_deimv2_one_class_for_training
from forestpersons.deimv2_training import (
    CHECKPOINT_ARCHITECTURE,
    PilotFlatCosineScheduler,
    atomic_torch_save,
    capture_rng_state,
    input_and_target,
    make_official_criterion,
    official_parameter_groups,
    restore_rng_state,
    samples_from_manifest,
    validate_training_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--backbone-lr", type=float, default=2.5e-5)
    parser.add_argument("--detector-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.999)
    parser.add_argument("--adamw-eps", type=float, default=1e-8)
    parser.add_argument("--gradient-clip", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--flat-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.5)
    parser.add_argument("--checkpoint-every-steps", type=int, default=25)
    parser.add_argument("--log-every-steps", type=int, default=5)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--technical-validation", action="store_true")
    parser.add_argument("--noncommercial-research-confirmed", action="store_true")
    parser.add_argument("--supervisor-confirmed", action="store_true")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_run_scope(args: argparse.Namespace) -> None:
    if not args.noncommercial_research_confirmed:
        raise RuntimeError(
            "DEIMv2 is licensed for non-commercial research only. Pass "
            "--noncommercial-research-confirmed only after confirming this run is permitted."
        )
    if args.batch_size != 1:
        raise ValueError("This validated MPS pilot supports batch size 1 only")
    if args.epochs <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("epochs and gradient accumulation must be positive")
    limited = (
        args.technical_validation
        and args.max_images is not None
        and args.max_images <= 20
        and args.max_optimizer_steps is not None
        and args.max_optimizer_steps <= 5
        and args.epochs == 1
    )
    if args.technical_validation and not limited:
        raise ValueError("Technical validation is capped at 20 images, five optimizer steps, and one epoch")
    if not limited and not args.supervisor_confirmed:
        raise RuntimeError(
            "The full company-internship pilot requires --supervisor-confirmed after explicit approval."
        )
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "Set PYTORCH_ENABLE_MPS_FALLBACK=1 before starting Python; "
                "DEIMv2 deformable-attention backward requires CPU fallback on PyTorch 2.9 MPS."
            )


def ensure_output_directory(args: argparse.Namespace) -> None:
    if args.resume is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        args.output_dir.mkdir(parents=True, exist_ok=True)


def disjoint_calibration_check(training_samples: list[Any], calibration_samples: list[Any]) -> None:
    training_names = {sample.file_name for sample in training_samples}
    calibration_names = {sample.file_name for sample in calibration_samples}
    overlap = training_names & calibration_names
    if overlap:
        raise ValueError(f"Training/calibration filename overlap: {sorted(overlap)[:5]}")


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: PilotFlatCosineScheduler,
    config: dict[str, Any],
    load_metadata: dict[str, Any],
    device: torch.device,
    epoch: int,
    next_sample_index: int,
    completed_epochs: int,
    physical_steps: int,
    optimizer_steps: int,
    history: list[dict[str, Any]],
    epoch_accumulator: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "architecture": CHECKPOINT_ARCHITECTURE,
        "status": status,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "training_config": config,
        "source_load_metadata": load_metadata,
        "class_mapping": "ForestPersons categories -> internal person label 0",
        "epoch": epoch,
        "next_sample_index": next_sample_index,
        "completed_epochs": completed_epochs,
        "physical_steps": physical_steps,
        "optimizer_steps": optimizer_steps,
        "history": history,
        "epoch_accumulator": epoch_accumulator,
        "rng_state": capture_rng_state(device),
    }


def main() -> None:
    args = parse_args()
    validate_run_scope(args)
    ensure_output_directory(args)
    device = torch.device(args.device)
    set_seed(args.seed)

    training_samples = samples_from_manifest(
        args.root, args.manifest, split="train", max_images=args.max_images
    )
    calibration_samples = samples_from_manifest(
        args.root, args.calibration_manifest, split="train"
    )
    disjoint_calibration_check(training_samples, calibration_samples)
    annotation_count = sum(len(sample.annotations) for sample in training_samples)
    calibration_annotation_count = sum(len(sample.annotations) for sample in calibration_samples)
    physical_steps_per_epoch = len(training_samples)
    optimizer_steps_per_epoch = math.ceil(physical_steps_per_epoch / args.gradient_accumulation)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    if args.max_optimizer_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, args.max_optimizer_steps)
    if not (0 <= args.warmup_steps <= args.flat_steps <= total_optimizer_steps):
        raise ValueError(
            "Scheduler must satisfy 0 <= warmup_steps <= flat_steps <= total optimizer steps; "
            f"got {args.warmup_steps}, {args.flat_steps}, {total_optimizer_steps}"
        )

    source_model = args.checkpoint_dir / "model.safetensors"
    source_config = args.checkpoint_dir / "config.json"
    config: dict[str, Any] = {
        "label": "DEIMv2-S one-class ForestPersons full-model fine-tuning pilot",
        "technical_validation": args.technical_validation,
        "license_scope": "non-commercial research only; supervisor confirmation required for full pilot",
        "official_repo": str(args.official_repo.resolve()),
        "official_repo_commit": "1d2ca42171570c713e78fc6a766ec5104b7f4724",
        "source_checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "source_model_sha256": sha256(source_model),
        "source_config_sha256": sha256(source_config),
        "manifest": str(args.manifest),
        "training_images": len(training_samples),
        "training_annotations": annotation_count,
        "calibration_manifest": str(args.calibration_manifest),
        "calibration_images": len(calibration_samples),
        "calibration_annotations": calibration_annotation_count,
        "calibration_usage": "operating threshold selection only; never used for gradient updates",
        "input_size": [args.input_size, args.input_size],
        "preprocessing": "fixed square resize; ImageNet normalization; no AMP",
        "augmentation": "none in controlled two-epoch pilot",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "seed": args.seed,
        "optimizer": {
            "type": "AdamW",
            "backbone_lr": args.backbone_lr,
            "detector_lr": args.detector_lr,
            "betas": [args.adamw_beta1, args.adamw_beta2],
            "eps": args.adamw_eps,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip,
        },
        "scheduler": {
            "type": "pilot-scaled official quadratic-warmup flat-cosine",
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": args.warmup_steps,
            "flat_steps": args.flat_steps,
            "min_lr_ratio": args.min_lr_ratio,
        },
        "criterion": {
            "matcher_costs": {"class": 2, "bbox": 5, "giou": 2},
            "loss_weights": {"mal": 1, "bbox": 5, "giou": 2, "fgl": 0.15, "ddf": 1.5},
            "mal_alpha": 0.75,
            "mal_gamma": 1.5,
            "reg_max": 32,
        },
        "class_mapping": "all ForestPersons pose/visibility categories -> internal label 0",
        "precision": "float32",
        "device": str(device),
        "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "checkpoint_policy": f"atomic latest every {args.checkpoint_every_steps} optimizer steps plus every epoch",
        "model_selection": "no val600 model selection; evaluate the final epoch only",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
    }
    (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model, load_metadata = load_official_deimv2_one_class_for_training(
        args.official_repo, args.checkpoint_dir, device
    )
    model.train()
    criterion = make_official_criterion(args.official_repo, device).train()
    groups = official_parameter_groups(
        model,
        backbone_lr=args.backbone_lr,
        detector_lr=args.detector_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(args.adamw_beta1, args.adamw_beta2),
        eps=args.adamw_eps,
    )
    scheduler = PilotFlatCosineScheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=args.warmup_steps,
        flat_steps=args.flat_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    start_epoch = 0
    next_sample_index = 0
    completed_epochs = 0
    physical_steps = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    epoch_accumulator: dict[str, Any] = {"samples": 0, "loss_sum": 0.0, "components": {}}
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_training_checkpoint(checkpoint)
        if checkpoint["training_config"] != config:
            raise ValueError("Resume configuration differs from the saved training configuration")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"])
        next_sample_index = int(checkpoint["next_sample_index"])
        completed_epochs = int(checkpoint["completed_epochs"])
        physical_steps = int(checkpoint["physical_steps"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        history = list(checkpoint["history"])
        epoch_accumulator = dict(checkpoint["epoch_accumulator"])
        restore_rng_state(checkpoint["rng_state"], device)

    log_path = args.output_dir / "training_log.jsonl"
    log_handle = log_path.open("a", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    peak_mps_allocated = 0
    peak_mps_driver = 0
    optimizer_step_times: list[float] = []
    run_started = time.perf_counter()
    stop_requested = False

    try:
        for epoch in range(start_epoch, args.epochs):
            order = list(range(len(training_samples)))
            random.Random(args.seed + epoch).shuffle(order)
            sample_start = next_sample_index if epoch == start_epoch else 0
            accumulation_started: float | None = None
            for order_position in range(sample_start, len(order)):
                if args.max_optimizer_steps is not None and optimizer_steps >= args.max_optimizer_steps:
                    stop_requested = True
                    break
                sample = training_samples[order[order_position]]
                if accumulation_started is None:
                    synchronize(device)
                    accumulation_started = time.perf_counter()
                images, targets = input_and_target(sample, device, args.input_size)
                outputs = model.forward_raw(images, targets=targets)
                loss_dict = criterion(
                    outputs,
                    targets,
                    epoch=epoch,
                    step=order_position,
                    global_step=physical_steps,
                    epoch_step=len(order),
                )
                loss = sum(loss_dict.values())
                loss_value = float(loss.detach().cpu())
                component_values = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
                if not math.isfinite(loss_value) or not all(math.isfinite(value) for value in component_values.values()):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch} sample={order_position}")
                (loss / args.gradient_accumulation).backward()
                physical_steps += 1
                epoch_accumulator["samples"] = int(epoch_accumulator.get("samples", 0)) + 1
                epoch_accumulator["loss_sum"] = float(epoch_accumulator.get("loss_sum", 0.0)) + loss_value
                components = defaultdict(float, epoch_accumulator.get("components", {}))
                for name, value in component_values.items():
                    components[name] += value
                epoch_accumulator["components"] = dict(components)

                should_step = (
                    physical_steps % args.gradient_accumulation == 0
                    or order_position + 1 == len(order)
                )
                log_row: dict[str, Any] = {
                    "event": "physical_step",
                    "epoch": epoch,
                    "sample_index": order_position,
                    "physical_step": physical_steps,
                    "optimizer_step": optimizer_steps,
                    "image_id": sample.image_id,
                    "file_name": sample.file_name,
                    "ground_truth": len(targets[0]["labels"]),
                    "loss": loss_value,
                    "losses": component_values,
                }
                if should_step:
                    next_optimizer_step = optimizer_steps + 1
                    lrs = scheduler.step(next_optimizer_step)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    grad_norm_value = float(grad_norm.detach().cpu())
                    if not math.isfinite(grad_norm_value):
                        raise FloatingPointError(f"Non-finite gradient norm at optimizer step {next_optimizer_step}")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    synchronize(device)
                    step_seconds = time.perf_counter() - accumulation_started
                    accumulation_started = None
                    optimizer_steps = next_optimizer_step
                    optimizer_step_times.append(step_seconds)
                    if device.type == "mps":
                        peak_mps_allocated = max(peak_mps_allocated, int(torch.mps.current_allocated_memory()))
                        peak_mps_driver = max(peak_mps_driver, int(torch.mps.driver_allocated_memory()))
                    log_row.update(
                        {
                            "optimizer_update": True,
                            "optimizer_step": optimizer_steps,
                            "learning_rates": lrs,
                            "gradient_norm_before_clip": grad_norm_value,
                            "optimizer_step_seconds": step_seconds,
                            "mps_allocated_bytes": peak_mps_allocated,
                            "mps_driver_bytes": peak_mps_driver,
                        }
                    )
                log_handle.write(json.dumps(log_row) + "\n")
                log_handle.flush()
                next_sample_index = order_position + 1

                if should_step and (
                    optimizer_steps % args.checkpoint_every_steps == 0
                    or (args.max_optimizer_steps is not None and optimizer_steps >= args.max_optimizer_steps)
                ):
                    payload = checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        load_metadata=load_metadata,
                        device=device,
                        epoch=epoch,
                        next_sample_index=order_position + 1,
                        completed_epochs=completed_epochs,
                        physical_steps=physical_steps,
                        optimizer_steps=optimizer_steps,
                        history=history,
                        epoch_accumulator=epoch_accumulator,
                        status="running",
                    )
                    atomic_torch_save(payload, args.output_dir / "latest.pt")

                if should_step and optimizer_steps % args.log_every_steps == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} sample={order_position + 1}/{len(order)} "
                        f"optimizer_step={optimizer_steps}/{total_optimizer_steps} loss={loss_value:.4f}",
                        flush=True,
                    )

            if stop_requested:
                break
            samples_seen = int(epoch_accumulator["samples"])
            epoch_summary = {
                "epoch": epoch + 1,
                "samples": samples_seen,
                "mean_total_loss": float(epoch_accumulator["loss_sum"]) / max(samples_seen, 1),
                "mean_components": {
                    name: value / max(samples_seen, 1)
                    for name, value in epoch_accumulator["components"].items()
                },
                "optimizer_steps_completed": optimizer_steps,
            }
            history.append(epoch_summary)
            completed_epochs = epoch + 1
            epoch_accumulator = {"samples": 0, "loss_sum": 0.0, "components": {}}
            next_sample_index = 0
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                load_metadata=load_metadata,
                device=device,
                epoch=epoch + 1,
                next_sample_index=0,
                completed_epochs=completed_epochs,
                physical_steps=physical_steps,
                optimizer_steps=optimizer_steps,
                history=history,
                epoch_accumulator=epoch_accumulator,
                status="epoch_complete",
            )
            atomic_torch_save(payload, args.output_dir / "latest.pt")
            atomic_torch_save(payload, args.output_dir / f"epoch_{epoch + 1:03d}.pt")
    except BaseException:
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            load_metadata=load_metadata,
            device=device,
            epoch=min(args.epochs - 1, max(start_epoch, completed_epochs)),
            next_sample_index=next_sample_index,
            completed_epochs=completed_epochs,
            physical_steps=physical_steps,
            optimizer_steps=optimizer_steps,
            history=history,
            epoch_accumulator=epoch_accumulator,
            status="interrupted_or_failed",
        )
        atomic_torch_save(payload, args.output_dir / "interrupted.pt")
        raise
    finally:
        log_handle.close()

    status = "technical_validation_complete" if args.technical_validation else "complete"
    final_payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        load_metadata=load_metadata,
        device=device,
        epoch=completed_epochs,
        next_sample_index=0,
        completed_epochs=completed_epochs,
        physical_steps=physical_steps,
        optimizer_steps=optimizer_steps,
        history=history,
        epoch_accumulator=epoch_accumulator,
        status=status,
    )
    final_name = "timing_validation.pt" if args.technical_validation else "final.pt"
    atomic_torch_save(final_payload, args.output_dir / final_name)
    steady_times = optimizer_step_times[1:] if len(optimizer_step_times) > 1 else optimizer_step_times
    steady_mean = float(statistics.mean(steady_times)) if steady_times else None
    steady_median = float(statistics.median(steady_times)) if steady_times else None
    summary = {
        "status": status,
        "full_training_started": not args.technical_validation,
        "completed_epochs": completed_epochs,
        "physical_steps": physical_steps,
        "optimizer_steps": optimizer_steps,
        "history": history,
        "runtime_seconds": time.perf_counter() - run_started,
        "optimizer_step_seconds": optimizer_step_times,
        "steady_optimizer_step_mean_seconds": steady_mean,
        "steady_optimizer_step_median_seconds": steady_median,
        "projected_200_step_seconds_from_median": steady_median * 200 if steady_median is not None else None,
        "peak_mps_allocated_bytes_observed": peak_mps_allocated,
        "peak_mps_driver_bytes_observed": peak_mps_driver,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cpu_fallback_expected_operation": "aten::grid_sampler_2d_backward",
        "output_checkpoint": str(args.output_dir / final_name),
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
