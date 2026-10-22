#!/usr/bin/env python3
"""Train one resumable DTSS ForestPersons arm on train5000 and evaluate val600."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import resource
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forestpersons.dtss_detector import DTSSDetectorConfig, DTSSPatchDetector, set_dtss_learning
from scripts.run_dtss_forestpersons_detection_pilot import (
    PilotImageDataset,
    _active_steps,
    _batches,
    _sync,
    evaluate_arm,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_VISIBILITIES = {20, 40, 70, 100}


@dataclass(frozen=True)
class FullTrainingConfig:
    seed: int = 20261020
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 0.002
    weight_decay: float = 0.0
    box_loss_weight: float = 5.0
    mask_loss_weight: float = 0.02
    confidence_threshold: float = 0.05
    nms_iou: float = 0.50
    gradient_clip_norm: float = 5.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_records(
    manifest_path: Path,
    annotation_path: Path,
    image_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Resolve samples by image_id and exclude annotations without valid visibility."""

    manifest = _load_json(manifest_path)
    coco = _load_json(annotation_path)
    coco_images = {int(image["id"]): image for image in coco["images"]}
    selected = {int(item["image_id"]): item for item in manifest["images"]}
    if len(selected) != len(manifest["images"]):
        raise ValueError(f"Duplicate image_id in {manifest_path}")

    annotations_by_id: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in selected}
    excluded_annotations = 0
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in selected:
            continue
        attributes = annotation.get("attributes", {})
        visibility = attributes.get("visibility_level") if isinstance(attributes, dict) else None
        if visibility not in VALID_VISIBILITIES:
            excluded_annotations += 1
            continue
        normalized = dict(annotation)
        normalized["metadata"] = dict(attributes)
        annotations_by_id[image_id].append(normalized)

    records: list[dict[str, Any]] = []
    missing_files = []
    for item in manifest["images"]:
        image_id = int(item["image_id"])
        official = coco_images.get(image_id)
        if official is None:
            raise ValueError(f"Manifest image_id {image_id} is absent from {annotation_path}")
        if str(official["file_name"]) != str(item["file_name"]):
            raise ValueError(
                f"Filename mismatch for image_id {image_id}: "
                f"{item['file_name']} != {official['file_name']}"
            )
        image_path = image_root / str(item["file_name"])
        if not image_path.is_file():
            missing_files.append(str(image_path))
        records.append(
            {
                **item,
                "image_id": image_id,
                "width": int(official.get("width", item["width"])),
                "height": int(official.get("height", item["height"])),
                "annotations": annotations_by_id[image_id],
            }
        )
    if missing_files:
        preview = "\n".join(missing_files[:5])
        raise FileNotFoundError(f"Missing {len(missing_files)} manifest images; first paths:\n{preview}")
    if any(not record["annotations"] for record in records):
        empty_ids = [record["image_id"] for record in records if not record["annotations"]]
        raise ValueError(f"Images without evaluable person annotations: {empty_ids[:10]}")
    return records, {
        "images": len(records),
        "annotations": sum(len(record["annotations"]) for record in records),
        "excluded_annotations": excluded_annotations,
        "distinct_filenames": len({record["file_name"] for record in records}),
    }


def epoch_order(length: int, seed: int, epoch: int) -> torch.Tensor:
    """Generate a resume-safe matched order without depending on global RNG state."""

    if epoch < 1:
        raise ValueError("epoch is one-based")
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path: Path,
    *,
    arm: str,
    epoch: int,
    model: DTSSPatchDetector,
    optimizer: torch.optim.Optimizer,
    model_config: DTSSDetectorConfig,
    training_config: FullTrainingConfig,
    history: list[dict[str, Any]],
    initial_state_sha256: str,
    cumulative_runtime_seconds: float,
    peak_mps_driver_bytes: int,
) -> None:
    atomic_torch_save(
        {
            "format_version": 1,
            "arm": arm,
            "completed_epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "history": history,
            "initial_state_sha256": initial_state_sha256,
            "cumulative_runtime_seconds": cumulative_runtime_seconds,
            "peak_mps_driver_bytes": peak_mps_driver_bytes,
            "active_timesteps": _active_steps(model),
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def _validate_resume(
    checkpoint: dict[str, Any],
    arm: str,
    model_config: DTSSDetectorConfig,
    training_config: FullTrainingConfig,
    initial_state_sha256: str,
) -> None:
    if checkpoint.get("arm") != arm:
        raise ValueError(f"Checkpoint arm {checkpoint.get('arm')} does not match {arm}")
    if checkpoint.get("model_config") != asdict(model_config):
        raise ValueError("Checkpoint model configuration does not match this run")
    saved_training = checkpoint.get("training_config", {})
    current = asdict(training_config)
    for key, value in current.items():
        if key == "epochs":
            continue
        if saved_training.get(key) != value:
            raise ValueError(f"Checkpoint training setting differs for {key}")
    if checkpoint.get("initial_state_sha256") != initial_state_sha256:
        raise ValueError("Checkpoint was not initialized from the expected deterministic state")


def train(args: argparse.Namespace) -> None:
    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    model_config = DTSSDetectorConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        t_max=4,
        initial_timesteps=4,
    )
    config = FullTrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        box_loss_weight=args.box_loss_weight,
        mask_loss_weight=args.mask_loss_weight,
        confidence_threshold=args.confidence_threshold,
        nms_iou=args.nms_iou,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_records, train_stats = load_manifest_records(
        args.train_manifest, args.train_annotations, args.image_root
    )
    val_records, val_stats = load_manifest_records(
        args.val_manifest, args.val_annotations, args.image_root
    )
    if len(train_records) != 5000 or len(val_records) != 600:
        raise ValueError(
            f"Expected train5000/val600, received {len(train_records)}/{len(val_records)}"
        )
    train_keys = {(record["image_id"], record["file_name"]) for record in train_records}
    val_keys = {(record["image_id"], record["file_name"]) for record in val_records}
    if train_keys & val_keys:
        raise ValueError("Split-qualified train/validation identity overlap detected")

    train_dataset = PilotImageDataset(train_records, args.image_root, model_config.image_size)
    val_dataset = PilotImageDataset(val_records, args.image_root, model_config.image_size)
    initial_model = DTSSPatchDetector(model_config)
    initial_state = copy.deepcopy(initial_model.state_dict())
    initial_sha = state_digest(initial_state)
    model = DTSSPatchDetector(model_config).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    set_dtss_learning(model, args.arm == "learned_dtss")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output = args.output_dir.resolve()
    last_checkpoint = output / "last.pt"
    if args.resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint is missing: {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        _validate_resume(checkpoint, args.arm, model_config, config, initial_sha)
        model.load_state_dict(checkpoint["model"])
        set_dtss_learning(model, args.arm == "learned_dtss")
        optimizer.load_state_dict(checkpoint["optimizer"])
        _optimizer_to_device(optimizer, device)
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        cumulative_runtime = float(checkpoint.get("cumulative_runtime_seconds", 0.0))
        peak_mps = int(checkpoint.get("peak_mps_driver_bytes", 0))
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    else:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
        output.mkdir(parents=True)
        history = []
        start_epoch = 1
        cumulative_runtime = 0.0
        peak_mps = 0
        (output / "run_config.json").write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "device": str(device),
                    "model_config": asdict(model_config),
                    "training_config": asdict(config),
                    "initial_state_sha256": initial_sha,
                    "train_manifest": str(args.train_manifest.resolve()),
                    "val_manifest": str(args.val_manifest.resolve()),
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                    "compute_caveat": (
                        "DTSS masks outputs, but eager PyTorch still computes all T=4 timesteps."
                    ),
                },
                indent=2,
            )
            + "\n"
        )

    invocation_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        order = epoch_order(len(train_dataset), config.seed, epoch)
        totals = {"detection": 0.0, "classification": 0.0, "regression": 0.0}
        examples = 0
        dtss_gradient_norms: list[float] = []
        epoch_started = time.perf_counter()
        for images, boxes, _ in _batches(train_dataset, order, config.batch_size):
            images = images.to(device)
            boxes = [item.to(device) for item in boxes]
            optimizer.zero_grad(set_to_none=True)
            components = model.compute_loss(
                model(images), boxes, box_weight=config.box_loss_weight
            )
            objective = components["detection"]
            if args.arm == "learned_dtss":
                objective = objective + config.mask_loss_weight * model.mask_loss() / (
                    len(model.timestep_masks()) * model_config.t_max
                )
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Non-finite objective at epoch {epoch}")
            objective.backward()
            if args.arm == "learned_dtss":
                gradients = [
                    module.raw_timestep_parameters.grad.flatten()
                    for module in model.timestep_masks().values()
                    if module.raw_timestep_parameters.grad is not None
                ]
                if gradients:
                    dtss_gradient_norms.append(float(torch.cat(gradients).norm().detach().cpu()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            count = len(boxes)
            examples += count
            for key in totals:
                totals[key] += float(components[key].detach().cpu()) * count
            if device.type == "mps":
                peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))
        _sync(device)
        epoch_runtime = time.perf_counter() - epoch_started
        cumulative_runtime += epoch_runtime
        history.append(
            {
                "epoch": epoch,
                **{key: value / examples for key, value in totals.items()},
                "active_timesteps": _active_steps(model),
                "mean_dtss_gradient_norm": (
                    float(np.mean(dtss_gradient_norms)) if dtss_gradient_norms else 0.0
                ),
                "runtime_seconds": epoch_runtime,
            }
        )
        checkpoint_path = output / f"epoch_{epoch:03d}.pt"
        checkpoint_kwargs = {
            "arm": args.arm,
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "model_config": model_config,
            "training_config": config,
            "history": history,
            "initial_state_sha256": initial_sha,
            "cumulative_runtime_seconds": cumulative_runtime,
            "peak_mps_driver_bytes": peak_mps,
        }
        save_checkpoint(checkpoint_path, **checkpoint_kwargs)
        save_checkpoint(last_checkpoint, **checkpoint_kwargs)
        (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"{args.arm} epoch {epoch}/{config.epochs} "
            f"loss={history[-1]['detection']:.6f} seconds={epoch_runtime:.1f}",
            flush=True,
        )

    if start_epoch > config.epochs:
        print(f"Checkpoint already completed {config.epochs} epochs; evaluating it.")
    model.eval()
    evaluation = evaluate_arm(model, val_dataset, config, device)
    fixed = evaluation["fixed_operating_point"]
    result = {
        "arm": args.arm,
        "completed_epochs": len(history),
        "AP50": evaluation["AP50"],
        "AR100": evaluation["AR100"],
        "recall_iou50": fixed["overall"]["recall"],
        "precision_iou50": fixed["overall"]["precision"],
        "false_positives_per_image": fixed["false_positives_per_image"],
        "visibility20_recall": evaluation["visibility20_recall"],
        "confidence_threshold": config.confidence_threshold,
        "active_timesteps": _active_steps(model),
        "cumulative_training_runtime_seconds": cumulative_runtime,
        "this_invocation_seconds": time.perf_counter() - invocation_started,
        "peak_mps_driver_gib": peak_mps / (1024**3) if device.type == "mps" else None,
        "process_high_water_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2),
        "train_stats": train_stats,
        "validation_stats": val_stats,
    }
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    final_checkpoint = output / "final.pt"
    temporary_final = final_checkpoint.with_suffix(".pt.tmp")
    shutil.copy2(last_checkpoint, temporary_final)
    temporary_final.replace(final_checkpoint)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("fixed_t4", "learned_dtss"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=ROOT / "outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=ROOT / "outputs/subsets/forestpersons_val_600_seed20261010_manifest.json",
    )
    parser.add_argument("--train-annotations", type=Path, default=ROOT / "data/annotations/train.json")
    parser.add_argument("--val-annotations", type=Path, default=ROOT / "data/annotations/val.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--box-loss-weight", type=float, default=5.0)
    parser.add_argument("--mask-loss-weight", type=float, default=0.02)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
