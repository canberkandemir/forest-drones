#!/usr/bin/env python3
"""Run a matched Fixed-T versus learned-DTSS ForestPersons detection pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from forestpersons.dtss_detector import (
    DTSSDetectorConfig,
    DTSSPatchDetector,
    scale_xywh_to_normalized_xyxy,
    set_dtss_learning,
)
from forestpersons.evaluation import (
    EvaluationPrediction,
    annotations_to_eval_ground_truth,
    evaluate_coco_style,
    evaluate_custom_metrics,
)


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 20261020
    train_per_visibility: int = 8
    val_per_visibility: int = 4
    epochs: int = 8
    batch_size: int = 4
    learning_rate: float = 0.002
    weight_decay: float = 0.0
    box_loss_weight: float = 5.0
    mask_loss_weight: float = 0.02
    confidence_threshold: float = 0.05
    nms_iou: float = 0.50
    overfit_steps: int = 50


class PilotImageDataset:
    def __init__(self, records: list[dict[str, Any]], image_root: Path, image_size: int) -> None:
        self.records = records
        self.image_root = image_root
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.image_root / record["file_name"]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            original_size = image.size
            resized = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        boxes = torch.stack(
            [
                scale_xywh_to_normalized_xyxy(
                    annotation["bbox"], original_size, (self.image_size, self.image_size)
                )
                for annotation in record["annotations"]
            ]
        )
        return {"image": (tensor - mean) / std, "boxes": boxes, "record": record}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _visibility(annotation: dict[str, Any]) -> int:
    return int(annotation["attributes"]["visibility_level"])


def select_records(
    manifest_path: Path,
    annotation_path: Path,
    per_visibility: int,
) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    coco = _load_json(annotation_path)
    manifest_by_id = {int(item["image_id"]): item for item in manifest["images"]}
    annotations_by_id: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in manifest_by_id}
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in annotations_by_id:
            normalized = dict(annotation)
            attributes = annotation.get("attributes", {})
            normalized["metadata"] = dict(attributes) if isinstance(attributes, dict) else {}
            annotations_by_id[image_id].append(normalized)
    selected: list[dict[str, Any]] = []
    for visibility in (20, 40, 70, 100):
        candidates = []
        for image_id, item in manifest_by_id.items():
            annotations = annotations_by_id[image_id]
            if len(annotations) == 1 and _visibility(annotations[0]) == visibility:
                candidates.append({**item, "annotations": annotations})
        candidates.sort(key=lambda item: (item["file_name"], int(item["image_id"])))
        if len(candidates) < per_visibility:
            raise RuntimeError(f"only {len(candidates)} single-person images at visibility {visibility}")
        selected.extend(candidates[:per_visibility])
    return selected


def _hashes(records: list[dict[str, Any]], image_root: Path) -> set[str]:
    result = set()
    for record in records:
        digest = hashlib.sha256((image_root / record["file_name"]).read_bytes()).hexdigest()
        result.add(digest)
    return result


def _batches(dataset: PilotImageDataset, order: torch.Tensor, batch_size: int):
    for offset in range(0, len(order), batch_size):
        items = [dataset[int(index)] for index in order[offset : offset + batch_size]]
        yield (
            torch.stack([item["image"] for item in items]),
            [item["boxes"] for item in items],
            [item["record"] for item in items],
        )


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _active_steps(model: DTSSPatchDetector) -> dict[str, int]:
    return {name: module().active_timesteps for name, module in model.timestep_masks().items()}


def run_overfit_gate(
    initial_state: dict[str, torch.Tensor],
    model_config: DTSSDetectorConfig,
    dataset: PilotImageDataset,
    config: PilotConfig,
    device: torch.device,
) -> dict[str, float | bool]:
    model = DTSSPatchDetector(model_config).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    set_dtss_learning(model, False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.005,
    )
    images, boxes, _ = next(_batches(dataset, torch.arange(4), 4))
    images = images.to(device)
    boxes = [item.to(device) for item in boxes]
    losses = []
    model.train()
    for _ in range(config.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(
            model(images), boxes, box_weight=config.box_loss_weight
        )["detection"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    passed = losses[-1] < 0.70 * losses[0] and np.isfinite(losses).all()
    return {"initial_loss": losses[0], "final_loss": losses[-1], "passed": bool(passed)}


def train_arm(
    name: str,
    initial_state: dict[str, torch.Tensor],
    model_config: DTSSDetectorConfig,
    pilot_config: PilotConfig,
    train_dataset: PilotImageDataset,
    orders: list[torch.Tensor],
    device: torch.device,
    output_dir: Path,
) -> tuple[DTSSPatchDetector, dict[str, Any]]:
    model = DTSSPatchDetector(model_config).to(device)
    model.load_state_dict(copy.deepcopy(initial_state))
    learned = name == "learned_dtss"
    set_dtss_learning(model, learned)
    initial_timestep = {
        key: module.raw_timestep_parameters.detach().cpu().clone()
        for key, module in model.timestep_masks().items()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=pilot_config.learning_rate,
        weight_decay=pilot_config.weight_decay,
    )
    history = []
    peak_mps = 0
    start = time.perf_counter()
    for epoch, order in enumerate(orders, start=1):
        model.train()
        totals = {"detection": 0.0, "classification": 0.0, "regression": 0.0}
        examples = 0
        dtss_gradient_norms = []
        for images, boxes, _ in _batches(train_dataset, order, pilot_config.batch_size):
            images = images.to(device)
            boxes = [item.to(device) for item in boxes]
            optimizer.zero_grad(set_to_none=True)
            components = model.compute_loss(
                model(images), boxes, box_weight=pilot_config.box_loss_weight
            )
            objective = components["detection"]
            if learned:
                objective = objective + pilot_config.mask_loss_weight * model.mask_loss() / (
                    len(model.timestep_masks()) * model_config.t_max
                )
            objective.backward()
            if learned:
                gradients = [
                    module.raw_timestep_parameters.grad.flatten()
                    for module in model.timestep_masks().values()
                    if module.raw_timestep_parameters.grad is not None
                ]
                dtss_gradient_norms.append(float(torch.cat(gradients).norm().detach().cpu()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(boxes)
            examples += count
            for key in totals:
                totals[key] += float(components[key].detach().cpu()) * count
            if device.type == "mps":
                peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))
        history.append(
            {
                "epoch": epoch,
                **{key: value / examples for key, value in totals.items()},
                "active_timesteps": _active_steps(model),
                "mean_dtss_gradient_norm": float(np.mean(dtss_gradient_norms)) if dtss_gradient_norms else 0.0,
            }
        )
    _sync(device)
    runtime = time.perf_counter() - start
    delta = {
        key: float((model.timestep_masks()[key].raw_timestep_parameters.detach().cpu() - value).norm())
        for key, value in initial_timestep.items()
    }
    checkpoint = output_dir / f"{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": asdict(model_config),
            "pilot_config": asdict(pilot_config),
            "arm": name,
        },
        checkpoint,
    )
    return model, {
        "name": name,
        "checkpoint": str(checkpoint),
        "runtime_seconds": runtime,
        "peak_mps_driver_gib": peak_mps / (1024**3) if device.type == "mps" else None,
        "process_high_water_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        "history": history,
        "final_active_timesteps": _active_steps(model),
        "timestep_parameter_delta_norms": delta,
    }


def evaluate_arm(
    model: DTSSPatchDetector,
    dataset: PilotImageDataset,
    config: PilotConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    ground_truth_by_image = {}
    predictions_by_image = {}
    image_ids = []
    latency_start = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(dataset)):
            item = dataset[index]
            record = item["record"]
            image_id = int(record["image_id"])
            image_ids.append(image_id)
            ground_truth_by_image[image_id] = annotations_to_eval_ground_truth(
                image_id, record["annotations"]
            )
            prediction = model.predict(
                item["image"].unsqueeze(0).to(device),
                confidence_threshold=0.001,
                nms_iou=config.nms_iou,
            )[0]
            width, height = float(record["width"]), float(record["height"])
            predictions_by_image[image_id] = [
                EvaluationPrediction(
                    image_id=image_id,
                    box=[
                        float(box[0].cpu()) * width,
                        float(box[1].cpu()) * height,
                        float(box[2].cpu()) * width,
                        float(box[3].cpu()) * height,
                    ],
                    confidence=float(score.cpu()),
                )
                for box, score in zip(prediction["boxes"], prediction["scores"])
            ]
    _sync(device)
    latency = (time.perf_counter() - latency_start) / len(dataset)
    ground_truth = [item for values in ground_truth_by_image.values() for item in values]
    predictions = [item for values in predictions_by_image.values() for item in values]
    coco = evaluate_coco_style(ground_truth, predictions, image_ids)
    custom = evaluate_custom_metrics(
        ground_truth_by_image,
        predictions_by_image,
        image_ids,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=0.50,
    )
    return {
        "images": len(image_ids),
        "ground_truth": len(ground_truth),
        "AP50": coco["AP50"],
        "AR100": coco["AR_by_max_dets"]["100"],
        "fixed_operating_point": custom,
        "visibility20_recall": custom["by_visibility"].get("20", {}).get("recall", 0.0),
        "latency_seconds_per_image": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dtss_detection/train32_val16_seed20261020"),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.backends.mps.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pilot_config = PilotConfig()
    model_config = DTSSDetectorConfig()
    random.seed(pilot_config.seed)
    np.random.seed(pilot_config.seed)
    torch.manual_seed(pilot_config.seed)

    train_records = select_records(
        Path("outputs/subsets/forestpersons_train_400_seed20261011_manifest_proposal.json"),
        Path("data/annotations/train.json"),
        pilot_config.train_per_visibility,
    )
    val_records = select_records(
        Path("outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"),
        Path("data/annotations/val.json"),
        pilot_config.val_per_visibility,
    )
    train_hashes = _hashes(train_records, Path("data/images"))
    val_hashes = _hashes(val_records, Path("data/images"))
    if train_hashes & val_hashes:
        raise RuntimeError("train/validation image hash overlap detected")
    train_dataset = PilotImageDataset(train_records, Path("data/images"), model_config.image_size)
    val_dataset = PilotImageDataset(val_records, Path("data/images"), model_config.image_size)
    initial_model = DTSSPatchDetector(model_config)
    initial_state = copy.deepcopy(initial_model.state_dict())
    overfit = run_overfit_gate(
        initial_state, model_config, train_dataset, pilot_config, device
    )
    if not overfit["passed"]:
        raise RuntimeError(f"tiny overfit gate failed: {overfit}")
    generator = torch.Generator().manual_seed(pilot_config.seed + 1)
    orders = [torch.randperm(len(train_dataset), generator=generator) for _ in range(pilot_config.epochs)]
    arms = []
    for name in ("fixed_t4", "learned_dtss"):
        model, training = train_arm(
            name,
            initial_state,
            model_config,
            pilot_config,
            train_dataset,
            orders,
            device,
            args.output_dir,
        )
        training["evaluation"] = evaluate_arm(model, val_dataset, pilot_config, device)
        arms.append(training)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    summary = {
        "scope": "tiny image-disjoint ForestPersons detection pilot; metrics are inconclusive",
        "device": str(device),
        "model_config": asdict(model_config),
        "pilot_config": asdict(pilot_config),
        "train_images": len(train_records),
        "validation_images": len(val_records),
        "train_validation_hash_overlap": 0,
        "overfit_gate": overfit,
        "compute_caveat": "DTSS masks outputs, but eager PyTorch still computes all T=4 timesteps.",
        "arms": arms,
    }
    (args.output_dir / "train_manifest.json").write_text(json.dumps(train_records, indent=2), encoding="utf-8")
    (args.output_dir / "validation_manifest.json").write_text(json.dumps(val_records, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
