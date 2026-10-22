#!/usr/bin/env python3
"""Prepare, train, and evaluate matched frozen-DINOv3 AOS RGB/thermal arms.

The three commands are intentionally separate so the official TEST products are
never consulted during feature preparation choices or operating-point calibration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forestpersons.aos_multimodal_detection import (  # noqa: E402
    ARM_NAMES,
    SUPPORTED_ARM_NAMES,
    fcos_head_loss,
    feature_for_arm,
    select_evenly,
    verify_scene_disjoint,
    yolo_rows_to_ground_truth,
)
from src.forestpersons.dinov3_detector import (  # noqa: E402
    DINOv3FCOSHead,
    PreprocessInfo,
    build_fcos_targets,
    choose_device,
    decode_fcos_predictions,
)
from src.forestpersons.dinov3_lora import (  # noqa: E402
    extract_lora_patch_features,
    load_lora_fcos_checkpoint,
)
from src.forestpersons.evaluation import (  # noqa: E402
    EvaluationGroundTruth,
    EvaluationPrediction,
    evaluate_coco_style,
    evaluate_confidence_sweep,
    evaluate_custom_metrics,
)


EXPECTED_SCENES = {
    "TRAIN": ("F8", "F10", "O1", "O2", "O3", "O4", "O5"),
    "VALID": ("O6",),
    "TEST": ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F9", "F11"),
}
DEFAULT_CHECKPOINT = ROOT / (
    "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/"
    "dinov3_lora_fcos.pt"
)
DEFAULT_DATA_ROOT = ROOT / "data/aos/integral_images/Integral Images/data"
DEFAULT_OUTPUT = ROOT / "outputs/aos/dinov3_rgb_thermal_detection_20261020"
CANONICAL_SUFFIX = "_rot000_fpH0.00_rx0.0_ry0.0.png"
DEFAULT_THRESHOLDS = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)


def _parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(item.strip() for item in value.split(",") if item.strip())
    if not arms:
        raise ValueError("At least one arm is required")
    if len(set(arms)) != len(arms):
        raise ValueError(f"Duplicate arms are not allowed: {arms}")
    unsupported = [arm for arm in arms if arm not in SUPPORTED_ARM_NAMES]
    if unsupported:
        raise ValueError(f"Unsupported arms {unsupported}; choose from {SUPPORTED_ARM_NAMES}")
    return arms


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_yolo(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5 or int(values[0]) != 0:
            raise ValueError(f"Invalid one-class YOLO row at {path}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in values[1:]):
            raise ValueError(f"Out-of-range YOLO coordinate at {path}:{line_number}")
        rows.append(values)
    return rows


def _scene_paths(data_root: Path, modality: str, split: str, scene: str) -> tuple[Path, Path]:
    collection = "SARAOS" if modality == "rgb" else "SARAOS_thermal"
    scene_root = data_root / collection / "AOS" / split / scene
    return scene_root, scene_root


def _discover_split(
    data_root: Path,
    split: str,
    products_per_scene: int,
    canonical_only: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in EXPECTED_SCENES[split]:
        rgb_images, rgb_labels = _scene_paths(data_root, "rgb", split, scene)
        thermal_images, thermal_labels = _scene_paths(data_root, "thermal", split, scene)
        if not all(path.is_dir() for path in (rgb_images, rgb_labels, thermal_images, thermal_labels)):
            raise FileNotFoundError(f"Missing official AOS product directory for {split}/{scene}")

        images = sorted(rgb_images.glob("*.png"))
        if canonical_only:
            exact = [path for path in images if path.name.endswith(CANONICAL_SUFFIX)]
            if len(exact) != 1:
                raise ValueError(f"Expected one canonical no-AHE product for {scene}, found {len(exact)}")
            selected = exact
        else:
            selected = select_evenly(images, products_per_scene)

        for rgb_path in selected:
            stem = rgb_path.stem
            thermal_path = thermal_images / rgb_path.name
            rgb_label = rgb_labels / f"{stem}.txt"
            thermal_label = thermal_labels / f"{stem}.txt"
            for path in (thermal_path, rgb_label, thermal_label):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if rgb_label.read_bytes() != thermal_label.read_bytes():
                raise ValueError(f"RGB/thermal label mismatch: {split}/{scene}/{stem}")
            with Image.open(rgb_path) as rgb_image, Image.open(thermal_path) as thermal_image:
                if rgb_image.size != thermal_image.size:
                    raise ValueError(f"RGB/thermal size mismatch: {split}/{scene}/{stem}")
                width, height = rgb_image.size
            rows = _read_yolo(rgb_label)
            records.append(
                {
                    "split": split,
                    "scene": scene,
                    "stem": stem,
                    "rgb_path": str(rgb_path.resolve()),
                    "thermal_path": str(thermal_path.resolve()),
                    "label_path": str(rgb_label.resolve()),
                    "width": width,
                    "height": height,
                    "labels": rows,
                }
            )
    return records


def discover_protocol(
    data_root: Path,
    train_per_scene: int,
    valid_per_scene: int,
) -> list[dict[str, Any]]:
    verify_scene_disjoint({key: set(value) for key, value in EXPECTED_SCENES.items()})
    records = _discover_split(data_root, "TRAIN", train_per_scene, False)
    records += _discover_split(data_root, "VALID", valid_per_scene, False)
    records += _discover_split(data_root, "TEST", 1, True)
    for image_id, record in enumerate(records, start=1):
        record["image_id"] = image_id
    return records


def _serialize_info(info: PreprocessInfo) -> dict[str, int | float]:
    return {
        "original_width": info.original_width,
        "original_height": info.original_height,
        "resized_width": info.resized_width,
        "resized_height": info.resized_height,
        "scale": info.scale,
        "grid_width": info.grid_width,
        "grid_height": info.grid_height,
    }


def _deserialize_info(payload: dict[str, int | float]) -> PreprocessInfo:
    return PreprocessInfo(**payload)


def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {manifest_path}; pass --resume to verify/reuse cache")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    records = discover_protocol(args.data_root.resolve(), args.train_per_scene, args.valid_per_scene)
    device = choose_device(args.device)
    processor, backbone, _, checkpoint_meta = load_lora_fcos_checkpoint(checkpoint, device=device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        cache_path = output / "features" / record["split"].lower() / record["scene"] / f"{record['stem']}.pt"
        record["cache_path"] = str(cache_path.resolve())
        if cache_path.is_file() and args.resume:
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(record["rgb_path"]) as image:
            rgb = image.convert("RGB")
        with Image.open(record["thermal_path"]) as image:
            thermal = image.convert("L").convert("RGB")
        with torch.inference_mode():
            rgb_features, rgb_info = extract_lora_patch_features(
                rgb, processor, backbone, device, short_side=args.short_side
            )
            thermal_features, thermal_info = extract_lora_patch_features(
                thermal, processor, backbone, device, short_side=args.short_side
            )
        if rgb_features.shape != thermal_features.shape or rgb_info != thermal_info:
            raise ValueError(f"Feature geometry mismatch for {record['scene']}/{record['stem']}")
        torch.save(
            {
                "rgb": rgb_features.half().cpu(),
                "thermal": thermal_features.half().cpu(),
                "preprocess": _serialize_info(rgb_info),
            },
            cache_path,
        )
        if index % 25 == 0:
            print(f"cached {index}/{len(records)}")

    split_summary: dict[str, dict[str, int]] = {}
    for split in EXPECTED_SCENES:
        selected = [record for record in records if record["split"] == split]
        split_summary[split] = {
            "products": len(selected),
            "scene_count": len({record["scene"] for record in selected}),
            "person_boxes": sum(len(record["labels"]) for record in selected),
        }
    manifest = {
        "protocol": "AOS integral-image frozen DINOv3 matched RGB/thermal pilot",
        "data_root": str(args.data_root.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_metadata": {
            "model_id": checkpoint_meta["model_id"],
            "head_input_dim": checkpoint_meta["head_input_dim"],
            "target_names": checkpoint_meta["target_names"],
            "lora_config": asdict(checkpoint_meta["config"]),
        },
        "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "short_side": args.short_side,
        "thermal_encoding": "official grayscale thermal integral image replicated to RGB",
        "selection": {
            "TRAIN": f"{args.train_per_scene} evenly spaced official products per scene",
            "VALID": f"{args.valid_per_scene} evenly spaced official products from O6",
            "TEST": "one canonical product named after each scene; untouched by calibration",
        },
        "split_summary": split_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "splits": split_summary}, indent=2))


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    split_scenes = {
        split: {record["scene"] for record in manifest["records"] if record["split"] == split}
        for split in EXPECTED_SCENES
    }
    verify_scene_disjoint(split_scenes)
    return manifest


def _load_cache(record: dict[str, Any], arm: str, device: torch.device) -> tuple[torch.Tensor, PreprocessInfo]:
    payload = torch.load(record["cache_path"], map_location="cpu", weights_only=True)
    features = feature_for_arm(payload["rgb"].float(), payload["thermal"].float(), arm)
    return features.to(device), _deserialize_info(payload["preprocess"])


def _ground_truth(records: list[dict[str, Any]]) -> list[EvaluationGroundTruth]:
    result: list[EvaluationGroundTruth] = []
    annotation_id = 1
    for record in records:
        boxes = yolo_rows_to_ground_truth(record["labels"], record["width"], record["height"])
        for box in boxes:
            result.append(
                EvaluationGroundTruth(
                    image_id=record["image_id"],
                    annotation_id=annotation_id,
                    box=[float(value) for value in box["box"]],
                    visibility_level=100,
                )
            )
            annotation_id += 1
    return result


def _group_ground_truth(
    ground_truth: list[EvaluationGroundTruth],
) -> dict[int | str, list[EvaluationGroundTruth]]:
    grouped: dict[int | str, list[EvaluationGroundTruth]] = {}
    for item in ground_truth:
        grouped.setdefault(item.image_id, []).append(item)
    return grouped


def _group_predictions(
    predictions: list[EvaluationPrediction],
) -> dict[int | str, list[EvaluationPrediction]]:
    grouped: dict[int | str, list[EvaluationPrediction]] = {}
    for item in predictions:
        grouped.setdefault(item.image_id, []).append(item)
    return grouped


def _predict(
    head: DINOv3FCOSHead,
    records: list[dict[str, Any]],
    arm: str,
    device: torch.device,
    confidence: float = 0.001,
) -> list[EvaluationPrediction]:
    predictions: list[EvaluationPrediction] = []
    head.eval()
    with torch.inference_mode():
        for record in records:
            features, info = _load_cache(record, arm, device)
            class_logits, box_raw, centerness_logits = head(features)
            decoded = decode_fcos_predictions(
                class_logits,
                box_raw,
                centerness_logits,
                info,
                conf_threshold=confidence,
                nms_threshold=0.5,
                topk=300,
            )
            for prediction in decoded[:100]:
                predictions.append(
                    EvaluationPrediction(
                        image_id=record["image_id"],
                        box=[float(value) for value in prediction["box"]],
                        confidence=float(prediction["confidence"]),
                    )
                )
    return predictions


def _train_one_arm(
    arm: str,
    initial_state: dict[str, torch.Tensor],
    train_records: list[dict[str, Any]],
    valid_records: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    head = DINOv3FCOSHead(input_dim=384).to(device)
    head.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    orders: list[list[int]] = []
    generator = random.Random(args.seed)
    for _ in range(args.epochs):
        order = list(range(len(train_records)))
        generator.shuffle(order)
        orders.append(order)

    history: list[dict[str, float | int]] = []
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch, order in enumerate(orders, start=1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {"total": 0.0, "classification": 0.0, "regression": 0.0, "centerness": 0.0}
        positives = 0
        for position, record_index in enumerate(order, start=1):
            record = train_records[record_index]
            features, info = _load_cache(record, arm, device)
            boxes = yolo_rows_to_ground_truth(record["labels"], record["width"], record["height"])
            classification, box_targets, centerness, positive_mask, _ = build_fcos_targets(
                boxes,
                info,
                grid_height=features.shape[0],
                grid_width=features.shape[1],
            )
            classification = classification.to(device)
            box_targets = box_targets.to(device)
            centerness = centerness.to(device)
            positive_mask = positive_mask.to(device)
            class_logits, box_raw, centerness_logits = head(features)
            loss, parts = fcos_head_loss(
                class_logits,
                box_raw,
                centerness_logits,
                classification,
                box_targets,
                centerness,
                positive_mask,
                box_weight=2.0,
            )
            (loss / args.gradient_accumulation).backward()
            for key in sums:
                sums[key] += float(parts[key])
            positives += int(positive_mask.sum())
            should_step = position % args.gradient_accumulation == 0 or position == len(order)
            if should_step:
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        row: dict[str, float | int] = {
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "positive_locations": positives,
        }
        row.update({key: value / len(order) for key, value in sums.items()})
        history.append(row)
        print(f"{arm} epoch {epoch}/{args.epochs}: loss={row['total']:.6f}")

    arm_dir = output / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = arm_dir / "fcos_head.pt"
    torch.save(
        {
            "arm": arm,
            "head_state_dict": head.state_dict(),
            "initial_state_sha256": _state_digest(initial_state),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "func"
            },
            "history": history,
        },
        checkpoint_path,
    )
    valid_predictions = _predict(head, valid_records, arm, device)
    valid_gt = _ground_truth(valid_records)
    valid_ids = [record["image_id"] for record in valid_records]
    gt_by_image = _group_ground_truth(valid_gt)
    pred_by_image = _group_predictions(valid_predictions)
    sweep = evaluate_confidence_sweep(
        gt_by_image,
        pred_by_image,
        valid_ids,
        confidence_thresholds=DEFAULT_THRESHOLDS,
        iou_threshold=0.5,
        fp_budgets=(args.fp_budget,),
    )
    operating = sweep["matched_false_positive_budgets"].get(str(args.fp_budget))
    if operating is None:
        operating = min(sweep["rows"], key=lambda row: row["false_positives_per_image"])
    summary = {
        "arm": arm,
        "checkpoint": str(checkpoint_path.resolve()),
        "train_seconds": time.perf_counter() - started,
        "optimizer_steps": optimizer_steps,
        "history": history,
        "validation_calibration": {
            "scene": "O6 only",
            "threshold": operating["confidence_threshold"],
            "metrics": operating,
            "fp_budget": args.fp_budget,
            "fp_budget_satisfied": (
                operating["false_positives_per_image"] <= args.fp_budget
            ),
            "warning": "Validation products repeat five underlying people; calibration is feasibility-only.",
        },
    }
    (arm_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def train(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest.resolve())
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing training directory: {output}")
    output.mkdir(parents=True)
    train_records = [record for record in manifest["records"] if record["split"] == "TRAIN"]
    valid_records = [record for record in manifest["records"] if record["split"] == "VALID"]
    if not train_records or not valid_records:
        raise ValueError("Manifest requires non-empty TRAIN and VALID records")
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    initial_head = DINOv3FCOSHead(input_dim=384)
    initial_state = copy.deepcopy(initial_head.state_dict())
    arms = _parse_arms(args.arms)
    summaries = [
        _train_one_arm(arm, initial_state, train_records, valid_records, args, device, output)
        for arm in arms
    ]
    # This rule is fixed before TEST evaluation: maximize validation recall
    # under the FP budget, then precision, retaining arm order for exact ties.
    preferred = max(
        summaries,
        key=lambda item: (
            bool(item["validation_calibration"]["fp_budget_satisfied"]),
            float(item["validation_calibration"]["metrics"]["overall"]["recall"]),
            float(item["validation_calibration"]["metrics"]["overall"]["precision"]),
        ),
    )
    (output / "matched_training_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "initial_state_sha256": _state_digest(initial_state),
                "arms": summaries,
                "validation_only_selection": {
                    "preferred_arm": preferred["arm"],
                    "preferred_threshold": preferred["validation_calibration"]["threshold"],
                    "rule": (
                        "Prefer arms satisfying the O6 FP/image budget, then choose highest "
                        "validation recall, precision, and predeclared arm order. If no arm "
                        "satisfies the budget, the lowest-FP sweep point is used per arm."
                    ),
                    "test_metrics_consulted": False,
                },
            },
            indent=2,
        )
        + "\n"
    )


def _prediction_json(predictions: list[EvaluationPrediction]) -> list[dict[str, Any]]:
    return [
        {
            "image_id": prediction.image_id,
            "category_id": 1,
            "bbox_xyxy": list(prediction.box),
            "score": prediction.confidence,
        }
        for prediction in predictions
    ]


def evaluate(args: argparse.Namespace) -> None:
    manifest = _load_manifest(args.manifest.resolve())
    test_records = [record for record in manifest["records"] if record["split"] == "TEST"]
    if len(test_records) != len(EXPECTED_SCENES["TEST"]):
        raise ValueError("Evaluation requires exactly one canonical TEST product per official scene")
    if not all(Path(record["rgb_path"]).name.endswith(CANONICAL_SUFFIX) for record in test_records):
        raise ValueError("TEST records are not canonical no-AHE products")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation directory: {output}")
    output.mkdir(parents=True)
    device = choose_device(args.device)
    ground_truth = _ground_truth(test_records)
    training_summary = json.loads(
        (args.training_dir.resolve() / "matched_training_summary.json").read_text()
    )
    trained_arms = tuple(item["arm"] for item in training_summary["arms"])
    requested_arms = _parse_arms(args.arms) if args.arms else trained_arms
    if set(requested_arms) != set(trained_arms):
        raise ValueError(
            f"Evaluation arms {requested_arms} must exactly match trained arms {trained_arms}"
        )
    selection = training_summary.get("validation_only_selection")
    if args.require_validation_selection and selection is None:
        raise ValueError("Training summary lacks a validation-only preferred-arm selection")
    rows: list[dict[str, Any]] = []
    for arm in requested_arms:
        arm_train = args.training_dir.resolve() / arm
        summary = json.loads((arm_train / "training_summary.json").read_text())
        threshold = float(summary["validation_calibration"]["threshold"])
        checkpoint = torch.load(arm_train / "fcos_head.pt", map_location="cpu", weights_only=False)
        head = DINOv3FCOSHead(input_dim=384).to(device)
        head.load_state_dict(checkpoint["head_state_dict"])
        started = time.perf_counter()
        predictions = _predict(head, test_records, arm, device)
        elapsed = time.perf_counter() - started
        image_ids = [record["image_id"] for record in test_records]
        custom = evaluate_custom_metrics(
            _group_ground_truth(ground_truth),
            _group_predictions(predictions),
            image_ids,
            confidence_threshold=threshold,
            iou_threshold=0.5,
        )
        coco = evaluate_coco_style(
            ground_truth,
            predictions,
            image_ids=image_ids,
            max_dets=(1, 10, 100),
        )
        row = {
            "arm": arm,
            "validation_preferred": bool(selection and arm == selection["preferred_arm"]),
            "threshold_selected_on_valid_o6": threshold,
            "images": len(test_records),
            "ground_truth_people": len(ground_truth),
            "precision": custom["overall"]["precision"],
            "recall": custom["overall"]["recall"],
            "fp_per_image": custom["false_positives_per_image"],
            "tp": custom["overall"]["true_positives"],
            "fp": custom["overall"]["false_positives"],
            "fn": custom["overall"]["false_negatives"],
            "ap": coco["AP"],
            "ap50": coco["AP50"],
            "ap75": coco["AP75"],
            "ar100": coco["AR_by_max_dets"]["100"],
            "seconds": elapsed,
            "seconds_per_image": elapsed / len(test_records),
        }
        rows.append(row)
        (output / f"{arm}_predictions.json").write_text(
            json.dumps(_prediction_json(predictions), indent=2) + "\n"
        )
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "selection.json").write_text(
        json.dumps(
            {
                "validation_only_selection": selection,
                "preferred_test_result": next(
                    (row for row in rows if row["validation_preferred"]), None
                ),
                "warning": "The preferred arm and threshold were locked before TEST evaluation.",
            },
            indent=2,
        )
        + "\n"
    )
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common_device = argparse.ArgumentParser(add_help=False)
    common_device.add_argument("--device", default="mps", choices=("mps", "cpu", "auto"))

    prepare_parser = subparsers.add_parser("prepare", parents=[common_device])
    prepare_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    prepare_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "prepared")
    prepare_parser.add_argument("--short-side", type=int, default=672)
    prepare_parser.add_argument("--train-per-scene", type=int, default=32)
    prepare_parser.add_argument("--valid-per-scene", type=int, default=32)
    prepare_parser.add_argument("--resume", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    train_parser = subparsers.add_parser("train", parents=[common_device])
    train_parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT / "prepared/manifest.json")
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "training")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--gradient-accumulation", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=20261020)
    train_parser.add_argument("--fp-budget", type=float, default=0.425)
    train_parser.add_argument(
        "--arms",
        default=",".join(ARM_NAMES),
        help=f"Comma-separated matched arms; supported: {','.join(SUPPORTED_ARM_NAMES)}",
    )
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser("evaluate", parents=[common_device])
    evaluate_parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT / "prepared/manifest.json")
    evaluate_parser.add_argument("--training-dir", type=Path, default=DEFAULT_OUTPUT / "training")
    evaluate_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "evaluation")
    evaluate_parser.add_argument(
        "--arms",
        default=None,
        help="Optional exact trained-arm list; defaults to matched training summary",
    )
    evaluate_parser.add_argument("--require-validation-selection", action="store_true")
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
