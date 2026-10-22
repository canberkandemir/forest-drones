#!/usr/bin/env python3
"""Evaluate frozen DINOv3 correspondence on official AOS integral pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from forestpersons.aos_correspondence import (
    global_patch_retrieval,
    local_cosine_correspondence,
    yolo_person_patch_mask,
)
from forestpersons.dinov3_detector import choose_device
from forestpersons.dinov3_lora import extract_lora_patch_features, load_lora_fcos_checkpoint


CANONICAL_SUFFIX = "_rot000_fpH0.00_rx0.0_ry0.0.png"
EXPECTED_SCENES = {
    "TRAIN": ("F8", "F10", "O1", "O2", "O3", "O4", "O5"),
    "VALID": ("O6",),
    "TEST": ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F9", "F11"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--integral-root",
        type=Path,
        default=Path("data/aos/integral_images/Integral Images/data"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/"
            "dinov3_lora_fcos.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aos/dinov3_rgb_thermal_correspondence_20261020"),
    )
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="mps")
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--visual-scenes", default="F0,F8,O6")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_labels(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = [float(value) for value in line.split()]
        if len(fields) != 5 or int(fields[0]) != 0:
            raise ValueError(f"Invalid one-class YOLO row in {path}: {line}")
        rows.append(fields)
    return rows


def discover_pairs(root: Path) -> list[dict[str, Any]]:
    rgb_root = root / "SARAOS" / "AOS"
    thermal_root = root / "SARAOS_thermal" / "AOS"
    pairs = []
    for split, scenes in EXPECTED_SCENES.items():
        actual = {path.name for path in (rgb_root / split).iterdir() if path.is_dir()}
        if actual != set(scenes):
            raise ValueError(f"Unexpected {split} scenes: {actual}")
        for scene in scenes:
            rgb_matches = sorted((rgb_root / split / scene).glob(f"*{CANONICAL_SUFFIX}"))
            if len(rgb_matches) != 1:
                raise ValueError(f"Expected one canonical RGB image for {split}/{scene}")
            rgb_path = rgb_matches[0]
            thermal_path = thermal_root / split / scene / rgb_path.name
            rgb_label = rgb_path.with_suffix(".txt")
            thermal_label = thermal_path.with_suffix(".txt")
            if not thermal_path.is_file() or not rgb_label.is_file() or not thermal_label.is_file():
                raise FileNotFoundError(f"Incomplete canonical pair for {split}/{scene}")
            if rgb_label.read_bytes() != thermal_label.read_bytes():
                raise ValueError(f"RGB/thermal labels differ for {split}/{scene}")
            with Image.open(rgb_path) as rgb, Image.open(thermal_path) as thermal:
                if rgb.size != (512, 512) or thermal.size != (512, 512):
                    raise ValueError(f"Unexpected dimensions for {split}/{scene}")
            pairs.append(
                {
                    "split": split,
                    "scene": scene,
                    "rgb_path": rgb_path,
                    "thermal_path": thermal_path,
                    "label_path": rgb_label,
                    "labels": parse_labels(rgb_label),
                }
            )
    return pairs


def thermal_as_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("L").convert("RGB")


def annotated_rgb(path: Path, labels: list[list[float]]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for _, xc, yc, width, height in labels:
        draw.rectangle(
            (
                (xc - width / 2) * image.width,
                (yc - height / 2) * image.height,
                (xc + width / 2) * image.width,
                (yc + height / 2) * image.height,
            ),
            outline=(255, 45, 45),
            width=3,
        )
    return image


def save_visualization(
    pair: dict[str, Any],
    same: torch.Tensor,
    local: torch.Tensor,
    displacement: torch.Tensor,
    output: Path,
) -> None:
    magnitude = displacement.float().square().sum(dim=-1).sqrt()
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(annotated_rgb(pair["rgb_path"], pair["labels"]))
    axes[0].set_title("RGB integral + GT")
    axes[1].imshow(thermal_as_rgb(pair["thermal_path"]), cmap="gray")
    axes[1].set_title("Thermal integral")
    same_plot = axes[2].imshow(same.cpu(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[2].set_title("Same-patch cosine")
    local_plot = axes[3].imshow(local.cpu(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[3].set_title("Best cosine in 3x3")
    displacement_plot = axes[4].imshow(magnitude.cpu(), vmin=0, vmax=1.5, cmap="magma")
    axes[4].set_title("Best-match displacement")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(same_plot, ax=axes[2], fraction=0.046)
    fig.colorbar(local_plot, ax=axes[3], fraction=0.046)
    fig.colorbar(displacement_plot, ax=axes[4], fraction=0.046)
    fig.suptitle(
        f"AOS {pair['split']}/{pair['scene']} frozen DINOv3 RGB-thermal correspondence"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def finite_mean(values: torch.Tensor) -> float | None:
    return float(values.mean().item()) if values.numel() else None


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visualizations"
    visual_dir.mkdir()
    pairs = discover_pairs(args.integral_root)
    device = choose_device(args.device)
    processor, backbone, _head, metadata = load_lora_fcos_checkpoint(args.checkpoint, device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    extracted = []
    peak_mps = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for pair in pairs:
            rgb = Image.open(pair["rgb_path"]).convert("RGB")
            thermal = thermal_as_rgb(pair["thermal_path"])
            rgb_features, rgb_info = extract_lora_patch_features(
                rgb, processor, backbone, device, short_side=args.short_side
            )
            thermal_features, thermal_info = extract_lora_patch_features(
                thermal, processor, backbone, device, short_side=args.short_side
            )
            if (rgb_info.grid_height, rgb_info.grid_width) != (
                thermal_info.grid_height,
                thermal_info.grid_width,
            ):
                raise RuntimeError(f"Feature-grid mismatch for {pair['scene']}")
            extracted.append(
                {
                    **pair,
                    "rgb_features": rgb_features.cpu(),
                    "thermal_features": thermal_features.cpu(),
                    "grid_height": rgb_info.grid_height,
                    "grid_width": rgb_info.grid_width,
                }
            )
            if device.type == "mps":
                torch.mps.synchronize()
                peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))
    inference_seconds = time.perf_counter() - start

    # Predeclared deterministic wrong-pair control: next scene in the complete
    # split-then-scene ordering, wrapping at the end.
    rows = []
    visual_scenes = {value.strip() for value in args.visual_scenes.split(",") if value.strip()}
    for index, item in enumerate(extracted):
        control = extracted[(index + 1) % len(extracted)]
        same, local, displacement = local_cosine_correspondence(
            item["rgb_features"], item["thermal_features"], radius=args.local_radius
        )
        control_same = (item["rgb_features"] * control["thermal_features"]).sum(dim=-1)
        person_mask = yolo_person_patch_mask(
            item["labels"], item["grid_height"], item["grid_width"]
        )
        retrieval = global_patch_retrieval(item["rgb_features"], item["thermal_features"])
        row = {
            "split": item["split"],
            "scene": item["scene"],
            "gt_people": len(item["labels"]),
            "grid_height": item["grid_height"],
            "grid_width": item["grid_width"],
            "paired_same_location_mean_cosine": float(same.mean()),
            "paired_same_location_median_cosine": float(same.median()),
            "paired_local_best_mean_cosine": float(local.mean()),
            "local_alignment_gain": float((local - same).mean()),
            "wrong_scene_control": control["scene"],
            "wrong_scene_same_location_mean_cosine": float(control_same.mean()),
            "paired_minus_control_mean_cosine": float(same.mean() - control_same.mean()),
            "person_patch_count": int(person_mask.sum()),
            "person_patch_mean_cosine": finite_mean(same[person_mask]),
            "background_patch_mean_cosine": finite_mean(same[~person_mask]),
            **retrieval,
        }
        rows.append(row)
        if item["scene"] in visual_scenes:
            save_visualization(
                item,
                same,
                local,
                displacement,
                visual_dir / f"{item['split']}_{item['scene']}.png",
            )

    numeric_keys = (
        "paired_same_location_mean_cosine",
        "paired_local_best_mean_cosine",
        "local_alignment_gain",
        "wrong_scene_same_location_mean_cosine",
        "paired_minus_control_mean_cosine",
        "exact_patch_fraction",
        "within_one_patch_fraction",
        "median_chebyshev_patch_error",
    )
    aggregate = {
        key: float(np.mean([row[key] for row in rows])) for key in numeric_keys
    }
    summary = {
        "status": "completed",
        "experiment": "frozen DINOv3 RGB-thermal patch correspondence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "backbone": metadata["model_id"],
        "lora_checkpoint_loaded": True,
        "head_used": False,
        "device": str(device),
        "short_side": args.short_side,
        "local_radius_patches": args.local_radius,
        "canonical_pairs": len(rows),
        "inference_seconds": inference_seconds,
        "seconds_per_modality_image": inference_seconds / (2 * len(rows)),
        "peak_mps_driver_gib": peak_mps / (1024**3) if device.type == "mps" else None,
        "control_definition": "Thermal features from the next scene in fixed split-then-scene order.",
        "alignment_scope": (
            "Official RGB and thermal integral products share stems, dimensions, labels, and "
            "virtual coordinates. Residual modality-specific blur, rendering, and local "
            "misregistration are measured diagnostically by the 3x3 local search; no warp is fitted."
        ),
        "interpretation": (
            "Feature similarity and retrieval are correspondence diagnostics, not person detection "
            "metrics and not evidence that feature fusion improves recall."
        ),
        "aggregate": aggregate,
        "per_scene": rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: summary[key] for key in ("status", "canonical_pairs", "aggregate")}, indent=2))


if __name__ == "__main__":
    main()
