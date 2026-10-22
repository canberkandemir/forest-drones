"""Run a classical RX anomaly baseline on ForestPersons RGB patches."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import PATCH_SIZE, annotations_to_ground_truth
from forestpersons.patch_knn import (
    background_patch_mask,
    binary_classification_metrics,
    labels_from_ground_truth,
)
from forestpersons.rx_anomaly import fit_rx_model, rgb_patch_features, rx_scores, serialize_rx_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--background-images", type=int, default=500)
    parser.add_argument("--patches-per-image", type=int, default=100)
    parser.add_argument("--max-background-patches", type=int, default=50000)
    parser.add_argument("--safety-margin-patches", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--max-val-images", type=int)
    parser.add_argument("--max-heatmaps", type=int, default=16)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def samples_from_manifest(root: Path, split: str, manifest_path: Path) -> list[Any]:
    dataset = ForestPersonsDataset(root, split=split)
    by_file = {sample.file_name: sample for sample in dataset.samples}
    return [by_file[item["file_name"]] for item in load_manifest(manifest_path)]


def sample_background_features(
    samples: list[Any],
    background_images: int,
    patches_per_image: int,
    max_background_patches: int,
    safety_margin_pixels: float,
    short_side: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = random.Random(seed)
    selected = samples.copy()
    rng.shuffle(selected)
    selected = selected[:background_images]
    chunks: list[np.ndarray] = []
    candidate_total = 0
    used_images = 0
    for sample in selected:
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image: {sample.image_path}")
        image = Image.open(sample.image_path).convert("RGB")
        features, info = rgb_patch_features(image, short_side=short_side)
        ground_truth = annotations_to_ground_truth(sample.annotations)
        mask = background_patch_mask(ground_truth, info, safety_margin=safety_margin_pixels).reshape(-1)
        indices = np.flatnonzero(mask)
        candidate_total += int(indices.size)
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        chosen = indices[:patches_per_image]
        chunks.append(features.reshape(-1, features.shape[-1])[chosen])
        used_images += 1
        if sum(chunk.shape[0] for chunk in chunks) >= max_background_patches:
            break

    if not chunks:
        raise RuntimeError("No background patches available for RX fitting")
    background = np.concatenate(chunks, axis=0)[:max_background_patches].astype(np.float32)
    metadata = {
        "sampled_images": len(selected),
        "used_images": used_images,
        "background_patches": int(background.shape[0]),
        "candidate_background_patches": candidate_total,
        "feature_dim": int(background.shape[1]),
        "safety_margin_pixels_resized": safety_margin_pixels,
    }
    return background, metadata


def save_heatmap(
    image_path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heatmap = Image.fromarray((normalized * 255).astype(np.uint8)).resize(
        image.size,
        Image.Resampling.BILINEAR,
    )
    label_map = Image.fromarray((labels.astype(np.uint8) * 255)).resize(
        image.size,
        Image.Resampling.NEAREST,
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(image)
    axes[0].set_title("image", fontsize=9)
    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="magma", alpha=0.55)
    axes[1].set_title("RX anomaly", fontsize=9)
    axes[2].imshow(image)
    axes[2].imshow(label_map, cmap="Greens", alpha=0.45)
    axes[2].set_title("GT person patches", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.output_dir / "heatmaps"
    safety_margin_pixels = float(args.safety_margin_patches * PATCH_SIZE)

    train_samples = samples_from_manifest(args.root, "train", args.train_manifest)
    val_samples = samples_from_manifest(args.root, "val", args.val_manifest)
    if args.max_val_images is not None:
        val_samples = val_samples[: args.max_val_images]

    background_features, bank_metadata = sample_background_features(
        train_samples,
        background_images=args.background_images,
        patches_per_image=args.patches_per_image,
        max_background_patches=args.max_background_patches,
        safety_margin_pixels=safety_margin_pixels,
        short_side=args.short_side,
        seed=args.seed,
    )
    model = fit_rx_model(background_features, regularization=args.regularization)
    (args.output_dir / "rx_model.json").write_text(
        json.dumps(serialize_rx_model(model), indent=2),
        encoding="utf-8",
    )
    np.save(args.output_dir / "background_features.npy", background_features)

    all_scores: list[float] = []
    all_labels: list[bool] = []
    all_visibility: list[int] = []
    image_rows: list[dict[str, Any]] = []
    heatmaps_written = 0
    patch_total = 0

    for sample_index, sample in enumerate(val_samples):
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing image: {sample.image_path}")
        image = Image.open(sample.image_path).convert("RGB")
        features, info = rgb_patch_features(image, short_side=args.short_side)
        ground_truth = annotations_to_ground_truth(sample.annotations)
        labels = labels_from_ground_truth(ground_truth, info)
        scores = rx_scores(features, model)

        flat_scores = scores.reshape(-1)
        flat_labels = labels.person_mask.reshape(-1)
        flat_visibility = labels.visibility.reshape(-1)
        all_scores.extend(float(value) for value in flat_scores)
        all_labels.extend(bool(value) for value in flat_labels)
        all_visibility.extend(int(value) for value in flat_visibility)
        patch_total += int(flat_scores.size)

        image_metrics = binary_classification_metrics(flat_labels, flat_scores)
        image_rows.append(
            {
                "file_name": sample.file_name,
                "image_id": sample.image_id,
                "patches": int(flat_scores.size),
                "person_patches": int(flat_labels.sum()),
                "background_patches": int((~flat_labels).sum()),
                "auroc": image_metrics["auroc"],
                "average_precision": image_metrics["average_precision"],
                "score_mean": float(flat_scores.mean()),
                "person_score_mean": float(flat_scores[flat_labels].mean()) if flat_labels.any() else None,
                "background_score_mean": float(flat_scores[~flat_labels].mean()) if (~flat_labels).any() else None,
            }
        )

        if heatmaps_written < args.max_heatmaps and (sample_index < 4 or flat_labels.any()):
            output_name = sample.file_name.replace("/", "__")
            save_heatmap(
                sample.image_path,
                scores,
                labels.person_mask,
                heatmap_dir / output_name,
                title=f"{sample.file_name} AUROC={image_metrics['auroc']}",
            )
            heatmaps_written += 1

    labels_np = np.asarray(all_labels, dtype=bool)
    scores_np = np.asarray(all_scores, dtype=np.float64)
    visibility_np = np.asarray(all_visibility, dtype=np.int16)
    overall = binary_classification_metrics(labels_np, scores_np)
    by_visibility = {}
    for level in [20, 40, 70, 100]:
        selected = (visibility_np == level) | (~labels_np)
        by_visibility[str(level)] = binary_classification_metrics(labels_np[selected], scores_np[selected])

    elapsed = time.perf_counter() - started
    summary = {
        "settings": {
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "short_side": args.short_side,
            "patch_size": PATCH_SIZE,
            "feature_representation": "per-patch RGB mean and RGB standard deviation in [0, 1]",
            "background_images": args.background_images,
            "patches_per_image": args.patches_per_image,
            "max_background_patches": args.max_background_patches,
            "safety_margin_patches": args.safety_margin_patches,
            "safety_margin_pixels_resized": safety_margin_pixels,
            "regularization": args.regularization,
            "seed": args.seed,
            "max_val_images": args.max_val_images,
        },
        "background_model": {
            **bank_metadata,
            "mean": model.mean.tolist(),
            "covariance_regularization": model.regularization,
            "feature_names": list(model.feature_names),
            "background_feature_bytes_float32": int(background_features.nbytes),
            "background_feature_mib_float32": float(background_features.nbytes / (1024 * 1024)),
        },
        "validation": {
            "images": len(val_samples),
            "patches": patch_total,
            "person_patches": int(labels_np.sum()),
            "background_patches": int((~labels_np).sum()),
        },
        "overall": overall,
        "by_visibility": by_visibility,
        "runtime": {
            "seconds_total": elapsed,
            "seconds_per_image": elapsed / len(val_samples) if val_samples else 0.0,
        },
        "interpretation_note": (
            "RX scores are patch-level RGB distribution anomalies, not bounding-box detections. "
            "High scores often reflect color/texture rarity and may correspond to vegetation, "
            "illumination, shadows, roads, or people."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "image_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {args.output_dir / 'summary.json'}")
    print(f"Saved heatmaps: {heatmap_dir}")


if __name__ == "__main__":
    main()
