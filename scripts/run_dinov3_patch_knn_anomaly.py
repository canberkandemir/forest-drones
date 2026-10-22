"""Run patch-kNN anomaly scoring on cached DINOv3 ForestPersons features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.dinov3_detector import (
    annotations_to_ground_truth,
    cache_path_for_file,
    load_feature_cache,
)
from forestpersons.patch_knn import (
    background_patch_mask,
    binary_classification_metrics,
    cosine_knn_anomaly_scores,
    labels_from_ground_truth,
    normalize_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--train-cache-root", type=Path, required=True)
    parser.add_argument("--val-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank-size", type=int, default=5000)
    parser.add_argument("--bank-images", type=int, default=500)
    parser.add_argument("--patches-per-image", type=int, default=20)
    parser.add_argument("--safety-margin-patches", type=float, default=1.0)
    parser.add_argument("--knn-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20261014)
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


def build_reference_bank(
    samples: list[Any],
    cache_root: Path,
    bank_size: int,
    bank_images: int,
    patches_per_image: int,
    safety_margin: float,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rng = random.Random(seed)
    selected = samples.copy()
    rng.shuffle(selected)
    selected = selected[:bank_images]

    chunks: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    candidate_total = 0
    for sample in selected:
        features, info = load_feature_cache(cache_path_for_file(cache_root, sample.file_name))
        ground_truth = annotations_to_ground_truth(sample.annotations)
        mask = background_patch_mask(ground_truth, info, safety_margin=safety_margin).reshape(-1)
        indices = np.flatnonzero(mask)
        candidate_total += int(indices.size)
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        chosen = indices[:patches_per_image]
        flat = features.reshape(-1, features.shape[-1])
        chunks.append(flat[chosen])
        for index in chosen:
            rows.append({"file_name": sample.file_name, "patch_index": int(index)})
        if sum(chunk.shape[0] for chunk in chunks) >= bank_size:
            break

    if not chunks:
        raise RuntimeError("No background patches were available for the reference bank")
    bank = normalize_features(torch.cat(chunks, dim=0)[:bank_size]).contiguous()
    metadata = {
        "bank_size": int(bank.shape[0]),
        "feature_dim": int(bank.shape[1]),
        "sampled_images": len(selected),
        "candidate_background_patches": candidate_total,
        "safety_margin_pixels_resized": safety_margin,
        "rows": rows[: int(bank.shape[0])],
    }
    return bank, metadata


def save_heatmap(
    image_path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    heatmap = Image.fromarray(
        ((scores - scores.min()) / (scores.max() - scores.min() + 1e-8) * 255).astype(np.uint8)
    ).resize(image.size, Image.Resampling.BILINEAR)
    label_map = Image.fromarray((labels.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(image)
    axes[0].set_title("image", fontsize=9)
    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="magma", alpha=0.55)
    axes[1].set_title("kNN anomaly", fontsize=9)
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


def aggregate_visibility_metrics(scores: list[float], labels: list[bool], visibility: list[int]) -> dict[str, Any]:
    result = {}
    for level in [20, 40, 70, 100]:
        selected = np.asarray([(v == level) or (not y) for y, v in zip(labels, visibility)], dtype=bool)
        if not selected.any():
            continue
        selected_labels = np.asarray(labels, dtype=bool)[selected]
        selected_scores = np.asarray(scores, dtype=np.float64)[selected]
        result[str(level)] = binary_classification_metrics(selected_labels, selected_scores)
    return result


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.output_dir / "heatmaps"
    safety_margin = float(args.safety_margin_patches * 16.0)

    train_samples = samples_from_manifest(args.root, "train", args.train_manifest)
    val_samples = samples_from_manifest(args.root, "val", args.val_manifest)
    if args.max_val_images is not None:
        val_samples = val_samples[: args.max_val_images]

    bank, bank_metadata = build_reference_bank(
        train_samples,
        args.train_cache_root,
        bank_size=args.bank_size,
        bank_images=args.bank_images,
        patches_per_image=args.patches_per_image,
        safety_margin=safety_margin,
        seed=args.seed,
    )
    torch.save({"features": bank, "metadata": bank_metadata}, args.output_dir / "reference_bank.pt")
    (args.output_dir / "reference_bank_metadata.json").write_text(
        json.dumps({k: v for k, v in bank_metadata.items() if k != "rows"}, indent=2),
        encoding="utf-8",
    )

    all_scores: list[float] = []
    all_labels: list[bool] = []
    all_visibility: list[int] = []
    image_rows = []
    heatmaps_written = 0
    patch_total = 0

    for sample_index, sample in enumerate(val_samples):
        features, info = load_feature_cache(cache_path_for_file(args.val_cache_root, sample.file_name))
        ground_truth = annotations_to_ground_truth(sample.annotations)
        labels = labels_from_ground_truth(ground_truth, info)
        scores = cosine_knn_anomaly_scores(
            features,
            bank,
            batch_size=args.knn_batch_size,
        ).detach().cpu().numpy()
        flat_scores = scores.reshape(-1)
        flat_labels = labels.person_mask.reshape(-1)
        flat_visibility = labels.visibility.reshape(-1)
        all_scores.extend(float(value) for value in flat_scores)
        all_labels.extend(bool(value) for value in flat_labels)
        all_visibility.extend(int(value) for value in flat_visibility)
        patch_total += flat_scores.size

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

        if heatmaps_written < args.max_heatmaps and (
            sample_index < 4 or int(flat_labels.sum()) > 0
        ):
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

    elapsed = time.perf_counter() - start_time
    summary = {
        "settings": {
            "train_manifest": str(args.train_manifest),
            "val_manifest": str(args.val_manifest),
            "train_cache_root": str(args.train_cache_root),
            "val_cache_root": str(args.val_cache_root),
            "bank_size": args.bank_size,
            "bank_images": args.bank_images,
            "patches_per_image": args.patches_per_image,
            "safety_margin_patches": args.safety_margin_patches,
            "safety_margin_pixels_resized": safety_margin,
            "knn_batch_size": args.knn_batch_size,
            "seed": args.seed,
            "max_val_images": args.max_val_images,
        },
        "reference_bank": {k: v for k, v in bank_metadata.items() if k != "rows"},
        "memory_estimate": {
            "reference_bank_bytes_float32": int(bank.numel() * 4),
            "reference_bank_mib_float32": float(bank.numel() * 4 / (1024 * 1024)),
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
            "Scores are patch anomaly scores, not bounding-box detections. High scores can "
            "come from people, unusual vegetation, shadows, blur, or other rare visual content."
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
