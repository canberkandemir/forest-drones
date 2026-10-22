"""Create plots, per-group diagnostics, and real-image galleries for learned fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from forestpersons import ForestPersonsDataset
from forestpersons.detection import draw_detection_comparison, evaluate_image_matches, xywh_to_xyxy
from forestpersons.evaluation import EvaluationPrediction, annotations_to_eval_ground_truth
from forestpersons.multiview import TrainableReferencePreservingFusion
from scripts.train_dinov3_learnable_multiview_fusion import (
    ensure_group_ids,
    load_cached_group,
    matched_ids,
)


METHODS = ("single", "equal_mean", "fixed_selective", "learned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[int, list[EvaluationPrediction]]:
    output: dict[int, list[EvaluationPrediction]] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        image_id = int(item["image_id"])
        output.setdefault(image_id, []).append(
            EvaluationPrediction(image_id, xywh_to_xyxy(item["bbox"]), float(item["score"]))
        )
    return output


def save_curves(summary: dict, output: Path) -> None:
    history = summary["history"]
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(epochs, [row["train"]["loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], marker="o", label="validation")
    axes[0].set_title("Detection loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_fixed_threshold"]["overall"]["recall"] for row in history], marker="o", label="recall")
    axes[1].plot(epochs, [row["validation_coco"]["AR@100"] for row in history], marker="o", label="AR@100")
    axes[1].set_title("Validation detection")
    axes[1].legend()
    axes[2].plot(epochs, [row["validation_gate"]["mean_total_source_weight"] for row in history], marker="o")
    axes[2].axhline(0.005, color="black", linestyle="--", linewidth=1, label="collapse flag")
    axes[2].set_title("Mean source contribution")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def panel(image_path: Path, gt: list, predictions: list[EvaluationPrediction], confidence: float, title: str, output: Path) -> None:
    gt_rows = [
        {"box": item.box, "visibility_level": item.visibility_level, "annotation_id": item.annotation_id}
        for item in gt
    ]
    pred_rows = [
        {"box": item.box, "confidence": item.confidence}
        for item in predictions
        if item.confidence >= confidence
    ]
    matches = evaluate_image_matches(gt_rows, pred_rows, 0.50)[0]
    draw_detection_comparison(image_path, gt_rows, pred_rows, matches, output, title)


def thumbnail(path: Path, width: int = 520) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, 390), Image.Resampling.LANCZOS)
    return image


def compose_gallery(paths: list[tuple[str, Path]], output: Path) -> None:
    cells = [(label, thumbnail(path)) for label, path in paths]
    cell_width, cell_height = 540, 430
    columns = 4
    rows = int(np.ceil(len(cells) / columns))
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(cells):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        draw.text((x + 8, y + 8), label, fill="black")
        canvas.paste(image, (x + 8, y + 32))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)


def main() -> None:
    args = parse_args()
    summary = json.loads((args.experiment_dir / "summary.json").read_text(encoding="utf-8"))
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    groups = split["validation_groups"]
    ensure_group_ids(groups)
    dataset = ForestPersonsDataset(args.root, split="train")
    samples = {int(sample.image_id): sample for sample in dataset.samples}
    predictions = {
        method: load_predictions(args.experiment_dir / f"{method}_coco_results.json")
        for method in METHODS
    }

    rows = []
    for group in groups:
        image_id = int(group["reference"]["image_id"])
        gt = annotations_to_eval_ground_truth(image_id, samples[image_id].annotations)
        row = {
            "group_id": group["group_id"],
            "sequence": group["sequence"],
            "reference_image_id": image_id,
            "ground_truth": len(gt),
        }
        for method in METHODS:
            detected = matched_ids(gt, predictions[method].get(image_id, []), args.confidence)
            kept = [item for item in predictions[method].get(image_id, []) if item.confidence >= args.confidence]
            row[f"{method}_TP"] = len(detected)
            row[f"{method}_FP"] = len(kept) - len(detected)
            row[f"{method}_FN"] = len(gt) - len(detected)
        rows.append(row)
    with (args.experiment_dir / "per_group_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_curves(summary, args.experiment_dir / "training_curves.png")
    best = torch.load(args.experiment_dir / "best_fusion.pt", map_location="cpu", weights_only=False)
    fusion = TrainableReferencePreservingFusion()
    fusion.load_state_dict(best["fusion_state"])
    fusion.eval()

    ranked = sorted(
        rows,
        key=lambda row: (
            row["single_TP"] - row["equal_mean_TP"],
            row["single_TP"] - row["fixed_selective_TP"],
            row["ground_truth"],
        ),
        reverse=True,
    )[: args.examples]
    gallery_index = []
    temporary = args.experiment_dir / "visual_examples" / "parts"
    temporary.mkdir(parents=True, exist_ok=True)
    groups_by_id = {group["group_id"]: group for group in groups}
    for row in ranked:
        group = groups_by_id[row["group_id"]]
        image_id = int(group["reference"]["image_id"])
        gt = annotations_to_eval_ground_truth(image_id, samples[image_id].annotations)
        paths: list[tuple[str, Path]] = [("reference", samples[image_id].image_path)]
        for source_index, source in enumerate(group["sources"], start=1):
            paths.append((f"source {source_index}", samples[int(source["image_id"])].image_path))
        for method in METHODS:
            part = temporary / f"{row['group_id'].replace('/', '_')}_{method}.jpg"
            panel(samples[image_id].image_path, gt, predictions[method].get(image_id, []), args.confidence, method, part)
            paths.append((method, part))

        cached = load_cached_group(args.experiment_dir / "feature_cache", group, torch.device("cpu"))
        with torch.inference_mode():
            _, diagnostics = fusion(cached["reference"], cached["sources"])
        heat = diagnostics["total_source_weight"].numpy()
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.imshow(Image.open(samples[image_id].image_path))
        axis.imshow(heat, cmap="magma", alpha=0.55, vmin=0.0, vmax=0.5, extent=(0, samples[image_id].width, samples[image_id].height, 0))
        axis.set_title("Learned total source weight (0 to 0.5)")
        axis.axis("off")
        heat_path = temporary / f"{row['group_id'].replace('/', '_')}_gate.jpg"
        figure.tight_layout()
        figure.savefig(heat_path, dpi=140)
        plt.close(figure)
        paths.append(("learned gate", heat_path))

        output = args.experiment_dir / "visual_examples" / f"{row['group_id'].replace('/', '_')}.jpg"
        compose_gallery(paths, output)
        gallery_index.append({"group_id": row["group_id"], "path": str(output), **row})
    (args.experiment_dir / "visual_examples" / "index.json").write_text(
        json.dumps(gallery_index, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
