"""Summarize YOLO11 fine-tuning evaluation sweeps and compare against baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


VISIBILITY_LEVELS = (20, 40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dinov3-comparison",
        type=Path,
        default=Path("outputs/dinov3_detector/head_v3_400_vs_5000_yolo11s_comparison.csv"),
    )
    parser.add_argument("--target-fp-per-image", type=float, default=0.425)
    return parser.parse_args()


def pct(value: float) -> float:
    return 100.0 * value


def row_from_metrics(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    overall = metrics["overall"]
    settings = metrics["settings"]
    row: dict[str, Any] = {
        "model": "YOLO11s fine-tuned",
        "family": "yolo11s_finetuned_train5000",
        "train_images": 5000,
        "confidence": float(settings["conf_threshold"]),
        "ground_truth": int(overall["ground_truth"]),
        "true_positives": int(overall["true_positives"]),
        "false_positives": int(overall["false_positives"]),
        "false_negatives": int(overall["false_negatives"]),
        "recall": float(overall["recall"]),
        "precision": float(overall["precision"]),
        "fp_per_image": float(metrics["false_positives_per_image"]),
        "seconds_per_image": float(metrics["inference"]["seconds_per_image"]),
        "metrics_path": str(path),
    }
    for visibility in VISIBILITY_LEVELS:
        group = metrics["by_visibility"].get(str(visibility), {})
        row[f"recall_visibility_{visibility}"] = float(group.get("recall", 0.0))
        row[f"gt_visibility_{visibility}"] = int(group.get("ground_truth", 0))
    return row


def read_finetuned_rows(grid_root: Path) -> list[dict[str, Any]]:
    rows = [row_from_metrics(path) for path in sorted(grid_root.glob("*/metrics.json"))]
    rows.sort(key=lambda row: row["confidence"])
    return rows


def read_comparison_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            converted: dict[str, Any] = dict(row)
            for key in (
                "train_images",
                "ground_truth",
                "true_positives",
                "false_positives",
                "false_negatives",
            ):
                converted[key] = int(float(converted[key]))
            for key in (
                "confidence",
                "recall",
                "precision",
                "fp_per_image",
                "recall_visibility_20",
                "recall_visibility_40",
                "recall_visibility_70",
                "recall_visibility_100",
                "seconds_per_image",
            ):
                converted[key] = float(converted[key])
            rows.append(converted)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "family",
        "train_images",
        "confidence",
        "ground_truth",
        "true_positives",
        "false_positives",
        "false_negatives",
        "recall",
        "precision",
        "fp_per_image",
        "recall_visibility_20",
        "recall_visibility_40",
        "recall_visibility_70",
        "recall_visibility_100",
        "seconds_per_image",
        "metrics_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_curve(
    rows_by_family: dict[str, list[dict[str, Any]]],
    y_key: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, rows in rows_by_family.items():
        ordered = sorted(rows, key=lambda row: row["fp_per_image"])
        ax.plot(
            [row["fp_per_image"] for row in ordered],
            [pct(row[y_key]) for row in ordered],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=ordered[0]["model"] if ordered else family,
        )
    ax.set_xlabel("False positives per image")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pr(rows_by_family: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, rows in rows_by_family.items():
        ordered = sorted(rows, key=lambda row: row["recall"])
        ax.plot(
            [pct(row["recall"]) for row in ordered],
            [pct(row["precision"]) for row in ordered],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=ordered[0]["model"] if ordered else family,
        )
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    finetuned_rows = read_finetuned_rows(args.grid_root)
    baseline_rows = read_comparison_rows(args.dinov3_comparison)
    comparison_rows = baseline_rows + finetuned_rows

    rows_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in comparison_rows:
        rows_by_family.setdefault(str(row["family"]), []).append(row)

    matched = min(
        finetuned_rows,
        key=lambda row: abs(row["fp_per_image"] - args.target_fp_per_image),
    )
    matched_rows = [
        row
        for row in comparison_rows
        if (
            row["family"] in {"yolo11s_pretrained", "dino_v3_5000"}
            and abs(row["fp_per_image"] - args.target_fp_per_image) <= 0.08
        )
    ]
    matched_rows.append(matched)
    matched_rows.sort(key=lambda row: (row["family"], row["fp_per_image"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "yolo11s_finetuned_threshold_summary.csv", finetuned_rows)
    write_csv(args.output_dir / "model_comparison_all_thresholds.csv", comparison_rows)
    write_csv(args.output_dir / "model_comparison_matched_fp.csv", matched_rows)

    plots_dir = args.output_dir / "plots"
    plot_curve(rows_by_family, "recall", "Recall (%)", plots_dir / "recall_vs_fp.png")
    plot_curve(
        rows_by_family,
        "recall_visibility_20",
        "Recall at 20% visibility (%)",
        plots_dir / "recall20_vs_fp.png",
    )
    plot_pr(rows_by_family, plots_dir / "precision_recall.png")

    print(json.dumps({"matched_finetuned_row": matched}, indent=2))
    print(f"Saved summary: {args.output_dir / 'yolo11s_finetuned_threshold_summary.csv'}")
    print(f"Saved comparison: {args.output_dir / 'model_comparison_all_thresholds.csv'}")
    print(f"Saved matched FP comparison: {args.output_dir / 'model_comparison_matched_fp.csv'}")
    print(f"Saved plots: {plots_dir}")


if __name__ == "__main__":
    main()
