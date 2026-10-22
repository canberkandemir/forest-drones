"""Create compact tables and figures for selective benchmark49 fusion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.experiment_dir / "summary.json").read_text(encoding="utf-8"))
    figures = args.experiment_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = ("single_view", "equal_mean", "selective")
    names = {"single_view": "A single", "equal_mean": "B equal mean", "selective": "C selective"}
    rows = []
    for label in labels:
        item = summary[label]
        fixed = item["fixed_threshold"]
        coco = item["pycocotools"]
        rows.append({
            "method": names[label],
            "TP": fixed["overall"]["true_positives"],
            "FP": fixed["overall"]["false_positives"],
            "FN": fixed["overall"]["false_negatives"],
            "recall": fixed["overall"]["recall"],
            "precision": fixed["overall"]["precision"],
            "FP_per_image": fixed["false_positives_per_image"],
            "R20": fixed["by_visibility"]["20"]["recall"],
            "R40": fixed["by_visibility"]["40"]["recall"],
            "R70": fixed["by_visibility"]["70"]["recall"],
            "R100": fixed["by_visibility"]["100"]["recall"],
            "AP": coco["AP"],
            "AP50": coco["AP50"],
            "AP75": coco["AP75"],
            "AR100": coco["AR@100"],
        })
    with (figures / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(5)
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 4.8))
    for index, (label, row) in enumerate(zip(labels, rows, strict=True)):
        values = [row["recall"], row["R20"], row["R40"], row["R70"], row["R100"]]
        axis.bar(x + (index - 1) * width, values, width, label=names[label])
    axis.set_xticks(x, ["Overall", "20%", "40%", "70%", "100%"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Recall at confidence 0.55")
    axis.set_title("ForestPersons benchmark49: fixed operating point")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "recall_by_visibility.png", dpi=170)
    plt.close(fig)

    with (args.experiment_dir / "per_gt_diagnostics.csv").open(encoding="utf-8") as handle:
        per_gt = list(csv.DictReader(handle))
    weights = np.asarray([float(row["posthoc_gate_weight_at_gt_center"]) for row in per_gt])
    changes = np.asarray([float(row["selective_best_iou"]) - float(row["single_best_iou"]) for row in per_gt])
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.scatter(weights, changes, s=25, alpha=0.7)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Post-hoc source weight at GT center")
    axis.set_ylabel("Selective minus single best IoU")
    axis.set_title("Gate activity does not guarantee target correspondence")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "gt_center_gate_vs_iou_change.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
