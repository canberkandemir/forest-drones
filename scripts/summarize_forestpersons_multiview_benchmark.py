"""Create compact tables and figures from a completed multi-view benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preparation-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    preparation = json.loads(args.preparation_summary.read_text(encoding="utf-8"))
    methods = {"single_view": summary["single_view"], "multi_view": summary["multi_view"]}
    rows = []
    for name, result in methods.items():
        fixed = result["fixed_threshold"]
        coco = result["pycocotools"]
        rows.append({
            "method": name,
            "confidence": fixed["confidence_threshold"],
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
    with (args.output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["Overall", "20", "40", "70", "100"]
    single = [rows[0]["recall"], rows[0]["R20"], rows[0]["R40"], rows[0]["R70"], rows[0]["R100"]]
    fused = [rows[1]["recall"], rows[1]["R20"], rows[1]["R40"], rows[1]["R70"], rows[1]["R100"]]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar([value - 0.18 for value in x], single, width=0.36, label="Single view")
    ax.bar([value + 0.18 for value in x], fused, width=0.36, label="Aligned feature mean")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Recall at IoU 0.50, confidence 0.55")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "recall_by_visibility.png", dpi=180)
    plt.close(fig)

    reasons = preparation["source_rejection_reasons"]
    sorted_reasons = sorted(reasons.items(), key=lambda item: item[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([item[0].replace("_", " ") for item in reversed(sorted_reasons)], [item[1] for item in reversed(sorted_reasons)])
    ax.set_xlabel("Rejected source-frame registration attempts")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "registration_rejection_reasons.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
