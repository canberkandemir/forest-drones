"""Build comparison tables and recall/FP plots for the FCOS quality pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_at_threshold(summary: dict[str, Any], threshold: float) -> dict[str, Any]:
    return min(
        summary["custom_confidence_sweep"]["rows"],
        key=lambda row: abs(float(row["confidence_threshold"]) - threshold),
    )


def flatten(name: str, summary: dict[str, Any], row: dict[str, Any], selection: str) -> dict[str, Any]:
    coco = summary["pycocotools"]
    vis = row["by_visibility"]
    return {
        "model": name,
        "selection": selection,
        "AP": coco["AP"],
        "AP50": coco["AP50"],
        "AP75": coco["AP75"],
        "AR@100": coco["AR@100"],
        "threshold": row["confidence_threshold"],
        "recall": row["overall"]["recall"],
        "precision": row["overall"]["precision"],
        "FP_per_image": row["false_positives_per_image"],
        "R20": vis["20"]["recall"],
        "R40": vis["40"]["recall"],
        "R70": vis["70"]["recall"],
        "R100": vis["100"]["recall"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/dinov3_lora_fcos_quality"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("control", args.root / "val600_control/summary.json", 0.515),
        ("quality_head_original_score", args.root / "val600_quality_ablation/summary.json", 0.515),
        ("quality_head_quality_power_0.25", args.root / "val600_quality_aware/summary.json", 0.480),
    ]
    summaries = [(name, load(path), locked) for name, path, locked in specs]
    comparison = []
    sweep_rows = []
    for name, summary, locked in summaries:
        comparison.append(flatten(name, summary, row_at_threshold(summary, locked), "calibration_locked"))
        matched = summary["custom_confidence_sweep"]["matched_false_positive_budgets"]["0.425"]
        comparison.append(flatten(name, summary, matched, "exploratory_val600_FP<=0.425"))
        for row in summary["custom_confidence_sweep"]["rows"]:
            sweep_rows.append(
                {
                    "model": name,
                    "threshold": row["confidence_threshold"],
                    "recall": row["overall"]["recall"],
                    "precision": row["overall"]["precision"],
                    "FP_per_image": row["false_positives_per_image"],
                    "R20": row["by_visibility"]["20"]["recall"],
                    "R40": row["by_visibility"]["40"]["recall"],
                    "R70": row["by_visibility"]["70"]["recall"],
                    "R100": row["by_visibility"]["100"]["recall"],
                }
            )
    write_csv(args.output_dir / "val600_comparison.csv", comparison)
    write_csv(args.output_dir / "recall_fp_sweeps.csv", sweep_rows)

    for metric, filename, ylabel in (
        ("recall", "recall_vs_fp_per_image.png", "Overall recall"),
        ("R20", "r20_vs_fp_per_image.png", "Recall at 20% visibility"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for name, _, _ in summaries:
            rows = sorted(
                (row for row in sweep_rows if row["model"] == name),
                key=lambda row: row["FP_per_image"],
            )
            rows = [row for row in rows if row["FP_per_image"] <= 2.0]
            ax.plot([row["FP_per_image"] for row in rows], [row[metric] for row in rows], label=name)
        ax.axvline(0.425, color="black", linestyle="--", linewidth=1, label="FP/image budget 0.425")
        ax.set_xlabel("False positives per image")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=170)
        plt.close(fig)

    calibration_rows = []
    for power in ("0", "0.25", "0.5", "1.0"):
        summary = load(args.root / f"calibration100_quality_power_{power}/summary.json")
        matched = summary["custom_confidence_sweep"]["matched_false_positive_budgets"]["0.425"]
        calibration_rows.append(
            {
                "quality_power": power,
                "threshold": matched["confidence_threshold"],
                "recall": matched["overall"]["recall"],
                "precision": matched["overall"]["precision"],
                "FP_per_image": matched["false_positives_per_image"],
                "R20": matched["by_visibility"]["20"]["recall"],
                "AP": summary["pycocotools"]["AP"],
            }
        )
    write_csv(args.output_dir / "calibration100_score_selection.csv", calibration_rows)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
