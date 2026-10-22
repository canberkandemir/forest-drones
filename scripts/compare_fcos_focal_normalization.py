#!/usr/bin/env python3
"""Compare the controlled FCOS focal-normalization experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from forestpersons.detection import greedy_match, xywh_to_xyxy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "reference": ROOT
    / "outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600",
    "mean": ROOT
    / "outputs/dinov3_lora_focal_norm/train400_mean_seed20261014_from_lora5k_e2/eval_val600",
    "positive_count": ROOT
    / "outputs/dinov3_lora_focal_norm/train400_positive_seed20261014_from_lora5k_e2/eval_val600",
}
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/dinov3_lora_focal_norm/train400_mean_vs_positive_seed20261014_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data/annotations/val.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs/subsets/forestpersons_val_600_seed20261010_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def selected_row(summary: dict[str, Any], budget: float = 0.425) -> dict[str, Any]:
    return summary["custom_confidence_sweep"]["matched_false_positive_budgets"][str(budget)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_metric_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    selected = selected_row(summary)
    coco = summary["pycocotools"]
    row: dict[str, Any] = {
        "model": name,
        "AP": coco["AP"],
        "AP50": coco["AP50"],
        "AP75": coco["AP75"],
        "AR100": coco["AR@100"],
        "matched_fp_budget": 0.425,
        "confidence_threshold": selected["confidence_threshold"],
        "recall": selected["overall"]["recall"],
        "precision": selected["overall"]["precision"],
        "false_positives_per_image": selected["false_positives_per_image"],
        "true_positives": selected["overall"]["true_positives"],
        "false_positives": selected["overall"]["false_positives"],
    }
    for visibility in (20, 40, 70, 100):
        row[f"recall_visibility_{visibility}"] = selected["by_visibility"][str(visibility)][
            "recall"
        ]
    maximum = summary["custom_confidence_sweep"]["maximum_recall"]
    row["maximum_recall"] = maximum["overall"]["recall"]
    row["maximum_recall_fp_per_image"] = maximum["false_positives_per_image"]
    return row


def load_ground_truth(
    annotations_path: Path, manifest_path: Path
) -> tuple[dict[int, list[dict[str, Any]]], list[int]]:
    annotations = load_json(annotations_path)
    manifest = load_json(manifest_path)
    image_ids = [int(item["image_id"]) for item in manifest["images"]]
    selected = set(image_ids)
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in selected:
            continue
        attributes = annotation.get("attributes", {})
        visibility = attributes.get("visibility_level")
        if visibility is None:
            continue
        by_image[image_id].append(
            {
                "annotation_id": int(annotation["id"]),
                "box": xywh_to_xyxy(annotation["bbox"]),
                "visibility_level": int(visibility),
            }
        )
    return by_image, image_ids


def matched_annotation_ids(
    prediction_path: Path,
    ground_truth: dict[int, list[dict[str, Any]]],
    image_ids: list[int],
    threshold: float,
) -> set[int]:
    predictions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in load_json(prediction_path):
        if float(prediction["score"]) < threshold:
            continue
        predictions[int(prediction["image_id"])].append(
            {
                "box": xywh_to_xyxy(prediction["bbox"]),
                "confidence": float(prediction["score"]),
            }
        )
    matched: set[int] = set()
    for image_id in image_ids:
        image_gt = ground_truth.get(image_id, [])
        matches, _, _ = greedy_match(image_gt, predictions.get(image_id, []), 0.50)
        for match in matches:
            matched.add(int(image_gt[match["gt_index"]]["annotation_id"]))
    return matched


def visibility_by_annotation(ground_truth: dict[int, list[dict[str, Any]]]) -> dict[int, int]:
    return {
        int(gt["annotation_id"]): int(gt["visibility_level"])
        for image_ground_truth in ground_truth.values()
        for gt in image_ground_truth
    }


def plot_recall_curves(
    summaries: dict[str, dict[str, Any]], output_path: Path, visibility: int | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    labels = {
        "reference": "Original train5000 checkpoint",
        "mean": "Mean focal continuation",
        "positive_count": "Positive-count continuation",
    }
    for name, summary in summaries.items():
        rows = summary["custom_confidence_sweep"]["rows"]
        rows = sorted(rows, key=lambda row: row["false_positives_per_image"])
        x = [row["false_positives_per_image"] for row in rows]
        if visibility is None:
            y = [row["overall"]["recall"] for row in rows]
        else:
            y = [row["by_visibility"][str(visibility)]["recall"] for row in rows]
        ax.plot(x, y, marker=".", markersize=2, linewidth=1.5, label=labels[name])
    ax.axvline(0.425, color="black", linestyle="--", linewidth=1, label="FP/image budget 0.425")
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Recall" if visibility is None else f"Recall at visibility {visibility}%")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_score_distributions(summaries: dict[str, dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=False)
    labels = {"mean": "Mean focal", "positive_count": "Positive-count focal"}
    for axis, score_name in zip(axes, ("classification", "centerness", "combined")):
        for run_name in ("mean", "positive_count"):
            distribution = summaries[run_name]["timing"]["pre_nms_score_distributions"][score_name]
            edges = distribution["bin_edges"]
            counts = distribution["counts"]
            total = max(sum(counts), 1)
            centers = [(left + right) / 2 for left, right in zip(edges[:-1], edges[1:])]
            axis.step(
                centers,
                [count / total for count in counts],
                where="mid",
                linewidth=1.5,
                label=f"{labels[run_name]} (mean={distribution['mean']:.3f})",
            )
        axis.set_yscale("log")
        axis.set_xlabel(f"{score_name.capitalize()} probability")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Fraction of pre-NMS grid locations (log scale)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: load_json(path / "summary.json") for name, path in DEFAULT_RUNS.items()}

    metrics = [flatten_metric_row(name, summary) for name, summary in summaries.items()]
    write_csv(args.output_dir / "metrics_comparison.csv", metrics)

    budget_rows: list[dict[str, Any]] = []
    for name in ("mean", "positive_count"):
        for budget, row in summaries[name]["custom_confidence_sweep"][
            "matched_false_positive_budgets"
        ].items():
            budget_rows.append(
                {
                    "model": name,
                    "fp_budget": float(budget),
                    "confidence_threshold": row["confidence_threshold"],
                    "recall": row["overall"]["recall"],
                    "precision": row["overall"]["precision"],
                    "false_positives_per_image": row["false_positives_per_image"],
                    **{
                        f"recall_visibility_{visibility}": row["by_visibility"][str(visibility)][
                            "recall"
                        ]
                        for visibility in (20, 40, 70, 100)
                    },
                }
            )
    write_csv(args.output_dir / "fp_budget_comparison.csv", budget_rows)

    ground_truth, image_ids = load_ground_truth(args.annotations, args.manifest)
    matched: dict[str, set[int]] = {}
    for name, run_path in DEFAULT_RUNS.items():
        threshold = float(selected_row(summaries[name])["confidence_threshold"])
        matched[name] = matched_annotation_ids(
            run_path / "coco_results.json", ground_truth, image_ids, threshold
        )
    visibility = visibility_by_annotation(ground_truth)
    positive_only = matched["positive_count"] - matched["mean"]
    mean_only = matched["mean"] - matched["positive_count"]
    changes = {
        "matching": {
            "iou_threshold": 0.50,
            "fp_budget": 0.425,
            "note": "Each model uses its exploratory val600-selected threshold at or below the budget.",
        },
        "matched_counts": {name: len(ids) for name, ids in matched.items()},
        "positive_count_only_vs_mean": sorted(positive_only),
        "mean_only_vs_positive_count": sorted(mean_only),
        "positive_count_only_by_visibility": {
            str(level): sum(visibility[item] == level for item in positive_only)
            for level in (20, 40, 70, 100)
        },
        "mean_only_by_visibility": {
            str(level): sum(visibility[item] == level for item in mean_only)
            for level in (20, 40, 70, 100)
        },
    }
    with (args.output_dir / "matched_detection_changes.json").open("w") as handle:
        json.dump(changes, handle, indent=2)

    plot_recall_curves(summaries, args.output_dir / "recall_vs_fp_per_image.png")
    plot_recall_curves(summaries, args.output_dir / "recall20_vs_fp_per_image.png", visibility=20)
    plot_score_distributions(summaries, args.output_dir / "pre_nms_score_distributions.png")

    compact = {
        "inputs": {name: str(path.relative_to(ROOT)) for name, path in DEFAULT_RUNS.items()},
        "metrics": metrics,
        "matched_detection_changes": changes,
    }
    with (args.output_dir / "comparison_summary.json").open("w") as handle:
        json.dump(compact, handle, indent=2)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
