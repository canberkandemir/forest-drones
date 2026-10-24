#!/usr/bin/env python3
"""Evaluate nested-subset Full MVR checkpoints with the canonical protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from forestpersons.evaluation import predictions_to_coco_results
from scripts.evaluate_dinov3_mvr_fcos import (
    collect_samples,
    evaluate_arm,
    flatten_sweep_rows,
    matched_budget_rows,
    write_csv,
)


def checkpoint_specs(values: list[str]) -> list[tuple[int, Path]]:
    specs = []
    for value in values:
        size_text, separator, path_text = value.partition("=")
        if not separator or not size_text.isdigit() or not path_text:
            raise ValueError(f"Expected SIZE=CHECKPOINT, got {value!r}")
        specs.append((int(size_text), Path(path_text)))
    if len({size for size, _ in specs}) != len(specs):
        raise ValueError("Duplicate training subset size")
    return sorted(specs)


def checkpoint_initialization(path: Path, expected_size: int) -> tuple[str, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != "mvr":
        raise ValueError(f"Checkpoint {path} is not the full MVR arm")
    actual_size = checkpoint.get("training_samples")
    if actual_size != expected_size:
        raise ValueError(
            f"Checkpoint {path} has training_samples={actual_size}, expected {expected_size}"
        )
    fcos_digest = checkpoint.get("fcos_initial_sha256")
    model_digest = checkpoint.get("model_initial_sha256")
    if not fcos_digest or not model_digest:
        raise ValueError(f"Checkpoint {path} lacks matched initialization metadata")
    return str(fcos_digest), str(model_digest)


def report(results: list[dict], matched: list[dict]) -> str:
    lines = [
        "# DINOv3 Full MVR label-efficiency study on ForestPersons val600",
        "",
        "All checkpoints use the canonical val600 protocol: IoU 0.50 one-to-one "
        "matching, confidence floor 0.001, NMS IoU 0.50, top-300, and the same "
        "predeclared confidence sweep. Matched-budget thresholds are exploratory "
        "because they are selected on val600.",
        "",
        "| Training images | AP | AP50 | AP75 | AR@100 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['training_images']} | {result['AP']:.4f} | "
            f"{result['AP50']:.4f} | {result['AP75']:.4f} | {result['AR100']:.4f} |"
        )
    lines += [
        "",
        "| Training images | Budget | Threshold | Recall | R20 | R40 | R70 | R100 | FP/image |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matched:
        values = [row.get(key) for key in ("confidence_threshold", "recall", "R20", "R40", "R70", "R100", "false_positives_per_image")]
        if any(value is None for value in values):
            lines.append(f"| {row['training_images']} | {row['fp_per_image_budget']:.3f} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {row['training_images']} | {row['fp_per_image_budget']:.3f} | "
                f"{row['confidence_threshold']:.3f} | {row['recall']:.4f} | "
                f"{row['R20']:.4f} | {row['R40']:.4f} | {row['R70']:.4f} | "
                f"{row['R100']:.4f} | {row['false_positives_per_image']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final-cache-root", type=Path, required=True)
    parser.add_argument("--mvr-cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="SIZE=CHECKPOINT")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    samples = collect_samples(args.root, args.manifest)
    results = []
    specs = checkpoint_specs(args.checkpoint)
    initialization_digests = {
        checkpoint_initialization(checkpoint, size) for size, checkpoint in specs
    }
    if len(initialization_digests) != 1:
        raise ValueError("Label-efficiency checkpoints do not share initialization")
    for size, checkpoint in specs:
        metrics, predictions = evaluate_arm(
            "mvr", checkpoint, samples, args.final_cache_root, args.mvr_cache_root, device
        )
        metrics["arm"] = f"train{size}"
        metrics["training_images"] = size
        results.append(metrics)
        (args.output_dir / f"train{size}_predictions_coco.json").write_text(
            json.dumps(predictions_to_coco_results(predictions), indent=2) + "\n"
        )
    sweep = flatten_sweep_rows(results)
    matched = matched_budget_rows(results)
    size_by_arm = {f"train{result['training_images']}": result["training_images"] for result in results}
    for row in (*sweep, *matched):
        row["training_images"] = size_by_arm[row["arm"]]
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    write_csv(args.output_dir / "confidence_sweep.csv", sweep)
    write_csv(args.output_dir / "matched_fp_budgets.csv", matched)
    (args.output_dir / "comparison_report.md").write_text(report(results, matched), encoding="utf-8")
    print(json.dumps(matched, indent=2))


if __name__ == "__main__":
    main()
