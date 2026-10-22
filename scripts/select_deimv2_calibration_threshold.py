"""Lock a DEIMv2 confidence threshold from a train-derived calibration summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--fp-budget", type=float, default=0.425)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-threshold", action="store_true")
    return parser.parse_args()


def select_threshold(summary: dict, fp_budget: float) -> dict:
    settings = summary.get("settings", {})
    if settings.get("split") != "train":
        raise ValueError("Operating thresholds must be selected from a training-derived calibration split")
    key = str(fp_budget)
    selected = summary["custom_sweep"]["matched_false_positive_budgets"].get(key)
    if selected is None:
        raise ValueError(f"No calibration operating point satisfies FP/image <= {fp_budget}")
    return {
        "selection_source": "train-derived calibration subset",
        "source_summary": settings.get("manifest"),
        "fp_budget": fp_budget,
        "confidence_threshold": float(selected["confidence_threshold"]),
        "calibration_metrics": selected,
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = select_threshold(summary, args.fp_budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.print_threshold:
        print(result["confidence_threshold"])
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
