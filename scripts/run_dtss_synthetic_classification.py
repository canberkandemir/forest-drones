"""Run the controlled fixed-T4 versus learned-DTSS synthetic experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from forestpersons.dtss_experiment import SyntheticExperimentConfig, run_controlled_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    config = SyntheticExperimentConfig(epochs=args.epochs)
    result = run_controlled_experiment(config, torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        arm["name"]: {
            "accuracy": arm["final"]["test_accuracy"],
            "loss": arm["final"]["test_loss"],
            "active_timesteps": arm["final_active_timesteps"],
            "runtime_seconds": arm["runtime_seconds"],
        }
        for arm in result["arms"]
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

