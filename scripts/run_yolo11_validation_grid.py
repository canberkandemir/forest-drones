"""Run YOLO11 validation evaluations over models and confidence thresholds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["models/yolo11n.pt", "models/yolo11s.pt"],
    )
    parser.add_argument(
        "--conf-thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25],
    )
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/yolo11_validation_grid"))
    parser.add_argument("--max-visualizations", type=int, default=24)
    return parser.parse_args()


def slug_model(model_path: str) -> str:
    return Path(model_path).stem


def slug_conf(confidence: float) -> str:
    return f"conf{int(round(confidence * 100)):03d}"


def main() -> None:
    args = parse_args()
    for model in args.models:
        for confidence in args.conf_thresholds:
            output_dir = (
                args.output_root
                / f"{slug_model(model)}_{slug_conf(confidence)}_iou{int(args.iou * 100):03d}_img{args.imgsz}"
            )
            command = [
                sys.executable,
                "scripts/evaluate_yolo11_forestpersons.py",
                "--root",
                str(args.root),
                "--split",
                args.split,
                "--manifest",
                str(args.manifest),
                "--model",
                model,
                "--conf",
                str(confidence),
                "--iou",
                str(args.iou),
                "--imgsz",
                str(args.imgsz),
                "--device",
                args.device,
                "--output-dir",
                str(output_dir),
                "--max-visualizations",
                str(args.max_visualizations),
            ]
            print("Running:", " ".join(command))
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "run.log"
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
