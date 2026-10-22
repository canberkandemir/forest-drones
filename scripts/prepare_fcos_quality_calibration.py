"""Create a deterministic calibration holdout outside ForestPersons train400."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def load_images(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["images"] if isinstance(data, dict) else data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train5000", type=Path, required=True)
    parser.add_argument("--train400", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()

    train5000 = load_images(args.train5000)
    excluded_ids = {item["image_id"] for item in load_images(args.train400)}
    candidates = [item for item in train5000 if item["image_id"] not in excluded_ids]
    if len(candidates) < args.images:
        raise ValueError(f"Only {len(candidates)} calibration candidates remain")
    selected = random.Random(args.seed).sample(candidates, args.images)
    selected.sort(key=lambda item: int(item["image_id"]))
    keys = [f"{item['image_id']}\t{item['file_name']}" for item in selected]
    visibility_counts: dict[str, int] = {}
    for item in selected:
        for visibility, count in item.get("visibility_counts", {}).items():
            visibility_counts[str(visibility)] = visibility_counts.get(str(visibility), 0) + int(count)

    payload = {
        "source_manifest": str(args.train5000),
        "excluded_training_manifest": str(args.train400),
        "selection": {
            "purpose": "FCOS quality-score calibration only; never used for gradient updates",
            "method": "uniform random sample without replacement from train5000 minus train400",
            "seed": args.seed,
            "requested_images": args.images,
            "candidate_images": len(candidates),
            "sample_key_sha256": hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest(),
        },
        "summary": {
            "images": len(selected),
            "annotations": sum(int(item.get("annotation_count", 0)) for item in selected),
            "visibility_counts": visibility_counts,
        },
        "images": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["selection"] | payload["summary"], indent=2))


if __name__ == "__main__":
    main()
