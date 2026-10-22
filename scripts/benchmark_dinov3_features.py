"""Benchmark frozen DINOv3 feature extraction on a small ForestPersons subset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from forestpersons.dinov3_detector import choose_device, extract_patch_features, load_dinov3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dinov3_detector/benchmark.json"))
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def mps_memory() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": int(torch.mps.current_allocated_memory()),
        "driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["images"][: args.limit]
    device = choose_device(args.device)
    processor, model = load_dinov3(device)

    rows = []
    total_seconds = 0.0
    for item in manifest:
        image_path = args.image_root / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        start = time.perf_counter()
        features, info = extract_patch_features(
            image,
            processor=processor,
            model=model,
            device=device,
            short_side=args.short_side,
        )
        synchronize_device(device)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        rows.append(
            {
                "file_name": item["file_name"],
                "seconds": elapsed,
                "original_size": [info.original_width, info.original_height],
                "resized_size": [info.resized_width, info.resized_height],
                "grid_size": [info.grid_width, info.grid_height],
                "feature_shape": list(features.shape),
            }
        )
        print(f"{len(rows):03d}/{len(manifest)} {elapsed:.3f}s {item['file_name']}")

    result = {
        "settings": {
            "manifest": str(args.manifest),
            "image_root": str(args.image_root),
            "short_side": args.short_side,
            "limit": len(manifest),
            "device": str(device),
        },
        "images": len(rows),
        "seconds_total": total_seconds,
        "seconds_per_image": total_seconds / len(rows) if rows else 0.0,
        "mps_memory_bytes": mps_memory(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved benchmark: {args.output}")


if __name__ == "__main__":
    main()
