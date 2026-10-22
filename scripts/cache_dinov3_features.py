"""Extract and cache frozen DINOv3 patch features for a manifest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image
import torch

from forestpersons.dinov3_detector import (
    cache_path_for_file,
    choose_device,
    extract_patch_features,
    load_dinov3,
    save_feature_cache,
)


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
    device = choose_device(args.device)
    processor, model = load_dinov3(device)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    seconds_total = 0.0
    for index, item in enumerate(manifest, 1):
        output_path = cache_path_for_file(args.cache_root, item["file_name"])
        if output_path.exists() and not args.force:
            skipped += 1
            continue
        image = Image.open(args.image_root / item["file_name"]).convert("RGB")
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
        seconds_total += elapsed
        save_feature_cache(output_path, features, info)
        extracted += 1
        if extracted <= 5 or index % 25 == 0:
            print(f"{index:04d}/{len(manifest)} cached {elapsed:.3f}s {item['file_name']}")

    summary = {
        "manifest": str(args.manifest),
        "cache_root": str(args.cache_root),
        "short_side": args.short_side,
        "device": str(device),
        "images_in_manifest": len(manifest),
        "extracted": extracted,
        "skipped_existing": skipped,
        "seconds_total_new_extractions": seconds_total,
        "seconds_per_new_extraction": seconds_total / extracted if extracted else 0.0,
    }
    summary_path = args.cache_root / "cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved cache summary: {summary_path}")


if __name__ == "__main__":
    main()
