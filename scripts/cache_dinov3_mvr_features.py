#!/usr/bin/env python3
"""Cache frozen DINOv3 intermediate patch layers for the MVR readout."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from forestpersons.dinov3_detector import (
    cache_path_for_file,
    choose_device,
    extract_intermediate_patch_features,
    load_dinov3,
    load_feature_cache,
    save_mvr_feature_cache,
)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--final-cache-root", type=Path, required=True)
    parser.add_argument("--mvr-cache-root", type=Path, required=True)
    parser.add_argument("--layers", default="3,6,9")
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    layers = tuple(int(value) for value in args.layers.split(",") if value.strip())
    if layers != (3, 6, 9):
        raise ValueError("The controlled experiment requires --layers 3,6,9")
    items = json.loads(args.manifest.read_text(encoding="utf-8"))["images"]
    device = choose_device(args.device)
    processor, model = load_dinov3(device)
    args.mvr_cache_root.mkdir(parents=True, exist_ok=True)
    extracted = skipped = 0
    seconds = 0.0
    for index, item in enumerate(items, start=1):
        final_path = cache_path_for_file(args.final_cache_root, item["file_name"])
        output_path = cache_path_for_file(args.mvr_cache_root, item["file_name"])
        if not final_path.is_file():
            raise FileNotFoundError(f"Canonical final-layer cache is missing: {final_path}")
        if output_path.is_file() and not args.force:
            skipped += 1
            continue
        image_path = args.image_root / item["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        started = time.perf_counter()
        features, info = extract_intermediate_patch_features(
            image,
            processor,
            model,
            device,
            short_side=args.short_side,
            layer_indices=layers,
        )
        synchronize(device)
        final_features, final_info = load_feature_cache(final_path)
        if info != final_info or tuple(features.shape[1:3]) != tuple(final_features.shape[:2]):
            raise ValueError(f"Intermediate/final cache geometry mismatch for {item['file_name']}")
        save_mvr_feature_cache(output_path, features, info, layers)
        extracted += 1
        seconds += time.perf_counter() - started
        if extracted <= 5 or index % 50 == 0:
            print(f"{index:04d}/{len(items)} cached {item['file_name']}", flush=True)
    summary = {
        "manifest": str(args.manifest.resolve()),
        "image_root": str(args.image_root.resolve()),
        "final_cache_root": str(args.final_cache_root.resolve()),
        "mvr_cache_root": str(args.mvr_cache_root.resolve()),
        "layers": list(layers),
        "short_side": args.short_side,
        "device": str(device),
        "images": len(items),
        "extracted": extracted,
        "skipped": skipped,
        "seconds_new_extractions": seconds,
        "seconds_per_new_extraction": seconds / extracted if extracted else 0.0,
    }
    (args.mvr_cache_root / "cache_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
