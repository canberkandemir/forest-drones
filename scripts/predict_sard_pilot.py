"""Run a frozen ForestPersons detector on the SARD mountain pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from PIL import Image

from scripts.predict_weitefeld_pilot import (
    FrozenPredictor,
    cross_tile_nms,
    tile_owns_center,
)
from forestpersons.weitefeld import image_tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-root", type=Path,
        default=Path("data/sard/official_v1.0/dataset/mountains/frames"),
    )
    parser.add_argument("--detector", choices=("yolo", "dinov3_lora_fcos"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--prediction-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--mode", choices=("full", "tiled"), default="full")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest["images"][: args.max_images]
    predictor = FrozenPredictor(args)
    predictions = []
    per_image = []
    started = time.perf_counter()
    for index, item in enumerate(images, start=1):
        with Image.open(args.image_root / item["file_name"]) as source:
            image = source.convert("RGB")
        image_started = time.perf_counter()
        if args.mode == "full":
            image_predictions = predictor.predict(image)
            tile_count = 1
        else:
            image_predictions = []
            tiles = image_tiles(
                image.width, image.height, tile_size=args.tile_size, overlap=args.tile_overlap
            )
            tile_count = len(tiles)
            for tile in tiles:
                x0, y0, x1, y1 = tile
                for prediction in predictor.predict(image.crop(tile)):
                    if not tile_owns_center(
                        prediction["bbox"], tile, image.width, image.height, args.tile_overlap
                    ):
                        continue
                    x_a, y_a, x_b, y_b = prediction["bbox"]
                    image_predictions.append(
                        {
                            **prediction,
                            "bbox": [x_a + x0, y_a + y0, x_b + x0, y_b + y0],
                        }
                    )
            image_predictions = cross_tile_nms(image_predictions, args.nms_iou)
        elapsed = time.perf_counter() - image_started
        for prediction in image_predictions:
            predictions.append(
                {
                    "image_id": item["image_id"],
                    "category_id": 1,
                    "bbox": prediction["bbox"],
                    "score": prediction["score"],
                    "source": "yolo" if args.detector == "yolo" else "dino",
                }
            )
        per_image.append(
            {
                "image_id": item["image_id"], "predictions": len(image_predictions),
                "tiles": tile_count, "seconds": elapsed,
            }
        )
        print(
            f"[{index}/{len(images)}] image={item['image_id']} "
            f"tiles={tile_count} predictions={len(image_predictions)} seconds={elapsed:.3f}",
            flush=True,
        )
    total = time.perf_counter() - started
    payload = {
        "settings": {
            "detector": args.detector,
            "checkpoint": str(args.checkpoint),
            "device": args.device,
            "imgsz": args.imgsz,
            "short_side": args.short_side,
            "prediction_floor": args.prediction_floor,
            "nms_iou": args.nms_iou,
            "mode": args.mode,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "pipeline": (
                "unchanged full-image ForestPersons detector preprocessing"
                if args.mode == "full" else
                "post-hoc equal-coverage tiling diagnostic; half-overlap center ownership and cross-tile NMS"
            ),
        },
        "images": len(images),
        "predictions": predictions,
        "per_image": per_image,
        "timing": {"seconds_total": total, "seconds_per_image": total / len(images)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(images), "predictions": len(predictions), **payload["timing"]}, indent=2))


if __name__ == "__main__":
    main()
