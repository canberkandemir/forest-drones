"""Run a frozen ForestPersons detector on the Weitefeld pilot, full or tiled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision.ops import nms

from forestpersons.weitefeld import image_tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/weitefeld/images/pilot200"))
    parser.add_argument("--detector", choices=("yolo", "dinov3_lora_fcos"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "tiled"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--prediction-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument("--tile-overlap", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def cross_tile_nms(predictions: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    if not predictions:
        return []
    boxes = torch.tensor([item["bbox"] for item in predictions], dtype=torch.float32)
    scores = torch.tensor([item["score"] for item in predictions], dtype=torch.float32)
    keep = nms(boxes, scores, threshold).tolist()
    return [predictions[index] for index in keep]


def tile_owns_center(
    box: list[float], tile: tuple[int, int, int, int], width: int, height: int, overlap: int
) -> bool:
    x0, y0, x1, y1 = tile
    center_x = x0 + (box[0] + box[2]) / 2.0
    center_y = y0 + (box[1] + box[3]) / 2.0
    half = overlap / 2.0
    owner_x0 = x0 if x0 == 0 else x0 + half
    owner_y0 = y0 if y0 == 0 else y0 + half
    owner_x1 = x1 if x1 == width else x1 - half
    owner_y1 = y1 if y1 == height else y1 - half
    return owner_x0 <= center_x <= owner_x1 and owner_y0 <= center_y <= owner_y1


class FrozenPredictor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.detector == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(args.checkpoint)
            self.processor = self.backbone = self.head = self.metadata = None
        else:
            from forestpersons.dinov3_detector import choose_device
            from forestpersons.dinov3_lora import load_lora_fcos_checkpoint
            self.device = choose_device(args.device)
            self.processor, self.backbone, self.head, self.metadata = load_lora_fcos_checkpoint(
                args.checkpoint, self.device
            )
            if self.metadata.get("head_kind", "fcos") != "fcos":
                raise ValueError("Expected canonical single-scale FCOS checkpoint")
            self.backbone.eval()
            self.head.eval()

    def predict(self, image: Image.Image) -> list[dict[str, Any]]:
        if self.args.detector == "yolo":
            result = self.model.predict(
                source=np.asarray(image), conf=self.args.prediction_floor,
                iou=self.args.nms_iou, imgsz=self.args.imgsz,
                device=self.args.device, verbose=False,
            )[0]
            output = []
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:
                        output.append({
                            "bbox": [float(value) for value in box.xyxy[0].cpu().tolist()],
                            "score": float(box.conf[0]),
                        })
            return output

        from forestpersons.dinov3_detector import decode_fcos_predictions
        from forestpersons.dinov3_lora import extract_lora_patch_features
        with torch.inference_mode():
            features, info = extract_lora_patch_features(
                image, processor=self.processor, model=self.backbone,
                device=self.device, short_side=self.args.short_side,
            )
            logits, raw_box, centerness = self.head(features)
            decoded = decode_fcos_predictions(
                logits.detach().cpu(), raw_box.detach().cpu(), centerness.detach().cpu(), info,
                conf_threshold=self.args.prediction_floor, nms_threshold=self.args.nms_iou,
            )
        return [
            {"bbox": [float(value) for value in item["box"]], "score": float(item["confidence"])}
            for item in decoded
        ]


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest["images"][: args.max_images]
    predictor = FrozenPredictor(args)
    records = []
    per_image = []
    started = time.perf_counter()
    for index, item in enumerate(images, 1):
        path = args.image_root / f"strip{item['strip']}" / item["file_name"]
        with Image.open(path) as source:
            image = source.convert("RGB")
        image_start = time.perf_counter()
        if args.mode == "full":
            predictions = predictor.predict(image)
            tile_count = 1
        else:
            predictions = []
            tiles = image_tiles(
                image.width, image.height, tile_size=args.tile_size, overlap=args.tile_overlap
            )
            tile_count = len(tiles)
            for tile in tiles:
                x0, y0, x1, y1 = tile
                tile_predictions = predictor.predict(image.crop(tile))
                for prediction in tile_predictions:
                    if not tile_owns_center(
                        prediction["bbox"], tile, image.width, image.height, args.tile_overlap
                    ):
                        continue
                    x_a, y_a, x_b, y_b = prediction["bbox"]
                    predictions.append({
                        **prediction,
                        "bbox": [x_a + x0, y_a + y0, x_b + x0, y_b + y0],
                    })
            predictions = cross_tile_nms(predictions, args.nms_iou)
        elapsed = time.perf_counter() - image_start
        for prediction in predictions:
            records.append({
                "image_id": item["image_id"], "category_id": 1,
                "bbox": prediction["bbox"], "score": prediction["score"],
                "source": "yolo" if args.detector == "yolo" else "dino",
            })
        per_image.append({
            "image_id": item["image_id"], "predictions": len(predictions),
            "tiles": tile_count, "seconds": elapsed,
        })
        print(
            f"[{index}/{len(images)}] image={item['image_id']} tiles={tile_count} "
            f"predictions={len(predictions)} seconds={elapsed:.3f}", flush=True
        )

    elapsed = time.perf_counter() - started
    output = {
        "settings": {
            "detector": args.detector, "checkpoint": str(args.checkpoint),
            "mode": args.mode, "device": args.device, "imgsz": args.imgsz,
            "short_side": args.short_side, "prediction_floor": args.prediction_floor,
            "nms_iou": args.nms_iou, "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "tile_ownership": "prediction center in non-overlapping half-overlap ownership region",
        },
        "images": len(images), "predictions": records, "per_image": per_image,
        "timing": {"seconds_total": elapsed, "seconds_per_image": elapsed / len(images)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(images), "predictions": len(records), **output["timing"]}, indent=2))


if __name__ == "__main__":
    main()

