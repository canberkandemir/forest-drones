"""Run MMDetection inference on a ForestPersons COCO subset.

The script is intended for CUDA-side published-baseline smoke tests. It reads a
COCO annotation file exported by ``scripts/export_forestpersons_coco_subset.py``
and writes standard COCO detection results:

    [{"image_id": int, "category_id": 1, "bbox": [x, y, w, h], "score": float}]

For generic COCO-pretrained models, use ``--source-class-index 0`` because the
COCO person class is index 0 in MMDetection's class-ordered output. For
ForestPersons fine-tuned one-class checkpoints, keep the same default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument(
        "--source-class-index",
        type=int,
        default=0,
        help="Index in MMDetection result list to map to ForestPersons person category_id=1.",
    )
    return parser.parse_args()


def load_images(annotation_path: Path, max_images: int | None) -> list[dict[str, Any]]:
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = list(coco["images"])
    if max_images is not None:
        images = images[:max_images]
    return images


def result_to_array(result: Any, class_index: int) -> np.ndarray:
    """Extract an Nx5 xyxy+score array from MMDetection inference output.

    MMDetection 2.x returns a list of per-class arrays. MMDetection 3.x returns
    a DetDataSample with ``pred_instances``. The Colab workflow targets 2.x for
    architecture fidelity, but accepting both shapes makes accidental version
    mismatches easier to diagnose.
    """

    if isinstance(result, tuple):
        result = result[0]
    if hasattr(result, "pred_instances"):
        instances = result.pred_instances
        boxes = instances.bboxes.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        labels = instances.labels.detach().cpu().numpy()
        keep = labels == class_index
        if not np.any(keep):
            return np.zeros((0, 5), dtype=np.float32)
        return np.concatenate([boxes[keep], scores[keep, None]], axis=1).astype(np.float32)
    if isinstance(result, list):
        if class_index >= len(result):
            return np.zeros((0, 5), dtype=np.float32)
        boxes = result[class_index]
    else:
        boxes = result
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 5:
        raise ValueError(f"Expected Nx5 detection array, got shape {boxes.shape}")
    return boxes[:, :5]


def main() -> None:
    args = parse_args()
    from mmdet.apis import init_detector, inference_detector

    images = load_images(args.annotation, args.max_images)
    model = init_detector(str(args.config), str(args.checkpoint), device=args.device)
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for image in tqdm(images, desc="MMDetection inference"):
        image_path = args.image_root / image["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        result = inference_detector(model, str(image_path))
        boxes = result_to_array(result, args.source_class_index)
        for x1, y1, x2, y2, score in boxes.tolist():
            if score < args.score_thr:
                continue
            width = max(0.0, float(x2) - float(x1))
            height = max(0.0, float(y2) - float(y1))
            if width <= 0.0 or height <= 0.0:
                continue
            predictions.append(
                {
                    "image_id": image["id"],
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), width, height],
                    "score": float(score),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions), encoding="utf-8")
    elapsed = time.perf_counter() - started
    metadata = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "annotation": str(args.annotation),
        "image_root": str(args.image_root),
        "device": args.device,
        "images": len(images),
        "predictions": len(predictions),
        "score_threshold": args.score_thr,
        "source_class_index": args.source_class_index,
        "seconds_total": elapsed,
        "seconds_per_image": elapsed / len(images) if images else None,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
