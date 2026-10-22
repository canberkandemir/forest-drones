"""Validate and prepare the deterministic SARD v1.0 mountain pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from forestpersons.sard import build_sard_mountains_manifest, manifest_to_coco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/sard/official_v1.0/dataset/mountains"),
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-id-start", type=int)
    parser.add_argument("--image-id-end", type=int)
    return parser.parse_args()


def save_samples(manifest: dict, dataset_root: Path, output_dir: Path) -> None:
    positive = [item for item in manifest["images"] if item["person_annotations"]]
    # Spread examples across the sequential source video instead of selecting by box quality.
    positions = [0, len(positive) // 4, len(positive) // 2, 3 * len(positive) // 4, len(positive) - 1]
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in [positive[index] for index in positions]:
        with Image.open(dataset_root / "frames" / item["file_name"]) as source:
            canvas = source.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for annotation in item["person_annotations"]:
            draw.rectangle(annotation["bbox_xyxy"], outline="lime", width=8)
        canvas.thumbnail((1600, 1000))
        canvas.save(output_dir / item["file_name"], quality=90)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = build_sard_mountains_manifest(
        args.dataset_root, source_revision=args.source_revision
    )
    if args.image_id_start is not None or args.image_id_end is not None:
        if args.image_id_start is None or args.image_id_end is None:
            raise ValueError("Both --image-id-start and --image-id-end are required")
        manifest["images"] = [
            item for item in manifest["images"]
            if args.image_id_start <= int(item["image_id"]) <= args.image_id_end
        ]
        manifest["selection"] = (
            f"post-hoc tiling diagnostic, contiguous frames "
            f"{args.image_id_start}-{args.image_id_end}; not a primary benchmark"
        )
        manifest["summary"] = {
            "images": len(manifest["images"]),
            "positive_images": sum(bool(item["person_annotations"]) for item in manifest["images"]),
            "negative_images": sum(not item["person_annotations"] for item in manifest["images"]),
            "person_annotations": sum(len(item["person_annotations"]) for item in manifest["images"]),
        }
    (args.output_dir / "pilot200_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output_dir / "pilot200_coco.json").write_text(
        json.dumps(manifest_to_coco(manifest), indent=2), encoding="utf-8"
    )
    save_samples(manifest, args.dataset_root, args.output_dir / "annotation_samples")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
