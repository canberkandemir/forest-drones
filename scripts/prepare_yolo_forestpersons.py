"""Prepare ForestPersons subsets in Ultralytics YOLO format.

The conversion maps all ForestPersons pose/visibility categories to one class:
``person``. Visibility and pose are preserved in sidecar metadata files for
custom evaluation, but YOLO training sees a single detection class.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from forestpersons import ForestPersonsDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/yolo11_finetune/forestpersons_yolo_dataset"),
    )
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest_files(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["images"] if isinstance(data, dict) and "images" in data else data
    files = [str(item["file_name"]) for item in items]
    if len(files) != len(set(files)):
        duplicates = [name for name, count in Counter(files).items() if count > 1]
        raise ValueError(f"Manifest {path} contains duplicate files, e.g. {duplicates[:3]}")
    return files


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Convert COCO xywh pixels to clipped YOLO normalized xywh."""

    x, y, width, height = [float(value) for value in bbox]
    x1 = max(0.0, min(x, float(image_width)))
    y1 = max(0.0, min(y, float(image_height)))
    x2 = max(0.0, min(x + width, float(image_width)))
    y2 = max(0.0, min(y + height, float(image_height)))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    x_center = (x1 + 0.5 * clipped_width) / image_width
    y_center = (y1 + 0.5 * clipped_height) / image_height
    norm_width = clipped_width / image_width
    norm_height = clipped_height / image_height
    values = (x_center, y_center, norm_width, norm_height)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"YOLO bbox normalization escaped [0, 1]: {values}")
    return values


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_files = load_manifest_files(args.train_manifest)
    val_files = load_manifest_files(args.val_manifest)
    overlap = sorted(set(train_files) & set(val_files))
    if overlap:
        raise ValueError(f"Train/val manifest overlap: {len(overlap)} files, e.g. {overlap[:3]}")

    train_dataset = ForestPersonsDataset(args.root, split="train")
    val_dataset = ForestPersonsDataset(args.root, split="val")
    summary = {
        "train": write_split(args, "train", train_dataset, train_files),
        "val": write_split(args, "val", val_dataset, val_files),
        "overlap_images": 0,
        "class_mapping": {"0": "person"},
        "copy_images": bool(args.copy_images),
    }

    yaml_path = args.output_dir / "forestpersons_yolo.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {args.output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary["yaml"] = str(yaml_path)
    (args.output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved YOLO dataset YAML: {yaml_path}")


def write_split(
    args: argparse.Namespace,
    split_name: str,
    dataset: ForestPersonsDataset,
    manifest_files: list[str],
) -> dict[str, Any]:
    samples_by_file = {sample.file_name: sample for sample in dataset.samples}
    missing_annotations = [file_name for file_name in manifest_files if file_name not in samples_by_file]
    if missing_annotations:
        raise ValueError(
            f"{split_name} manifest references files missing from annotations, "
            f"e.g. {missing_annotations[:3]}"
        )

    image_out_root = args.output_dir / "images" / split_name
    label_out_root = args.output_dir / "labels" / split_name
    metadata_path = args.output_dir / f"{split_name}_annotation_metadata.csv"
    image_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    visibility_counts: Counter[int] = Counter()
    pose_counts: Counter[str] = Counter()
    empty_images = 0
    clipped_boxes = 0
    skipped_without_visibility = 0

    for file_name in manifest_files:
        sample = samples_by_file[file_name]
        source_image = sample.image_path
        if not source_image.exists():
            raise FileNotFoundError(f"Missing local image for {split_name}: {source_image}")
        width, height = verify_image_dimensions(source_image, sample.width, sample.height)

        dest_image = image_out_root / file_name
        write_image_reference(source_image, dest_image, copy=args.copy_images)
        label_path = label_out_root / Path(file_name).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)

        label_lines: list[str] = []
        for annotation in sample.annotations:
            metadata = annotation.get("metadata", {})
            visibility = metadata.get("visibility_level")
            if visibility not in {20, 40, 70, 100}:
                skipped_without_visibility += 1
                continue
            yolo_box = coco_bbox_to_yolo(annotation["bbox"], width, height)
            if yolo_box is None:
                clipped_boxes += 1
                continue
            x, y, box_width, box_height = [float(value) for value in annotation["bbox"]]
            clipped = (
                x < 0
                or y < 0
                or x + box_width > width
                or y + box_height > height
            )
            clipped_boxes += int(clipped)
            label_lines.append("0 " + " ".join(f"{value:.8f}" for value in yolo_box))
            pose = str(metadata.get("pose", "unknown"))
            visibility_counts[int(visibility)] += 1
            pose_counts[pose] += 1
            annotation_rows.append(
                {
                    "split": split_name,
                    "file_name": file_name,
                    "image_id": sample.image_id,
                    "annotation_id": annotation.get("id"),
                    "class_id": 0,
                    "class_name": "person",
                    "visibility_level": visibility,
                    "pose": pose,
                    "bbox_coco_x": x,
                    "bbox_coco_y": y,
                    "bbox_coco_width": box_width,
                    "bbox_coco_height": box_height,
                    "yolo_x_center": yolo_box[0],
                    "yolo_y_center": yolo_box[1],
                    "yolo_width": yolo_box[2],
                    "yolo_height": yolo_box[3],
                    "was_clipped": clipped,
                }
            )

        if not label_lines:
            empty_images += 1
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        image_rows.append(
            {
                "file_name": file_name,
                "image_id": sample.image_id,
                "width": width,
                "height": height,
                "annotations": len(label_lines),
                "image_path": str(dest_image),
                "label_path": str(label_path),
            }
        )

    write_csv(metadata_path, annotation_rows)
    write_csv(args.output_dir / f"{split_name}_images.csv", image_rows)
    return {
        "images": len(manifest_files),
        "annotations": len(annotation_rows),
        "empty_label_images": empty_images,
        "clipped_or_dropped_boxes": clipped_boxes,
        "skipped_without_visibility": skipped_without_visibility,
        "visibility_counts": dict(sorted(visibility_counts.items())),
        "pose_counts": dict(sorted(pose_counts.items())),
        "metadata_csv": str(metadata_path),
    }


def verify_image_dimensions(path: Path, expected_width: int | None, expected_height: int | None) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    if mode != "RGB":
        raise ValueError(f"Expected RGB image, got {mode}: {path}")
    if expected_width is not None and width != int(expected_width):
        raise ValueError(f"Width mismatch for {path}: {width} != {expected_width}")
    if expected_height is not None and height != int(expected_height):
        raise ValueError(f"Height mismatch for {path}: {height} != {expected_height}")
    return width, height


def write_image_reference(source: Path, dest: Path, copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if copy:
        shutil.copy2(source, dest)
        return
    relative_source = os.path.relpath(source.resolve(), start=dest.parent.resolve())
    dest.symlink_to(relative_source)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
