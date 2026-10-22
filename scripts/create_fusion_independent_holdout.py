"""Create and integrity-check an independent ForestPersons fusion holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from create_forestpersons_val_subset import image_records, select_subset, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/val.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--exclude-manifest", type=Path, action="append", required=True)
    parser.add_argument("--protected-manifest", type=Path, action="append", default=[])
    parser.add_argument("--images-per-visibility", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def manifest_images(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    excluded_files: set[str] = set()
    excluded_ids: set[int | str] = set()
    exclusion_counts = {}
    for path in args.exclude_manifest:
        images = manifest_images(path)
        excluded_files.update(str(item["file_name"]) for item in images)
        excluded_ids.update(item["image_id"] for item in images if "image_id" in item)
        exclusion_counts[str(path)] = len(images)

    protected_paths: dict[str, Path] = {}
    protected_folders: set[str] = set()
    for path in [*args.exclude_manifest, *args.protected_manifest]:
        for item in manifest_images(path):
            file_name = str(item["file_name"])
            local_path = args.image_root / file_name
            if local_path.is_file():
                protected_paths.setdefault(file_name, local_path)
            protected_folders.add(file_name.split("/")[0])

    records = [
        record
        for record in image_records(coco)
        if record["file_name"] not in excluded_files
        and record["image_id"] not in excluded_ids
        and (args.image_root / record["file_name"]).is_file()
    ]
    selected, selection = select_subset(records, args.images_per_visibility, args.seed)
    expected = 4 * args.images_per_visibility
    if len(selected) != expected:
        raise RuntimeError(f"Selected {len(selected)} images, expected {expected}")

    protected_hashes: dict[str, list[str]] = {}
    for file_name, path in protected_paths.items():
        protected_hashes.setdefault(sha256_file(path), []).append(file_name)

    selected_hashes: set[str] = set()
    duplicate_hashes = []
    dimension_mismatches = []
    unreadable = []
    for item in selected:
        path = args.image_root / item["file_name"]
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                actual_size = image.size
        except Exception as exc:
            unreadable.append({"file_name": item["file_name"], "error": repr(exc)})
            continue
        expected_size = (int(item["width"]), int(item["height"]))
        if actual_size != expected_size:
            dimension_mismatches.append({
                "file_name": item["file_name"], "expected": expected_size, "actual": actual_size,
            })
        digest = sha256_file(path)
        item["sha256"] = digest
        if digest in selected_hashes:
            duplicate_hashes.append({"file_name": item["file_name"], "kind": "within_holdout"})
        if digest in protected_hashes:
            duplicate_hashes.append({
                "file_name": item["file_name"],
                "kind": "protected_overlap",
                "protected_files": protected_hashes[digest],
            })
        selected_hashes.add(digest)

    if unreadable or dimension_mismatches or duplicate_hashes:
        raise RuntimeError(
            f"Holdout integrity failed: unreadable={len(unreadable)}, "
            f"dimension_mismatches={len(dimension_mismatches)}, hash_duplicates={len(duplicate_hashes)}"
        )

    annotation_ids = [
        annotation["id"]
        for annotation in coco["annotations"]
        if annotation["image_id"] in {item["image_id"] for item in selected}
    ]
    selected_folders = {item["folder"] for item in selected}
    output = {
        "source_annotations": str(args.annotations),
        "selection": {
            **selection,
            "purpose": "Independent frozen YOLO11s + DINOv3 FCOS fusion evaluation",
            "local_images_only": True,
            "exclude_manifests": [str(path) for path in args.exclude_manifest],
            "protected_manifests_for_hash_audit": [str(path) for path in args.protected_manifest],
            "exclusion_manifest_image_counts": exclusion_counts,
        },
        "summary": {
            **summarize(selected, args.image_root),
            "annotations": len(annotation_ids),
            "unique_annotation_ids": len(set(annotation_ids)),
            "unique_image_ids": len({item["image_id"] for item in selected}),
            "unique_filenames": len({item["file_name"] for item in selected}),
            "unique_sha256": len(selected_hashes),
            "excluded_filename_overlap": len({item["file_name"] for item in selected} & excluded_files),
            "excluded_image_id_overlap": len({item["image_id"] for item in selected} & excluded_ids),
            "protected_sha256_overlap": 0,
            "unreadable_images": 0,
            "dimension_mismatches": 0,
            "selected_folder_count": len(selected_folders),
            "folders_also_present_in_protected_manifests": len(selected_folders & protected_folders),
            "selection_is_image_disjoint_not_scene_disjoint": True,
            "sha256_algorithm": "SHA-256 over original local image bytes",
            "visibility_target_counts": dict(Counter(str(item["target_visibility"]) for item in selected)),
        },
        "images": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
