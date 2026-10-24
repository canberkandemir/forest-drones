#!/usr/bin/env python3
"""Create and verify nested ForestPersons manifests for MVR label efficiency."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VISIBILITIES = (20, 40, 70, 100)
DEFAULT_SIZES = (100, 400, 1000, 5000)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_key(seed: int, *parts: object) -> str:
    text = ":".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def folder_round_robin(
    items: list[dict[str, Any]], seed: int, visibility: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["folder"])].append(item)
    folders = sorted(grouped, key=lambda name: stable_key(seed, visibility, "folder", name))
    for folder in folders:
        grouped[folder].sort(
            key=lambda item: stable_key(seed, visibility, "file", item["file_name"])
        )
    ordered = []
    offset = 0
    while True:
        added = False
        for folder in folders:
            if offset < len(grouped[folder]):
                ordered.append(grouped[folder][offset])
                added = True
        if not added:
            return ordered
        offset += 1


def build_nested_order(items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if len({str(item["file_name"]) for item in items}) != len(items):
        raise ValueError("Source manifest contains duplicate filenames")
    by_visibility: dict[int, list[dict[str, Any]]] = {}
    for visibility in VISIBILITIES:
        group = [item for item in items if int(item["target_visibility"]) == visibility]
        by_visibility[visibility] = folder_round_robin(group, seed, visibility)
    counts = {visibility: len(group) for visibility, group in by_visibility.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Target-visibility pool is not balanced: {counts}")
    ordered = []
    for index in range(next(iter(counts.values()))):
        ordered.extend(by_visibility[visibility][index] for visibility in VISIBILITIES)
    if {str(item["file_name"]) for item in ordered} != {
        str(item["file_name"]) for item in items
    }:
        raise AssertionError("Nested ordering changed the source image set")
    return ordered


def validate_no_leakage(
    train_items: list[dict[str, Any]], val_items: list[dict[str, Any]]
) -> dict[str, int]:
    train_files = {str(item["file_name"]) for item in train_items}
    val_files = {str(item["file_name"]) for item in val_items}
    train_folders = {str(item["folder"]) for item in train_items}
    val_folders = {str(item["folder"]) for item in val_items}
    file_overlap = train_files & val_files
    folder_overlap = train_folders & val_folders
    if file_overlap or folder_overlap:
        raise ValueError(
            f"Train/val leakage: {len(file_overlap)} filenames, "
            f"{len(folder_overlap)} sequence folders"
        )
    return {
        "filename_overlap": 0,
        "sequence_folder_overlap": 0,
        "numeric_image_id_overlap_ignored": len(
            {int(item["image_id"]) for item in train_items}
            & {int(item["image_id"]) for item in val_items}
        ),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    target = Counter(int(item["target_visibility"]) for item in items)
    annotations: Counter[int] = Counter()
    for item in items:
        annotations.update(
            {int(key): int(value) for key, value in item["visibility_counts"].items()}
        )
    folder_counts = Counter(str(item["folder"]) for item in items)
    return {
        "selected_images": len(items),
        "annotations": sum(int(item["annotation_count"]) for item in items),
        "selected_images_by_target_visibility": dict(sorted(target.items())),
        "annotation_visibility_counts": dict(sorted(annotations.items())),
        "sequence_folders": len(folder_counts),
        "maximum_images_per_sequence": max(folder_counts.values()),
        "season_image_counts": dict(sorted(Counter(item["season"] for item in items).items())),
        "camera_image_counts": dict(sorted(Counter(item["camera"] for item in items).items())),
        "place_image_counts": dict(sorted(Counter(item["place"] for item in items).items())),
        "weather_image_counts": dict(sorted(Counter(item["weather"] for item in items).items())),
    }


def verify_assets(
    items: list[dict[str, Any]],
    image_root: Path,
    final_cache_root: Path,
    mvr_cache_root: Path,
) -> dict[str, int]:
    counts = {"images": 0, "final_caches": 0, "mvr_caches": 0}
    missing: list[str] = []
    for item in items:
        relative = Path(str(item["file_name"]))
        paths = {
            "images": image_root / relative,
            "final_caches": final_cache_root / relative.with_suffix(".npz"),
            "mvr_caches": mvr_cache_root / relative.with_suffix(".npz"),
        }
        for kind, path in paths.items():
            if path.is_file() and path.stat().st_size > 0:
                counts[kind] += 1
            else:
                missing.append(f"{kind}:{relative}")
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} assets; first: {missing[:5]}")
    return counts


def prepare_manifests(
    source_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    seed: int,
    sizes: tuple[int, ...],
    image_root: Path,
    final_cache_root: Path,
    mvr_cache_root: Path,
) -> dict[str, Any]:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    val = json.loads(val_manifest.read_text(encoding="utf-8"))
    ordered = build_nested_order(source["images"], seed)
    if tuple(sorted(sizes)) != sizes or sizes[-1] != len(ordered):
        raise ValueError("Sizes must be increasing and end at the source-pool size")
    if any(size % len(VISIBILITIES) for size in sizes):
        raise ValueError("Every subset size must be divisible by four")
    leakage = validate_no_leakage(ordered, val["images"])
    assets = verify_assets(ordered, image_root, final_cache_root, mvr_cache_root)
    output_paths = {
        size: output_dir / f"forestpersons_train_nested_{size}_seed{seed}_manifest.json"
        for size in sizes
    }
    audit_path = output_dir / f"label_efficiency_nested_seed{seed}_audit.json"
    existing = [path for path in (*output_paths.values(), audit_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source_manifest)
    manifests = {}
    for size, path in output_paths.items():
        selected = ordered[:size]
        expected = size // len(VISIBILITIES)
        target_counts = Counter(int(item["target_visibility"]) for item in selected)
        if any(target_counts[level] != expected for level in VISIBILITIES):
            raise AssertionError(f"Unbalanced nested prefix {size}: {target_counts}")
        payload = {
            "source_annotations": source["source_annotations"],
            "selection": {
                "experiment": "dinov3_mvr_label_efficiency",
                "seed": seed,
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": source_sha,
                "strategy": (
                    "Fixed train5000 pool; deterministic folder round-robin within "
                    "each target-visibility stratum; visibility-interleaved order "
                    "makes 100/400/1000/5000 exact nested prefixes."
                ),
                "nested_sizes": list(sizes),
                "prefix_size": size,
            },
            "summary": summarize(selected),
            "images": selected,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifests[str(size)] = {"path": str(path), "sha256": sha256_file(path), **payload["summary"]}
    audit = {
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_sha,
        "validation_manifest": str(val_manifest),
        "selection_seed": seed,
        "sizes": list(sizes),
        "nested_prefixes_verified": True,
        "full_pool_preserved": True,
        "leakage": leakage,
        "asset_counts": assets,
        "manifests": manifests,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--final-cache-root", type=Path, required=True)
    parser.add_argument("--mvr-cache-root", type=Path, required=True)
    args = parser.parse_args()
    audit = prepare_manifests(
        args.source_manifest,
        args.val_manifest,
        args.output_dir,
        args.seed,
        tuple(args.sizes),
        args.image_root,
        args.final_cache_root,
        args.mvr_cache_root,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
