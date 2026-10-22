"""Audit official Weitefeld metadata and create a deterministic pilot200 manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from forestpersons.weitefeld import (
    PRIORITY_ZONE_RANGES,
    WEITEFELD_HEIGHT,
    WEITEFELD_PERSON_CLASS,
    WEITEFELD_WIDTH,
    assign_strip_track_ids,
    deduplicate_annotations,
    is_priority_zone_image,
    parse_data_txt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=Path("data/weitefeld/metadata/official"))
    parser.add_argument("--output", type=Path, default=Path("outputs/weitefeld/pilot200_manifest.json"))
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--positive-views-per-track", type=int, default=3)
    parser.add_argument("--total-images", type=int, default=200)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_indexes(root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    result = {}
    for strip in range(1, 16):
        entries = json.loads((root / f"strip{strip}_index.json").read_text(encoding="utf-8"))
        for entry in entries:
            key = strip, int(entry["id"])
            result[key] = {**entry, "strip": strip, "image_number": int(entry["id"])}
    return result


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round((index + 1) * (len(values) - 1) / (count + 1)) for index in range(count)]
    return [values[index] for index in sorted(set(indices))]


def main() -> None:
    args = parse_args()
    raw = parse_data_txt(args.metadata_root / "data.txt")
    unique = assign_strip_track_ids(deduplicate_annotations(raw))
    indexes = load_indexes(args.metadata_root)
    by_image: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for annotation in unique:
        by_image[annotation.image_key].append(annotation)

    person = [item for item in unique if item.category_id == WEITEFELD_PERSON_CLASS]
    valid_person = [item for item in person if item.bbox_valid]
    tracks: dict[str, list[Any]] = defaultdict(list)
    for annotation in valid_person:
        assert annotation.track_id is not None
        tracks[annotation.track_id].append(annotation)

    selected_positive: set[tuple[int, int]] = set()
    for track_id in sorted(tracks):
        image_numbers = sorted({item.image_number for item in tracks[track_id]})
        strip = tracks[track_id][0].image_strip
        selected_positive.update(
            (strip, number) for number in evenly_spaced(image_numbers, args.positive_views_per_track)
        )

    if len(selected_positive) >= args.total_images:
        raise RuntimeError("Positive-view selection fills the whole pilot; no context images remain")

    # Context-only means no published anomaly annotation, not confirmed negative.
    priority_keys = {
        key for key in indexes
        if is_priority_zone_image(*key)
    }
    context_pool = sorted(priority_keys - set(by_image) - selected_positive)
    context_needed = args.total_images - len(selected_positive)
    rng = random.Random(args.seed)
    by_strip: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in context_pool:
        by_strip[key[0]].append(key)
    selected_context: list[tuple[int, int]] = []
    strips = list(range(1, 16))
    rng.shuffle(strips)
    while len(selected_context) < context_needed:
        progress = False
        for strip in strips:
            candidates = by_strip[strip]
            if not candidates:
                continue
            # Deterministic pseudo-random choice while discouraging adjacent frames.
            rng.shuffle(candidates)
            chosen_index = next(
                (index for index, key in enumerate(candidates)
                 if all(key[0] != old[0] or abs(key[1] - old[1]) >= 10 for old in selected_context)),
                0,
            )
            selected_context.append(candidates.pop(chosen_index))
            progress = True
            if len(selected_context) == context_needed:
                break
        if not progress:
            raise RuntimeError("Insufficient context candidates")

    selected = sorted(selected_positive) + sorted(selected_context)
    images = []
    for image_id, key in enumerate(selected, 1):
        annotations = by_image.get(key, [])
        person_annotations = [
            item for item in annotations
            if item.category_id == WEITEFELD_PERSON_CLASS and item.bbox_valid
        ]
        index = indexes[key]
        images.append({
            "image_id": image_id,
            "strip": key[0],
            "image_number": key[1],
            "file_name": Path(index["rgb"]).name,
            "source_url": index["rgb"],
            "width": WEITEFELD_WIDTH,
            "height": WEITEFELD_HEIGHT,
            "selection_category": "person_class_track_view" if key in selected_positive else "context_unannotated_unknown",
            "person_annotation_count": len(person_annotations),
            "person_track_ids": sorted({item.track_id for item in person_annotations}),
            "person_annotations": [item.to_dict() for item in person_annotations],
            "all_annotation_count": len(annotations),
            "annotation_completeness": "published anomaly labels are non-exhaustive",
        })

    class_counts = Counter(item.category_id for item in unique)
    person_tiers = Counter(item.verification_tier for item in person)
    manifest = {
        "schema_version": 1,
        "dataset": "Weitefeld core priority-zone pilot",
        "seed": args.seed,
        "source": {
            "zenodo_record": "https://zenodo.org/records/15848419",
            "web_index": "https://weitefeld.cg.jku.at/strip{strip}/index.json",
            "paper_doi": "10.1038/s41597-026-07101-w",
            "data_txt_sha256": sha256_file(args.metadata_root / "data.txt"),
        },
        "selection": {
            "positive_strategy": f"up to {args.positive_views_per_track} interior evenly spaced images from every strip-local class-3 track proxy",
            "context_strategy": "deterministic strip-round-robin sampling from priority-zone images with no published anomaly row; not treated as person-negative",
            "physical_identity_warning": "data.txt has no physical finding ID; track IDs are strip-local contiguous proxies",
        },
        "audit": {
            "raw_annotation_rows": len(raw),
            "deduplicated_annotations": len(unique),
            "class_counts_deduplicated": {str(key): value for key, value in sorted(class_counts.items())},
            "annotated_images": len(by_image),
            "priority_zone_images": sum(
                end - start + 1 for ranges in PRIORITY_ZONE_RANGES.values() for start, end in ranges
            ),
            "person_class_annotations": len(person),
            "valid_person_class_annotations": len(valid_person),
            "invalid_person_class_annotations": len(person) - len(valid_person),
            "person_class_images": len({item.image_key for item in person}),
            "person_strip_track_proxies": len(tracks),
            "person_verification_tiers": dict(person_tiers),
        },
        "summary": {
            "images": len(images),
            "person_track_view_images": len(selected_positive),
            "context_unannotated_unknown_images": len(selected_context),
            "person_annotations_in_pilot": sum(item["person_annotation_count"] for item in images),
            "person_tracks_represented": len({track for item in images for track in item["person_track_ids"]}),
            "images_by_strip": dict(Counter(str(item["strip"]) for item in images)),
            "selection_categories": dict(Counter(item["selection_category"] for item in images)),
        },
        "images": images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["audit"] | manifest["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

