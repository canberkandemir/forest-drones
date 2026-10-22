"""Audit train5000 for leakage-safe, registration-qualified three-view groups.

Group discovery uses only split-qualified filenames, sequence/frame metadata,
local image availability, and ORB/RANSAC registration. COCO annotations are
loaded only after the qualified and frame-disjoint group lists are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

OFFSETS_EARLIER = (-10, -8, -5, -3)
OFFSETS_LATER = (3, 5, 8, 10)
SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--train5000-manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json"),
    )
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/train.json"))
    parser.add_argument(
        "--benchmark49-manifest",
        type=Path,
        default=Path("outputs/multiview/forestpersons_val_benchmark50_seed20261016/preparation/manifest.json"),
    )
    parser.add_argument(
        "--pilot8-manifest",
        type=Path,
        default=Path("outputs/multiview/forestpersons_val_pilot8_seed20261016/preparation/manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--registration-scale", type=float, default=0.5)
    parser.add_argument("--split-qualified-validation-fraction", type=float, default=0.20)
    return parser.parse_args()


def frame_number(file_name: str) -> int | None:
    match = re.search(r"-(\d+)\.[^.]+$", file_name)
    return int(match.group(1)) if match else None


def identity(file_name: str) -> str:
    return f"{SPLIT}::{file_name}"


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest_sequences(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(group["sequence"]) for group in data["groups"]}


def disjoint_groups(groups: list[dict[str, Any]], seed: int, trials: int = 100) -> list[dict[str, Any]]:
    """Return the best deterministic greedy frame-disjoint packing found."""

    best: list[dict[str, Any]] = []
    canonical = sorted(groups, key=lambda item: (item["sequence"], item["reference"]["frame"]))
    for trial in range(trials):
        shuffled = list(canonical)
        random.Random(seed + trial).shuffle(shuffled)
        used: set[str] = set()
        selected = []
        for group in shuffled:
            keys = [group["reference"]["identity"], *[source["identity"] for source in group["sources"]]]
            if used.isdisjoint(keys):
                selected.append(group)
                used.update(keys)
        if len(selected) > len(best):
            best = selected
    return sorted(best, key=lambda item: (item["sequence"], item["reference"]["frame"]))


def annotation_summary(groups: list[dict[str, Any]], annotations_by_image: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    references = {int(group["reference"]["image_id"]): group["reference"]["file_name"] for group in groups}
    visibility = Counter()
    annotations = []
    for image_id, file_name in references.items():
        for annotation in annotations_by_image.get(image_id, []):
            level = annotation.get("attributes", {}).get("visibility_level")
            if level is None:
                level = annotation.get("visibility_level")
            if level is not None:
                visibility[int(level)] += 1
            annotations.append({
                "image_id": image_id,
                "split_qualified_file": identity(file_name),
                "annotation_id": annotation.get("id"),
                "visibility": level,
            })
    return {
        "reference_images": len(references),
        "person_annotations": len(annotations),
        "visibility_counts": {str(level): visibility[level] for level in (20, 40, 70, 100)},
        "annotations": annotations,
    }


def main() -> None:
    args = parse_args()
    try:
        from scripts.prepare_forestpersons_multiview_benchmark import estimate_homography
    except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
        from prepare_forestpersons_multiview_benchmark import estimate_homography
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    source = json.loads(args.train5000_manifest.read_text(encoding="utf-8"))
    # Deliberately strip annotation-derived manifest fields before selection.
    images = [
        {
            "image_id": int(item["image_id"]),
            "file_name": str(item["file_name"]),
            "identity": identity(str(item["file_name"])),
        }
        for item in source["images"]
    ]
    if len({item["identity"] for item in images}) != len(images):
        raise ValueError("Duplicate split-qualified filename in train5000")

    sequences: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    missing_files = []
    for image in images:
        sequence = image["file_name"].split("/")[0]
        frame = frame_number(image["file_name"])
        if frame is None:
            continue
        image["frame"] = frame
        image["sequence"] = sequence
        sequences[sequence][frame] = image
        if not (args.root / "images" / image["file_name"]).is_file():
            missing_files.append(image["identity"])
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} train5000 images; first={missing_files[0]}")

    candidate_rows = []
    candidate_triplet_combinations = 0
    for sequence, frames in sorted(sequences.items()):
        for reference_frame in sorted(frames):
            earlier = [offset for offset in OFFSETS_EARLIER if reference_frame + offset in frames]
            later = [offset for offset in OFFSETS_LATER if reference_frame + offset in frames]
            if earlier and later:
                candidate_triplet_combinations += len(earlier) * len(later)
                candidate_rows.append({
                    "sequence": sequence,
                    "reference_frame": reference_frame,
                    "reference_image_id": frames[reference_frame]["image_id"],
                    "reference_identity": frames[reference_frame]["identity"],
                    "earlier_candidates": ";".join(map(str, earlier)),
                    "later_candidates": ";".join(map(str, later)),
                    "triplet_combinations": len(earlier) * len(later),
                })

    source_attempts = []
    qualified = []
    for index, candidate in enumerate(candidate_rows, start=1):
        sequence = candidate["sequence"]
        reference_frame = int(candidate["reference_frame"])
        frames = sequences[sequence]
        reference = frames[reference_frame]
        reference_path = args.root / "images" / reference["file_name"]
        sources = []
        for direction, offsets in (("earlier", OFFSETS_EARLIER), ("later", OFFSETS_LATER)):
            accepted = None
            for offset in offsets:
                source_image = frames.get(reference_frame + offset)
                if source_image is None:
                    continue
                homography, registration, reason = estimate_homography(
                    args.root / "images" / source_image["file_name"],
                    reference_path,
                    args.registration_scale,
                )
                source_attempts.append({
                    "sequence": sequence,
                    "reference_identity": reference["identity"],
                    "source_identity": source_image["identity"],
                    "direction": direction,
                    "offset": offset,
                    "accepted": reason is None,
                    "reason": reason or "accepted",
                    **registration,
                })
                if reason is None and homography is not None:
                    accepted = {
                        "image_id": source_image["image_id"],
                        "file_name": source_image["file_name"],
                        "identity": source_image["identity"],
                        "frame": source_image["frame"],
                        "frame_offset": offset,
                        "homography_source_to_reference": homography.tolist(),
                        "registration": registration,
                    }
                    break
            if accepted is not None:
                sources.append(accepted)
        if len(sources) == 2:
            qualified.append({
                "sequence": sequence,
                "reference": {
                    "image_id": reference["image_id"],
                    "file_name": reference["file_name"],
                    "identity": reference["identity"],
                    "frame": reference_frame,
                },
                "sources": sources,
            })
        if index % 100 == 0 or index == len(candidate_rows):
            print(f"[{index}/{len(candidate_rows)}] qualified={len(qualified)}", flush=True)

    # Freeze registration-qualified groups before opening annotation JSON.
    qualified_sha256 = sha256_json(qualified)
    frame_disjoint = disjoint_groups(qualified, args.seed)
    sequence_names = sorted({group["sequence"] for group in qualified})
    shuffled_sequences = list(sequence_names)
    random.Random(args.seed).shuffle(shuffled_sequences)
    validation_count = max(1, round(len(shuffled_sequences) * args.split_qualified_validation_fraction))
    validation_sequences = set(shuffled_sequences[:validation_count])
    training_sequences = set(shuffled_sequences[validation_count:])
    proposed_train = disjoint_groups([group for group in qualified if group["sequence"] in training_sequences], args.seed)
    proposed_validation_all = disjoint_groups([group for group in qualified if group["sequence"] in validation_sequences], args.seed)
    # One validation group per sequence avoids pseudo-replication in checkpoint selection.
    proposed_validation = []
    seen_validation_sequences = set()
    for group in proposed_validation_all:
        if group["sequence"] not in seen_validation_sequences:
            proposed_validation.append(group)
            seen_validation_sequences.add(group["sequence"])

    annotation_data = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotation_data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    benchmark_sequences = manifest_sequences(args.benchmark49_manifest)
    pilot_sequences = manifest_sequences(args.pilot8_manifest)
    train5000_sequences = set(sequences)
    summaries = {
        "all_registration_qualified_references": annotation_summary(qualified, annotations_by_image),
        "frame_disjoint_groups": annotation_summary(frame_disjoint, annotations_by_image),
        "proposed_training_groups": annotation_summary(proposed_train, annotations_by_image),
        "proposed_validation_groups_one_per_sequence": annotation_summary(proposed_validation, annotations_by_image),
    }
    for value in summaries.values():
        value.pop("annotations")

    result = {
        "dataset": "ForestPersons train5000 seed20261012",
        "selection_uses_annotations": False,
        "selection_uses_model_predictions": False,
        "annotation_join_after_qualified_groups_frozen": True,
        "identity_rule": "split-qualified filename (train::<relative file_name>); numeric image IDs are not global identity",
        "registration": {
            "method": "unchanged benchmark49 ORB ratio-test plus RANSAC homography",
            "scale": args.registration_scale,
            "source_offsets": {"earlier": list(OFFSETS_EARLIER), "later": list(OFFSETS_LATER)},
        },
        "counts": {
            "images": len(images),
            "sequences": len(sequences),
            "metadata_candidate_references": len(candidate_rows),
            "metadata_candidate_triplet_combinations": candidate_triplet_combinations,
            "sequences_with_metadata_candidate": len({row["sequence"] for row in candidate_rows}),
            "registration_qualified_references": len(qualified),
            "sequences_with_registration_qualified_group": len(sequence_names),
            "frame_disjoint_groups": len(frame_disjoint),
            "proposed_training_groups": len(proposed_train),
            "proposed_training_sequences": len(training_sequences),
            "proposed_validation_groups": len(proposed_validation),
            "proposed_validation_sequences": len(validation_sequences),
            "source_registration_attempts": len(source_attempts),
        },
        "leakage": {
            "train5000_vs_benchmark49_sequence_overlap": len(train5000_sequences & benchmark_sequences),
            "train5000_vs_pilot8_sequence_overlap": len(train5000_sequences & pilot_sequences),
            "proposed_train_validation_sequence_overlap": len(training_sequences & validation_sequences),
            "proposed_train_validation_identity_overlap": len(
                {view["identity"] for group in proposed_train for view in [group["reference"], *group["sources"]]}
                & {view["identity"] for group in proposed_validation for view in [group["reference"], *group["sources"]]}
            ),
            "physical_scene_warning": "ForestPersons exposes sequence folders but no globally unique physical-scene/site identifier; sequence-disjoint is verifiable, site-disjoint is not.",
        },
        "annotation_summaries": summaries,
        "source_rejection_reasons": dict(Counter(row["reason"] for row in source_attempts if not row["accepted"])),
        "qualified_groups_sha256_before_annotation_join": qualified_sha256,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "qualified_groups_registration_only.json").write_text(json.dumps(qualified, indent=2), encoding="utf-8")
    (args.output_dir / "frame_disjoint_groups.json").write_text(json.dumps(frame_disjoint, indent=2), encoding="utf-8")
    (args.output_dir / "proposed_sequence_split.json").write_text(json.dumps({
        "seed": args.seed,
        "training_sequences": sorted(training_sequences),
        "validation_sequences": sorted(validation_sequences),
        "training_groups": proposed_train,
        "validation_groups": proposed_validation,
    }, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "metadata_candidates.csv", candidate_rows)
    write_csv(args.output_dir / "source_registration_attempts.csv", source_attempts)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
