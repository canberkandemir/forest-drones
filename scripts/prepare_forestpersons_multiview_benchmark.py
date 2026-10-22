"""Build a registration-only ForestPersons multi-view benchmark manifest.

Selection reads the COCO image table but never reads annotations, visibility,
boxes, or detector predictions. Ground truth is joined only after the manifest
has been frozen by the evaluation script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from forestpersons.multiview import (
    DEFAULT_REGISTRATION_THRESHOLDS,
    registration_rejection_reasons,
    validate_unique_group_images,
)


OFFSETS_EARLIER = (-10, -8, -5, -3)
OFFSETS_LATER = (3, 5, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/val.json"))
    parser.add_argument("--train-annotations", type=Path, default=Path("data/annotations/train.json"))
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=50)
    parser.add_argument("--minimum-groups", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--registration-scale", type=float, default=0.5)
    parser.add_argument("--max-reference-attempts-per-sequence", type=int, default=40)
    return parser.parse_args()


def frame_number(file_name: str) -> int | None:
    match = re.search(r"-(\d+)\.[^.]+$", file_name)
    return int(match.group(1)) if match else None


def sha256_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def bbox_spread_fraction(points: np.ndarray, width: int, height: int) -> float:
    if points.size == 0:
        return 0.0
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(width * height, 1))


def estimate_homography(
    source_path: Path,
    reference_path: Path,
    scale: float,
) -> tuple[np.ndarray | None, dict[str, Any], str | None]:
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if source is None or reference is None:
        return None, {}, "unreadable_image"
    source_small = cv2.resize(source, None, fx=scale, fy=scale)
    reference_small = cv2.resize(reference, None, fx=scale, fy=scale)
    detector = cv2.ORB_create(nfeatures=3500, fastThreshold=10)
    source_keypoints, source_descriptors = detector.detectAndCompute(source_small, None)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_small, None)
    if source_descriptors is None or reference_descriptors is None:
        return None, {}, "missing_descriptors"
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_descriptors, reference_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        return None, {"ratio_test_matches": len(good)}, "too_few_matches_for_homography"
    source_points = np.float32([source_keypoints[item.queryIdx].pt for item in good])
    reference_points = np.float32([reference_keypoints[item.trainIdx].pt for item in good])
    homography_small, mask = cv2.findHomography(source_points, reference_points, cv2.RANSAC, 3.0)
    if homography_small is None or mask is None or not np.isfinite(homography_small).all():
        return None, {"ratio_test_matches": len(good)}, "homography_estimation_failed"
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < 4:
        return None, {"ratio_test_matches": len(good), "ransac_inliers": inliers}, "too_few_homography_inliers"
    projected_raw = cv2.perspectiveTransform(
        source_points[inlier_mask].reshape(-1, 1, 2), homography_small
    )
    if projected_raw is None or not np.isfinite(projected_raw).all():
        return None, {}, "nonfinite_projection"
    projected = projected_raw.reshape(-1, 2)
    errors = np.linalg.norm(projected - reference_points[inlier_mask], axis=1)
    down = np.diag([scale, scale, 1.0])
    up = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    homography = up @ homography_small @ down

    source_height, source_width = source.shape
    reference_height, reference_width = reference.shape
    source_corners = np.float32(
        [[0, 0], [source_width, 0], [source_width, source_height], [0, source_height]]
    ).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(source_corners, homography).reshape(-1, 2)
    if not np.isfinite(projected_corners).all():
        return None, {}, "nonfinite_corner_projection"
    projected_hull = cv2.convexHull(projected_corners.astype(np.float32))
    reference_polygon = np.float32(
        [[0, 0], [reference_width, 0], [reference_width, reference_height], [0, reference_height]]
    ).reshape(-1, 1, 2)
    intersection_area, _ = cv2.intersectConvexConvex(projected_hull, reference_polygon)
    projected_area = polygon_area(projected_hull)
    reference_area = float(reference_width * reference_height)
    metrics: dict[str, Any] = {
        "ratio_test_matches": len(good),
        "ransac_inliers": inliers,
        "inlier_ratio": inliers / len(good),
        "median_reprojection_error_registration_pixels": float(np.median(errors)),
        "median_reprojection_error_original_pixels": float(np.median(errors) / scale),
        "reference_overlap_fraction": float(intersection_area / max(reference_area, 1.0)),
        "projected_area_ratio": float(projected_area / max(reference_area, 1.0)),
        "source_inlier_spread_fraction": bbox_spread_fraction(
            source_points[inlier_mask], source_small.shape[1], source_small.shape[0]
        ),
        "reference_inlier_spread_fraction": bbox_spread_fraction(
            reference_points[inlier_mask], reference_small.shape[1], reference_small.shape[0]
        ),
    }
    reasons = registration_rejection_reasons(metrics)
    return homography, metrics, reasons[0] if reasons else None


def load_image_table(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["images"]


def previous_images_and_sequences(path: Path) -> tuple[set[int], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    image_ids: set[int] = set()
    sequences: set[str] = set()
    for group in data["groups"]:
        sequences.add(group["sequence"])
        image_ids.add(int(group["reference"]["image_id"]))
        image_ids.update(int(source["image_id"]) for source in group["sources"])
    return image_ids, sequences


def save_alignment_gallery(group: dict[str, Any], image_root: Path, output_path: Path) -> None:
    reference = cv2.imread(str(image_root / group["reference"]["file_name"]))
    panels = []
    for label, view in [("reference", group["reference"]), *(("source", item) for item in group["sources"])]:
        image = cv2.imread(str(image_root / view["file_name"]))
        display = image.copy()
        cv2.putText(display, f"{label} frame {view['frame']}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        panels.append(display)
        if label == "source":
            homography = np.asarray(view["homography_source_to_reference"], dtype=np.float64)
            warped = cv2.warpPerspective(image, homography, (reference.shape[1], reference.shape[0]))
            overlay = cv2.addWeighted(reference, 0.5, warped, 0.5, 0.0)
            cv2.putText(overlay, "reference / warped source", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            panels.append(overlay)
    target_width = 640
    resized = [cv2.resize(panel, (target_width, round(panel.shape[0] * target_width / panel.shape[1]))) for panel in panels]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(resized))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_images = load_image_table(args.annotations)
    train_images = load_image_table(args.train_annotations)
    train_files = {str(item["file_name"]) for item in train_images}
    train_sequences = {file_name.split("/")[0] for file_name in train_files}
    previous_ids, previous_sequences = previous_images_and_sequences(args.previous_manifest)
    image_root = args.root / "images"

    sequences: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for image in val_images:
        frame = frame_number(str(image["file_name"]))
        if frame is not None:
            sequences[str(image["file_name"]).split("/")[0]][frame] = image
    val_sequences = set(sequences)
    if val_sequences.intersection(train_sequences):
        raise RuntimeError("Train/validation sequence overlap detected")
    if any(str(item["file_name"]) in train_files for item in val_images):
        raise RuntimeError("Train/validation filename overlap detected")

    rng = random.Random(args.seed)
    new_sequences = sorted(val_sequences - previous_sequences)
    old_sequences = sorted(val_sequences & previous_sequences)
    rng.shuffle(new_sequences)
    rng.shuffle(old_sequences)
    sequence_order = new_sequences + old_sequences
    groups: list[dict[str, Any]] = []
    reference_rejections: list[dict[str, Any]] = []
    source_attempts: list[dict[str, Any]] = []
    used_images: set[int] = set(previous_ids)

    for sequence in sequence_order:
        if len(groups) >= args.groups:
            break
        frames = sequences[sequence]
        reference_frames = sorted(frames)
        rng.shuffle(reference_frames)
        accepted_group = None
        for reference_frame in reference_frames[: args.max_reference_attempts_per_sequence]:
            reference = frames[reference_frame]
            reference_id = int(reference["id"])
            reference_path = image_root / reference["file_name"]
            if reference_id in used_images:
                reference_rejections.append({"sequence": sequence, "reference_image_id": reference_id, "reference_frame": reference_frame, "reason": "previous_or_reused_image"})
                continue
            if not reference_path.exists():
                reference_rejections.append({"sequence": sequence, "reference_image_id": reference_id, "reference_frame": reference_frame, "reason": "missing_reference_file"})
                continue
            accepted_sources = []
            for direction, offsets in (("earlier", OFFSETS_EARLIER), ("later", OFFSETS_LATER)):
                direction_source = None
                for offset in offsets:
                    source = frames.get(reference_frame + offset)
                    if source is None:
                        source_attempts.append({"sequence": sequence, "reference_image_id": reference_id, "source_image_id": "", "offset": offset, "direction": direction, "accepted": False, "reason": "frame_missing"})
                        continue
                    source_id = int(source["id"])
                    source_path = image_root / source["file_name"]
                    if source_id in used_images:
                        source_attempts.append({"sequence": sequence, "reference_image_id": reference_id, "source_image_id": source_id, "offset": offset, "direction": direction, "accepted": False, "reason": "previous_or_reused_image"})
                        continue
                    homography, metrics, reason = estimate_homography(source_path, reference_path, args.registration_scale)
                    source_attempts.append({
                        "sequence": sequence,
                        "reference_image_id": reference_id,
                        "source_image_id": source_id,
                        "offset": offset,
                        "direction": direction,
                        "accepted": reason is None,
                        "reason": reason or "accepted",
                        **metrics,
                    })
                    if reason is None and homography is not None:
                        direction_source = {
                            "image_id": source_id,
                            "file_name": source["file_name"],
                            "frame": reference_frame + offset,
                            "frame_offset": offset,
                            "homography_source_to_reference": homography.tolist(),
                            "registration": metrics,
                        }
                        break
                if direction_source is not None:
                    accepted_sources.append(direction_source)
            if len(accepted_sources) != 2:
                missing = "earlier_and_later" if not accepted_sources else (
                    "earlier" if accepted_sources[0]["frame_offset"] > 0 else "later"
                )
                reference_rejections.append({"sequence": sequence, "reference_image_id": reference_id, "reference_frame": reference_frame, "reason": f"missing_reliable_{missing}_source"})
                continue
            accepted_group = {
                "group_id": f"group_{len(groups) + 1:03d}",
                "sequence": sequence,
                "sequence_seen_in_pilot8": sequence in previous_sequences,
                "reference": {
                    "image_id": reference_id,
                    "file_name": reference["file_name"],
                    "frame": reference_frame,
                },
                "sources": accepted_sources,
            }
            break
        if accepted_group is None:
            continue
        groups.append(accepted_group)
        group_ids = [accepted_group["reference"]["image_id"], *[item["image_id"] for item in accepted_group["sources"]]]
        used_images.update(group_ids)
        save_alignment_gallery(
            accepted_group,
            image_root,
            args.output_dir / "alignment_gallery" / f"{accepted_group['group_id']}.jpg",
        )

    validate_unique_group_images(groups)
    manifest = {
        "dataset": "ForestPersons official validation split",
        "purpose": "Registration-only DINOv3 aligned feature-fusion benchmark",
        "seed": args.seed,
        "selection_uses_annotations": False,
        "selection_uses_model_predictions": False,
        "groups_requested": args.groups,
        "groups_accepted": len(groups),
        "views_per_group": 3,
        "one_group_per_sequence": True,
        "source_policy": "one earlier and one later frame from fixed offsets",
        "source_offsets": {"earlier": list(OFFSETS_EARLIER), "later": list(OFFSETS_LATER)},
        "registration": {
            "method": "ORB ratio-test correspondences plus RANSAC homography",
            "scale": args.registration_scale,
            "thresholds": DEFAULT_REGISTRATION_THRESHOLDS,
        },
        "split_integrity": {
            "train_val_filename_overlap": 0,
            "train_val_sequence_overlap": 0,
            "previous_pilot_image_overlap": 0,
            "previous_pilot_sequences_available_as_fallback": len(old_sequences),
            "warning": "COCO numeric image IDs are split-local and overlap numerically; identity checks use split-qualified filenames and sequence names.",
        },
        "groups": groups,
    }
    manifest["groups_sha256"] = sha256_payload(groups)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    accepted_rows = []
    for group in groups:
        accepted_rows.append({
            "group_id": group["group_id"],
            "sequence": group["sequence"],
            "sequence_seen_in_pilot8": group["sequence_seen_in_pilot8"],
            "reference_image_id": group["reference"]["image_id"],
            "reference_frame": group["reference"]["frame"],
            "earlier_image_id": group["sources"][0]["image_id"],
            "earlier_offset": group["sources"][0]["frame_offset"],
            "later_image_id": group["sources"][1]["image_id"],
            "later_offset": group["sources"][1]["frame_offset"],
        })
    write_csv(args.output_dir / "accepted_groups.csv", accepted_rows)
    write_csv(args.output_dir / "rejected_references.csv", reference_rejections)
    write_csv(args.output_dir / "source_registration_attempts.csv", source_attempts)
    summary = {
        "groups_requested": args.groups,
        "groups_accepted": len(groups),
        "unique_sequences": len({group["sequence"] for group in groups}),
        "views": 3 * len(groups),
        "sequences_seen_in_pilot8": sum(bool(group["sequence_seen_in_pilot8"]) for group in groups),
        "reference_attempts_rejected": len(reference_rejections),
        "source_registration_attempts": len(source_attempts),
        "source_rejection_reasons": Counter(row["reason"] for row in source_attempts if not row["accepted"]),
    }
    summary["source_rejection_reasons"] = dict(summary["source_rejection_reasons"])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if len(groups) < args.minimum_groups:
        raise RuntimeError(
            f"Only {len(groups)} registration-qualified groups found; "
            f"minimum required is {args.minimum_groups}"
        )


if __name__ == "__main__":
    main()
