"""Select registration-qualified ForestPersons multi-view groups.

This preparation step requires OpenCV. It uses ground-truth boxes only to
verify that a proposed source/reference pair depicts the same annotated target
region; the saved homographies themselves are estimated from image features.
Consequently the resulting pilot is exploratory and selection-biased, not an
independent benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/val.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--registration-scale", type=float, default=0.5)
    return parser.parse_args()


def frame_number(file_name: str) -> int | None:
    match = re.search(r"-(\d+)\.[^.]+$", file_name)
    return int(match.group(1)) if match else None


def bbox_xyxy(annotation: dict[str, Any]) -> list[float]:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    return [x, y, x + width, y + height]


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def project_box(annotation: dict[str, Any], homography: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy(annotation)
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return [
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    ]


def estimate_homography(
    source_path: Path,
    reference_path: Path,
    scale: float,
) -> tuple[np.ndarray, dict[str, float | int]] | None:
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    if source is None or reference is None:
        return None
    source_small = cv2.resize(source, None, fx=scale, fy=scale)
    reference_small = cv2.resize(reference, None, fx=scale, fy=scale)
    detector = cv2.ORB_create(nfeatures=3500, fastThreshold=10)
    source_keypoints, source_descriptors = detector.detectAndCompute(source_small, None)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_small, None)
    if source_descriptors is None or reference_descriptors is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(source_descriptors, reference_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 20:
        return None
    source_points = np.float32([source_keypoints[item.queryIdx].pt for item in good])
    reference_points = np.float32([reference_keypoints[item.trainIdx].pt for item in good])
    homography_small, mask = cv2.findHomography(source_points, reference_points, cv2.RANSAC, 3.0)
    if homography_small is None or mask is None:
        return None
    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    if inliers < 4 or not np.isfinite(homography_small).all():
        return None
    projected_raw = cv2.perspectiveTransform(
        source_points[inlier_mask].reshape(-1, 1, 2), homography_small
    )
    if projected_raw is None or not np.isfinite(projected_raw).all():
        return None
    projected = projected_raw.reshape(-1, 2)
    errors = np.linalg.norm(projected - reference_points[inlier_mask], axis=1)
    down = np.diag([scale, scale, 1.0])
    up = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    homography = up @ homography_small @ down
    return homography, {
        "ratio_test_matches": len(good),
        "ransac_inliers": inliers,
        "inlier_ratio": inliers / len(good),
        "median_reprojection_error_registration_pixels": float(np.median(errors)),
        "median_reprojection_error_original_pixels": float(np.median(errors) / scale),
    }


def excluded_val_ids(subset_root: Path) -> set[int]:
    excluded: set[int] = set()
    for path in subset_root.glob("*val*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("images", data) if isinstance(data, dict) else data
        for item in items:
            if isinstance(item, dict) and "image_id" in item:
                excluded.add(int(item["image_id"]))
    return excluded


def draw_box(image: np.ndarray, annotation: dict[str, Any], label: str) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy(annotation)]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)
    cv2.putText(image, label, (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def save_gallery(group: dict[str, Any], image_root: Path, output_path: Path) -> None:
    reference = cv2.imread(str(image_root / group["reference"]["file_name"]))
    panels = []
    ref_drawn = reference.copy()
    draw_box(ref_drawn, group["reference"]["annotation"], f"reference v{group['reference']['visibility']}")
    panels.append(ref_drawn)
    for source in group["sources"]:
        image = cv2.imread(str(image_root / source["file_name"]))
        drawn = image.copy()
        draw_box(drawn, source["annotation"], f"source v{source['visibility']}")
        panels.append(drawn)
        homography = np.asarray(source["homography_source_to_reference"], dtype=np.float64)
        warped = cv2.warpPerspective(image, homography, (reference.shape[1], reference.shape[0]))
        overlay = cv2.addWeighted(reference, 0.5, warped, 0.5, 0.0)
        cv2.putText(overlay, "reference/source alignment overlay", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        panels.append(overlay)
    target_width = 640
    resized = [cv2.resize(panel, (target_width, int(panel.shape[0] * target_width / panel.shape[1]))) for panel in panels]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(resized))


def main() -> None:
    args = parse_args()
    if args.views < 2:
        raise ValueError("Multi-view groups require at least two views")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    sequences: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for image in data["images"]:
        frame = frame_number(image["file_name"])
        if frame is not None:
            sequences[image["file_name"].split("/")[0]][frame] = image

    excluded = excluded_val_ids(Path("outputs/subsets"))
    candidates = []
    for sequence, frames in sequences.items():
        for frame, image in frames.items():
            annotations = annotations_by_image[int(image["id"])]
            if len(annotations) != 1 or int(image["id"]) in excluded:
                continue
            attributes = annotations[0]["attributes"]
            if int(attributes["visibility_level"]) not in (20, 40):
                continue
            candidates.append((sequence, frame, image, annotations[0]))
    random.Random(args.seed).shuffle(candidates)

    groups: list[dict[str, Any]] = []
    used_sequences: set[str] = set()
    offsets = (-10, -8, -5, -3, 3, 5, 8, 10)
    image_root = args.root / "images"
    for sequence, reference_frame, reference_image, reference_annotation in candidates:
        if sequence in used_sequences:
            continue
        reference_path = image_root / reference_image["file_name"]
        if not reference_path.exists():
            continue
        accepted_sources = []
        for offset in offsets:
            source_image = sequences[sequence].get(reference_frame + offset)
            if source_image is None or int(source_image["id"]) in excluded:
                continue
            source_annotations = annotations_by_image[int(source_image["id"])]
            if len(source_annotations) != 1:
                continue
            source_annotation = source_annotations[0]
            if source_annotation["attributes"]["pose"] != reference_annotation["attributes"]["pose"]:
                continue
            source_path = image_root / source_image["file_name"]
            if not source_path.exists():
                continue
            registration = estimate_homography(source_path, reference_path, args.registration_scale)
            if registration is None:
                continue
            homography, metrics = registration
            projected = project_box(source_annotation, homography)
            projected_iou = box_iou(projected, bbox_xyxy(reference_annotation))
            reference_box = bbox_xyxy(reference_annotation)
            reference_diagonal = math.hypot(reference_box[2] - reference_box[0], reference_box[3] - reference_box[1])
            projected_center = ((projected[0] + projected[2]) * 0.5, (projected[1] + projected[3]) * 0.5)
            reference_center = ((reference_box[0] + reference_box[2]) * 0.5, (reference_box[1] + reference_box[3]) * 0.5)
            center_error = math.dist(projected_center, reference_center) / max(reference_diagonal, 1e-6)
            qualified = (
                metrics["ransac_inliers"] >= 30
                and metrics["inlier_ratio"] >= 0.5
                and metrics["median_reprojection_error_registration_pixels"] <= 2.0
                and projected_iou >= 0.5
            )
            if not qualified:
                continue
            accepted_sources.append(
                {
                    "image_id": int(source_image["id"]),
                    "file_name": source_image["file_name"],
                    "frame": reference_frame + offset,
                    "frame_offset": offset,
                    "visibility": int(source_annotation["attributes"]["visibility_level"]),
                    "pose": source_annotation["attributes"]["pose"],
                    "annotation": source_annotation,
                    "homography_source_to_reference": homography.tolist(),
                    "registration": metrics,
                    "projected_gt_iou": projected_iou,
                    "projected_gt_center_error_normalized": center_error,
                }
            )
            if len(accepted_sources) >= args.views - 1:
                break
        if len(accepted_sources) < args.views - 1:
            continue
        group = {
            "group_id": f"group_{len(groups) + 1:02d}",
            "sequence": sequence,
            "reference": {
                "image_id": int(reference_image["id"]),
                "file_name": reference_image["file_name"],
                "frame": reference_frame,
                "visibility": int(reference_annotation["attributes"]["visibility_level"]),
                "pose": reference_annotation["attributes"]["pose"],
                "annotation": reference_annotation,
            },
            "sources": accepted_sources[: args.views - 1],
        }
        groups.append(group)
        used_sequences.add(sequence)
        save_gallery(group, image_root, args.output_dir / "alignment_gallery" / f"{group['group_id']}.jpg")
        if len(groups) >= args.groups:
            break
    if len(groups) < args.groups:
        raise RuntimeError(f"Only {len(groups)} qualified groups found; requested {args.groups}")

    manifest = {
        "dataset": "ForestPersons official validation split",
        "purpose": "Exploratory DINOv3 aligned feature-fusion pilot",
        "seed": args.seed,
        "groups_requested": args.groups,
        "views_per_group": args.views,
        "selection": {
            "reference_visibility": [20, 40],
            "one_annotated_person_per_frame": True,
            "same_sequence_and_pose": True,
            "source_offsets_considered": list(offsets),
            "excluded_prior_val_subset_images": len(excluded),
            "registration": "ORB ratio-test matches plus RANSAC homography, image-only",
            "minimum_ransac_inliers": 30,
            "minimum_inlier_ratio": 0.5,
            "maximum_median_reprojection_error_registration_pixels": 2.0,
            "minimum_projected_gt_iou": 0.5,
            "warning": "GT is used to qualify correspondence, so detection metrics are exploratory and selection-biased.",
        },
        "groups": groups,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "groups": len(groups),
                "views": sum(1 + len(group["sources"]) for group in groups),
                "reference_visibility_counts": {
                    str(value): sum(group["reference"]["visibility"] == value for group in groups)
                    for value in (20, 40)
                },
                "unique_sequences": len({group["sequence"] for group in groups}),
                "excluded_prior_subset_images": len(excluded),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "groups": len(groups)}, indent=2))


if __name__ == "__main__":
    main()
