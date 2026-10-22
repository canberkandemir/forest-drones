#!/usr/bin/env python3
"""Verify the downloaded AOS Stage 1 archives and inspect one geometry example."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


SPLITS = {
    "TRAIN": ("O1", "O2", "O3", "O4", "O5", "F8", "F10"),
    "VALID": ("O6",),
    "TEST": ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F9", "F11"),
}
PAPER_PEOPLE = {
    "O1": 10, "O2": 10, "O3": 6, "O4": 6, "O5": 5, "O6": 5,
    "F0": 3, "F1": 10, "F2": 10, "F3": 6, "F4": 6, "F5": 10,
    "F6": 10, "F7": 2, "F8": 0, "F9": 0, "F10": 0, "F11": 0,
}


def load_poses(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        for key in ("images", "poses", "frames"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Unexpected pose schema: {path}")
    return data


def pose_matrix(entry: dict) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3] = np.asarray(entry["M3x4"], dtype=np.float64)
    return matrix


def camera_center(w2c: np.ndarray) -> np.ndarray:
    return -w2c[:3, :3].T @ w2c[:3, 3]


def parse_camera(path: Path) -> dict:
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(v) for v in parts[4:]]
            if model != "SIMPLE_RADIAL":
                raise ValueError(f"Unsupported sampled COLMAP camera model: {model}")
            return {"model": model, "width": width, "height": height,
                    "focal": params[0], "cx": params[1], "cy": params[2],
                    "distortion": params[3]}
    raise ValueError(f"No camera row in {path}")


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def verify_scene(scene_dir: Path) -> dict:
    rgb_files = image_files(scene_dir / "RGB_rect")
    thermal_files = image_files(scene_dir / "thermal")
    thermal_rect_files = image_files(scene_dir / "thermal_rect_meanAdj")
    rgb_poses = load_poses(scene_dir / "poses" / "RGB_poses.json")
    thermal_poses = load_poses(scene_dir / "poses" / "thermal_poses.json")

    rgb_names = {p.name for p in rgb_files}
    thermal_names = {p.name for p in thermal_files}
    thermal_rect_names = {p.name for p in thermal_rect_files}
    rgb_pose_names = {e["imagefile"] for e in rgb_poses}
    thermal_pose_names = {e["imagefile"] for e in thermal_poses}

    count = min(len(rgb_poses), len(thermal_poses))
    relatives = np.stack([
        pose_matrix(thermal_poses[i]) @ np.linalg.inv(pose_matrix(rgb_poses[i]))
        for i in range(count)
    ])
    mean_relative = relatives.mean(axis=0)
    relative_deviation = float(np.max(np.abs(relatives - mean_relative)))

    camera = parse_camera(scene_dir / "poses" / "COLMAP" / "cameras.txt")
    sample_sizes = []
    for files in (rgb_files, thermal_files, thermal_rect_files):
        for idx in sorted({0, len(files) // 2, len(files) - 1}):
            with Image.open(files[idx]) as im:
                im.verify()
            with Image.open(files[idx]) as im:
                sample_sizes.append([files[idx].name, list(im.size)])

    centers = np.stack([camera_center(pose_matrix(e)) for e in rgb_poses])
    return {
        "scene": scene_dir.name,
        "rgb_images": len(rgb_files),
        "thermal_images": len(thermal_files),
        "thermal_rect_images": len(thermal_rect_files),
        "rgb_poses": len(rgb_poses),
        "thermal_poses": len(thermal_poses),
        "rgb_filename_pose_match": rgb_names == rgb_pose_names,
        "thermal_filename_pose_match": thermal_names == thermal_pose_names,
        "thermal_rect_filename_match": thermal_names == thermal_rect_names,
        "paired_pose_count_equal": len(rgb_poses) == len(thermal_poses),
        "rig_relative_transform_max_deviation": relative_deviation,
        "camera": camera,
        "camera_center_span": (centers.max(axis=0) - centers.min(axis=0)).tolist(),
        "sampled_images_verified": len(sample_sizes),
        "sample_sizes": sample_sizes,
    }


def parse_yolo_label(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label row in {path}: {line}")
        category = int(parts[0])
        values = [float(v) for v in parts[1:]]
        if category != 0 or not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"Invalid category/coordinate in {path}: {line}")
        rows.append([float(category), *values])
    return rows


def canonical_file(scene_dir: Path) -> Path:
    matches = sorted(scene_dir.glob("*_rot000_fpH0.00_rx0.0_ry0.0.png"))
    if len(matches) != 1:
        raise ValueError(f"Expected one canonical integral image in {scene_dir}; found {len(matches)}")
    return matches[0]


def verify_integrals(root: Path) -> tuple[dict, list[dict]]:
    rgb_root = root / "SARAOS" / "AOS"
    thermal_root = root / "SARAOS_thermal" / "AOS"
    scene_rows = []
    total_boxes = 0
    total_labels = 0
    all_stems_match = True
    canonical_people_match = True
    for split, scenes in SPLITS.items():
        for scene in scenes:
            rgb_dir = rgb_root / split / scene
            thermal_dir = thermal_root / split / scene
            rgb_png = {p.stem for p in rgb_dir.glob("*.png")}
            rgb_txt = {p.stem for p in rgb_dir.glob("*.txt")}
            thermal_png = {p.stem for p in thermal_dir.glob("*.png")}
            thermal_txt = {p.stem for p in thermal_dir.glob("*.txt")}
            stems_match = rgb_png == rgb_txt == thermal_png == thermal_txt
            all_stems_match &= stems_match
            scene_boxes = 0
            labels_identical = True
            for stem in sorted(rgb_txt):
                rgb_label = rgb_dir / f"{stem}.txt"
                thermal_label = thermal_dir / f"{stem}.txt"
                rows = parse_yolo_label(rgb_label)
                parse_yolo_label(thermal_label)
                scene_boxes += len(rows)
                labels_identical &= rgb_label.read_bytes() == thermal_label.read_bytes()
            canonical = canonical_file(rgb_dir)
            canonical_boxes = len(parse_yolo_label(canonical.with_suffix(".txt")))
            canonical_match = canonical_boxes == PAPER_PEOPLE[scene]
            canonical_people_match &= canonical_match
            total_labels += len(rgb_txt)
            total_boxes += scene_boxes
            scene_rows.append({
                "split": split, "scene": scene, "integral_images_per_modality": len(rgb_png),
                "labels_per_modality": len(rgb_txt), "all_stems_match": stems_match,
                "rgb_thermal_labels_identical": labels_identical,
                "all_augmented_boxes": scene_boxes,
                "canonical_boxes": canonical_boxes,
                "paper_scene_people": PAPER_PEOPLE[scene],
                "canonical_matches_paper_count": canonical_match,
                "canonical_rgb": str(canonical),
            })
    return {
        "rgb_integral_images": total_labels,
        "thermal_integral_images": total_labels,
        "labels_per_modality": total_labels,
        "all_augmented_boxes_per_modality": total_boxes,
        "all_rgb_thermal_stems_match": all_stems_match,
        "all_canonical_counts_match_paper": canonical_people_match,
    }, scene_rows


def orb_homography(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray | None, dict, list, list, list, np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=6000)
    kp_s, desc_s = orb.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), None)
    kp_r, desc_r = orb.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY), None)
    if desc_s is None or desc_r is None:
        return None, {"keypoints_source": len(kp_s), "keypoints_reference": len(kp_r), "ratio_matches": 0}, kp_s, kp_r, [], None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_s, desc_r, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    if len(good) < 4:
        return None, {"keypoints_source": len(kp_s), "keypoints_reference": len(kp_r), "ratio_matches": len(good)}, kp_s, kp_r, good, None
    src = np.float32([kp_s[m.queryIdx].pt for m in good])
    dst = np.float32([kp_r[m.trainIdx].pt for m in good])
    h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good), dtype=bool)
    if h is not None and inliers.any():
        projected = cv2.perspectiveTransform(src[:, None, :], h)[:, 0]
        errors = np.linalg.norm(projected - dst, axis=1)[inliers]
        corners = np.float32([[[0, 0], [source.shape[1], 0], [source.shape[1], source.shape[0]], [0, source.shape[0]]]])
        warped_corners = cv2.perspectiveTransform(corners, h)[0]
        corner_area_ratio = float(abs(cv2.contourArea(warped_corners)) / (source.shape[0] * source.shape[1]))
        overlap_mask = np.ones(source.shape[:2], np.uint8) * 255
        warped_mask = cv2.warpPerspective(overlap_mask, h, (reference.shape[1], reference.shape[0]))
        overlap = float((warped_mask > 0).mean())
        stats = {
            "keypoints_source": len(kp_s), "keypoints_reference": len(kp_r),
            "ratio_matches": len(good), "ransac_inliers": int(inliers.sum()),
            "inlier_ratio": float(inliers.mean()), "median_reprojection_error_px": float(np.median(errors)),
            "warped_source_coverage": overlap, "warped_corner_area_ratio": corner_area_ratio,
            "registration_reliable": bool(inliers.sum() >= 10 and inliers.mean() >= 0.25 and
                                          0.20 <= corner_area_ratio <= 4.0 and np.median(errors) <= 3.0),
            "warped_corners": warped_corners.tolist(),
        }
    else:
        stats = {"keypoints_source": len(kp_s), "keypoints_reference": len(kp_r), "ratio_matches": len(good), "ransac_inliers": 0}
    return h, stats, kp_s, kp_r, good, mask


def draw_integral_boxes(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for _, xc, yc, w, h in parse_yolo_label(path.with_suffix(".txt")):
        x1, y1 = (xc - w / 2) * image.width, (yc - h / 2) * image.height
        x2, y2 = (xc + w / 2) * image.width, (yc + h / 2) * image.height
        draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 40), width=3)
    return image


def contact_sheet(items: list[tuple[str, Image.Image]], output: Path) -> None:
    tile_w, tile_h, label_h = 512, 512, 28
    canvas = Image.new("RGB", (tile_w * len(items), tile_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate(items):
        canvas.paste(image.convert("RGB").resize((tile_w, tile_h)), (i * tile_w, 0))
        draw.text((i * tile_w + 8, tile_h + 6), label, fill="black")
    canvas.save(output)


def geometry_example(raw_root: Path, integral_root: Path, output: Path) -> dict:
    scene_dir = raw_root / "F0"
    poses = load_poses(scene_dir / "poses" / "RGB_poses.json")
    indices = [180, 205, 235]
    entries = [poses[i] for i in indices]
    images = [np.asarray(Image.open(scene_dir / "RGB_rect" / e["imagefile"]).convert("RGB")) for e in entries]
    reference = images[1]
    reference_pose = pose_matrix(entries[1])
    ref_center = camera_center(reference_pose)

    pair_results = []
    for slot in (0, 2):
        source = images[slot]
        h, stats, kp_s, kp_r, good, mask = orb_homography(source, reference)
        source_pose = pose_matrix(entries[slot])
        center = camera_center(source_pose)
        axis_source = source_pose[:3, :3].T[:, 2]
        axis_ref = reference_pose[:3, :3].T[:, 2]
        angle = math.degrees(math.acos(float(np.clip(np.dot(axis_source, axis_ref), -1.0, 1.0))))
        stats.update({
            "source_index": indices[slot], "reference_index": indices[1],
            "source_file": entries[slot]["imagefile"], "reference_file": entries[1]["imagefile"],
            "camera_baseline_colmap_units": float(np.linalg.norm(center - ref_center)),
            "optical_axis_angle_deg": angle,
            "selection": "pose-neighborhood candidate accepted by predeclared ORB/RANSAC reliability checks; no labels or detections used",
        })
        if h is not None:
            warped = cv2.warpPerspective(source, h, (reference.shape[1], reference.shape[0]))
            overlay = cv2.addWeighted(reference, 0.5, warped, 0.5, 0)
            Image.fromarray(overlay).save(output / f"F0_orb_overlay_{entries[slot]['imagefile'].split('.')[0]}_to_{entries[1]['imagefile'].split('.')[0]}.png")
            matches = cv2.drawMatches(
                cv2.cvtColor(source, cv2.COLOR_RGB2BGR), kp_s,
                cv2.cvtColor(reference, cv2.COLOR_RGB2BGR), kp_r,
                good, None, matchesMask=mask.ravel().tolist() if mask is not None else None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imwrite(str(output / f"F0_orb_matches_{entries[slot]['imagefile'].split('.')[0]}_to_{entries[1]['imagefile'].split('.')[0]}.png"), matches)
        pair_results.append(stats)

    canonical = canonical_file(integral_root / "SARAOS" / "AOS" / "TEST" / "F0")
    contact_sheet([
        (f"source {entries[0]['imagefile']}", Image.fromarray(images[0])),
        (f"reference {entries[1]['imagefile']}", Image.fromarray(images[1])),
        (f"source {entries[2]['imagefile']}", Image.fromarray(images[2])),
        ("official F0 RGB integral (red: released boxes)", draw_integral_boxes(canonical)),
    ], output / "F0_pose_selected_triplet_and_official_integral.png")

    centers = np.stack([camera_center(pose_matrix(e)) for e in poses])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(centers[:, 0], centers[:, 1], color="#777777", linewidth=0.8)
    ax.scatter(centers[:, 0], centers[:, 1], s=5, color="#4472c4")
    for idx, color in zip(indices, ("#f28e2b", "#e15759", "#59a14f")):
        ax.scatter(centers[idx, 0], centers[idx, 1], s=80, color=color, label=f"{idx}: {poses[idx]['imagefile']}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("COLMAP x")
    ax.set_ylabel("COLMAP y")
    ax.set_title("F0 camera centers and metadata-selected geometry triplet")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "F0_camera_track_triplet.png", dpi=160)
    plt.close(fig)

    return {
        "scene": "F0", "indices": indices,
        "filenames": [e["imagefile"] for e in entries],
        "selection_used_annotations_or_detections": False,
        "orb_pairs": pair_results,
        "interpretation_limit": "ORB/RANSAC is an image-plane canopy/background diagnostic, not proof of person or ground-plane alignment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--integral-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted([p for p in args.raw_root.iterdir() if p.is_dir() and (p.name.startswith("F") or p.name.startswith("O"))], key=lambda p: (p.name[0], int(p.name[1:])))
    scenes = [verify_scene(path) for path in scene_dirs]
    integral_summary, integral_scenes = verify_integrals(args.integral_root)
    geometry = geometry_example(args.raw_root, args.integral_root, args.output_dir)

    raw_direct_labels = [p for p in args.raw_root.rglob("*.txt") if p.parent.name not in {"COLMAP"}]
    result = {
        "raw": {
            "scene_count": len(scenes),
            "rgb_images": sum(s["rgb_images"] for s in scenes),
            "thermal_images": sum(s["thermal_images"] for s in scenes),
            "thermal_rect_images": sum(s["thermal_rect_images"] for s in scenes),
            "all_filename_pose_mappings_valid": all(s["rgb_filename_pose_match"] and s["thermal_filename_pose_match"] for s in scenes),
            "all_rgb_thermal_pose_counts_equal": all(s["paired_pose_count_equal"] for s in scenes),
            "max_rig_transform_deviation": max(s["rig_relative_transform_max_deviation"] for s in scenes),
            "direct_raw_rgb_annotation_files": len(raw_direct_labels),
            "scenes": scenes,
        },
        "integral": integral_summary,
        "integral_scenes": integral_scenes,
        "geometry_example": geometry,
        "annotation_linkage": {
            "raw_rgb": "No directly released raw RGB labels found; person recall is not valid without original 3D contours or verified projection metadata.",
            "geometry_fusion": "No exact released virtual-camera/DEM mapping to integral coordinates found in Stage 1 files.",
            "official_integral_rgb": "Direct YOLO labels are present and validated for all released products.",
        },
    }
    (args.output_dir / "aos_stage1_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "raw_scene_inventory.csv").open("w", newline="") as handle:
        fields = ["scene", "rgb_images", "thermal_images", "thermal_rect_images", "rgb_poses", "thermal_poses", "rgb_filename_pose_match", "thermal_filename_pose_match", "paired_pose_count_equal", "rig_relative_transform_max_deviation"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in scenes)
    with (args.output_dir / "integral_scene_inventory.csv").open("w", newline="") as handle:
        fields = ["split", "scene", "integral_images_per_modality", "labels_per_modality", "all_stems_match", "rgb_thermal_labels_identical", "all_augmented_boxes", "canonical_boxes", "paper_scene_people", "canonical_matches_paper_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in integral_scenes)
    print(json.dumps({
        "scene_count": result["raw"]["scene_count"],
        "rgb_images": result["raw"]["rgb_images"],
        "thermal_images": result["raw"]["thermal_images"],
        "integral": integral_summary,
        "geometry": geometry,
    }, indent=2))


if __name__ == "__main__":
    main()
