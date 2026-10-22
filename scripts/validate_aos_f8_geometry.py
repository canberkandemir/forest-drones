#!/usr/bin/env python3
"""Validate released F8 poses and three-view image overlap without rendering AOS."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_audit_module(project_root: Path):
    path = project_root / "scripts" / "audit_aos_stage1.py"
    spec = importlib.util.spec_from_file_location("aos_stage1_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def colmap_images(path: Path) -> dict[str, np.ndarray]:
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 10 and fields[-1].lower().endswith((".jpg", ".jpeg", ".png")):
            qw, qx, qy, qz = map(float, fields[1:5])
            translation = np.asarray(list(map(float, fields[5:8])))
            matrix = np.concatenate((quaternion_rotation(qw, qx, qy, qz), translation[:, None]), axis=1)
            result[fields[-1]] = matrix
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    audit = load_audit_module(project_root)
    poses = audit.load_poses(args.scene_root / "poses" / "RGB_poses.json")
    colmap = colmap_images(args.scene_root / "poses" / "COLMAP" / "images.txt")
    indices = [59, 60, 61]
    reference_slot = 1
    entries = [poses[index] for index in indices]
    images = [np.asarray(Image.open(args.scene_root / "RGB_rect" / e["imagefile"]).convert("RGB")) for e in entries]
    reference = images[reference_slot]
    reference_pose = audit.pose_matrix(entries[reference_slot])
    reference_center = audit.camera_center(reference_pose)
    pairs = []
    overlays = []
    for slot in (0, 2):
        homography, stats, *_ = audit.orb_homography(images[slot], reference)
        if homography is None or not stats.get("registration_reliable", False):
            raise RuntimeError(f"F8 pair failed registration: {entries[slot]['imagefile']} {stats}")
        pose = audit.pose_matrix(entries[slot])
        center = audit.camera_center(pose)
        warped = cv2.warpPerspective(images[slot], homography, (512, 512))
        overlay = cv2.addWeighted(reference, 0.5, warped, 0.5, 0)
        overlays.append(overlay)
        pairs.append({
            "source": entries[slot]["imagefile"],
            "reference": entries[reference_slot]["imagefile"],
            "camera_baseline_colmap_units": float(np.linalg.norm(center - reference_center)),
            **stats,
        })

    pose_differences = {}
    for entry in entries:
        released = np.asarray(entry["M3x4"], dtype=np.float64)
        pose_differences[entry["imagefile"]] = float(np.max(np.abs(released - colmap[entry["imagefile"]])))

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis, image, entry in zip(axes[0], images, entries):
        axis.imshow(image)
        axis.set_title(entry["imagefile"])
        axis.axis("off")
    axes[1, 0].imshow(overlays[0])
    axes[1, 0].set_title(f"{entries[0]['imagefile']} warped to reference")
    axes[1, 1].imshow(overlays[1])
    axes[1, 1].set_title(f"{entries[2]['imagefile']} warped to reference")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.95,
        "Image-plane ORB/RANSAC only\n"
        f"left: {pairs[0]['ransac_inliers']} inliers, {pairs[0]['median_reprojection_error_px']:.2f}px\n"
        f"right: {pairs[1]['ransac_inliers']} inliers, {pairs[1]['median_reprojection_error_px']:.2f}px\n\n"
        "Not an official virtual-camera\nor terrain reprojection.",
        va="top", fontsize=11,
    )
    fig.suptitle("AOS F8 training scene: three-view overlap validation")
    fig.tight_layout()
    figure_path = args.output_dir / "f8_three_view_alignment.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    summary = {
        "scene": "F8", "split": "TRAIN", "indices": indices,
        "filenames": [entry["imagefile"] for entry in entries],
        "released_rectified_intrinsics": {
            "width": 512, "height": 512, "fx": 512 / 0.95, "fy": 512 / 0.95,
            "cx": 256.0, "cy": 256.0, "fov_degrees": 50.815436217896945,
        },
        "pose_convention": "released M3x4 equals COLMAP world-to-camera [R|t]",
        "max_pose_difference_by_file": pose_differences,
        "pairs": pairs,
        "terrain_files_in_downloaded_stage1": 0,
        "official_projection_reproduced": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
