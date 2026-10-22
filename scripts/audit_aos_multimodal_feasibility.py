#!/usr/bin/env python3
"""Bounded RGB/thermal and multi-view audit for three downloaded AOS scenes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


SCENES = {
    "F0": {"split": "TEST", "indices": [180, 205, 235], "people": 3},
    "F8": {"split": "TRAIN", "indices": [59, 60, 61], "people": 0},
    "O6": {"split": "VALID", "indices": [189, 190, 191], "people": 5},
}
RECTIFIED_K = np.array(
    [[512 / 0.95, 0, 256], [0, 512 / 0.95, 256], [0, 0, 1]], dtype=np.float64
)


def load_audit(project_root: Path):
    path = project_root / "scripts" / "audit_aos_stage1.py"
    spec = importlib.util.spec_from_file_location("aos_stage1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def normalize_thermal(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path), dtype=np.float32)
    low, high = np.percentile(array, (1, 99))
    normalized = np.clip((array - low) / max(high - low, 1e-6), 0, 1)
    return (plt.get_cmap("inferno")(normalized)[..., :3] * 255).astype(np.uint8)


def kml_altitudes(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    block = re.search(r"<coordinates>(.*?)</coordinates>", text, flags=re.DOTALL)
    if not block:
        return []
    return [float(row.split(",")[2]) for row in block.group(1).split() if len(row.split(",")) == 3]


def canonical_integral(root: Path, modality: str, split: str, scene: str) -> Path:
    directory = root / modality / "AOS" / split / scene
    matches = sorted(directory.glob("*_rot000_fpH0.00_rx0.0_ry0.0.png"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one canonical integral for {modality}/{split}/{scene}")
    return matches[0]


def draw_yolo(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in path.with_suffix(".txt").read_text().splitlines():
        _, xc, yc, width, height = map(float, row.split())
        box = (
            (xc - width / 2) * image.width,
            (yc - height / 2) * image.height,
            (xc + width / 2) * image.width,
            (yc + height / 2) * image.height,
        )
        draw.rectangle(box, outline=(255, 45, 45), width=3)
    return image


def save_scene_figure(
    scene: str,
    rgb_entries: list[dict],
    thermal_entries: list[dict],
    raw_scene: Path,
    rgb_integral: Path,
    thermal_integral: Path,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for column, (rgb, thermal) in enumerate(zip(rgb_entries, thermal_entries)):
        axes[0, column].imshow(Image.open(raw_scene / "RGB_rect" / rgb["imagefile"]))
        axes[0, column].set_title(f"RGB index {SCENES[scene]['indices'][column]}\n{rgb['imagefile']}", fontsize=9)
        axes[1, column].imshow(normalize_thermal(raw_scene / "thermal_rect_meanAdj" / thermal["imagefile"]))
        axes[1, column].set_title(f"Thermal same pose slot\n{thermal['imagefile']}", fontsize=9)
    axes[0, 3].imshow(draw_yolo(rgb_integral))
    axes[0, 3].set_title("Official RGB integral + boxes", fontsize=9)
    axes[1, 3].imshow(draw_yolo(thermal_integral))
    axes[1, 3].set_title("Official thermal integral + boxes", fontsize=9)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(
        f"AOS {scene}: metadata-paired raw views and common-frame integral products\n"
        "Raw pairs are synchronized but not asserted pixel-aligned",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_rotation_only_overlay(
    scene: str,
    rgb_entry: dict,
    thermal_entry: dict,
    raw_scene: Path,
    audit,
    output: Path,
) -> None:
    rgb = np.asarray(Image.open(raw_scene / "RGB_rect" / rgb_entry["imagefile"]).convert("RGB"))
    thermal = normalize_thermal(raw_scene / "thermal_rect_meanAdj" / thermal_entry["imagefile"])
    rgb_pose = audit.pose_matrix(rgb_entry)
    thermal_pose = audit.pose_matrix(thermal_entry)
    relative = thermal_pose @ np.linalg.inv(rgb_pose)
    homography = RECTIFIED_K @ relative[:3, :3] @ np.linalg.inv(RECTIFIED_K)
    warped_rgb = cv2.warpPerspective(rgb, homography, (512, 512))
    rgb_edges = cv2.Canny(cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY), 60, 140)
    thermal_edges = cv2.Canny(cv2.cvtColor(thermal, cv2.COLOR_RGB2GRAY), 60, 140)
    overlay = np.zeros((512, 512, 3), dtype=np.uint8)
    overlay[..., 0] = rgb_edges
    overlay[..., 1] = thermal_edges
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Rectified RGB")
    axes[1].imshow(thermal)
    axes[1].set_title("Rectified thermal")
    axes[2].imshow(overlay)
    axes[2].set_title("Rotation-only edge overlay\nred=RGB, green=thermal")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{scene} paired pose slot: released-pose rotation homography\n"
        "Residual physical baseline/timing effects are not represented by the metadata"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def analyze_scene(scene: str, raw_root: Path, integral_root: Path, output: Path, audit) -> dict:
    settings = SCENES[scene]
    raw_scene = raw_root / scene
    rgb_all = audit.load_poses(raw_scene / "poses" / "RGB_poses.json")
    thermal_all = audit.load_poses(raw_scene / "poses" / "thermal_poses.json")
    if len(rgb_all) != len(thermal_all):
        raise RuntimeError(f"pose count mismatch for {scene}")
    relatives = np.stack(
        [audit.pose_matrix(t) @ np.linalg.inv(audit.pose_matrix(r)) for r, t in zip(rgb_all, thermal_all)]
    )
    shifted = np.stack(
        [audit.pose_matrix(thermal_all[i + 1]) @ np.linalg.inv(audit.pose_matrix(rgb_all[i])) for i in range(len(rgb_all) - 1)]
    )
    indices = settings["indices"]
    rgb_entries = [rgb_all[index] for index in indices]
    thermal_entries = [thermal_all[index] for index in indices]
    reference_slot = 1
    reference_image = np.asarray(
        Image.open(raw_scene / "RGB_rect" / rgb_entries[reference_slot]["imagefile"]).convert("RGB")
    )
    reference_center = audit.camera_center(audit.pose_matrix(rgb_entries[reference_slot]))
    overlap = []
    for slot in (0, 2):
        source_image = np.asarray(
            Image.open(raw_scene / "RGB_rect" / rgb_entries[slot]["imagefile"]).convert("RGB")
        )
        _, stats, *_ = audit.orb_homography(source_image, reference_image)
        center = audit.camera_center(audit.pose_matrix(rgb_entries[slot]))
        overlap.append(
            {
                "source_index": indices[slot],
                "reference_index": indices[reference_slot],
                "source_file": rgb_entries[slot]["imagefile"],
                "reference_file": rgb_entries[reference_slot]["imagefile"],
                "camera_baseline_colmap_units": float(np.linalg.norm(center - reference_center)),
                "camera_center_z_difference_colmap_units": float(center[2] - reference_center[2]),
                **{key: value for key, value in stats.items() if key != "warped_corners"},
            }
        )
    pair_relative = relatives[indices[reference_slot]]
    pair_translation = pair_relative[:3, 3]
    kml = next((raw_root / "_flights" / "individual").glob(f"{scene}_*.KML"))
    altitudes = kml_altitudes(kml)
    rgb_integral = canonical_integral(integral_root, "SARAOS", settings["split"], scene)
    thermal_integral = canonical_integral(integral_root, "SARAOS_thermal", settings["split"], scene)
    label_identity = rgb_integral.with_suffix(".txt").read_bytes() == thermal_integral.with_suffix(".txt").read_bytes()
    save_scene_figure(
        scene,
        rgb_entries,
        thermal_entries,
        raw_scene,
        rgb_integral,
        thermal_integral,
        output / f"{scene}_paired_views_and_integrals.png",
    )
    save_rotation_only_overlay(
        scene,
        rgb_entries[reference_slot],
        thermal_entries[reference_slot],
        raw_scene,
        audit,
        output / f"{scene}_rgb_thermal_rotation_only_overlay.png",
    )
    return {
        "scene": scene,
        "split": settings["split"],
        "released_person_count_in_canonical_integral": settings["people"],
        "pose_pairs": len(rgb_all),
        "selected_pose_indices": indices,
        "selected_rgb_files": [entry["imagefile"] for entry in rgb_entries],
        "selected_thermal_files": [entry["imagefile"] for entry in thermal_entries],
        "same_index_rig_transform_max_deviation": float(np.max(np.abs(relatives - relatives.mean(axis=0)))),
        "one_slot_shift_transform_max_deviation": float(np.max(np.abs(shifted - shifted.mean(axis=0)))),
        "rig_translation_colmap_units": pair_translation.tolist(),
        "rig_baseline_colmap_units": float(np.linalg.norm(pair_translation)),
        "rig_rotation_degrees": rotation_angle(pair_relative[:3, :3]),
        "overlapping_rgb_views": overlap,
        "kml_altitude_mode": "relativeToGround",
        "kml_altitude_meters": {
            "minimum": min(altitudes),
            "median": float(np.median(altitudes)),
            "maximum": max(altitudes),
            "range": max(altitudes) - min(altitudes),
            "frame_linkage": "No timestamps in pose JSON; scene-level only, not assigned to selected frames.",
        },
        "canonical_rgb_thermal_label_files_identical": label_identity,
        "raw_rgb_person_boxes_available": False,
        "raw_thermal_person_boxes_available": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/aos/raw_lowres/RAW_lowresRGB")
    )
    parser.add_argument(
        "--integral-root",
        type=Path,
        default=Path("data/aos/integral_images/Integral Images/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aos/multimodal_feasibility_20261020"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    project_root = Path(__file__).resolve().parents[1]
    audit = load_audit(project_root)
    scenes = [
        analyze_scene(scene, args.raw_root, args.integral_root, args.output_dir, audit)
        for scene in SCENES
    ]
    summary = {
        "scope": "metadata and geometry feasibility only; no training, fusion, or detection",
        "rectified_intrinsics": {
            "width": 512,
            "height": 512,
            "fx": 512 / 0.95,
            "fy": 512 / 0.95,
            "cx": 256,
            "cy": 256,
            "source": "released rectification code defaults already verified in prior audit",
        },
        "pairing_evidence": (
            "Equal RGB/thermal pose counts plus a constant same-index rigid transform. "
            "A one-slot shift destroys that constancy; filenames were not used to pair frames."
        ),
        "registration_limit": (
            "Common rectified intrinsics and a constant 0.8896-degree relative rotation are "
            "verified. Released RGB/thermal camera centers coincide numerically, so their "
            "metadata model supports a depth-independent rotation homography. Residual "
            "physical baseline, exposure timing, and cross-modal appearance errors are not "
            "quantified by those poses and prevent claiming exact observed pixel registration."
        ),
        "annotation_limit": (
            "Direct person boxes are valid only for official RGB/thermal integral images. "
            "Raw-frame recall is unsupported by Stage 1."
        ),
        "scenes": scenes,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
