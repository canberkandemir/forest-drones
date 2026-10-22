"""Create target-scale, localization, sequence, and curve diagnostics for SARD."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from forestpersons.detection import box_iou
from scripts.evaluate_sard_pilot import _greedy_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dhash(path: Path, size: int = 16) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((size + 1, size)))
    return pixels[:, 1:] > pixels[:, :-1]


def derive_segments(images: list[dict[str, Any]], image_root: Path) -> tuple[dict[int, int], dict[str, Any]]:
    hashes = [dhash(image_root / item["file_name"]) for item in images]
    distances = [int(np.count_nonzero(hashes[index] != hashes[index - 1])) for index in range(1, len(hashes))]
    # Large discontinuities reliably identify changes between the source clips.
    # ``index`` is already the zero-based position of the new frame because
    # distances[0] compares image 1 with image 2.
    boundary_indices = [index for index, distance in enumerate(distances, start=1) if distance >= 80]
    starts = [0, *boundary_indices]
    mapping: dict[int, int] = {}
    for segment, (start, end) in enumerate(zip(starts, [*starts[1:], len(images)]), start=1):
        for item in images[start:end]:
            mapping[int(item["image_id"])] = segment
    return mapping, {
        "method": "16x16 dHash; new segment when adjacent Hamming distance >=80/256",
        "segment_starts": [images[index]["image_id"] for index in starts],
        "segments": len(starts),
        "adjacent_hamming_min": min(distances),
        "adjacent_hamming_median": median(distances),
        "adjacent_hamming_mean": float(np.mean(distances)),
        "adjacent_hamming_max": max(distances),
    }


def load_method(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["predictions"] if isinstance(data, dict) else data


def fixed_matches(
    images: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float
) -> set[int]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[int(prediction["image_id"])].append(prediction)
    matched_ids: set[int] = set()
    for image in images:
        matched_gt, _, _ = _greedy_match(
            image["person_annotations"], grouped.get(int(image["image_id"]), []), threshold
        )
        matched_ids.update(
            int(image["person_annotations"][index]["annotation_id"]) for index in matched_gt
        )
    return matched_ids


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = json.loads((args.evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    images = manifest["images"]
    segments, sequence = derive_segments(images, args.image_root)
    methods = {
        name: load_method(args.evaluation_dir / f"{name}_predictions.json")
        for name in ("yolo", "dino", "fusion_B")
    }
    thresholds = summary["thresholds_frozen_before_sard"]
    matched = {name: fixed_matches(images, predictions, thresholds[name]) for name, predictions in methods.items()}
    by_method: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for name, predictions in methods.items():
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for prediction in predictions:
            grouped[int(prediction["image_id"])].append(prediction)
        by_method[name] = grouped

    diagnostics = []
    for image in images:
        image_id = int(image["image_id"])
        for annotation in image["person_annotations"]:
            x1, y1, x2, y2 = annotation["bbox_xyxy"]
            width, height = x2 - x1, y2 - y1
            row: dict[str, Any] = {
                "image_id": image_id,
                "annotation_id": annotation["annotation_id"],
                "derived_segment": segments[image_id],
                "gt_width_source_px": width,
                "gt_height_source_px": height,
                "gt_area_source_px": width * height,
                "gt_width_yolo640_px": width * 640 / image["width"],
                "gt_height_yolo640_px": height * 640 / image["width"],
                "gt_width_dino672_px": width * 672 / image["height"],
                "gt_height_dino672_px": height * 672 / image["height"],
            }
            for name, predictions in by_method.items():
                candidates = predictions.get(image_id, [])
                best = max(candidates, key=lambda item: box_iou(annotation["bbox_xyxy"], item["bbox"]), default=None)
                row[f"{name}_best_saved_iou"] = box_iou(annotation["bbox_xyxy"], best["bbox"]) if best else 0.0
                row[f"{name}_best_saved_score"] = float(best["fused_score"]) if best else 0.0
                row[f"{name}_detected_fixed"] = int(annotation["annotation_id"] in matched[name])
            diagnostics.append(row)
    with (args.output_dir / "per_gt_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader(); writer.writerows(diagnostics)

    segment_table = []
    for segment in sorted(set(segments.values())):
        rows = [row for row in diagnostics if row["derived_segment"] == segment]
        segment_table.append({
            "derived_segment": segment,
            "images": sum(value == segment for value in segments.values()),
            "gt": len(rows),
            **{
                f"{name}_recall": sum(row[f"{name}_detected_fixed"] for row in rows) / len(rows) if rows else None
                for name in methods
            },
        })
    with (args.output_dir / "segment_recall.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segment_table[0]))
        writer.writeheader(); writer.writerows(segment_table)

    comparison_rows = []
    for name in methods:
        result = summary["results"][name]
        fixed = result["fixed_operating_point"]
        coco = result["coco"]
        comparison_rows.append(
            {
                "method": name,
                "threshold": fixed["threshold"],
                "tp": fixed["tp"], "fp": fixed["fp"], "fn": fixed["fn"],
                "recall": fixed["recall"], "precision": fixed["precision"],
                "fp_per_image": fixed["fp_per_image"],
                "AP": coco["AP"], "AP50": coco["AP50"], "AP75": coco["AP75"],
                "AR@100": coco["AR@100"],
            }
        )
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader(); writer.writerows(comparison_rows)

    plt.figure(figsize=(7, 5))
    for name in methods:
        curve = np.genfromtxt(
            args.evaluation_dir / f"{name}_recall_fp_curve.csv", delimiter=",", names=True
        )
        order = np.argsort(curve["fp_per_image"])
        plt.plot(curve["fp_per_image"][order], curve["recall"][order], label=name)
    plt.xlim(0, 10); plt.ylim(0, 1)
    plt.xlabel("False positives per image"); plt.ylabel("Recall at IoU 0.50")
    plt.title("SARD mountain200: frozen-score sweep")
    plt.grid(alpha=0.25); plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "recall_vs_fp_per_image.png", dpi=180); plt.close()

    widths = [row["gt_width_source_px"] for row in diagnostics]
    heights = [row["gt_height_source_px"] for row in diagnostics]
    analysis = {
        "sequence_dependence": sequence,
        "target_scale": {
            "source_width_min_median_max": [min(widths), median(widths), max(widths)],
            "source_height_min_median_max": [min(heights), median(heights), max(heights)],
            "median_yolo640_width_height": [
                median(row["gt_width_yolo640_px"] for row in diagnostics),
                median(row["gt_height_yolo640_px"] for row in diagnostics),
            ],
            "median_dino672_width_height": [
                median(row["gt_width_dino672_px"] for row in diagnostics),
                median(row["gt_height_dino672_px"] for row in diagnostics),
            ],
        },
        "saved_candidate_opportunity_iou50": {
            name: sum(row[f"{name}_best_saved_iou"] >= 0.5 for row in diagnostics) / len(diagnostics)
            for name in methods
        },
        "fixed_matches": {name: len(values) for name, values in matched.items()},
        "segment_recall": segment_table,
        "warning": "Saved-floor candidate opportunity is diagnostic, not a deployable operating point.",
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
