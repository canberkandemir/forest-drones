"""Build a manual Weitefeld class-3 review gallery and localization audit.

Only existing pilot images and saved prediction dumps are read.  No inference,
training, downloading, or threshold selection is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import html
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

from forestpersons.detection import box_iou


REFERENCE_THRESHOLDS = {"yolo": 0.04, "dino": 0.55, "fusion_B": 0.0}
COLORS = {"gt": "#43ff64", "yolo": "#24d8ff", "dino": "#ff4fd8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join("".join(character.lower() if character.isalnum() else " " for character in value).split())


def provisional_review(police_comment: str) -> tuple[str, str]:
    """Return a conservative, police-comment-only provisional status.

    No text is auto-promoted to ``confirmed_person``.  Comments that explicitly
    say a person/cyclist was not checked remain uncertain.
    """

    comment = normalize_text(police_comment)
    if not comment or comment == "na":
        return "uncertain", "police comment unavailable"
    unverified_person_terms = ("group of people", "cyclist")
    if any(term in comment for term in unverified_person_terms):
        return "uncertain", "person-like interpretation explicitly marked not checked"
    nonperson_terms = (
        "no abnormalities", "no findings", "stone", "tree stump", "tree stumps",
        "tree trunk", "tree trunks", "uprooted tree", "trash bag", "feeding site",
        "hunting stand", "fallen tree", "no clear object",
    )
    if any(term in comment for term in nonperson_terms):
        return "non_person", "explicit police non-person/no-abnormality note (provisional)"
    return "not_reviewed", "police comment does not support an automatic decision"


def target_size_bin(area: float) -> str:
    """COCO pixel-area bins applied to original-resolution annotation boxes."""

    if area < 32**2:
        return "small_lt_32sq"
    if area < 96**2:
        return "medium_32_96sq"
    return "large_ge_96sq"


def load_predictions(path: Path, score_field: str = "score") -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    data = json.loads(path.read_text(encoding="utf-8"))
    for prediction in data["predictions"] if isinstance(data, dict) else data:
        item = dict(prediction)
        item["score"] = float(item.get(score_field, item.get("fused_score", 0.0)))
        grouped[int(item["image_id"])].append(item)
    return grouped


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def center_distance(first: list[float], second: list[float]) -> tuple[float, float]:
    x1, y1 = center(first)
    x2, y2 = center(second)
    distance = math.hypot(x1 - x2, y1 - y2)
    diagonal = math.hypot(first[2] - first[0], first[3] - first[1])
    return distance, distance / max(diagonal, 1e-9)


def best_iou_candidate(
    target: list[float], predictions: list[dict[str, Any]], threshold: float | None = None
) -> tuple[dict[str, Any] | None, float]:
    candidates = predictions if threshold is None else [p for p in predictions if p["score"] >= threshold]
    if not candidates:
        return None, 0.0
    item = max(candidates, key=lambda prediction: box_iou(target, prediction["bbox"]))
    return item, float(box_iou(target, item["bbox"]))


def nearest_center_candidate(
    target: list[float], predictions: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float, float]:
    if not predictions:
        return None, math.inf, math.inf
    item = min(predictions, key=lambda prediction: center_distance(target, prediction["bbox"])[0])
    distance, normalized = center_distance(target, item["bbox"])
    return item, distance, normalized


def greedy_match_count(
    ground_truth: list[tuple[int, int, list[float]]],
    predictions: dict[int, list[dict[str, Any]]],
    *,
    confidence: float,
    iou_threshold: float,
) -> int:
    total = 0
    by_image_gt: dict[int, list[list[float]]] = defaultdict(list)
    for image_id, _, box in ground_truth:
        by_image_gt[image_id].append(box)
    for image_id, targets in by_image_gt.items():
        candidates = [item for item in predictions.get(image_id, []) if item["score"] >= confidence]
        pairs = []
        for gt_index, target in enumerate(targets):
            for pred_index, prediction in enumerate(candidates):
                overlap = float(box_iou(target, prediction["bbox"]))
                if overlap >= iou_threshold:
                    pairs.append((overlap, gt_index, pred_index))
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for _, gt_index, pred_index in sorted(pairs, reverse=True):
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
        total += len(used_gt)
    return total


def font(size: int) -> ImageFont.ImageFont:
    path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def fit_panel(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    contained = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "#12161c")
    offset_x = (size[0] - contained.width) // 2
    offset_y = (size[1] - contained.height) // 2
    panel.paste(contained, (offset_x, offset_y))
    return panel, contained.width / image.width, offset_x, offset_y


def draw_box(
    draw: ImageDraw.ImageDraw, box: list[float], origin: tuple[float, float], scale: float,
    offset: tuple[int, int], color: str, width: int, bounds: tuple[float, float, float, float],
) -> None:
    clipped = [
        max(box[0], bounds[0]), max(box[1], bounds[1]),
        min(box[2], bounds[2]), min(box[3], bounds[3]),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return
    transformed = [
        (clipped[0] - origin[0]) * scale + offset[0],
        (clipped[1] - origin[1]) * scale + offset[1],
        (clipped[2] - origin[0]) * scale + offset[0],
        (clipped[3] - origin[1]) * scale + offset[1],
    ]
    draw.rectangle(transformed, outline=color, width=width)


def make_review_panel(
    source: Image.Image, row: dict[str, Any], output: Path,
) -> None:
    gt = row["gt_bbox"]
    yolo = row.get("yolo_tiled_best_box")
    dino = row.get("dino_tiled_best_box")
    candidate_boxes = [box for box in (yolo, dino) if box]
    cx, cy = center(gt)

    # Larger context contains the GT and nearby best candidates, but is capped
    # so a distant candidate cannot make the actual target invisible.
    union = [gt] + candidate_boxes
    union_width = max(box[2] for box in union) - min(box[0] for box in union)
    union_height = max(box[3] for box in union) - min(box[1] for box in union)
    context_span = min(1800.0, max(768.0, 1.2 * union_width, 1.2 * union_height))
    context_left = max(0.0, min(source.width - context_span, cx - context_span / 2))
    context_top = max(0.0, min(source.height - context_span, cy - context_span / 2))
    context_bounds = (
        int(context_left), int(context_top), int(context_left + context_span),
        int(context_top + context_span),
    )
    target_span = max(160.0, 5.0 * max(gt[2] - gt[0], gt[3] - gt[1]))
    target_span = min(target_span, 420.0)
    target_left = max(0.0, min(source.width - target_span, cx - target_span / 2))
    target_top = max(0.0, min(source.height - target_span, cy - target_span / 2))
    target_bounds = (
        int(target_left), int(target_top), int(target_left + target_span), int(target_top + target_span),
    )

    canvas = Image.new("RGB", (1800, 760), "#0b0e12")
    full, full_scale, full_x, full_y = fit_panel(source, (560, 560))
    context_crop = source.crop(context_bounds)
    context, context_scale, context_x, context_y = fit_panel(context_crop, (560, 560))
    target_crop = source.crop(target_bounds)
    target, target_scale, target_x, target_y = fit_panel(target_crop, (560, 560))
    canvas.paste(full, (20, 135)); canvas.paste(context, (620, 135)); canvas.paste(target, (1220, 135))
    draw = ImageDraw.Draw(canvas)
    title_font, body_font, small_font = font(25), font(18), font(15)
    draw.text((20, 15), f"Image {row['image_id']} | {row['file_name']} | strip {row['strip']} | annotation {row['annotation_id']}", fill="white", font=title_font)
    draw.text((20, 50), f"Review: {row['review_status']} (provisional) | track: {row['track_id']} | GT {row['gt_width']:.0f}x{row['gt_height']:.0f}", fill="#f5d76e", font=body_font)
    draw.text((20, 80), f"Police: {row['police_comment']}", fill="#d9dee7", font=small_font)
    draw.text((20, 108), "Full image context", fill="white", font=body_font)
    draw.text((620, 108), "Larger context crop", fill="white", font=body_font)
    draw.text((1220, 108), "Magnified target crop", fill="white", font=body_font)

    panel_specs = (
        ((20, 135), (0.0, 0.0), full_scale, (full_x, full_y), (0.0, 0.0, source.width, source.height)),
        ((620, 135), (context_bounds[0], context_bounds[1]), context_scale, (context_x, context_y), context_bounds),
        ((1220, 135), (target_bounds[0], target_bounds[1]), target_scale, (target_x, target_y), target_bounds),
    )
    for panel_offset, origin, scale, internal_offset, bounds in panel_specs:
        offset = (panel_offset[0] + internal_offset[0], panel_offset[1] + internal_offset[1])
        draw_box(draw, gt, origin, scale, offset, COLORS["gt"], 4, bounds)
        if yolo:
            draw_box(draw, yolo, origin, scale, offset, COLORS["yolo"], 3, bounds)
        if dino:
            draw_box(draw, dino, origin, scale, offset, COLORS["dino"], 3, bounds)
    # Ensure the tiny full-image target remains findable.
    full_cx = cx * full_scale + 20 + full_x
    full_cy = cy * full_scale + 135 + full_y
    draw.ellipse((full_cx - 9, full_cy - 9, full_cx + 9, full_cy + 9), outline=COLORS["gt"], width=3)
    draw.text((20, 714), "GT", fill=COLORS["gt"], font=body_font)
    draw.text((90, 714), f"YOLO best saved: IoU {row['yolo_tiled_best_iou']:.3f}, score {row['yolo_tiled_best_score']:.3f}", fill=COLORS["yolo"], font=body_font)
    draw.text((600, 714), f"DINO best saved: IoU {row['dino_tiled_best_iou']:.3f}, score {row['dino_tiled_best_score']:.3f}", fill=COLORS["dino"], font=body_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=91)


def save_contact_sheets(panel_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_size = (900, 380)
    for page_index, start in enumerate(range(0, len(panel_paths), 4), 1):
        page = Image.new("RGB", (1800, 800), "#05070a")
        for slot, path in enumerate(panel_paths[start:start + 4]):
            with Image.open(path) as panel:
                thumb = ImageOps.fit(panel.convert("RGB"), thumb_size, Image.Resampling.LANCZOS)
            page.paste(thumb, ((slot % 2) * 900, (slot // 2) * 400))
        page.save(output_dir / f"contact_sheet_{page_index:02d}.jpg", quality=90)


def grouped_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groupings = {
        "target_size": lambda row: row["target_size_bin"],
        "flight_strip": lambda row: str(row["strip"]),
        "track_proxy": lambda row: row["track_id"],
        "review_status": lambda row: row["review_status"],
    }
    for group_type, getter in groupings.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[getter(row)].append(row)
        for group_value, members in sorted(groups.items()):
            for model in ("yolo", "dino"):
                values = [float(row[f"{model}_tiled_best_iou"]) for row in members]
                center_values = [float(row[f"{model}_tiled_center_distance_norm"]) for row in members]
                output.append({
                    "group_type": group_type,
                    "group_value": group_value,
                    "model": model,
                    "annotations": len(members),
                    "candidate_iou_ge_0.10": sum(value >= 0.10 for value in values),
                    "candidate_iou_ge_0.25": sum(value >= 0.25 for value in values),
                    "candidate_iou_ge_0.50": sum(value >= 0.50 for value in values),
                    "median_best_iou": median(values),
                    "median_center_distance_gt_diagonal": median(center_values),
                })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    selected = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw = {
        f"{model}_{mode}": load_predictions(args.prediction_root / f"{model}_{mode}.json")
        for model in ("yolo", "dino") for mode in ("full", "tiled")
    }
    fixed = {
        f"{method}_{mode}": load_predictions(
            args.evaluation_root / mode / f"{method}_predictions.json", "fused_score"
        )
        for method in ("yolo", "dino", "fusion_B") for mode in ("full", "tiled")
    }

    rows: list[dict[str, Any]] = []
    images_by_id = {int(item["image_id"]): item for item in manifest["images"]}
    ground_truth: list[tuple[int, int, list[float]]] = []
    for image in manifest["images"]:
        image_id = int(image["image_id"])
        for annotation in image["person_annotations"]:
            gt = [float(value) for value in annotation["bbox_xyxy"]]
            gt_area = area(gt)
            status, evidence = provisional_review(annotation["police_comment"])
            row: dict[str, Any] = {
                "image_id": image_id,
                "file_name": image["file_name"],
                "strip": int(image["strip"]),
                "image_number": int(image["image_number"]),
                "annotation_id": int(annotation["annotation_id"]),
                "gt_bbox_xyxy": json.dumps(gt),
                "gt_bbox": gt,
                "gt_width": gt[2] - gt[0],
                "gt_height": gt[3] - gt[1],
                "gt_area": gt_area,
                "target_size_bin": target_size_bin(gt_area),
                "volunteer_comment": annotation["volunteer_comment"],
                "police_comment": annotation["police_comment"],
                "track_id": annotation["track_id"],
                "review_status": status,
                "review_evidence": evidence,
                "reviewer_status": "",
                "reviewer_notes": "",
            }
            ground_truth.append((image_id, int(annotation["annotation_id"]), gt))
            for key, grouped in raw.items():
                model = key.split("_", 1)[0]
                candidates = grouped.get(image_id, [])
                best, overlap = best_iou_candidate(gt, candidates)
                nearest, nearest_px, nearest_norm = nearest_center_candidate(gt, candidates)
                chosen = best if overlap > 0 else nearest
                if chosen is None:
                    candidate_box = None
                    width_ratio = height_ratio = area_ratio = center_px = center_norm = None
                    diagnosis = "no_saved_candidate"
                    score = None
                else:
                    candidate_box = [float(value) for value in chosen["bbox"]]
                    width_ratio = (candidate_box[2] - candidate_box[0]) / max(row["gt_width"], 1e-9)
                    height_ratio = (candidate_box[3] - candidate_box[1]) / max(row["gt_height"], 1e-9)
                    area_ratio = area(candidate_box) / max(gt_area, 1e-9)
                    center_px, center_norm = center_distance(gt, candidate_box)
                    score = float(chosen["score"])
                    nearby = nearest_norm <= 2.0
                    severe_scale = area_ratio > 4.0 or area_ratio < 0.25
                    if overlap >= 0.50:
                        diagnosis = "strict_iou_match"
                    elif overlap > 0 and severe_scale:
                        diagnosis = "overlap_wrong_scale"
                    elif overlap > 0:
                        diagnosis = "overlap_low_iou_other"
                    elif nearby and severe_scale:
                        diagnosis = "nearby_no_overlap_wrong_scale"
                    elif nearby:
                        diagnosis = "nearby_no_overlap"
                    else:
                        diagnosis = "no_nearby_prediction"
                row.update({
                    f"{key}_best_iou": overlap,
                    f"{key}_best_score": score,
                    f"{key}_best_box": candidate_box,
                    f"{key}_pred_width": None if candidate_box is None else candidate_box[2] - candidate_box[0],
                    f"{key}_pred_height": None if candidate_box is None else candidate_box[3] - candidate_box[1],
                    f"{key}_width_ratio": width_ratio,
                    f"{key}_height_ratio": height_ratio,
                    f"{key}_area_ratio": area_ratio,
                    f"{key}_center_distance_px": center_px,
                    f"{key}_center_distance_norm": center_norm,
                    f"{key}_nearest_center_distance_px": nearest_px,
                    f"{key}_nearest_center_distance_norm": nearest_norm,
                    f"{key}_diagnosis": diagnosis,
                })
            rows.append(row)

    review_fields = [
        "image_id", "file_name", "strip", "image_number", "annotation_id", "gt_bbox_xyxy",
        "gt_width", "gt_height", "gt_area", "volunteer_comment", "police_comment", "track_id",
        "review_status", "review_evidence", "reviewer_status", "reviewer_notes",
    ]
    write_csv(args.output_dir / "annotation_review.csv", rows, review_fields)
    diagnostic_fields = [key for key in rows[0] if key != "gt_bbox" and not key.endswith("_best_box")]
    write_csv(args.output_dir / "per_annotation_localization_diagnostics.csv", rows, diagnostic_fields)
    grouped = grouped_diagnostics(rows)
    write_csv(args.output_dir / "localization_group_summary.csv", grouped)

    annotation_pairs = []
    rows_for_pairs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_for_pairs[int(row["image_id"])].append(row)
    for image_id, image_rows in sorted(rows_for_pairs.items()):
        for first_index in range(len(image_rows)):
            for second_index in range(first_index + 1, len(image_rows)):
                first = image_rows[first_index]
                second = image_rows[second_index]
                overlap = float(box_iou(first["gt_bbox"], second["gt_bbox"]))
                annotation_pairs.append({
                    "image_id": image_id,
                    "first_annotation_id": first["annotation_id"],
                    "second_annotation_id": second["annotation_id"],
                    "first_track_id": first["track_id"],
                    "second_track_id": second["track_id"],
                    "annotation_iou": overlap,
                })
    if annotation_pairs:
        write_csv(args.output_dir / "overlapping_annotation_pairs.csv", annotation_pairs)

    panels = []
    rows_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_image[int(row["image_id"])].append(row)
    panel_root = args.output_dir / "review_panels"
    for image_id, image_rows in sorted(rows_by_image.items()):
        image = images_by_id[image_id]
        path = args.image_root / f"strip{image['strip']}" / image["file_name"]
        with Image.open(path) as source_image:
            source = source_image.convert("RGB")
            for row in image_rows:
                output = panel_root / f"image_{image_id:03d}_annotation_{row['annotation_id']:05d}.jpg"
                make_review_panel(source, row, output)
                panels.append(output)
    save_contact_sheets(panels, args.output_dir / "contact_sheets")

    status_counts = Counter(row["review_status"] for row in rows)
    uncertain_reason_counts = Counter(row["review_evidence"] for row in rows if row["review_status"] == "uncertain")
    size_counts = Counter(row["target_size_bin"] for row in rows)
    diagnostic_recall: dict[str, Any] = {}
    for mode in ("full", "tiled"):
        for method in ("yolo", "dino", "fusion_B"):
            key = f"{method}_{mode}"
            diagnostic_recall[key] = {
                f"one_to_one_recall_iou_{iou:.2f}": greedy_match_count(
                    ground_truth, fixed[key], confidence=REFERENCE_THRESHOLDS[method], iou_threshold=iou
                ) / len(ground_truth)
                for iou in (0.10, 0.25, 0.50)
            }
            diagnostic_recall[key]["reference_threshold"] = REFERENCE_THRESHOLDS[method]

    candidate_opportunity: dict[str, Any] = {}
    for key in raw:
        ious = [float(row[f"{key}_best_iou"]) for row in rows]
        candidate_opportunity[key] = {
            "prediction_floor": 0.001,
            "gt_with_candidate_iou_ge_0.10": sum(value >= 0.10 for value in ious),
            "gt_with_candidate_iou_ge_0.25": sum(value >= 0.25 for value in ious),
            "gt_with_candidate_iou_ge_0.50": sum(value >= 0.50 for value in ious),
            "maximum_best_iou": max(ious),
            "median_best_iou": median(ious),
            "nondeployable_warning": "per-GT best candidate; candidates are not one-to-one and include scores below the frozen operating threshold",
        }

    # Input-space scale audit.
    scales = {
        "yolo_full_640": 640 / 8416,
        "dino_full_short672": 672 / 6032,
        "yolo_tile2048_to640": 640 / 2048,
        "dino_tile2048_short672": 672 / 2048,
    }
    scale_summary = {}
    for name, scale in scales.items():
        widths = [row["gt_width"] * scale for row in rows]
        heights = [row["gt_height"] * scale for row in rows]
        scale_summary[name] = {
            "scale": scale,
            "median_width_px": median(widths),
            "median_height_px": median(heights),
            "boxes_with_both_dimensions_below_4px": sum(w < 4 and h < 4 for w, h in zip(widths, heights)),
            "boxes_with_both_dimensions_below_8px": sum(w < 8 and h < 8 for w, h in zip(widths, heights)),
            "boxes_with_both_dimensions_below_16px": sum(w < 16 and h < 16 for w, h in zip(widths, heights)),
        }

    localization_summary = {}
    for key in raw:
        overlaps = [row for row in rows if float(row[f"{key}_best_iou"]) > 0.0]
        localization_summary[key] = {
            "diagnosis_counts": dict(Counter(row[f"{key}_diagnosis"] for row in rows)),
            "median_selected_candidate_center_distance_px": median(
                float(row[f"{key}_center_distance_px"]) for row in rows
                if row[f"{key}_center_distance_px"] is not None
            ),
            "median_selected_candidate_center_distance_gt_diagonal": median(
                float(row[f"{key}_center_distance_norm"]) for row in rows
                if row[f"{key}_center_distance_norm"] is not None
            ),
            "median_nearest_candidate_center_distance_px": median(
                float(row[f"{key}_nearest_center_distance_px"]) for row in rows
                if math.isfinite(float(row[f"{key}_nearest_center_distance_px"]))
            ),
            "median_nearest_candidate_center_distance_gt_diagonal": median(
                float(row[f"{key}_nearest_center_distance_norm"]) for row in rows
                if math.isfinite(float(row[f"{key}_nearest_center_distance_norm"]))
            ),
            "annotations_with_any_overlap": len(overlaps),
            "median_area_ratio_among_overlapping_candidates": (
                median(float(row[f"{key}_area_ratio"]) for row in overlaps) if overlaps else None
            ),
            "overlapping_candidates_with_area_ratio_gt_4": sum(
                float(row[f"{key}_area_ratio"]) > 4.0 for row in overlaps
            ),
        }

    # Plots use fixed, predeclared thresholds and all saved candidates.
    plt.figure(figsize=(7.5, 4.8))
    bins = [index / 40 for index in range(21)]
    for key, color in (
        ("yolo_full", "#58a6ff"), ("yolo_tiled", "#0969da"),
        ("dino_full", "#f778ba"), ("dino_tiled", "#bf3989"),
    ):
        plt.hist([row[f"{key}_best_iou"] for row in rows], bins=bins, histtype="step", linewidth=2, label=key)
    plt.xlabel("Best saved-candidate IoU (diagnostic, floor 0.001)")
    plt.ylabel("Class-3 annotations")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "best_iou_histogram.png", dpi=180); plt.close()

    plt.figure(figsize=(7.5, 4.8))
    for key, color, marker in (
        ("yolo_tiled", "#0969da", "o"), ("dino_tiled", "#bf3989", "x"),
    ):
        x = [row["gt_area"] for row in rows if row[f"{key}_best_iou"] > 0]
        y = [row[f"{key}_area_ratio"] for row in rows if row[f"{key}_best_iou"] > 0]
        plt.scatter(x, y, s=18, alpha=0.55, color=color, marker=marker, label=key)
    plt.axhline(1.0, color="black", linewidth=1); plt.axhline(4.0, color="gray", linestyle="--", linewidth=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Official annotation area (source pixels)"); plt.ylabel("Best candidate / GT area ratio")
    plt.legend(); plt.tight_layout()
    plt.savefig(args.output_dir / "tiled_candidate_area_ratio.png", dpi=180); plt.close()

    summary = {
        "status": "analysis only; no inference, training, or download",
        "annotations": len(rows),
        "review_status_counts": dict(status_counts),
        "uncertain_reason_counts": dict(uncertain_reason_counts),
        "review_policy": {
            "confirmed_person_auto_assignment": False,
            "non_person": "only explicit police non-person/no-abnormality comments; provisional",
            "uncertain": "unverified person/cyclist wording or missing police comment",
        },
        "target_size_counts": dict(size_counts),
        "annotation_overlap_audit": {
            "images_with_multiple_class3_annotations": sum(len(items) > 1 for items in rows_for_pairs.values()),
            "maximum_annotations_in_one_image": max(len(items) for items in rows_for_pairs.values()),
            "within_image_annotation_pairs": len(annotation_pairs),
            "pairs_iou_ge_0.10": sum(float(item["annotation_iou"]) >= 0.10 for item in annotation_pairs),
            "pairs_iou_ge_0.50": sum(float(item["annotation_iou"]) >= 0.50 for item in annotation_pairs),
            "pairs_iou_ge_0.90": sum(float(item["annotation_iou"]) >= 0.90 for item in annotation_pairs),
        },
        "diagnostic_recall_at_frozen_operating_points": diagnostic_recall,
        "saved_candidate_opportunity": candidate_opportunity,
        "localization_summary": localization_summary,
        "input_space_scale": scale_summary,
        "diagnostic_rules": {
            "nearby": "nearest candidate center distance <= 2 GT diagonals",
            "severe_scale_mismatch": "candidate/GT area ratio outside [0.25, 4.0]",
            "strict_metric": "IoU >= 0.50",
            "lower_iou_metrics": "diagnostic only and fixed before calculation",
        },
        "gallery": {
            "individual_panels": len(panels),
            "contact_sheet_pages": math.ceil(len(panels) / 4),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    instructions = """# Weitefeld Annotation Review Instructions

Open `gallery.html` or the paged images in `contact_sheets/`. Every panel shows
the full source image, a larger context crop, and a magnified target crop.
Green is the official class-3 box, cyan is the best saved tiled YOLO candidate,
and magenta is the best saved tiled DINOv3 candidate. Candidate scores include
the saved low-score floor and are not all deployable detections.

Edit only `reviewer_status` and `reviewer_notes` in `annotation_review.csv`.
Allowed reviewer statuses are `confirmed_person`, `non_person`, `uncertain`, and
`not_reviewed`. The existing `review_status` is provisional and based only on
the official police comment. No annotation is automatically confirmed as a
person. Preserve image and annotation IDs when saving the CSV.
"""
    (args.output_dir / "REVIEW_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    cards = "\n".join(
        f'<article><a href="review_panels/{path.name}"><img src="review_panels/{path.name}" loading="lazy"></a></article>'
        for path in panels
    )
    gallery_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Weitefeld review</title>
<style>body{{background:#101318;color:#fff;font:16px Arial;margin:20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(680px,1fr));gap:18px}} article{{background:#191e26;padding:8px}} img{{width:100%;height:auto}}</style></head>
<body><h1>Weitefeld pilot200 class-3 annotation review</h1><p>155 panels. Green: official box; cyan: YOLO; magenta: DINOv3. Review instructions: <code>REVIEW_INSTRUCTIONS.md</code>.</p><div class="grid">{cards}</div></body></html>"""
    (args.output_dir / "gallery.html").write_text(gallery_html, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
