"""Diagnose DINOv3 LoRA FCOS candidate quality on ForestPersons val600.

This script uses saved post-NMS COCO-format predictions. It does not rerun
inference or training, so it cannot identify boxes that were removed before the
saved result file was produced.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, xywh_to_xyxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--manifest", type=Path, default=Path("outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"))
    parser.add_argument("--dino-predictions", type=Path, default=Path("outputs/dinov3_lora/train5000_seed20261012_r4_qv_fcosinit_e2_rerun1/eval_val600/coco_results.json"))
    parser.add_argument("--yolo-predictions", type=Path, default=Path("outputs/evaluation_framework/val600_yolo11s_finetuned/coco_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/error_analysis/dinov3_lora_candidate_failures_val600_20261013"))
    parser.add_argument("--dino-threshold", type=float, default=0.55)
    parser.add_argument("--yolo-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--meaningful-iou", type=float, default=0.10)
    parser.add_argument("--visualizations", type=int, default=12)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "images" in data:
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def load_predictions(path: Path, image_ids: set[int | str]) -> dict[int | str, list[dict[str, Any]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict) and "annotations" in rows:
        rows = rows["annotations"]
    predictions: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = row["image_id"]
        if image_id not in image_ids:
            continue
        if int(row.get("category_id", 1)) != 1:
            continue
        predictions[image_id].append(
            {
                "box": xywh_to_xyxy([float(value) for value in row["bbox"]]),
                "confidence": float(row.get("score", row.get("confidence", 0.0))),
            }
        )
    for image_id in predictions:
        predictions[image_id].sort(key=lambda item: item["confidence"], reverse=True)
    return predictions


def annotation_to_gt(sample: Any, annotation: dict[str, Any], image_meta: dict[str, Any]) -> dict[str, Any]:
    attrs = annotation.get("metadata", {})
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    camera = attrs.get("camera") or image_meta.get("camera") or "unknown"
    season = attrs.get("season") or image_meta.get("season") or "unknown"
    pose = attrs.get("pose") or "unknown"
    visibility = attrs.get("visibility_level")
    area = width * height
    max_side = max(width, height)
    if max_side < 96:
        size_bin = "small_max_side_lt96"
    elif max_side < 192:
        size_bin = "medium_max_side_96_192"
    else:
        size_bin = "large_max_side_ge192"
    return {
        "image_id": sample.image_id,
        "file_name": sample.file_name,
        "annotation_id": annotation.get("id"),
        "box": [x, y, x + width, y + height],
        "width": width,
        "height": height,
        "area": area,
        "max_side": max_side,
        "size_bin": size_bin,
        "visibility": int(visibility) if visibility is not None else None,
        "pose": str(pose),
        "camera": str(camera),
        "season": str(season),
        "folder": image_meta.get("folder") or str(Path(sample.file_name).parent),
    }


def greedy_matches(
    gt_by_image: dict[int | str, list[dict[str, Any]]],
    pred_by_image: dict[int | str, list[dict[str, Any]]],
    image_ids: list[int | str],
    threshold: float,
    iou_threshold: float,
) -> dict[tuple[int | str, int | str | None], dict[str, Any]]:
    matched: dict[tuple[int | str, int | str | None], dict[str, Any]] = {}
    for image_id in image_ids:
        gts = gt_by_image.get(image_id, [])
        preds = [p for p in pred_by_image.get(image_id, []) if p["confidence"] >= threshold]
        used_gt: set[int] = set()
        for pred_index, pred in enumerate(preds):
            best_iou = 0.0
            best_gt_index = -1
            for gt_index, gt in enumerate(gts):
                if gt_index in used_gt:
                    continue
                iou = box_iou(gt["box"], pred["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            if best_gt_index >= 0 and best_iou >= iou_threshold:
                used_gt.add(best_gt_index)
                gt = gts[best_gt_index]
                matched[(image_id, gt["annotation_id"])] = {
                    "prediction_index": pred_index,
                    "iou": best_iou,
                    "confidence": pred["confidence"],
                    "box": pred["box"],
                }
    return matched


def best_candidate(gt: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not predictions:
        return None
    best = max(predictions, key=lambda pred: box_iou(gt["box"], pred["box"]))
    return {
        "iou": box_iou(gt["box"], best["box"]),
        "confidence": best["confidence"],
        "box": best["box"],
    }


def categorize(
    gt: dict[str, Any],
    matched: dict[tuple[int | str, int | str | None], dict[str, Any]],
    best: dict[str, Any] | None,
    threshold: float,
    meaningful_iou: float,
    iou_threshold: float,
) -> str:
    if (gt["image_id"], gt["annotation_id"]) in matched:
        return "detected_at_operating_threshold"
    if best is None or best["iou"] < meaningful_iou:
        return "no_meaningful_surviving_candidate"
    if best["iou"] < iou_threshold:
        return "candidate_localization_below_0.50"
    if best["confidence"] < threshold:
        return "candidate_iou_ge_0.50_but_confidence_below_threshold"
    return "candidate_eligible_but_unmatched_one_to_one"


def summarize_numeric(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p25": sorted_values[int(0.25 * (len(sorted_values) - 1))],
        "p75": sorted_values[int(0.75 * (len(sorted_values) - 1))],
        "min": sorted_values[0],
        "max": sorted_values[-1],
    }


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = []
    for value, group in sorted(groups.items()):
        best_ious = [float(row["dino_best_iou"]) for row in group]
        output.append(
            {
                key: value,
                "gt": len(group),
                "detected": sum(row["dino_category"] == "detected_at_operating_threshold" for row in group),
                "missed": sum(row["dino_category"] != "detected_at_operating_threshold" for row in group),
                "best_iou_ge_050": sum(row["dino_best_iou"] >= 0.50 for row in group),
                "best_iou_ge_075": sum(row["dino_best_iou"] >= 0.75 for row in group),
                "mean_best_iou": mean(best_ious) if best_ious else 0.0,
                "median_best_iou": median(best_ious) if best_ious else 0.0,
                "mean_best_confidence": mean(float(row["dino_best_confidence"]) for row in group if row["dino_best_confidence"] != ""),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_example(
    image_path: Path,
    row: dict[str, Any],
    dino_candidate: dict[str, Any] | None,
    yolo_candidate: dict[str, Any] | None,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(
        f"ann {row['annotation_id']} vis {row['visibility']} | {row['dino_category']}",
        fontsize=9,
    )

    def rect(box: list[float], color: str, label: str, linestyle: str = "-") -> None:
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2.0,
                edgecolor=color,
                facecolor="none",
                linestyle=linestyle,
            )
        )
        ax.text(
            x1,
            max(0, y1 - 4),
            label,
            fontsize=7,
            color="black",
            bbox={"facecolor": color, "alpha": 0.78, "pad": 1},
        )

    rect(row["_gt_box"], "lime", f"GT v{row['visibility']}")
    if dino_candidate is not None:
        rect(
            dino_candidate["box"],
            "cyan",
            f"DINO best IoU {dino_candidate['iou']:.2f} s {dino_candidate['confidence']:.2f}",
            "--",
        )
    if yolo_candidate is not None:
        rect(
            yolo_candidate["box"],
            "magenta",
            f"YOLO best IoU {yolo_candidate['iou']:.2f} s {yolo_candidate['confidence']:.2f}",
            ":",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output_dir / "visualizations"

    dataset = ForestPersonsDataset(args.root, split=args.split)
    manifest_items = load_manifest(args.manifest)
    manifest_by_file = {item["file_name"]: item for item in manifest_items}
    samples_by_file = {sample.file_name: sample for sample in dataset.samples}
    samples = [samples_by_file[item["file_name"]] for item in manifest_items]
    image_ids = [sample.image_id for sample in samples]
    image_id_set = set(image_ids)

    gt_by_image: dict[int | str, list[dict[str, Any]]] = {}
    all_gt: list[dict[str, Any]] = []
    for sample in samples:
        image_meta = manifest_by_file.get(sample.file_name, {})
        gts = [
            annotation_to_gt(sample, ann, image_meta)
            for ann in sample.annotations
            if ann.get("metadata", {}).get("visibility_level") is not None
        ]
        gt_by_image[sample.image_id] = gts
        all_gt.extend(gts)

    dino_predictions = load_predictions(args.dino_predictions, image_id_set)
    yolo_predictions = load_predictions(args.yolo_predictions, image_id_set)
    dino_matches = greedy_matches(
        gt_by_image,
        dino_predictions,
        image_ids,
        args.dino_threshold,
        args.iou_threshold,
    )
    yolo_matches = greedy_matches(
        gt_by_image,
        yolo_predictions,
        image_ids,
        args.yolo_threshold,
        args.iou_threshold,
    )

    rows: list[dict[str, Any]] = []
    candidate_lookup: dict[tuple[int | str, int | str | None], tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    for gt in all_gt:
        dino_best = best_candidate(gt, dino_predictions.get(gt["image_id"], []))
        yolo_best = best_candidate(gt, yolo_predictions.get(gt["image_id"], []))
        dino_category = categorize(
            gt,
            dino_matches,
            dino_best,
            args.dino_threshold,
            args.meaningful_iou,
            args.iou_threshold,
        )
        yolo_category = categorize(
            gt,
            yolo_matches,
            yolo_best,
            args.yolo_threshold,
            args.meaningful_iou,
            args.iou_threshold,
        )
        key = (gt["image_id"], gt["annotation_id"])
        candidate_lookup[key] = (dino_best, yolo_best)
        row = {
            "image_id": gt["image_id"],
            "file_name": gt["file_name"],
            "annotation_id": gt["annotation_id"],
            "visibility": gt["visibility"],
            "pose": gt["pose"],
            "camera": gt["camera"],
            "season": gt["season"],
            "folder": gt["folder"],
            "bbox_width": gt["width"],
            "bbox_height": gt["height"],
            "bbox_area": gt["area"],
            "bbox_max_side": gt["max_side"],
            "size_bin": gt["size_bin"],
            "dino_category": dino_category,
            "dino_detected": dino_category == "detected_at_operating_threshold",
            "dino_best_iou": dino_best["iou"] if dino_best else 0.0,
            "dino_best_confidence": dino_best["confidence"] if dino_best else "",
            "dino_best_iou_ge_050": bool(dino_best and dino_best["iou"] >= 0.50),
            "dino_best_iou_ge_075": bool(dino_best and dino_best["iou"] >= 0.75),
            "yolo_category": yolo_category,
            "yolo_detected": yolo_category == "detected_at_operating_threshold",
            "yolo_best_iou": yolo_best["iou"] if yolo_best else 0.0,
            "yolo_best_confidence": yolo_best["confidence"] if yolo_best else "",
            "yolo_best_iou_ge_050": bool(yolo_best and yolo_best["iou"] >= 0.50),
            "yolo_best_iou_ge_075": bool(yolo_best and yolo_best["iou"] >= 0.75),
            "_gt_box": gt["box"],
        }
        rows.append(row)

    csv_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    write_csv(args.output_dir / "gt_candidate_analysis.csv", csv_rows)

    summaries = {
        "settings": {
            "manifest": str(args.manifest),
            "dino_predictions": str(args.dino_predictions),
            "yolo_predictions": str(args.yolo_predictions),
            "dino_threshold": args.dino_threshold,
            "yolo_threshold": args.yolo_threshold,
            "iou_threshold": args.iou_threshold,
            "meaningful_iou": args.meaningful_iou,
            "important_limitation": (
                "Input predictions are saved post-NMS COCO results. Pre-NMS candidates, "
                "raw classification logits, and raw centerness values are not available."
            ),
        },
        "counts": {
            "images": len(samples),
            "ground_truth": len(all_gt),
            "dino_predictions_post_nms": sum(len(v) for v in dino_predictions.values()),
            "yolo_predictions_post_nms": sum(len(v) for v in yolo_predictions.values()),
        },
        "dino_categories": dict(Counter(row["dino_category"] for row in rows)),
        "yolo_categories": dict(Counter(row["yolo_category"] for row in rows)),
        "dino_best_iou": summarize_numeric([float(row["dino_best_iou"]) for row in rows]),
        "yolo_best_iou": summarize_numeric([float(row["yolo_best_iou"]) for row in rows]),
        "dino_best_iou_ge_050": sum(row["dino_best_iou"] >= 0.50 for row in rows),
        "dino_best_iou_ge_075": sum(row["dino_best_iou"] >= 0.75 for row in rows),
        "yolo_best_iou_ge_050": sum(row["yolo_best_iou"] >= 0.50 for row in rows),
        "yolo_best_iou_ge_075": sum(row["yolo_best_iou"] >= 0.75 for row in rows),
        "dino_by_visibility": aggregate(rows, "visibility"),
        "dino_by_size_bin": aggregate(rows, "size_bin"),
        "dino_by_camera": aggregate(rows, "camera"),
        "dino_by_pose": aggregate(rows, "pose"),
        "dino_by_season": aggregate(rows, "season"),
        "cross_model": {
            "both_detected": sum(row["dino_detected"] and row["yolo_detected"] for row in rows),
            "yolo_only": sum((not row["dino_detected"]) and row["yolo_detected"] for row in rows),
            "dino_only": sum(row["dino_detected"] and (not row["yolo_detected"]) for row in rows),
            "both_missed": sum((not row["dino_detected"]) and (not row["yolo_detected"]) for row in rows),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for key in ["visibility", "size_bin", "camera", "pose", "season"]:
        write_csv(args.output_dir / f"dino_by_{key}.csv", summaries[f"dino_by_{key}"])

    examples: list[dict[str, Any]] = []
    for category in [
        "no_meaningful_surviving_candidate",
        "candidate_localization_below_0.50",
        "candidate_iou_ge_0.50_but_confidence_below_threshold",
        "candidate_eligible_but_unmatched_one_to_one",
    ]:
        category_rows = [row for row in rows if row["dino_category"] == category]
        if category == "candidate_localization_below_0.50":
            category_rows.sort(key=lambda row: abs(float(row["dino_best_iou"]) - 0.50))
        elif category == "candidate_iou_ge_0.50_but_confidence_below_threshold":
            category_rows.sort(key=lambda row: float(row["dino_best_confidence"] or 0.0), reverse=True)
        else:
            category_rows.sort(key=lambda row: (int(row["visibility"]), -float(row["bbox_area"])))
        examples.extend(category_rows[: max(1, args.visualizations // 4)])
    examples = examples[: args.visualizations]

    viz_rows = []
    for row in examples:
        key = (row["image_id"], row["annotation_id"])
        dino_best, yolo_best = candidate_lookup[key]
        image_path = args.root / "images" / row["file_name"]
        safe_name = f"ann_{row['annotation_id']}_vis{row['visibility']}_{Path(row['file_name']).name}"
        out_path = viz_dir / safe_name
        draw_example(image_path, row, dino_best, yolo_best, out_path)
        viz_rows.append(
            {
                "annotation_id": row["annotation_id"],
                "file_name": row["file_name"],
                "category": row["dino_category"],
                "visibility": row["visibility"],
                "dino_best_iou": row["dino_best_iou"],
                "dino_best_confidence": row["dino_best_confidence"],
                "yolo_best_iou": row["yolo_best_iou"],
                "yolo_best_confidence": row["yolo_best_confidence"],
                "visualization": str(out_path),
            }
        )
    write_csv(args.output_dir / "visualization_examples.csv", viz_rows)
    print(json.dumps(summaries, indent=2))
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
