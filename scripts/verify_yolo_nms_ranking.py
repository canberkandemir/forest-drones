"""Verify where YOLO-style high-IoU candidates are lost before/after NMS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import nms

from forestpersons import ForestPersonsDataset
from forestpersons.detection import box_iou, xywh_to_xyxy
from forestpersons.dinov3_detector import PATCH_SIZE, annotations_to_ground_truth, choose_device
from forestpersons.dinov3_lora import extract_lora_patch_features
from forestpersons.dinov3_yolo import (
    box_iou_matrix,
    dfl_expectation,
    load_lora_yolo_checkpoint,
    ltrb_to_xyxy,
    make_grid_centers,
    yolo_scores,
)


class OnlineForestPersonsDataset(Dataset):
    def __init__(self, root: Path, split: str, manifest: Path, max_images: int | None = None) -> None:
        self.dataset = ForestPersonsDataset(root, split=split)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        items = manifest_data["images"] if isinstance(manifest_data, dict) else manifest_data
        if max_images is not None:
            items = items[:max_images]
        by_file = {sample.file_name: sample for sample in self.dataset.samples}
        self.samples = [by_file[item["file_name"]] for item in items]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "file_name": sample.file_name,
            "image_id": sample.image_id,
            "image": Image.open(sample.image_path).convert("RGB"),
            "ground_truth": annotations_to_ground_truth(sample.annotations),
            "annotations": sample.annotations,
        }


def collate_one(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("This diagnostic expects batch_size=1.")
    return batch[0]


def decode_candidate_stages(
    outputs: dict[str, torch.Tensor],
    info: Any,
    score_mode: str,
    conf_threshold: float,
    nms_threshold: float,
    topk: int,
) -> dict[str, Any]:
    class_logits = outputs["class_logits"]
    distance_logits = outputs["distance_logits"]
    quality_logits = outputs.get("quality_logits")
    scores = yolo_scores(class_logits, quality_logits=quality_logits, score_mode=score_mode).reshape(-1)
    quality_scores = quality_logits.sigmoid().reshape(-1) if quality_logits is not None else None
    h, w = class_logits.shape
    centers = make_grid_centers(h, w, device=class_logits.device, dtype=class_logits.dtype).reshape(-1, 2)
    distances = dfl_expectation(distance_logits).reshape(-1, 4)
    boxes = ltrb_to_xyxy(centers, distances, stride=PATCH_SIZE)
    boxes = torch.stack(
        [
            boxes[:, 0].clamp(0, info.resized_width - 1),
            boxes[:, 1].clamp(0, info.resized_height - 1),
            boxes[:, 2].clamp(0, info.resized_width - 1),
            boxes[:, 3].clamp(0, info.resized_height - 1),
        ],
        dim=1,
    )
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    all_indices = torch.nonzero(valid, as_tuple=False).flatten()
    threshold_indices = torch.nonzero(valid & (scores >= conf_threshold), as_tuple=False).flatten()
    topk_indices = threshold_indices
    if topk_indices.numel() > topk:
        _top_scores, order = torch.topk(scores[topk_indices], k=topk)
        topk_indices = topk_indices[order]
    nms_indices = torch.empty((0,), device=class_logits.device, dtype=torch.long)
    if topk_indices.numel() > 0:
        selected_local = nms(boxes[topk_indices], scores[topk_indices], nms_threshold)
        nms_indices = topk_indices[selected_local]

    scale = torch.tensor(
        [info.scale_x, info.scale_y, info.scale_x, info.scale_y],
        device=boxes.device,
        dtype=boxes.dtype,
    )
    original_boxes = boxes / scale
    original_boxes[:, 0::2] = original_boxes[:, 0::2].clamp(0, info.original_width - 1)
    original_boxes[:, 1::2] = original_boxes[:, 1::2].clamp(0, info.original_height - 1)
    return {
        "boxes": original_boxes.detach().cpu(),
        "scores": scores.detach().cpu(),
        "quality_scores": quality_scores.detach().cpu() if quality_scores is not None else None,
        "all": set(int(index) for index in all_indices.detach().cpu().tolist()),
        "threshold": set(int(index) for index in threshold_indices.detach().cpu().tolist()),
        "topk": set(int(index) for index in topk_indices.detach().cpu().tolist()),
        "nms": set(int(index) for index in nms_indices.detach().cpu().tolist()),
    }


def pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2:
        return None
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-mode", choices=["class", "class_quality"], default="class")
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--short-side", type=int, default=672)
    parser.add_argument("--conf-threshold", type=float, default=0.001)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--high-iou", type=float, default=0.75)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    processor, backbone, head, _metadata = load_lora_yolo_checkpoint(args.checkpoint, device)
    backbone.eval()
    head.eval()
    dataset = OnlineForestPersonsDataset(args.root, args.split, args.manifest, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_one)

    totals = {
        "images": 0,
        "gt_instances": 0,
        "pre_nms_high_iou": 0,
        "lost_threshold": 0,
        "lost_topk": 0,
        "lost_nms": 0,
        "kept_after_nms": 0,
    }
    examples: list[dict[str, Any]] = []
    candidate_quality: list[float] = []
    candidate_max_iou: list[float] = []
    nms_quality: list[float] = []
    nms_max_iou: list[float] = []

    with torch.inference_mode():
        for batch in loader:
            features, info = extract_lora_patch_features(
                batch["image"],
                processor=processor,
                model=backbone,
                device=device,
                short_side=args.short_side,
            )
            outputs = head(features)
            stages = decode_candidate_stages(
                outputs,
                info,
                score_mode=args.score_mode,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                topk=args.topk,
            )
            gt_boxes = [gt["box"] for gt in batch["ground_truth"]]
            totals["images"] += 1
            totals["gt_instances"] += len(gt_boxes)
            if not gt_boxes:
                continue

            boxes = stages["boxes"]
            scores = stages["scores"]
            all_indices = sorted(stages["all"])
            topk_indices = sorted(stages["topk"])
            nms_indices = sorted(stages["nms"])
            for index in topk_indices:
                max_iou = max((box_iou(boxes[index].tolist(), gt_box) for gt_box in gt_boxes), default=0.0)
                if stages["quality_scores"] is not None:
                    candidate_quality.append(float(stages["quality_scores"][index]))
                    candidate_max_iou.append(float(max_iou))
            for index in nms_indices:
                max_iou = max((box_iou(boxes[index].tolist(), gt_box) for gt_box in gt_boxes), default=0.0)
                if stages["quality_scores"] is not None:
                    nms_quality.append(float(stages["quality_scores"][index]))
                    nms_max_iou.append(float(max_iou))

            for gt_index, gt_box in enumerate(gt_boxes):
                if not all_indices:
                    continue
                ious = [box_iou(boxes[index].tolist(), gt_box) for index in all_indices]
                best_local = int(np.argmax(ious))
                best_index = all_indices[best_local]
                best_iou = float(ious[best_local])
                if best_iou < args.high_iou:
                    continue
                totals["pre_nms_high_iou"] += 1
                if best_index not in stages["threshold"]:
                    totals["lost_threshold"] += 1
                    reason = "threshold"
                elif best_index not in stages["topk"]:
                    totals["lost_topk"] += 1
                    reason = "topk"
                elif best_index not in stages["nms"]:
                    totals["lost_nms"] += 1
                    reason = "nms"
                else:
                    totals["kept_after_nms"] += 1
                    reason = "kept"
                if len(examples) < 20:
                    examples.append(
                        {
                            "file_name": batch["file_name"],
                            "gt_index": gt_index,
                            "reason": reason,
                            "best_candidate_index": best_index,
                            "best_iou": best_iou,
                            "best_score": float(scores[best_index]),
                            "best_quality": (
                                float(stages["quality_scores"][best_index])
                                if stages["quality_scores"] is not None
                                else None
                            ),
                            "gt_box": gt_box,
                            "best_box": boxes[best_index].tolist(),
                        }
                    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "score_mode": args.score_mode,
        "conf_threshold": args.conf_threshold,
        "nms_threshold": args.nms_threshold,
        "topk": args.topk,
        "high_iou": args.high_iou,
        **totals,
        "quality_iou_correlation_topk_candidates": pearson(candidate_quality, candidate_max_iou),
        "quality_iou_correlation_nms_candidates": pearson(nms_quality, nms_max_iou),
        "candidate_quality_count": len(candidate_quality),
        "nms_quality_count": len(nms_quality),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
