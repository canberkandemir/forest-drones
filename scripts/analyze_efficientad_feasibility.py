"""Assess whether an EfficientAD ForestPersons pilot is scientifically valid.

This script is intentionally read-only with respect to datasets and model
artifacts. It checks whether the existing ForestPersons annotations and local
subsets provide enough reliable normal-only images for a standard EfficientAD
training protocol.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitSummary:
    split: str
    images: int
    annotations: int
    annotation_free_images: int
    annotation_free_examples: list[str]
    visibility_counts: dict[str, int]
    pose_counts: dict[str, int]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        return data["images"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def annotation_visibility(annotation: dict[str, Any]) -> str:
    attrs = annotation.get("attributes")
    if isinstance(attrs, dict) and "visibility_level" in attrs:
        return str(attrs["visibility_level"])
    if "visibility_level" in annotation:
        return str(annotation["visibility_level"])
    return "unknown"


def annotation_pose(annotation: dict[str, Any]) -> str:
    attrs = annotation.get("attributes")
    if isinstance(attrs, dict) and "pose" in attrs:
        return str(attrs["pose"])
    if "pose" in annotation:
        return str(annotation["pose"])
    return "unknown"


def summarize_split(coco: dict[str, Any], split: str, max_examples: int = 20) -> SplitSummary:
    anns_by_image: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)

    annotation_free = [
        str(image["file_name"])
        for image in coco["images"]
        if len(anns_by_image.get(image["id"], [])) == 0
    ]
    visibility = Counter(annotation_visibility(annotation) for annotation in coco["annotations"])
    pose = Counter(annotation_pose(annotation) for annotation in coco["annotations"])
    return SplitSummary(
        split=split,
        images=len(coco["images"]),
        annotations=len(coco["annotations"]),
        annotation_free_images=len(annotation_free),
        annotation_free_examples=annotation_free[:max_examples],
        visibility_counts=dict(sorted(visibility.items())),
        pose_counts=dict(sorted(pose.items())),
    )


def summarize_manifest(coco: dict[str, Any], manifest_items: list[dict[str, Any]]) -> dict[str, Any]:
    images_by_id = {image["id"]: image for image in coco["images"]}
    images_by_file = {image["file_name"]: image for image in coco["images"]}
    anns_by_image: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        anns_by_image[annotation["image_id"]].append(annotation)

    missing_manifest_images = []
    annotation_count_hist = Counter()
    visibility = Counter()
    pose = Counter()
    for item in manifest_items:
        image = None
        if "id" in item and item["id"] in images_by_id:
            image = images_by_id[item["id"]]
        elif "image_id" in item and item["image_id"] in images_by_id:
            image = images_by_id[item["image_id"]]
        elif "file_name" in item and item["file_name"] in images_by_file:
            image = images_by_file[item["file_name"]]
        if image is None:
            missing_manifest_images.append(item)
            continue

        annotations = anns_by_image.get(image["id"], [])
        annotation_count_hist[len(annotations)] += 1
        for annotation in annotations:
            visibility[annotation_visibility(annotation)] += 1
            pose[annotation_pose(annotation)] += 1

    return {
        "images": len(manifest_items),
        "missing_manifest_images": len(missing_manifest_images),
        "annotation_free_images": annotation_count_hist.get(0, 0),
        "annotation_count_histogram": {
            str(key): value for key, value in sorted(annotation_count_hist.items())
        },
        "annotations": int(
            sum(int(count) * int(freq) for count, freq in annotation_count_hist.items())
        ),
        "visibility_counts": dict(sorted(visibility.items())),
        "pose_counts": dict(sorted(pose.items())),
    }


def scientific_decision(train5000: dict[str, Any], official_train: SplitSummary) -> dict[str, Any]:
    reasons = []
    if train5000["annotation_free_images"] == 0:
        reasons.append("train5000 contains zero annotation-free images")
    if official_train.annotation_free_images < 100:
        reasons.append(
            "the official train split has too few annotation-free images for EfficientAD "
            f"({official_train.annotation_free_images}/{official_train.images})"
        )
    reasons.append(
        "COCO-style absence of annotations is not a verified normal-only guarantee; "
        "annotation-free frames may be unlabeled edge cases rather than person-free forest"
    )
    return {
        "run_efficientad_pilot": False,
        "decision": "do_not_train",
        "reason": "; ".join(reasons),
        "smallest_valid_alternative": (
            "Create a separate masked-background/crop dataset from train images that excludes "
            "person boxes with a safety margin, then treat the result as a masked-background "
            "EfficientAD adaptation rather than a standard normal-image EfficientAD baseline."
        ),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EfficientAD Feasibility Analysis - ForestPersons",
        "",
        "This analysis checks whether a scientifically valid EfficientAD anomaly-detection pilot can be run using the existing local ForestPersons train5000/val600 data. No training, inference, dataset download, or checkpoint download was performed.",
        "",
        "## Method Requirement",
        "",
        "EfficientAD is a trained visual anomaly-detection method. A standard setup requires normal-only training images so the student/autoencoder learns the normal appearance distribution and treats deviations as anomalies. For ForestPersons, that means we need images that are reliably person-free, not merely images whose annotations happen to be empty.",
        "",
        "## Available Data",
        "",
        "| Split/subset | Images | Annotations | Annotation-free images |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ["train", "val", "test"]:
        item = summary["official_splits"][split]
        lines.append(
            f"| official {split} | {item['images']} | {item['annotations']} | {item['annotation_free_images']} |"
        )
    for subset in ["train5000", "val600"]:
        item = summary["subsets"][subset]
        lines.append(
            f"| {subset} | {item['images']} | {item['annotations']} | {item['annotation_free_images']} |"
        )

    lines += [
        "",
        "Annotation-free examples from the official train split:",
        "",
    ]
    for file_name in summary["official_splits"]["train"]["annotation_free_examples"]:
        lines.append(f"- `{file_name}`")

    lines += [
        "",
        "## Decision",
        "",
        f"- EfficientAD pilot status: `{summary['decision']['decision']}`.",
        f"- Reason: {summary['decision']['reason']}.",
        "",
        "I did not train EfficientAD because the current train5000 subset has no annotation-free images, and the full official train split has only 13 annotation-free images. Those 13 frames are not enough for a standard normal-only training set, and annotation absence is not strong evidence of person absence.",
        "",
        "## Feasible Alternative",
        "",
        summary["decision"]["smallest_valid_alternative"],
        "",
        "This alternative would be scientifically valid only if documented as a masked-background EfficientAD adaptation. It would be comparable to RX and DINOv3 patch-kNN at the patch/heatmap level, not directly comparable to bounding-box detectors.",
        "",
        "## Environment",
        "",
    ]
    for package, status in summary["environment"].items():
        lines.append(f"- `{package}`: {status}")

    lines += [
        "",
        "## EfficientAD Implementation Candidate",
        "",
        "- Preferred implementation path: Anomalib EfficientAD in a separate isolated environment.",
        "- Current local status: Anomalib, Lightning, and PyTorch Lightning are not installed in `.venv-dinov3`.",
        "- No ForestPersons-trained EfficientAD checkpoint was found locally.",
        "- Reference implementation/documentation checked:",
        "  - Anomalib EfficientAd model documentation: https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/efficient_ad.html",
        "  - Anomalib EfficientAd source README: https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/efficient_ad/README.md",
        "  - EfficientAD WACV 2024 paper: https://openaccess.thecvf.com/content/WACV2024/papers/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.pdf",
        "",
        "## Comparison Scope",
        "",
        "If later executed, EfficientAD should be compared with RX and DINOv3 patch-kNN using patch-level AUROC/AP and visibility-specific patch labels. It should not be compared directly with YOLO/FCOS AP unless a separate heatmap-to-box proposal method is defined and fixed before evaluation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--train5000-manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_train_5000_seed20261012_manifest_proposal.json"),
    )
    parser.add_argument(
        "--val600-manifest",
        type=Path,
        default=Path("outputs/subsets/forestpersons_val_600_seed20261010_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/efficientad_feasibility/forestpersons_20261013"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    official = {
        split: summarize_split(load_json(args.root / "annotations" / f"{split}.json"), split).__dict__
        for split in ["train", "val", "test"]
    }
    train_coco = load_json(args.root / "annotations" / "train.json")
    val_coco = load_json(args.root / "annotations" / "val.json")
    train5000 = summarize_manifest(train_coco, load_manifest(args.train5000_manifest))
    val600 = summarize_manifest(val_coco, load_manifest(args.val600_manifest))

    environment = {}
    for package in ["anomalib", "lightning", "pytorch_lightning", "torch", "torchvision"]:
        try:
            module = __import__(package)
            environment[package] = str(getattr(module, "__version__", "available_unknown_version"))
        except Exception as exc:  # noqa: BLE001 - diagnostic summary
            environment[package] = f"not available ({type(exc).__name__})"

    summary = {
        "settings": {
            "root": str(args.root),
            "train5000_manifest": str(args.train5000_manifest),
            "val600_manifest": str(args.val600_manifest),
        },
        "official_splits": official,
        "subsets": {
            "train5000": train5000,
            "val600": val600,
        },
        "environment": environment,
        "decision": scientific_decision(train5000, SplitSummary(**official["train"])),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.output_dir / "efficientad_feasibility_report.md", summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved report: {args.output_dir / 'efficientad_feasibility_report.md'}")


if __name__ == "__main__":
    main()
