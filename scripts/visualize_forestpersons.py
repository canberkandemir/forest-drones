"""Visualize ForestPersons ground-truth person boxes from local COCO files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from forestpersons import ForestPersonsDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Sample from every annotated image, including files not present locally.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/forestpersons_gt_preview.png"),
        help="Path for the generated visualization grid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ForestPersonsDataset(
        args.root,
        split=args.split,
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No images found for split {args.split}")

    available_indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if args.all_images or sample.image_path.exists()
    ]
    if not available_indices:
        raise RuntimeError(
            f"No local image files found under {dataset.image_root}. "
            "Download images or pass --all-images to sample without checking."
        )

    count = min(args.count, len(available_indices))
    rng = random.Random(args.seed)
    indices = rng.sample(available_indices, count)

    cols = 2
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes_list = list(axes.flat if hasattr(axes, "flat") else [axes])

    for ax, index in zip(axes_list, indices):
        image, target = dataset[index]
        ax.imshow(image)
        ax.set_title(target["file_name"], fontsize=9)
        ax.axis("off")
        for annotation in target["annotations"]:
            x, y, width, height = annotation["bbox"]
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=1.8,
                edgecolor="lime",
                facecolor="none",
            )
            ax.add_patch(rect)
            metadata = annotation.get("metadata", {})
            label_parts = ["person"]
            for key in (
                "visibility_level",
                "visibility",
                "visibility_ratio",
                "pose",
                "difficulty",
            ):
                if key in metadata:
                    label_parts.append(f"{key}={metadata[key]}")
            ax.text(
                x,
                max(0, y - 3),
                " ".join(label_parts),
                color="black",
                fontsize=7,
                va="bottom",
                bbox={"facecolor": "lime", "alpha": 0.85, "pad": 1},
            )

    for ax in axes_list[count:]:
        ax.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
