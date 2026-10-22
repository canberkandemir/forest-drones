"""Prepare a minimal Google Drive upload bundle for Colab smoke inference.

The bundle contains only the files needed for the 30-image MMDetection smoke
test, avoiding virtual environments, cached DINOv3 features, old experiment
outputs, and the full ForestPersons dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_OUTPUT = Path("outputs/colab_upload/LogXY_colab_smoke30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-count", type=int, default=30)
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, patterns: tuple[str, ...] = ("*",)) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for pattern in patterns:
        for path in src.rglob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                copy_file(path, dst / path.relative_to(src))


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    val_coco_path = root / "outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json"
    val_coco = json.loads(val_coco_path.read_text(encoding="utf-8"))
    selected_images = val_coco["images"][: args.image_count]

    required_files = [
        "README.md",
        "AGENTS.md",
        "PROGRESS.md",
        "data/annotations/val.json",
        "outputs/baseline_feasibility/forestpersons_val600_oneclass_coco.json",
        "outputs/baseline_feasibility/forestpersons_val600_smoke30_manifest.json",
        "outputs/subsets/forestpersons_val_600_seed20261010_manifest.json",
        "scripts/mmdet_infer_forestpersons.py",
        "scripts/verify_mmdet_cuda_env.py",
        "scripts/evaluate_external_coco_results.py",
    ]
    for relative in required_files:
        copy_file(root / relative, output / relative)

    copy_tree(root / "src/forestpersons", output / "src/forestpersons", patterns=("*.py",))
    copy_tree(
        root / "experiments/published_baselines_cuda",
        output / "experiments/published_baselines_cuda",
        patterns=("*.md", "*.yml", "*.py", "*.sh", "*.ipynb"),
    )

    missing_images: list[str] = []
    for image in selected_images:
        relative = Path("data/images") / image["file_name"]
        src = root / relative
        if src.exists():
            copy_file(src, output / relative)
        else:
            missing_images.append(str(relative))
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} smoke images, e.g. {missing_images[:3]}")

    summary = {
        "bundle_root": str(output),
        "image_count": len(selected_images),
        "images": [image["file_name"] for image in selected_images],
        "upload_destination": "/content/drive/MyDrive/LogXY",
        "notes": [
            "Upload the contents of this folder as the LogXY folder in Google Drive.",
            "This bundle is for 30-image smoke inference only, not full val600 or training.",
            "Checkpoints are intentionally not included; the Colab notebook downloads them to Drive.",
        ],
    }
    (output / "COLAB_UPLOAD_README.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
