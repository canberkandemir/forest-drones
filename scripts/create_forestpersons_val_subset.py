"""Create a reproducible ForestPersons validation subset manifest.

The subset is image-based. Because ForestPersons images can contain multiple
persons with different visibility levels, image quotas do not guarantee equal
annotation counts. The script records both target-image counts and actual
annotation counts so later evaluation reports stay honest.
"""

from __future__ import annotations

import argparse
import json
import random
import ssl
import os
import urllib.parse
import urllib.request
from email.message import Message
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VISIBILITY_LEVELS = (20, 40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/val.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--images-per-visibility", type=int, default=150)
    parser.add_argument(
        "--hf-metadata",
        action="store_true",
        help="Fetch Hugging Face tree metadata to estimate image byte sizes.",
    )
    return parser.parse_args()


def image_records(coco: dict[str, Any]) -> list[dict[str, Any]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    records: list[dict[str, Any]] = []
    for image in coco["images"]:
        annotations = annotations_by_image[image["id"]]
        visibility_counts = Counter(
            annotation.get("attributes", {}).get("visibility_level")
            for annotation in annotations
            if annotation.get("attributes", {}).get("visibility_level") is not None
        )
        if not visibility_counts:
            continue

        poses = sorted(
            {
                annotation.get("attributes", {}).get("pose")
                for annotation in annotations
                if annotation.get("attributes", {}).get("pose")
            }
        )
        attrs = annotations[0].get("attributes", {}) if annotations else {}
        records.append(
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "folder": image["file_name"].split("/")[0],
                "annotation_count": len(annotations),
                "visibility_counts": {
                    str(key): visibility_counts[key]
                    for key in sorted(visibility_counts)
                },
                "visibilities": sorted(visibility_counts),
                "poses": poses,
                "season": attrs.get("season", "unknown"),
                "camera": attrs.get("camera", "unknown"),
                "place": attrs.get("place", "unknown"),
                "weather": attrs.get("weather", "unknown"),
            }
        )
    return records


def stratified_order(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda record: (
            record["season"],
            record["camera"],
            record["place"],
            record["weather"],
            ",".join(record["poses"]),
            record["folder"],
        )
    )
    return shuffled


def round_robin_strata(
    records: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Return records in a seeded round-robin over metadata/folder strata."""

    rng = random.Random(seed)
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["season"],
            record["camera"],
            record["place"],
            record["weather"],
            record["folder"],
        )
        groups[key].append(record)

    ordered_keys = list(groups)
    rng.shuffle(ordered_keys)
    for key in ordered_keys:
        rng.shuffle(groups[key])
        groups[key].sort(key=lambda record: (",".join(record["poses"]), record["file_name"]))

    ordered: list[dict[str, Any]] = []
    while ordered_keys:
        next_keys: list[tuple[str, str, str, str, str]] = []
        for key in ordered_keys:
            group = groups[key]
            if not group:
                continue
            ordered.append(group.pop(0))
            if group:
                next_keys.append(key)
        ordered_keys = next_keys
    return ordered


def select_subset(
    records: list[dict[str, Any]],
    images_per_visibility: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_by_visibility: dict[int, list[dict[str, Any]]] = {
        visibility: [] for visibility in VISIBILITY_LEVELS
    }
    selected_files: set[str] = set()

    # First pass: prefer single-visibility images for clearer per-level coverage.
    for visibility in VISIBILITY_LEVELS:
        candidates = [
            record
            for record in records
            if visibility in record["visibilities"] and len(record["visibilities"]) == 1
        ]
        for record in round_robin_strata(candidates, seed + visibility):
            if len(selected_by_visibility[visibility]) >= images_per_visibility:
                break
            if record["file_name"] in selected_files:
                continue
            item = dict(record)
            item["target_visibility"] = visibility
            selected_by_visibility[visibility].append(item)
            selected_files.add(item["file_name"])

    # Second pass: fill any shortfalls with mixed-visibility images.
    for visibility in VISIBILITY_LEVELS:
        if len(selected_by_visibility[visibility]) >= images_per_visibility:
            continue
        candidates = [
            record
            for record in records
            if visibility in record["visibilities"] and record["file_name"] not in selected_files
        ]
        for record in round_robin_strata(candidates, seed + visibility + 1000):
            if len(selected_by_visibility[visibility]) >= images_per_visibility:
                break
            item = dict(record)
            item["target_visibility"] = visibility
            selected_by_visibility[visibility].append(item)
            selected_files.add(item["file_name"])

    selected = [
        item
        for visibility in VISIBILITY_LEVELS
        for item in selected_by_visibility[visibility]
    ]
    summary = {
        "selection_strategy": (
            "Image-based stratified sample. For each visibility level, select "
            "single-visibility images first, then mixed-visibility images if "
            "needed. Candidate order is deterministic using season, camera, "
            "place, weather, pose set, folder, and seeded random tie-breaking."
        ),
        "seed": seed,
        "requested_images_per_visibility": images_per_visibility,
        "selected_images_by_target_visibility": {
            str(visibility): len(selected_by_visibility[visibility])
            for visibility in VISIBILITY_LEVELS
        },
    }
    return selected, summary


def fetch_hf_sizes(selected: list[dict[str, Any]]) -> dict[str, int]:
    folders = sorted({item["folder"] for item in selected})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    sizes: dict[str, int] = {}
    for folder in folders:
        encoded_folder = urllib.parse.quote(f"images/{folder}", safe="/")
        url = (
            "https://huggingface.co/api/datasets/etri/ForestPersons/tree/main/"
            f"{encoded_folder}?recursive=false&expand=true"
        )
        while url:
            with urllib.request.urlopen(url, timeout=30, context=ctx) as response:
                items = json.load(response)
                link = response.headers.get("Link")
            for item in items:
                if item.get("type") != "file" or item.get("size") is None:
                    continue
                relative = item["path"].split("images/", 1)[1]
                sizes[relative] = int(item["size"])
            url = next_link(link)
    return sizes


def next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    message = Message()
    message["Link"] = link_header
    for value in message.get_all("Link", []):
        parts = [part.strip() for part in value.split(",")]
        for part in parts:
            if 'rel="next"' not in part:
                continue
            if part.startswith("<") and ">" in part:
                return part[1 : part.index(">")]
    return None


def fetch_hf_size_with_head(relative_path: str, ctx: ssl.SSLContext) -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()

    url = (
        "https://huggingface.co/datasets/etri/ForestPersons/resolve/main/images/"
        + urllib.parse.quote(relative_path)
    )
    headers = {"User-Agent": "LogXY-ForestPersons-subset-metadata/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="HEAD")
    with urllib.request.urlopen(request, timeout=30, context=ctx) as response:
        content_length = response.headers.get("Content-Length")
        if not content_length:
            raise RuntimeError(f"Missing Content-Length for {relative_path}")
        return int(content_length)


def summarize(selected: list[dict[str, Any]], local_root: Path) -> dict[str, Any]:
    annotation_visibility_counts: Counter[int] = Counter()
    image_visibility_presence: Counter[int] = Counter()
    target_visibility_counts: Counter[int] = Counter()
    pose_counts: Counter[str] = Counter()
    season_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    place_counts: Counter[str] = Counter()
    weather_counts: Counter[str] = Counter()
    folder_counts: Counter[str] = Counter()
    existing_local_bytes = 0
    existing_local_images = 0

    for item in selected:
        target_visibility_counts[item["target_visibility"]] += 1
        folder_counts[item["folder"]] += 1
        season_counts[item["season"]] += 1
        camera_counts[item["camera"]] += 1
        place_counts[item["place"]] += 1
        weather_counts[item["weather"]] += 1
        for pose in item["poses"]:
            pose_counts[pose] += 1
        for visibility in item["visibilities"]:
            image_visibility_presence[visibility] += 1
        for visibility, count in item["visibility_counts"].items():
            annotation_visibility_counts[int(visibility)] += count

        local_path = local_root / item["file_name"]
        if local_path.exists():
            existing_local_images += 1
            existing_local_bytes += local_path.stat().st_size

    total_bytes = sum(item.get("hf_size", 0) for item in selected)
    missing_bytes = sum(
        item.get("hf_size", 0)
        for item in selected
        if not (local_root / item["file_name"]).exists()
    )
    return {
        "selected_images": len(selected),
        "selected_images_by_target_visibility": dict(sorted(target_visibility_counts.items())),
        "image_visibility_presence": dict(sorted(image_visibility_presence.items())),
        "annotation_visibility_counts": dict(sorted(annotation_visibility_counts.items())),
        "pose_image_counts": dict(pose_counts.most_common()),
        "season_image_counts": dict(season_counts.most_common()),
        "camera_image_counts": dict(camera_counts.most_common()),
        "place_image_counts": dict(place_counts.most_common()),
        "weather_image_counts": dict(weather_counts.most_common()),
        "folder_count": len(folder_counts),
        "folder_image_counts": dict(folder_counts.most_common()),
        "hf_total_bytes": total_bytes,
        "hf_total_mib": total_bytes / 1024 / 1024 if total_bytes else None,
        "already_local_images": existing_local_images,
        "already_local_bytes": existing_local_bytes,
        "additional_download_bytes": missing_bytes,
        "additional_download_mib": missing_bytes / 1024 / 1024 if missing_bytes else None,
    }


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    records = image_records(coco)
    selected, selection_summary = select_subset(
        records,
        images_per_visibility=args.images_per_visibility,
        seed=args.seed,
    )

    if args.hf_metadata:
        sizes = fetch_hf_sizes(selected)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        missing_sizes = []
        for item in selected:
            size = sizes.get(item["file_name"])
            if size is None:
                try:
                    size = fetch_hf_size_with_head(item["file_name"], ctx)
                except Exception:
                    missing_sizes.append(item["file_name"])
                    continue
            item["hf_size"] = size
        if missing_sizes:
            raise RuntimeError(f"Missing HF size metadata for {len(missing_sizes)} files")

    output = {
        "source_annotations": str(args.annotations),
        "selection": selection_summary,
        "summary": summarize(selected, local_root=Path("data/images")),
        "images": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
