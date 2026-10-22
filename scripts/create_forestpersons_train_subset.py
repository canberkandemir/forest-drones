"""Create a reproducible ForestPersons training subset manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from email.message import Message
from pathlib import Path
from typing import Any


VISIBILITY_LEVELS = (20, 40, 70, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/train.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--target-images", type=int, default=5000)
    parser.add_argument("--include-manifest", type=Path)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--hf-metadata", action="store_true")
    return parser.parse_args()


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    return token or None


def image_records(coco: dict[str, Any]) -> list[dict[str, Any]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    records = []
    for image in coco["images"]:
        annotations = annotations_by_image[image["id"]]
        visibility_counts = Counter(
            annotation.get("attributes", {}).get("visibility_level")
            for annotation in annotations
            if annotation.get("attributes", {}).get("visibility_level") in VISIBILITY_LEVELS
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


def round_robin_strata(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
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
    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
        groups[key].sort(key=lambda record: (",".join(record["poses"]), record["file_name"]))
    ordered = []
    while keys:
        next_keys = []
        for key in keys:
            if not groups[key]:
                continue
            ordered.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def load_included(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in data["images"]]


def select_subset(
    records: list[dict[str, Any]],
    include_records: list[dict[str, Any]],
    target_images: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_file = {record["file_name"]: record for record in records}
    selected = []
    selected_files = set()
    target_counts: Counter[int] = Counter()

    for item in include_records:
        record = dict(by_file.get(item["file_name"], item))
        target_visibility = int(item.get("target_visibility") or min(record["visibilities"]))
        record["target_visibility"] = target_visibility
        if record["file_name"] in selected_files:
            continue
        selected.append(record)
        selected_files.add(record["file_name"])
        target_counts[target_visibility] += 1

    per_visibility_target = target_images // len(VISIBILITY_LEVELS)
    remainder = target_images % len(VISIBILITY_LEVELS)
    desired = {
        visibility: per_visibility_target + (1 if index < remainder else 0)
        for index, visibility in enumerate(VISIBILITY_LEVELS)
    }

    for visibility in VISIBILITY_LEVELS:
        candidates = [
            record
            for record in records
            if visibility in record["visibilities"]
            and len(record["visibilities"]) == 1
            and record["file_name"] not in selected_files
        ]
        for record in round_robin_strata(candidates, seed + visibility):
            if len(selected) >= target_images or target_counts[visibility] >= desired[visibility]:
                break
            item = dict(record)
            item["target_visibility"] = visibility
            selected.append(item)
            selected_files.add(item["file_name"])
            target_counts[visibility] += 1

    for visibility in VISIBILITY_LEVELS:
        if len(selected) >= target_images or target_counts[visibility] >= desired[visibility]:
            continue
        candidates = [
            record
            for record in records
            if visibility in record["visibilities"] and record["file_name"] not in selected_files
        ]
        for record in round_robin_strata(candidates, seed + visibility + 1000):
            if len(selected) >= target_images or target_counts[visibility] >= desired[visibility]:
                break
            item = dict(record)
            item["target_visibility"] = visibility
            selected.append(item)
            selected_files.add(item["file_name"])
            target_counts[visibility] += 1

    if len(selected) < target_images:
        candidates = [record for record in records if record["file_name"] not in selected_files]
        for record in round_robin_strata(candidates, seed + 9000):
            if len(selected) >= target_images:
                break
            item = dict(record)
            item["target_visibility"] = min(item["visibilities"])
            selected.append(item)
            selected_files.add(item["file_name"])
            target_counts[item["target_visibility"]] += 1

    summary = {
        "selection_strategy": (
            "Training split only. Include prior manifest images first, then fill "
            "target image quotas per visibility using single-visibility images "
            "before mixed-visibility images. Candidate order round-robins over "
            "season, camera, place, weather, and sequence folder with seeded "
            "tie-breaking."
        ),
        "seed": seed,
        "target_images": target_images,
        "included_manifest_images": len(include_records),
        "selected_images_by_target_visibility": dict(sorted(target_counts.items())),
    }
    return selected[:target_images], summary


def fetch_hf_sizes(selected: list[dict[str, Any]]) -> None:
    folders = sorted({item["folder"] for item in selected})
    token = hf_token()
    headers = {"User-Agent": "LogXY-ForestPersons-train-subset-metadata/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    sizes = {}
    for folder_index, folder in enumerate(folders, 1):
        encoded_folder = urllib.parse.quote(f"images/{folder}", safe="/")
        url = (
            "https://huggingface.co/api/datasets/etri/ForestPersons/tree/main/"
            f"{encoded_folder}?recursive=false&expand=true"
        )
        while url:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                items = json.load(response)
                link = response.headers.get("Link")
            for item in items:
                if item.get("type") == "file" and item.get("size") is not None:
                    relative = item["path"].split("images/", 1)[1]
                    sizes[relative] = int(item["size"])
            url = next_link(link)
        if folder_index % 100 == 0:
            print(f"metadata folders {folder_index}/{len(folders)}")
    missing = []
    for item in selected:
        size = sizes.get(item["file_name"])
        if size is None:
            missing.append(item["file_name"])
        else:
            item["hf_size"] = size
    if missing:
        raise RuntimeError(f"Missing Hugging Face size metadata for {len(missing)} files")


def next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    message = Message()
    message["Link"] = link_header
    for value in message.get_all("Link", []):
        for part in [part.strip() for part in value.split(",")]:
            if 'rel="next"' in part and part.startswith("<") and ">" in part:
                return part[1 : part.index(">")]
    return None


def summarize(selected: list[dict[str, Any]], image_root: Path) -> dict[str, Any]:
    annotation_visibility_counts: Counter[int] = Counter()
    image_visibility_presence: Counter[int] = Counter()
    target_visibility_counts: Counter[int] = Counter()
    pose_counts: Counter[str] = Counter()
    season_counts: Counter[str] = Counter()
    camera_counts: Counter[str] = Counter()
    place_counts: Counter[str] = Counter()
    weather_counts: Counter[str] = Counter()
    folder_counts: Counter[str] = Counter()
    existing_local_images = 0
    existing_local_bytes = 0

    for item in selected:
        target_visibility_counts[int(item["target_visibility"])] += 1
        folder_counts[item["folder"]] += 1
        season_counts[item["season"]] += 1
        camera_counts[item["camera"]] += 1
        place_counts[item["place"]] += 1
        weather_counts[item["weather"]] += 1
        for pose in item["poses"]:
            pose_counts[pose] += 1
        for visibility in item["visibilities"]:
            image_visibility_presence[int(visibility)] += 1
        for visibility, count in item["visibility_counts"].items():
            annotation_visibility_counts[int(visibility)] += int(count)
        local_path = image_root / item["file_name"]
        if local_path.exists():
            existing_local_images += 1
            existing_local_bytes += local_path.stat().st_size

    total_bytes = sum(item.get("hf_size", 0) for item in selected)
    additional_bytes = sum(
        item.get("hf_size", 0)
        for item in selected
        if not (image_root / item["file_name"]).exists()
    )
    estimated_cache_bytes = estimate_feature_cache_bytes(selected)
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
        "folder_image_counts_top50": dict(folder_counts.most_common(50)),
        "hf_total_bytes": total_bytes,
        "hf_total_mib": total_bytes / 1024 / 1024 if total_bytes else None,
        "already_local_images": existing_local_images,
        "already_local_bytes": existing_local_bytes,
        "additional_download_bytes": additional_bytes,
        "additional_download_mib": additional_bytes / 1024 / 1024 if additional_bytes else None,
        "estimated_feature_cache_bytes": estimated_cache_bytes,
        "estimated_feature_cache_gib": estimated_cache_bytes / 1024 / 1024 / 1024,
    }


def estimate_feature_cache_bytes(selected: list[dict[str, Any]], short_side: int = 672) -> int:
    # np.savez_compressed overhead varies. This estimates raw float16 token bytes;
    # actual v1/v2 observed compressed cache size is close to this for DINO tokens.
    total = 0
    for item in selected:
        width = int(item["width"])
        height = int(item["height"])
        scale = short_side / min(width, height)
        resized_width = max(16, round(width * scale / 16) * 16)
        resized_height = max(16, round(height * scale / 16) * 16)
        grid_width = resized_width // 16
        grid_height = resized_height // 16
        total += grid_width * grid_height * 384 * 2
    return int(total)


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    records = image_records(coco)
    include_records = load_included(args.include_manifest)
    selected, selection = select_subset(records, include_records, args.target_images, args.seed)
    if args.hf_metadata:
        fetch_hf_sizes(selected)
    output = {
        "source_annotations": str(args.annotations),
        "selection": selection,
        "summary": summarize(selected, args.image_root),
        "images": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
