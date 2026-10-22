"""Add Hugging Face Content-Length sizes to a ForestPersons manifest."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    return token or None


def fetch_size(relative_path: str, headers: dict[str, str]) -> tuple[str, int]:
    url = (
        "https://huggingface.co/datasets/etri/ForestPersons/resolve/main/images/"
        + urllib.parse.quote(relative_path)
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers=headers, method="HEAD")
            with urllib.request.urlopen(request, timeout=45) as response:
                content_length = response.headers.get("Content-Length")
                if not content_length:
                    raise RuntimeError(f"Missing Content-Length for {relative_path}")
                return relative_path, int(content_length)
        except Exception as exc:
            last_error = exc
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch size for {relative_path}: {last_error}")


def recompute_size_summary(manifest: dict, image_root: Path = Path("data/images")) -> None:
    selected = manifest["images"]
    total = sum(int(item.get("hf_size", 0)) for item in selected)
    additional = sum(
        int(item.get("hf_size", 0))
        for item in selected
        if not (image_root / item["file_name"]).exists()
    )
    existing = sum(
        (image_root / item["file_name"]).stat().st_size
        for item in selected
        if (image_root / item["file_name"]).exists()
    )
    summary = manifest.setdefault("summary", {})
    summary["hf_total_bytes"] = total
    summary["hf_total_mib"] = total / 1024 / 1024
    summary["already_local_images"] = sum(
        1 for item in selected if (image_root / item["file_name"]).exists()
    )
    summary["already_local_bytes"] = existing
    summary["additional_download_bytes"] = additional
    summary["additional_download_mib"] = additional / 1024 / 1024


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    token = hf_token()
    headers = {"User-Agent": "LogXY-ForestPersons-size-metadata/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    missing = [item for item in manifest["images"] if "hf_size" not in item]
    print(f"fetching sizes for {len(missing)} files")
    sizes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_size, item["file_name"], headers) for item in missing]
        for index, future in enumerate(as_completed(futures), 1):
            file_name, size = future.result()
            sizes[file_name] = size
            if index % 250 == 0:
                print(f"sizes {index}/{len(futures)}")

    for item in manifest["images"]:
        if item["file_name"] in sizes:
            item["hf_size"] = sizes[item["file_name"]]
    recompute_size_summary(manifest)
    output = args.output or args.manifest
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
