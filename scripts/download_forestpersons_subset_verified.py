"""Download ForestPersons images from a manifest with normal SSL verification."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    return token or None


def verify_image(path: Path, item: dict) -> tuple[int, int, str]:
    expected_size = item.get("hf_size")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise RuntimeError(
            f"Size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}"
        )
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    if width != int(item["width"]) or height != int(item["height"]):
        raise RuntimeError(
            f"Dimension mismatch for {item['file_name']}: got {(width, height)}, "
            f"expected {(item['width'], item['height'])}"
        )
    if mode != "RGB":
        raise RuntimeError(f"Unexpected image mode for {item['file_name']}: {mode}")
    return width, height, mode


def download_one(item: dict, image_root: Path, headers: dict[str, str], force: bool) -> dict:
    relative = item["file_name"]
    output_path = image_root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False
    if force or not output_path.exists():
        url = (
            "https://huggingface.co/datasets/etri/ForestPersons/resolve/main/images/"
            + urllib.parse.quote(relative)
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    output_path.write_bytes(response.read())
                downloaded = True
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.75 * (attempt + 1))
        if last_error is not None and not output_path.exists():
            raise last_error
    verify_image(output_path, item)
    return {"file_name": relative, "downloaded": downloaded, "bytes": output_path.stat().st_size}


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    token = hf_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Run huggingface-cli login first.")
    headers = {
        "User-Agent": "LogXY-ForestPersons-subset-download/2.0",
        "Authorization": f"Bearer {token}",
    }
    downloaded = 0
    verified = 0
    bytes_written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, item, args.image_root, headers, args.force)
            for item in manifest["images"]
        ]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            verified += 1
            if result["downloaded"]:
                downloaded += 1
                bytes_written += int(result["bytes"])
            if index % 100 == 0:
                print(f"verified {index}/{len(futures)}")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "downloaded_images": downloaded,
                "verified_images": verified,
                "bytes_written": bytes_written,
                "mib_written": bytes_written / 1024 / 1024,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
