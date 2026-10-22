"""Download and verify only images listed in a Weitefeld pilot manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.request import Request, urlopen

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/weitefeld/images/pilot200"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        dimensions = image.size
    expected = int(item["width"]), int(item["height"])
    if dimensions != expected:
        raise ValueError(f"{path}: dimensions {dimensions} != {expected}")
    return {
        "path": str(path), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "width": dimensions[0], "height": dimensions[1],
    }


def download_one(item: dict[str, Any], root: Path, retries: int) -> dict[str, Any]:
    destination = root / f"strip{item['strip']}" / item["file_name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {**verify(destination, item), "status": "reused"}
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(item["source_url"], headers={"User-Agent": "LogXY-research-pilot/1.0"})
            with urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False, suffix=".part"
            ) as temporary:
                temporary_path = Path(temporary.name)
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            verified = verify(temporary_path, item)
            temporary_path.replace(destination)
            return {**verified, "path": str(destination), "status": "downloaded"}
        except Exception as exc:  # pragma: no cover - network failures are environment-specific.
            error = exc
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed {item['source_url']}: {error}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, item, args.output_root, args.retries): item
            for item in manifest["images"]
        }
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            print(f"[{index}/{len(futures)}] {record['status']} {record['path']}", flush=True)
    records.sort(key=lambda item: item["path"])
    report = {
        "manifest": str(args.manifest), "images": len(records),
        "downloaded": sum(item["status"] == "downloaded" for item in records),
        "reused": sum(item["status"] == "reused" for item in records),
        "bytes": sum(item["bytes"] for item in records),
        "seconds": time.perf_counter() - started, "files": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "download_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()

