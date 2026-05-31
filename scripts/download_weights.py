#!/usr/bin/env python3
"""Download published generator weights from GitHub Releases."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RELEASE_TAG = "v0.1.0"
BASE_URL = (
    "https://github.com/alikeivanmarz/pix2pix-rgb2nir-cgan/releases/download/"
    f"{RELEASE_TAG}"
)
WEIGHTS = {
    "fp32": {
        "filename": "rgb2nir-pix2pix-generator-fp32.safetensors",
        "sha256": "f0928bb058e9323dbdfe552ed0efa83f0cee335727a0ee76190759164bd26c4e",
    },
    "fp16": {
        "filename": "rgb2nir-pix2pix-generator-fp16.safetensors",
        "sha256": "a5537213b378fa6c747b84f45d3a0e9ec810185762282e7fef361192a23ce948",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(WEIGHTS), default="fp32")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = WEIGHTS[args.variant]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / info["filename"]
    if output_path.exists() and not args.force:
        print(output_path)
        return

    url = f"{BASE_URL}/{info['filename']}"
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {url}\n{exc}") from exc

    expected = info["sha256"]
    if expected and not expected.startswith("TO_BE_FILLED"):
        actual = sha256_file(output_path)
        if actual != expected:
            output_path.unlink(missing_ok=True)
            raise SystemExit(f"Checksum mismatch for {output_path.name}: {actual} != {expected}")
    print(output_path)


if __name__ == "__main__":
    main()
