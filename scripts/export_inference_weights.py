#!/usr/bin/env python3
"""Export generator-only inference weights from a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_size_mb(state_dict: dict[str, torch.Tensor]) -> float:
    total = sum(t.numel() * t.element_size() for t in state_dict.values() if torch.is_tensor(t))
    return total / 1024 / 1024


def save_variant(
    state_dict: dict[str, torch.Tensor],
    output_path: Path,
    metadata: dict[str, str],
    dtype: torch.dtype,
) -> dict[str, Any]:
    tensors = {
        key: value.detach().cpu().to(dtype=dtype).contiguous()
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata=metadata)
    return {
        "filename": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "dtype": "float16" if dtype == torch.float16 else "float32",
        "tensor_mb": round(tensor_size_mb(tensors), 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release_assets")
    parser.add_argument("--name", default="rgb2nir-pix2pix-generator")
    parser.add_argument("--generator-type", default="transposed")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--include-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "generator" not in payload:
        raise ValueError("Expected a training checkpoint with a 'generator' state dict")

    metrics = payload.get("metrics", {})
    metadata = {
        "model": "pix2pix-rgb2nir-cgan",
        "architecture": "U-Net generator",
        "generator_type": args.generator_type,
        "image_size": str(args.image_size),
        "epoch": str(payload.get("epoch", "")),
        "step": str(payload.get("step", "")),
        "metrics": json.dumps(metrics, sort_keys=True),
    }
    state_dict = payload["generator"]
    artifacts: list[dict[str, Any]] = []
    artifacts.append(
        save_variant(
            state_dict,
            args.output_dir / f"{args.name}-fp32.safetensors",
            metadata | {"dtype": "float32"},
            torch.float32,
        )
    )
    if args.include_fp16:
        artifacts.append(
            save_variant(
                state_dict,
                args.output_dir / f"{args.name}-fp16.safetensors",
                metadata | {"dtype": "float16"},
                torch.float16,
            )
        )

    manifest = {
        "source_checkpoint": args.checkpoint.name,
        "artifacts": artifacts,
    }
    manifest_path = args.output_dir / "weights_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{item['sha256']}  {item['filename']}\n" for item in artifacts),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
