#!/usr/bin/env python3
"""Run a dataloader and visual smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgb2nir_pix2pix.config import load_config
from rgb2nir_pix2pix.dataset import make_dataloader
from rgb2nir_pix2pix.utils import set_seed, write_json
from rgb2nir_pix2pix.visualize import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pix2pix_mps.yaml")
    parser.add_argument("--max-samples", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    loader = make_dataloader(
        cfg.path_value("train_csv"),
        batch_size=int(cfg.get("data", "batch_size", 8)),
        image_size=int(cfg.get("data", "image_size", 128)),
        augment=True,
        augmentation_cfg=cfg.raw.get("augmentation", {}),
        max_samples=args.max_samples,
        seed=cfg.seed,
        shuffle=True,
        num_workers=0,
        allowed_root=cfg.crop_root,
        expected_source_split="selected",
        expected_category="clean",
    )
    batch = next(iter(loader))
    rgb = batch["rgb"]
    nir = batch["nir"]
    checks = {
        "batch_rgb_shape": list(rgb.shape),
        "batch_nir_shape": list(nir.shape),
        "rgb_min": float(rgb.min()),
        "rgb_max": float(rgb.max()),
        "nir_min": float(nir.min()),
        "nir_max": float(nir.max()),
        "rgb_finite": bool(torch.isfinite(rgb).all()),
        "nir_finite": bool(torch.isfinite(nir).all()),
        "sample_pair_ids": list(batch["pair_id"][: min(5, len(batch["pair_id"]))]),
    }
    if checks["batch_rgb_shape"][1:] != [3, 128, 128]:
        raise ValueError(checks)
    if checks["batch_nir_shape"][1:] != [1, 128, 128]:
        raise ValueError(checks)
    save_prediction_grid(
        ROOT / "sample_grids" / "smoke_rgb_nir_grid.png",
        rgb,
        nir,
        nir,
        max_items=min(8, rgb.shape[0]),
    )
    write_json(ROOT / "reports" / "smoke_test.json", checks)
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
