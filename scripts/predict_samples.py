#!/usr/bin/env python3
"""Export qualitative prediction grids for a checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgb2nir_pix2pix.config import load_config
from rgb2nir_pix2pix.dataset import make_dataloader
from rgb2nir_pix2pix.train import load_generator_from_checkpoint
from rgb2nir_pix2pix.utils import select_device
from rgb2nir_pix2pix.visualize import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pix2pix_mps.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-items", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = select_device(str(cfg.get("train", "device", "mps")))
    generator = load_generator_from_checkpoint(
        checkpoint_path=args.checkpoint,
        generator_base=int(cfg.get("model", "generator_base_channels", 64)),
        discriminator_base=int(cfg.get("model", "discriminator_base_channels", 64)),
        dropout=float(cfg.get("model", "dropout", 0.5)),
        generator_type=str(cfg.get("model", "generator_type", "transposed")),
        device=device,
    )
    loader = make_dataloader(
        cfg.path_value("test_csv"),
        batch_size=max(args.max_items, 1),
        image_size=int(cfg.get("data", "image_size", 128)),
        augment=False,
        augmentation_cfg=None,
        max_samples=args.max_items,
        seed=cfg.seed + 777,
        shuffle=False,
        num_workers=0,
        allowed_root=cfg.crop_root,
        expected_source_split="selected",
        expected_category="clean",
    )
    batch = next(iter(loader))
    rgb = batch["rgb"].to(device)
    target = batch["nir"].to(device)
    with torch.no_grad():
        pred = generator(rgb)
    output = args.output or args.checkpoint.parent.parent / "samples" / "prediction_grid.png"
    save_prediction_grid(
        output,
        rgb,
        pred,
        target,
        max_items=args.max_items,
        vessel_sigmas=list(cfg.get("evaluation", "vessel_sigmas", [1, 2, 3])),
        vessel_threshold=float(cfg.get("evaluation", "vessel_threshold", 0.15)),
    )
    print(output)


if __name__ == "__main__":
    main()
