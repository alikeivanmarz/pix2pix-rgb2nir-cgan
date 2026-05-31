#!/usr/bin/env python3
"""Evaluate a pix2pix checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgb2nir_pix2pix.config import load_config
from rgb2nir_pix2pix.dataset import make_dataloader
from rgb2nir_pix2pix.train import evaluate_generator, load_generator_from_checkpoint, write_evaluation_outputs
from rgb2nir_pix2pix.utils import append_journal, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pix2pix_mps.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
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
    output_dir = args.output_dir or args.checkpoint.parent.parent / "evaluation"
    max_samples = args.max_samples
    if max_samples is None and cfg.get("evaluation", "max_samples") is not None:
        max_samples = int(cfg.get("evaluation", "max_samples"))
    loader = make_dataloader(
        cfg.path_value("test_csv"),
        batch_size=int(cfg.get("evaluation", "batch_size", 16)),
        image_size=int(cfg.get("data", "image_size", 128)),
        augment=False,
        augmentation_cfg=None,
        max_samples=max_samples,
        seed=cfg.seed + 111,
        shuffle=False,
        num_workers=0,
        allowed_root=cfg.crop_root,
        expected_source_split="selected",
        expected_category="clean",
    )
    result = evaluate_generator(
        generator=generator,
        dataloader=loader,
        device=device,
        vessel_sigmas=list(cfg.get("evaluation", "vessel_sigmas", [1, 2, 3])),
        vessel_threshold=float(cfg.get("evaluation", "vessel_threshold", 0.15)),
        include_vessel=True,
    )
    write_evaluation_outputs(result, output_dir)
    append_journal(cfg.project_root, f"Evaluation complete for {args.checkpoint}: {output_dir}")
    print(output_dir)
    print(result["overall"])


if __name__ == "__main__":
    main()
