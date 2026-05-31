#!/usr/bin/env python3
"""Predict a NIR-like image from one RGB crop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgb2nir_pix2pix.inference import load_generator, predict_image
from rgb2nir_pix2pix.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--generator-type", default="transposed")
    parser.add_argument("--generator-base", type=int, default=64)
    parser.add_argument("--discriminator-base", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-resize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = select_device(device_name)
    generator = load_generator(
        args.weights,
        generator_base=args.generator_base,
        discriminator_base=args.discriminator_base,
        dropout=args.dropout,
        generator_type=args.generator_type,
        device=device,
    )
    output = predict_image(
        generator,
        input_path=args.input,
        output_path=args.output,
        image_size=args.image_size,
        device=device,
        resize=not args.no_resize,
    )
    print(output)


if __name__ == "__main__":
    main()
