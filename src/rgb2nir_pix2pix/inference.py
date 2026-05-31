"""Inference helpers for the RGB-to-NIR generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import pil_rgb_to_tensor, tensor_to_unit
from .models import build_models


def load_generator(
    weights_path: str | Path,
    generator_base: int = 64,
    discriminator_base: int = 64,
    dropout: float = 0.5,
    generator_type: str = "transposed",
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load an inference-only generator from safetensors or a PyTorch checkpoint."""
    weights_path = Path(weights_path)
    generator, _ = build_models(
        generator_base=generator_base,
        discriminator_base=discriminator_base,
        dropout=dropout,
        generator_type=generator_type,
    )

    if weights_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(weights_path), device="cpu")
    else:
        payload = torch.load(weights_path, map_location="cpu")
        state_dict = payload["generator"] if isinstance(payload, dict) and "generator" in payload else payload

    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    return generator


def predict_image(
    generator: torch.nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    image_size: int = 128,
    device: str | torch.device = "cpu",
    resize: bool = True,
) -> Path:
    """Run generator inference on one RGB image and save an 8-bit grayscale PNG."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = Image.open(input_path).convert("RGB")
    if image.size != (image_size, image_size):
        if not resize:
            raise ValueError(f"Expected {image_size}x{image_size}, got {image.size}: {input_path}")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)

    tensor = pil_rgb_to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = generator(tensor)
    pred_unit = tensor_to_unit(pred.squeeze(0).cpu()).squeeze(0).numpy()
    out = np.clip(pred_unit * 255.0, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(output_path)
    return output_path
