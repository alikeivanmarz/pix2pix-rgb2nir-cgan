"""Visualization helpers for sample grids."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import tensor_to_unit
from .metrics import vesselness_map


def save_prediction_grid(
    path: Path,
    rgb: torch.Tensor,
    pred_nir: torch.Tensor,
    target_nir: torch.Tensor,
    max_items: int = 8,
    vessel_sigmas: list[int] | None = None,
    vessel_threshold: float = 0.15,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vessel_sigmas = vessel_sigmas or [1, 2, 3]
    rgb_np = tensor_to_unit(rgb.detach().cpu()).numpy()
    pred_np = tensor_to_unit(pred_nir.detach().cpu()).numpy()
    target_np = tensor_to_unit(target_nir.detach().cpu()).numpy()
    n = min(max_items, rgb_np.shape[0])
    cols = 7
    fig, axes = plt.subplots(n, cols, figsize=(cols * 2.2, n * 2.2), squeeze=False)
    titles = [
        "RGB",
        "Pred NIR",
        "Real NIR",
        "Abs Error",
        "Pred Vessel",
        "Real Vessel",
        "Overlay",
    ]
    for col, title in enumerate(titles):
        axes[0][col].set_title(title, fontsize=9)
    for i in range(n):
        rgb_img = np.transpose(rgb_np[i], (1, 2, 0))
        pred = pred_np[i, 0]
        target = target_np[i, 0]
        error = np.abs(pred - target)
        pred_v = vesselness_map(pred, vessel_sigmas)
        target_v = vesselness_map(target, vessel_sigmas)
        pred_m = pred_v >= vessel_threshold
        target_m = target_v >= vessel_threshold
        overlay = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
        overlay[..., 1] = np.logical_and(pred_m, target_m)
        overlay[..., 0] = np.logical_and(pred_m, ~target_m)
        overlay[..., 2] = np.logical_and(~pred_m, target_m)
        images = [rgb_img, pred, target, error, pred_v, target_v, overlay]
        cmaps = [None, "gray", "gray", "magma", "gray", "gray", None]
        for col, image in enumerate(images):
            axes[i][col].imshow(image, cmap=cmaps[col], vmin=0, vmax=1)
            axes[i][col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

