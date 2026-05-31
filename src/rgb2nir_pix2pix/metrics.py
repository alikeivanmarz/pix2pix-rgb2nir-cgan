"""Image and vessel metrics for predicted NIR images."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from skimage.filters import frangi, threshold_otsu
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize

from .dataset import tensor_to_unit


def batch_to_numpy_01(batch: torch.Tensor) -> np.ndarray:
    return tensor_to_unit(batch.detach().cpu()).numpy()


def image_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    target = np.clip(target.astype(np.float32), 0.0, 1.0)
    diff = pred - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff**2))
    rmse = float(math.sqrt(mse))
    psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-8)))
    ssim = float(structural_similarity(target, pred, data_range=1.0))
    return {"mae": mae, "rmse": rmse, "psnr": psnr, "ssim": ssim}


def vesselness_map(img: np.ndarray, sigmas: list[int] | tuple[int, ...]) -> np.ndarray:
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    vessel = frangi(img, sigmas=sigmas, black_ridges=True)
    vessel = np.nan_to_num(vessel, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(vessel.max())
    if vmax > 0:
        vessel = vessel / vmax
    return vessel.astype(np.float32)


def calibrate_threshold(images: list[np.ndarray], sigmas: list[int]) -> float:
    values = []
    for img in images:
        vessel = vesselness_map(img, sigmas)
        values.append(vessel.reshape(-1))
    if not values:
        return 0.15
    merged = np.concatenate(values)
    nonzero = merged[merged > 0]
    if len(nonzero) < 10:
        return 0.15
    return float(threshold_otsu(nonzero))


def vessel_metrics(
    pred: np.ndarray, target: np.ndarray, sigmas: list[int], threshold: float
) -> dict[str, float]:
    pv = vesselness_map(pred, sigmas)
    tv = vesselness_map(target, sigmas)
    if np.std(pv) < 1e-8 or np.std(tv) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(pv.reshape(-1), tv.reshape(-1))[0, 1])
    pm = pv >= threshold
    tm = tv >= threshold
    tp = float(np.logical_and(pm, tm).sum())
    fp = float(np.logical_and(pm, ~tm).sum())
    fn = float(np.logical_and(~pm, tm).sum())
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    ps = skeletonize(pm)
    ts = skeletonize(tm)
    stp = float(np.logical_and(ps, ts).sum())
    sfp = float(np.logical_and(ps, ~ts).sum())
    sfn = float(np.logical_and(~ps, ts).sum())
    skeleton_f1 = 2.0 * stp / max(2.0 * stp + sfp + sfn, 1.0)
    length_ratio = float(ps.sum()) / max(float(ts.sum()), 1.0)
    return {
        "vessel_corr": corr,
        "vessel_dice": float(dice),
        "vessel_iou": float(iou),
        "skeleton_f1": float(skeleton_f1),
        "vessel_length_ratio": length_ratio,
    }


def summarize_records(records: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    numeric_keys = [
        "mae",
        "rmse",
        "psnr",
        "ssim",
        "vessel_corr",
        "vessel_dice",
        "vessel_iou",
        "skeleton_f1",
        "vessel_length_ratio",
    ]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if group_key is None:
        groups["overall"] = records
    else:
        for record in records:
            groups[str(record.get(group_key, "unknown"))].append(record)
    summary = []
    for group, rows in groups.items():
        out: dict[str, Any] = {"group": group, "n": len(rows)}
        for key in numeric_keys:
            vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
            if vals:
                out[key] = float(np.mean(vals))
        summary.append(out)
    return sorted(summary, key=lambda row: str(row["group"]))

