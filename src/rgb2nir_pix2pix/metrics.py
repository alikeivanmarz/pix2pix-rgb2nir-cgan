"""Image and vessel metrics for predicted NIR images."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from skimage.filters import gaussian, meijering, sato, threshold_otsu
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_opening,
    disk,
    remove_small_objects,
    skeletonize,
)

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


def normalize01(arr: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.7) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [low_percentile, high_percentile])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _adaptive_line_mask(vessel: np.ndarray, min_size: int = 18) -> np.ndarray:
    nonzero = vessel[vessel > 0]
    if nonzero.size < 10:
        return np.zeros_like(vessel, dtype=bool)
    try:
        threshold = float(threshold_otsu(nonzero))
    except ValueError:
        threshold = float(np.percentile(nonzero, 90))
    threshold = max(threshold, float(np.percentile(nonzero, 85)))
    mask = vessel >= threshold
    mask = binary_opening(mask, footprint=disk(1))
    mask = remove_small_objects(mask, min_size=min_size)

    labeled = label(mask)
    keep = np.zeros_like(mask, dtype=bool)
    for region in regionprops(labeled):
        if region.area < min_size:
            continue
        component = labeled == region.label
        skeleton_len = int(skeletonize(component).sum())
        if region.major_axis_length >= 10 and (region.eccentricity >= 0.65 or skeleton_len >= 12):
            keep[component] = True
    keep = binary_closing(keep, footprint=disk(1))
    keep = binary_dilation(keep, footprint=disk(1))
    return keep


def _vein_scale_sigmas(sigmas: list[int] | tuple[int, ...]) -> list[int]:
    max_sigma = max([int(s) for s in sigmas] + [3])
    return list(range(2, max(max_sigma, 6) + 1))


def sato_smooth_map(img: np.ndarray, sigmas: list[int] | tuple[int, ...]) -> np.ndarray:
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    smooth = gaussian(img, sigma=0.8, preserve_range=True)
    response = sato(smooth, sigmas=[1, 2, 3, 4], black_ridges=True)
    return normalize01(response, 1.0, 99.7)


def meijering_smooth_map(img: np.ndarray, sigmas: list[int] | tuple[int, ...]) -> np.ndarray:
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    smooth = gaussian(img, sigma=1.25, preserve_range=True)
    response = meijering(smooth, sigmas=_vein_scale_sigmas(sigmas), black_ridges=True)
    return normalize01(response, 1.0, 99.7)


def dark_line_response_map(img: np.ndarray, sigmas: list[int] | tuple[int, ...]) -> np.ndarray:
    img = np.clip(img.astype(np.float32), 0.0, 1.0)
    smooth = gaussian(img, sigma=1.25, preserve_range=True)
    background = gaussian(smooth, sigma=14.0, preserve_range=True)
    dark_lines = np.maximum(background - smooth, 0.0)
    dark_lines = normalize01(dark_lines, 75.0, 99.5)
    ridge = sato(smooth, sigmas=_vein_scale_sigmas(sigmas), black_ridges=True)
    ridge = normalize01(ridge, 50.0, 99.7)
    return normalize01(0.75 * dark_lines + 0.25 * ridge, 1.0, 99.7)


def filtered_vein_map(img: np.ndarray, sigmas: list[int] | tuple[int, ...]) -> np.ndarray:
    """Continuous vein evidence map with light speckle suppression.

    This keeps the readable Sato/Meijering/dark-line response that exposes vein
    patterns, while dimming tiny isolated responses before thresholded metrics.
    """
    sato_map = sato_smooth_map(img, sigmas)
    meijering_map = meijering_smooth_map(img, sigmas)
    dark_map = dark_line_response_map(img, sigmas)
    combined = normalize01(0.35 * sato_map + 0.35 * meijering_map + 0.30 * dark_map, 1.0, 99.7)
    support = _adaptive_line_mask(combined, min_size=12)
    if support.any():
        support = binary_dilation(support, footprint=disk(1))
        combined = combined * (0.35 + 0.65 * support.astype(np.float32))
    combined = gaussian(combined, sigma=0.35, preserve_range=True)
    combined[:2, :] = 0.0
    combined[-2:, :] = 0.0
    combined[:, :2] = 0.0
    combined[:, -2:] = 0.0
    return normalize01(combined, 0.0, 99.7)


def vesselness_map(img: np.ndarray, sigmas: list[int] | tuple[int, ...]) -> np.ndarray:
    """Default visual/evaluation vein evidence map.

    The displayed map uses sensitive Sato settings because they preserve fine
    vein evidence. Filtering is reserved for binary masks, not the displayed
    map.
    """
    return sato_smooth_map(img, sigmas)


def vessel_mask(vessel: np.ndarray, threshold: float) -> np.ndarray:
    if float(vessel.max()) <= 0.0:
        return np.zeros_like(vessel, dtype=bool)
    mask = vessel >= threshold
    mask = binary_opening(mask, footprint=disk(1))
    mask = remove_small_objects(mask, min_size=18)
    return mask


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
    pm = vessel_mask(pv, threshold)
    tm = vessel_mask(tv, threshold)
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
