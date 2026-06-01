#!/usr/bin/env python3
"""Evaluate release weights against RGB-derived baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgb2nir_pix2pix.config import load_config
from rgb2nir_pix2pix.dataset import make_dataloader, tensor_to_unit
from rgb2nir_pix2pix.inference import load_generator
from rgb2nir_pix2pix.metrics import image_metrics, vessel_mask, vesselness_map
from rgb2nir_pix2pix.utils import environment_snapshot, select_device, write_json


METRIC_COLUMNS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pix2pix release weights and simple RGB baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pix2pix_mps.yaml")
    parser.add_argument("--weights", type=Path, required=True, help="Generator weights to evaluate.")
    parser.add_argument("--test-csv", type=Path, help="Test manifest CSV. Defaults to config path.")
    parser.add_argument("--train-csv", type=Path, help="Train manifest CSV for fitted baselines.")
    parser.add_argument("--crop-root", type=Path, help="Allowed crop root. Defaults to config path.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "reproducible_evaluation")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--include-vein", action="store_true", help="Compute Sato vein-map metrics.")
    parser.add_argument("--write-per-crop", action="store_true", help="Write all per-crop metric rows.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return select_device(name)


def fit_train_baselines(
    train_csv: Path,
    crop_root: Path,
    image_size: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    loader = make_dataloader(
        train_csv,
        batch_size=batch_size,
        image_size=image_size,
        augment=False,
        augmentation_cfg=None,
        max_samples=None,
        seed=seed,
        shuffle=False,
        num_workers=0,
        allowed_root=crop_root,
        expected_source_split="selected",
        expected_category="clean",
    )
    count = 0
    lum_sum = 0.0
    lum_sq_sum = 0.0
    nir_sum = 0.0
    nir_sq_sum = 0.0
    mean_nir_image = np.zeros((image_size, image_size), dtype=np.float64)

    for batch in tqdm(loader, desc="fit baselines", leave=False):
        rgb = tensor_to_unit(batch["rgb"]).numpy()
        nir = tensor_to_unit(batch["nir"]).numpy()[:, 0]
        luminance = rgb_luminance(rgb)
        n = int(np.prod(luminance.shape))
        count += n
        lum_sum += float(luminance.sum())
        lum_sq_sum += float((luminance**2).sum())
        nir_sum += float(nir.sum())
        nir_sq_sum += float((nir**2).sum())
        mean_nir_image += nir.sum(axis=0)

    if count == 0:
        raise ValueError(f"No training rows found in {train_csv}")
    lum_mean = lum_sum / count
    nir_mean = nir_sum / count
    lum_var = max(lum_sq_sum / count - lum_mean**2, 1e-12)
    nir_var = max(nir_sq_sum / count - nir_mean**2, 1e-12)
    return {
        "luminance_mean": float(lum_mean),
        "luminance_std": float(np.sqrt(lum_var)),
        "nir_mean": float(nir_mean),
        "nir_std": float(np.sqrt(nir_var)),
        "mean_nir_image": (mean_nir_image / len(loader.dataset)).astype(np.float32),
        "train_rows": len(loader.dataset),
    }


def rgb_luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]


def baseline_predictions(rgb: np.ndarray, stats: dict[str, Any]) -> dict[str, np.ndarray]:
    luminance = rgb_luminance(rgb)
    green = rgb[:, 1]
    adjusted = (luminance - stats["luminance_mean"]) / max(stats["luminance_std"], 1e-8)
    adjusted = adjusted * stats["nir_std"] + stats["nir_mean"]
    mean_image = np.broadcast_to(stats["mean_nir_image"], luminance.shape)
    return {
        "rgb_luminance": np.clip(luminance, 0.0, 1.0).astype(np.float32),
        "green_channel": np.clip(green, 0.0, 1.0).astype(np.float32),
        "mean_std_luminance": np.clip(adjusted, 0.0, 1.0).astype(np.float32),
        "train_mean_nir": np.clip(mean_image, 0.0, 1.0).astype(np.float32),
    }


def vessel_metrics_from_target(
    pred: np.ndarray,
    target_vessel: np.ndarray,
    target_mask: np.ndarray,
    target_skeleton: np.ndarray,
    sigmas: list[int],
    threshold: float,
) -> dict[str, float]:
    pred_vessel = vesselness_map(pred, sigmas)
    if np.std(pred_vessel) < 1e-8 or np.std(target_vessel) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(pred_vessel.reshape(-1), target_vessel.reshape(-1))[0, 1])
    pred_mask = vessel_mask(pred_vessel, threshold)
    tp = float(np.logical_and(pred_mask, target_mask).sum())
    fp = float(np.logical_and(pred_mask, ~target_mask).sum())
    fn = float(np.logical_and(~pred_mask, target_mask).sum())
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    pred_skeleton = skeletonize(pred_mask)
    stp = float(np.logical_and(pred_skeleton, target_skeleton).sum())
    sfp = float(np.logical_and(pred_skeleton, ~target_skeleton).sum())
    sfn = float(np.logical_and(~pred_skeleton, target_skeleton).sum())
    skeleton_f1 = 2.0 * stp / max(2.0 * stp + sfp + sfn, 1.0)
    length_ratio = float(pred_skeleton.sum()) / max(float(target_skeleton.sum()), 1.0)
    return {
        "vessel_corr": corr,
        "vessel_dice": float(dice),
        "vessel_iou": float(iou),
        "skeleton_f1": float(skeleton_f1),
        "vessel_length_ratio": length_ratio,
    }


def metric_summary(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [col for col in METRIC_COLUMNS if col in rows.columns]
    grouped = rows.groupby(group_cols, dropna=False)
    means = grouped[numeric_cols].mean().reset_index()
    means.insert(len(group_cols), "n", grouped.size().to_numpy())
    return means


def subject_average(rows: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in METRIC_COLUMNS if col in rows.columns]
    per_subject = metric_summary(rows, ["method", "participant"])
    grouped = per_subject.groupby("method", dropna=False)
    out = grouped[numeric_cols].mean().reset_index()
    for col in numeric_cols:
        out[f"{col}_subject_std"] = grouped[col].std(ddof=0).to_numpy()
    out.insert(1, "n_subjects", grouped.size().to_numpy())
    out.insert(2, "n_crops", rows.groupby("method").size().reindex(out["method"]).to_numpy())
    return out


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    lines = []
    header = [labels.get(col, col) for col in columns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |")
    for _, row in frame.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    crop_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    provenance: dict[str, Any],
    include_vein: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_cols = ["method", "n", "mae", "rmse", "psnr", "ssim"]
    subject_cols = ["method", "n_subjects", "n_crops", "mae", "rmse", "psnr", "ssim"]
    lines = [
        "# Reproducible Evaluation",
        "",
        "Evaluation uses the released fp32 generator weights and the participant-disjoint selected-clean test split.",
        "",
        "## Provenance",
        "",
        f"- Weights SHA256: `{provenance['weights_sha256']}`",
        f"- Test manifest SHA256: `{provenance['test_manifest_sha256']}`",
        f"- Train manifest SHA256: `{provenance['train_manifest_sha256']}`",
        f"- Test crops: {provenance['test_rows']}",
        f"- Test subjects: {provenance['test_subject_count']}",
        f"- Device: {provenance['device']}",
        f"- Evaluation generated: {provenance['generated_utc']}",
        "",
        "Reproduce with:",
        "",
        "```bash",
        provenance["command_template"],
        "```",
        "",
        "The training history CSV reports validation metrics during training. The tables below are independent test-set metrics.",
        "",
        "## Crop-Level Mean Metrics",
        "",
        markdown_table(crop_summary, image_cols),
        "",
        "## Subject-Averaged Metrics",
        "",
        markdown_table(subject_summary, subject_cols),
    ]
    if include_vein:
        vein_cols = ["method", "n", "vessel_corr", "vessel_dice", "vessel_iou", "skeleton_f1"]
        subject_vein_cols = ["method", "n_subjects", "vessel_corr", "vessel_dice", "vessel_iou", "skeleton_f1"]
        lines.extend(
            [
                "",
                "## Full-Test Sato Vein-Map Metrics",
                "",
                markdown_table(crop_summary, vein_cols),
                "",
                "## Subject-Averaged Sato Vein-Map Metrics",
                "",
                markdown_table(subject_summary, subject_vein_cols),
            ]
        )
    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "- `rgb_luminance`: standard RGB luminance converted directly to grayscale.",
            "- `green_channel`: the RGB green channel used directly as a grayscale estimate.",
            "- `mean_std_luminance`: RGB luminance matched to train-set NIR mean and standard deviation.",
            "- `train_mean_nir`: the train-set pixelwise mean NIR image.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_csv = args.test_csv or cfg.path_value("test_csv")
    train_csv = args.train_csv or cfg.path_value("train_fit_csv")
    crop_root = args.crop_root or cfg.crop_root
    device = choose_device(args.device)

    stats = fit_train_baselines(train_csv, crop_root, args.image_size, args.batch_size, cfg.seed)
    generator = load_generator(
        args.weights,
        generator_base=int(cfg.get("model", "generator_base_channels", 64)),
        discriminator_base=int(cfg.get("model", "discriminator_base_channels", 64)),
        dropout=float(cfg.get("model", "dropout", 0.5)),
        generator_type=str(cfg.get("model", "generator_type", "transposed")),
        device=device,
    )
    loader = make_dataloader(
        test_csv,
        batch_size=args.batch_size,
        image_size=args.image_size,
        augment=False,
        augmentation_cfg=None,
        max_samples=args.max_samples,
        seed=cfg.seed + 111,
        shuffle=False,
        num_workers=0,
        allowed_root=crop_root,
        expected_source_split="selected",
        expected_category="clean",
    )

    sigmas = list(cfg.get("evaluation", "vessel_sigmas", [1, 2, 3]))
    threshold = float(cfg.get("evaluation", "vessel_threshold", 0.15))
    records: list[dict[str, Any]] = []

    generator.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate benchmarks", leave=False):
            rgb = batch["rgb"].to(device)
            target = batch["nir"].to(device)
            pred_model = generator(rgb)
            rgb_np = tensor_to_unit(batch["rgb"]).numpy()
            target_np = tensor_to_unit(target.detach().cpu()).numpy()[:, 0]
            predictions = {"pix2pix_cgan": tensor_to_unit(pred_model.detach().cpu()).numpy()[:, 0]}
            predictions.update(baseline_predictions(rgb_np, stats))

            for i in range(target_np.shape[0]):
                target_vessel = None
                target_mask = None
                target_skeleton = None
                if args.include_vein:
                    target_vessel = vesselness_map(target_np[i], sigmas)
                    target_mask = vessel_mask(target_vessel, threshold)
                    target_skeleton = skeletonize(target_mask)
                base_record = {
                    "participant": batch["participant"][i],
                    "source_split": batch["source_split"][i],
                    "category": batch["category"][i],
                    "hand": batch["hand"][i],
                    "pair_id": batch["pair_id"][i],
                }
                for method, pred_batch in predictions.items():
                    record = {"method": method, **base_record}
                    record.update(image_metrics(pred_batch[i], target_np[i]))
                    if args.include_vein:
                        record.update(
                            vessel_metrics_from_target(
                                pred_batch[i],
                                target_vessel,
                                target_mask,
                                target_skeleton,
                                sigmas,
                                threshold,
                            )
                        )
                    records.append(record)

    rows = pd.DataFrame(records)
    method_order = ["pix2pix_cgan", "mean_std_luminance", "rgb_luminance", "green_channel", "train_mean_nir"]
    crop_summary = metric_summary(rows, ["method"]).set_index("method").reindex(method_order).dropna(how="all")
    crop_summary = crop_summary.reset_index()
    per_subject = metric_summary(rows, ["method", "participant"]).sort_values(["method", "participant"])
    subject_summary = subject_average(rows).set_index("method").reindex(method_order).dropna(how="all").reset_index()

    write_csv(output_dir / "benchmark_crop_mean.csv", crop_summary)
    write_csv(output_dir / "benchmark_subject_mean.csv", subject_summary)
    write_csv(output_dir / "per_subject_metrics.csv", per_subject)
    if args.write_per_crop:
        write_csv(output_dir / "per_crop_metrics.csv", rows)

    env = environment_snapshot()
    public_env = {
        "python": env.get("python"),
        "platform": env.get("platform"),
        "machine": env.get("machine"),
        "torch": env.get("torch"),
        "mps_available": env.get("mps_available"),
        "cuda_available": env.get("cuda_available"),
        "git_commit": env.get("git_commit"),
    }
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/evaluate_benchmarks.py",
        "command_template": (
            "python scripts/evaluate_benchmarks.py "
            "--weights weights/rgb2nir-pix2pix-generator-fp32.safetensors "
            "--train-csv manifests/train_fit_selected_clean.csv "
            "--test-csv manifests/test_selected_clean.csv "
            "--crop-root data/crops/selected/clean "
            "--output-dir reports/reproducible_evaluation "
            "--include-vein"
        ),
        "weights_file": args.weights.name,
        "weights_sha256": sha256_file(args.weights),
        "test_manifest_file": test_csv.name,
        "test_manifest_sha256": sha256_file(test_csv),
        "train_manifest_file": train_csv.name,
        "train_manifest_sha256": sha256_file(train_csv),
        "train_rows": count_csv_rows(train_csv),
        "test_rows": len(loader.dataset),
        "test_subject_count": int(rows["participant"].nunique()),
        "test_subjects": sorted(
            rows["participant"].astype(str).unique().tolist(),
            key=lambda x: int(x) if x.isdigit() else x,
        ),
        "dataset_scope": "data/crops/selected/clean",
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "include_vein": bool(args.include_vein),
        "vessel_sigmas": sigmas,
        "vessel_threshold": threshold,
        "model": {
            "generator_type": str(cfg.get("model", "generator_type", "transposed")),
            "generator_base_channels": int(cfg.get("model", "generator_base_channels", 64)),
            "dropout": float(cfg.get("model", "dropout", 0.5)),
        },
        "training_objective": {
            "adversarial": str(cfg.get("loss", "adversarial", "bce")),
            "lambda_l1": float(cfg.get("loss", "lambda_l1", 100.0)),
            "lambda_gradient": float(cfg.get("loss", "lambda_gradient", 0.0) or 0.0),
            "lambda_vessel": float(cfg.get("loss", "lambda_vessel", 0.0) or 0.0),
            "lambda_highpass": float(cfg.get("loss", "lambda_highpass", 0.0) or 0.0),
        },
        "baseline_fit": {
            key: value for key, value in stats.items() if key != "mean_nir_image"
        },
        "environment": public_env,
    }
    write_json(output_dir / "evaluation_provenance.json", provenance)
    write_report(output_dir / "evaluation_summary.md", crop_summary, subject_summary, provenance, args.include_vein)
    print(output_dir)


if __name__ == "__main__":
    main()
