"""Training and evaluation loops for pix2pix."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import Pix2PixConfig
from .dataset import make_dataloader
from .losses import Pix2PixLoss
from .metrics import batch_to_numpy_01, image_metrics, summarize_records, vessel_metrics
from .models import build_models
from .utils import AverageMeter, environment_snapshot, select_device, set_seed, write_json, write_yaml
from .visualize import save_prediction_grid


def config_int(cfg: Pix2PixConfig, section: str, key: str, default: int) -> int:
    value = cfg.get(section, key, default)
    return default if value is None else int(value)


def config_float(cfg: Pix2PixConfig, section: str, key: str, default: float) -> float:
    value = cfg.get(section, key, default)
    return default if value is None else float(value)


def config_bool(cfg: Pix2PixConfig, section: str, key: str, default: bool) -> bool:
    value = cfg.get(section, key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def choose_training_csv(cfg: Pix2PixConfig) -> Path:
    train_fit = cfg.raw.get("paths", {}).get("train_fit_csv")
    if train_fit:
        train_fit_path = Path(train_fit)
        if train_fit_path.exists():
            return train_fit_path
    return cfg.path_value("train_csv")


def choose_eval_csv(cfg: Pix2PixConfig) -> tuple[Path, str]:
    val_csv = cfg.raw.get("paths", {}).get("val_csv")
    if val_csv:
        val_path = Path(val_csv)
        if val_path.exists():
            return val_path, "validation"
    return cfg.path_value("test_csv"), "test"


def vein_checkpoint_score(metrics: dict[str, Any]) -> float:
    return (
        0.45 * float(metrics.get("vessel_dice", 0.0))
        + 0.25 * float(metrics.get("skeleton_f1", 0.0))
        + 0.20 * float(metrics.get("vessel_corr", 0.0))
        + 0.10 * float(metrics.get("ssim", 0.0))
    )


def make_run_dir(cfg: Pix2PixConfig, run_name: str) -> Path:
    run_dir = cfg.path_value("runs_dir") / run_name
    for child in ["checkpoints", "metrics", "samples", "logs"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config_snapshot.yaml", cfg.raw)
    write_json(run_dir / "environment.json", environment_snapshot())
    return run_dir


def save_checkpoint(
    path: Path,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def load_generator_from_checkpoint(
    checkpoint_path: Path,
    generator_base: int,
    discriminator_base: int,
    dropout: float,
    generator_type: str,
    device: torch.device,
) -> torch.nn.Module:
    generator, _ = build_models(generator_base, discriminator_base, dropout, generator_type)
    payload = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(payload["generator"])
    generator.to(device)
    generator.eval()
    return generator


def train_pix2pix(
    cfg: Pix2PixConfig,
    run_name: str,
    epochs: int | None = None,
    max_steps: int | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> Path:
    set_seed(cfg.seed)
    device = select_device(str(cfg.get("train", "device", "mps")))
    run_dir = make_run_dir(cfg, run_name)

    batch_size = config_int(cfg, "data", "batch_size", 8)
    image_size = config_int(cfg, "data", "image_size", 128)
    train_csv = choose_training_csv(cfg)
    eval_csv, eval_split_name = choose_eval_csv(cfg)

    train_limit = max_train_samples
    if train_limit is None:
        raw_limit = cfg.get("data", "max_train_samples")
        train_limit = None if raw_limit is None else int(raw_limit)
    eval_limit = max_eval_samples
    if eval_limit is None:
        raw_eval = cfg.get("train", "eval_samples_per_epoch", 512)
        eval_limit = None if raw_eval is None else int(raw_eval)

    train_loader = make_dataloader(
        train_csv,
        batch_size=batch_size,
        image_size=image_size,
        augment=True,
        augmentation_cfg=cfg.raw.get("augmentation", {}),
        max_samples=train_limit,
        seed=cfg.seed,
        shuffle=True,
        num_workers=config_int(cfg, "data", "num_workers", 0),
        pin_memory=bool(cfg.get("data", "pin_memory", False)),
        allowed_root=cfg.crop_root,
        expected_source_split="selected",
        expected_category="clean",
    )
    eval_loader = make_dataloader(
        eval_csv,
        batch_size=config_int(cfg, "evaluation", "batch_size", 16),
        image_size=image_size,
        augment=False,
        augmentation_cfg=None,
        max_samples=eval_limit,
        seed=cfg.seed + 99,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        allowed_root=cfg.crop_root,
        expected_source_split="selected",
        expected_category="clean",
    )

    generator, discriminator = build_models(
        generator_base=config_int(cfg, "model", "generator_base_channels", 64),
        discriminator_base=config_int(cfg, "model", "discriminator_base_channels", 64),
        dropout=config_float(cfg, "model", "dropout", 0.5),
        generator_type=str(cfg.get("model", "generator_type", "transposed")),
    )
    generator.to(device)
    discriminator.to(device)

    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=config_float(cfg, "train", "learning_rate", 0.0002),
        betas=(config_float(cfg, "train", "beta1", 0.5), config_float(cfg, "train", "beta2", 0.999)),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config_float(cfg, "train", "learning_rate", 0.0002),
        betas=(config_float(cfg, "train", "beta1", 0.5), config_float(cfg, "train", "beta2", 0.999)),
    )
    pix_loss = Pix2PixLoss(
        lambda_l1=config_float(cfg, "loss", "lambda_l1", 100.0),
        lambda_gradient=config_float(cfg, "loss", "lambda_gradient", 0.0),
        lambda_vessel=config_float(cfg, "loss", "lambda_vessel", 0.0),
        lambda_highpass=config_float(cfg, "loss", "lambda_highpass", 0.0),
    )

    n_epochs = epochs if epochs is not None else config_int(cfg, "train", "epochs", 25)
    configured_max_steps = cfg.get("train", "max_steps")
    if max_steps is None and configured_max_steps is not None:
        max_steps = int(configured_max_steps)
    include_vessel_eval = config_bool(cfg, "train", "include_vessel_metrics", False)
    early_patience_raw = cfg.get("train", "early_stopping_patience")
    early_stopping_patience = None if early_patience_raw is None else int(early_patience_raw)
    early_stopping_min_epochs = config_int(cfg, "train", "early_stopping_min_epochs", 1)
    early_stopping_min_delta = config_float(cfg, "train", "early_stopping_min_delta", 0.0)

    metrics_path = run_dir / "metrics" / "train_history.csv"
    train_fieldnames = [
        "epoch",
        "step",
        "seconds",
        "d_loss",
        "g_total",
        "g_adv",
        "g_l1",
        "g_gradient",
        "g_vessel",
        "g_highpass",
        "eval_mae",
        "eval_rmse",
        "eval_psnr",
        "eval_ssim",
        "eval_vessel_corr",
        "eval_vessel_dice",
        "eval_vessel_iou",
        "eval_skeleton_f1",
        "eval_vessel_length_ratio",
        "eval_vein_score",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=train_fieldnames)
        writer.writeheader()

    global_step = 0
    start_time = time.time()
    latest_metrics: dict[str, float] = {}
    best_score = float("-inf")
    best_epoch = 0
    best_checkpoint: str | None = None
    epochs_since_best = 0

    for epoch in range(1, n_epochs + 1):
        generator.train()
        discriminator.train()
        d_meter = AverageMeter()
        g_meter = AverageMeter()
        adv_meter = AverageMeter()
        l1_meter = AverageMeter()
        gradient_meter = AverageMeter()
        vessel_meter = AverageMeter()
        highpass_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{n_epochs}", leave=False)
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            real_nir = batch["nir"].to(device)

            with torch.no_grad():
                fake_nir_detached = generator(rgb).detach()
            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(rgb, real_nir)
            fake_logits = discriminator(rgb, fake_nir_detached)
            d_loss = pix_loss.discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_nir = generator(rgb)
            fake_logits_for_g = discriminator(rgb, fake_nir)
            g_loss, g_parts = pix_loss.generator_loss(fake_logits_for_g, fake_nir, real_nir)
            g_loss.backward()
            opt_g.step()

            global_step += 1
            batch_n = rgb.shape[0]
            d_meter.update(float(d_loss.detach().cpu()), batch_n)
            g_meter.update(g_parts["g_total"], batch_n)
            adv_meter.update(g_parts["g_adv"], batch_n)
            l1_meter.update(g_parts["g_l1"], batch_n)
            gradient_meter.update(g_parts["g_gradient"], batch_n)
            vessel_meter.update(g_parts["g_vessel"], batch_n)
            highpass_meter.update(g_parts["g_highpass"], batch_n)
            pbar.set_postfix(d=f"{d_meter.avg:.4f}", g=f"{g_meter.avg:.4f}", l1=f"{l1_meter.avg:.4f}")
            if max_steps is not None and global_step >= max_steps:
                break

        eval_summary = evaluate_generator(
            generator=generator,
            dataloader=eval_loader,
            device=device,
            vessel_sigmas=list(cfg.get("evaluation", "vessel_sigmas", [1, 2, 3])),
            vessel_threshold=float(cfg.get("evaluation", "vessel_threshold", 0.15)),
            include_vessel=include_vessel_eval,
        )
        overall = eval_summary["overall"]
        eval_vein_score = vein_checkpoint_score(overall)
        latest_metrics = {
            "d_loss": d_meter.avg,
            "g_total": g_meter.avg,
            "g_adv": adv_meter.avg,
            "g_l1": l1_meter.avg,
            "g_gradient": gradient_meter.avg,
            "g_vessel": vessel_meter.avg,
            "g_highpass": highpass_meter.avg,
            "eval_mae": float(overall.get("mae", 0.0)),
            "eval_rmse": float(overall.get("rmse", 0.0)),
            "eval_psnr": float(overall.get("psnr", 0.0)),
            "eval_ssim": float(overall.get("ssim", 0.0)),
            "eval_vessel_corr": float(overall.get("vessel_corr", 0.0)),
            "eval_vessel_dice": float(overall.get("vessel_dice", 0.0)),
            "eval_vessel_iou": float(overall.get("vessel_iou", 0.0)),
            "eval_skeleton_f1": float(overall.get("skeleton_f1", 0.0)),
            "eval_vessel_length_ratio": float(overall.get("vessel_length_ratio", 0.0)),
            "eval_vein_score": float(eval_vein_score),
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=train_fieldnames)
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "seconds": time.time() - start_time,
                    **latest_metrics,
                }
            )

        sample_every = config_int(cfg, "train", "sample_every_epochs", 1)
        if epoch % sample_every == 0:
            save_training_sample_grid(
                generator,
                eval_loader,
                device,
                run_dir / "samples" / f"epoch_{epoch:03d}.png",
                max_items=config_int(cfg, "train", "sample_grid_count", 8),
                sigmas=list(cfg.get("evaluation", "vessel_sigmas", [1, 2, 3])),
                threshold=float(cfg.get("evaluation", "vessel_threshold", 0.15)),
            )

        checkpoint_every = config_int(cfg, "train", "checkpoint_every_epochs", 1)
        epoch_checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        if epoch % checkpoint_every == 0:
            save_checkpoint(epoch_checkpoint, generator, discriminator, opt_g, opt_d, epoch, global_step, latest_metrics)
        save_checkpoint(
            run_dir / "checkpoints" / "latest.pt",
            generator,
            discriminator,
            opt_g,
            opt_d,
            epoch,
            global_step,
            latest_metrics,
        )
        if eval_vein_score > best_score + early_stopping_min_delta:
            best_score = eval_vein_score
            best_epoch = epoch
            best_checkpoint = str(epoch_checkpoint)
            epochs_since_best = 0
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                generator,
                discriminator,
                opt_g,
                opt_d,
                epoch,
                global_step,
                latest_metrics,
            )
            write_json(
                run_dir / "checkpoints" / "best_checkpoint.json",
                {
                    "epoch": best_epoch,
                    "checkpoint": best_checkpoint,
                    "best_pt": str(run_dir / "checkpoints" / "best.pt"),
                    "eval_split": eval_split_name,
                    "eval_csv": str(eval_csv),
                    "vein_score": best_score,
                    "metrics": latest_metrics,
                    "score_formula": "0.45*vessel_dice + 0.25*skeleton_f1 + 0.20*vessel_corr + 0.10*ssim",
                },
            )
        else:
            epochs_since_best += 1

        if (
            early_stopping_patience is not None
            and epoch >= early_stopping_min_epochs
            and epochs_since_best >= early_stopping_patience
        ):
            break
        if max_steps is not None and global_step >= max_steps:
            break

    write_json(
        run_dir / "run_summary.json",
        {
            "run_name": run_name,
            "epochs_requested": n_epochs,
            "global_step": global_step,
            "seconds": time.time() - start_time,
            "device": str(device),
            "generator_type": str(cfg.get("model", "generator_type", "transposed")),
            "latest_metrics": latest_metrics,
            "train_csv": str(train_csv),
            "eval_csv": str(eval_csv),
            "eval_split": eval_split_name,
            "train_samples": len(train_loader.dataset),
            "eval_samples_per_epoch": len(eval_loader.dataset),
            "best_epoch": best_epoch,
            "best_checkpoint": best_checkpoint,
            "best_vein_score": best_score,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_epochs": early_stopping_min_epochs,
        },
    )
    return run_dir


def save_training_sample_grid(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    path: Path,
    max_items: int,
    sigmas: list[int],
    threshold: float,
) -> None:
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        rgb = batch["rgb"].to(device)
        target = batch["nir"].to(device)
        pred = generator(rgb)
    save_prediction_grid(path, rgb, pred, target, max_items=max_items, vessel_sigmas=sigmas, vessel_threshold=threshold)


def evaluate_generator(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    vessel_sigmas: list[int],
    vessel_threshold: float,
    include_vessel: bool = True,
) -> dict[str, Any]:
    generator.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="evaluate", leave=False):
            rgb = batch["rgb"].to(device)
            target = batch["nir"].to(device)
            pred = generator(rgb)
            pred_np = batch_to_numpy_01(pred)[:, 0]
            target_np = batch_to_numpy_01(target)[:, 0]
            for i in range(pred_np.shape[0]):
                record: dict[str, Any] = {
                    "participant": batch["participant"][i],
                    "source_split": batch["source_split"][i],
                    "category": batch["category"][i],
                    "hand": batch["hand"][i],
                    "pair_id": batch["pair_id"][i],
                }
                record.update(image_metrics(pred_np[i], target_np[i]))
                if include_vessel:
                    record.update(vessel_metrics(pred_np[i], target_np[i], vessel_sigmas, vessel_threshold))
                records.append(record)
    overall = summarize_records(records)[0] if records else {"group": "overall", "n": 0}
    return {
        "overall": overall,
        "by_category": summarize_records(records, "category"),
        "by_source_split": summarize_records(records, "source_split"),
        "by_hand": summarize_records(records, "hand"),
        "by_participant": summarize_records(records, "participant"),
        "records": records,
    }


def write_evaluation_outputs(eval_result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = eval_result["records"]
    if records:
        with (output_dir / "per_crop_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    for key in ["by_category", "by_source_split", "by_hand", "by_participant"]:
        rows = eval_result[key]
        if rows:
            with (output_dir / f"{key}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    write_json(output_dir / "overall_metrics.json", eval_result["overall"])
