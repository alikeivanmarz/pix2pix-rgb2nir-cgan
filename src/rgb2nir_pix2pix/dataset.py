"""PyTorch dataset for paired RGB/NIR crop manifests."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class RGBNIRCropDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_size: int = 128,
        augment: bool = False,
        augmentation_cfg: dict[str, Any] | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        allowed_root: str | Path | None = None,
        expected_source_split: str | None = None,
        expected_category: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.allowed_root = Path(allowed_root).resolve() if allowed_root is not None else None
        self.expected_source_split = expected_source_split
        self.expected_category = expected_category
        if max_samples is not None:
            rng = random.Random(seed)
            indices = list(range(len(self.rows)))
            rng.shuffle(indices)
            keep = sorted(indices[: int(max_samples)])
            self.rows = [self.rows[i] for i in keep]
        self._validate_scope()
        self.image_size = image_size
        self.augment = augment
        self.augmentation_cfg = augmentation_cfg or {}
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        rgb = Image.open(row["rgb_path"]).convert("RGB")
        nir = Image.open(row["nir_path"]).convert("L")
        if rgb.size != (self.image_size, self.image_size):
            raise ValueError(f"Unexpected RGB size {rgb.size}: {row['rgb_path']}")
        if nir.size != (self.image_size, self.image_size):
            raise ValueError(f"Unexpected NIR size {nir.size}: {row['nir_path']}")
        if self.augment:
            rgb, nir = self._augment_pair(rgb, nir)
        rgb_tensor = pil_rgb_to_tensor(rgb)
        nir_tensor = pil_l_to_tensor(nir)
        return {
            "rgb": rgb_tensor,
            "nir": nir_tensor,
            "participant": row["participant"],
            "source_split": row["source_split"],
            "category": row["category"],
            "hand": row["hand"],
            "pair_id": row["pair_id"],
            "rgb_path": row["rgb_path"],
            "nir_path": row["nir_path"],
        }

    def _validate_scope(self) -> None:
        if self.allowed_root is None and self.expected_source_split is None and self.expected_category is None:
            return
        for row in self.rows:
            if self.expected_source_split is not None and row.get("source_split") != self.expected_source_split:
                raise ValueError(
                    f"Out-of-scope source_split in {self.csv_path}: "
                    f"{row.get('source_split')} != {self.expected_source_split}"
                )
            if self.expected_category is not None and row.get("category") != self.expected_category:
                raise ValueError(
                    f"Out-of-scope category in {self.csv_path}: "
                    f"{row.get('category')} != {self.expected_category}"
                )
            if self.allowed_root is not None:
                for key in ["rgb_path", "nir_path"]:
                    image_path = Path(row[key]).resolve()
                    if not image_path.is_relative_to(self.allowed_root):
                        raise ValueError(
                            f"Out-of-scope {key} in {self.csv_path}: "
                            f"{image_path} is not under {self.allowed_root}"
                        )
                    text_path = str(image_path)
                    if "/extra/" in text_path or "/challenging/" in text_path or "/unsynced/" in text_path:
                        raise ValueError(f"Forbidden crop scope in {self.csv_path}: {image_path}")

    def _augment_pair(self, rgb: Image.Image, nir: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.augmentation_cfg.get("horizontal_flip", False) and self.rng.random() < 0.5:
            rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            nir = nir.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.augmentation_cfg.get("vertical_flip", False) and self.rng.random() < 0.5:
            rgb = rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            nir = nir.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.augmentation_cfg.get("rotate_90", False):
            k = self.rng.choice([0, 1, 2, 3])
            if k:
                rgb = rgb.rotate(90 * k)
                nir = nir.rotate(90 * k)

        brightness = float(self.augmentation_cfg.get("rgb_brightness", 0.0) or 0.0)
        if brightness > 0:
            factor = 1.0 + self.rng.uniform(-brightness, brightness)
            rgb = ImageEnhance.Brightness(rgb).enhance(factor)
        contrast = float(self.augmentation_cfg.get("rgb_contrast", 0.0) or 0.0)
        if contrast > 0:
            factor = 1.0 + self.rng.uniform(-contrast, contrast)
            rgb = ImageEnhance.Contrast(rgb).enhance(factor)
        noise_std = float(self.augmentation_cfg.get("rgb_noise_std", 0.0) or 0.0)
        if noise_std > 0:
            arr = np.asarray(rgb).astype(np.float32) / 255.0
            noise = np.random.normal(0.0, noise_std, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)
            rgb = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
        return rgb, nir


def pil_rgb_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr * 2.0 - 1.0)


def pil_l_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr[None, :, :] * 2.0 - 1.0)


def tensor_to_unit(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def make_dataloader(
    csv_path: str | Path,
    batch_size: int,
    image_size: int,
    augment: bool,
    augmentation_cfg: dict[str, Any] | None,
    max_samples: int | None,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    allowed_root: str | Path | None = None,
    expected_source_split: str | None = None,
    expected_category: str | None = None,
) -> torch.utils.data.DataLoader:
    dataset = RGBNIRCropDataset(
        csv_path=csv_path,
        image_size=image_size,
        augment=augment,
        augmentation_cfg=augmentation_cfg,
        max_samples=max_samples,
        seed=seed,
        allowed_root=allowed_root,
        expected_source_split=expected_source_split,
        expected_category=expected_category,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
