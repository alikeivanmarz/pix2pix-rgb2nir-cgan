"""Configuration loading for the standalone pix2pix project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Pix2PixConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project_root(self) -> Path:
        root = Path(self.raw["paths"].get("project_root", "."))
        if root.is_absolute():
            return root
        return (self.path.parent.parent / root).resolve()

    @property
    def crop_root(self) -> Path:
        crop_root = Path(self.raw["paths"]["crop_root"])
        if crop_root.is_absolute():
            return crop_root
        return (self.project_root / crop_root).resolve()

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["seed"])

    def path_value(self, key: str) -> Path:
        value = Path(self.raw["paths"][key])
        if value.is_absolute():
            return value
        return (self.project_root / value).resolve()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)


def load_config(config_path: str | Path) -> Pix2PixConfig:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return Pix2PixConfig(raw=raw, path=path)


def ensure_project_dirs(cfg: Pix2PixConfig) -> None:
    for key in ["runs_dir", "reports_dir", "sample_grids_dir"]:
        cfg.path_value(key).mkdir(parents=True, exist_ok=True)
    cfg.path_value("manifest_csv").parent.mkdir(parents=True, exist_ok=True)
