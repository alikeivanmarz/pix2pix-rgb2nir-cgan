#!/usr/bin/env python3
"""Build and validate selected/clean-only manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgb2nir_pix2pix.config import ensure_project_dirs, load_config
from rgb2nir_pix2pix.manifest import discover_pairs, validate_manifest, write_manifest_files
from rgb2nir_pix2pix.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pix2pix_mps.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    pairs = discover_pairs(
        crop_root=cfg.crop_root,
        train_fraction=float(cfg.get("data", "train_fraction", 0.8)),
        seed=cfg.seed,
    )
    summary = write_manifest_files(
        pairs=pairs,
        manifest_csv=cfg.path_value("manifest_csv"),
        train_csv=cfg.path_value("train_csv"),
        test_csv=cfg.path_value("test_csv"),
        participant_split_csv=cfg.path_value("participant_split_csv"),
        summary_md=cfg.path_value("manifest_summary_md"),
    )
    audit = validate_manifest(
        cfg.path_value("manifest_csv"),
        expected_crop_root=cfg.crop_root,
        image_size=int(cfg.get("data", "image_size", 128)),
    )
    payload = {"summary": summary, "audit": audit}
    write_json(ROOT / "manifests" / "manifest_validation.json", payload)
    report = ROOT / "manifests" / "VALIDATION_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Manifest Validation Report\n\n")
        f.write("Scope: `data/crops/selected/clean/` only.\n\n")
        f.write("No `extra`, `challenging`, or `unsynced` paths are allowed.\n\n")
        f.write("## Summary\n\n")
        for key, value in summary.items():
            f.write(f"- {key}: `{value}`\n")
        f.write("\n## Audit\n\n")
        f.write(f"```json\n{json.dumps(audit, indent=2, sort_keys=True)}\n```\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
