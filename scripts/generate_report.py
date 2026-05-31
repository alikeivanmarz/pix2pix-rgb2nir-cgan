#!/usr/bin/env python3
"""Generate a Markdown report for a run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgb2nir_pix2pix.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pix2pix_mps.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def last_train_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "REPORT.md"
    manifest_summary = cfg.path_value("manifest_summary_md").read_text(encoding="utf-8")
    run_summary = read_json(run_dir / "run_summary.json")
    env = read_json(run_dir / "environment.json")
    overall = read_json(run_dir / "evaluation_test_best" / "overall_metrics.json")
    evaluation_dir = "evaluation_test_best"
    if not overall:
        overall = read_json(run_dir / "evaluation" / "overall_metrics.json")
        evaluation_dir = "evaluation"
    last_row = last_train_row(run_dir / "metrics" / "train_history.csv")

    with output.open("w", encoding="utf-8") as f:
        f.write(f"# Pix2Pix Run Report: {run_dir.name}\n\n")
        f.write("## Dataset\n\n")
        f.write(manifest_summary)
        f.write("\n## Environment\n\n")
        f.write(f"- python executable: `{env.get('executable', 'unknown')}`\n")
        f.write(f"- torch: `{env.get('torch', 'unknown')}`\n")
        f.write(f"- MPS available: `{env.get('mps_available', 'unknown')}`\n")
        f.write("\n## Training Summary\n\n")
        if run_summary:
            for key, value in run_summary.items():
                f.write(f"- {key}: `{value}`\n")
        if last_row:
            f.write("\nLast training row:\n\n")
            f.write("```json\n")
            f.write(json.dumps(last_row, indent=2, sort_keys=True))
            f.write("\n```\n")
        f.write("\n## Evaluation Summary\n\n")
        f.write(f"Evaluation directory: `{evaluation_dir}`\n\n")
        if overall:
            f.write("```json\n")
            f.write(json.dumps(overall, indent=2, sort_keys=True))
            f.write("\n```\n")
        else:
            f.write("No evaluation output found yet.\n")
        f.write("\n## Qualitative Outputs\n\n")
        for image in sorted((run_dir / "samples").glob("*.png")):
            f.write(f"- `{image.relative_to(run_dir)}`\n")
        f.write("\n## Notes\n\n")
        f.write("- The model is a reconstruction-dominant pix2pix cGAN.\n")
        f.write("- Sato vein maps are used for qualitative vein-pattern visualisation.\n")
    print(output)


if __name__ == "__main__":
    main()
