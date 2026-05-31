"""Manifest building and validation for selected/clean RGB/NIR crops."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

RGB_RE = re.compile(r"^(?P<base>.+)_y(?P<y>\d+)_x(?P<x>\d+)_RGB\.png$")


@dataclass(frozen=True)
class CropPair:
    experiment_split: str
    participant: str
    source_split: str
    category: str
    base: str
    hand: str
    x: int
    y: int
    pair_id: str
    rgb_path: str
    nir_path: str

    def as_row(self) -> dict[str, str | int]:
        return {
            "experiment_split": self.experiment_split,
            "participant": self.participant,
            "source_split": self.source_split,
            "category": self.category,
            "base": self.base,
            "hand": self.hand,
            "x": self.x,
            "y": self.y,
            "pair_id": self.pair_id,
            "rgb_path": self.rgb_path,
            "nir_path": self.nir_path,
        }


FIELDNAMES = [
    "experiment_split",
    "participant",
    "source_split",
    "category",
    "base",
    "hand",
    "x",
    "y",
    "pair_id",
    "rgb_path",
    "nir_path",
]


def parse_hand(base: str) -> str:
    parts = base.split("-")
    if len(parts) >= 3 and parts[-1] in {"L", "R"}:
        return parts[-1]
    return "unknown"


def participant_splits(participants: list[str], train_fraction: float, seed: int) -> dict[str, str]:
    ordered = sorted(participants, key=lambda p: int(p) if p.isdigit() else p)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * train_fraction)
    train = set(shuffled[:n_train])
    return {participant: ("train" if participant in train else "test") for participant in ordered}


def discover_pairs(crop_root: Path, train_fraction: float, seed: int) -> list[CropPair]:
    if not crop_root.exists():
        raise FileNotFoundError(f"Crop root does not exist: {crop_root}")
    if crop_root.name != "clean" or crop_root.parent.name != "selected":
        raise ValueError(f"Expected crop_root to end with data/crops/selected/clean: {crop_root}")

    rgb_files = sorted(crop_root.glob("*/*/*_RGB.png"))
    participants = sorted({p.relative_to(crop_root).parts[0] for p in rgb_files})
    split_map = participant_splits(participants, train_fraction=train_fraction, seed=seed)

    pairs: list[CropPair] = []
    for rgb_path in rgb_files:
        rel = rgb_path.relative_to(crop_root)
        participant, base_dir, filename = rel.parts
        match = RGB_RE.match(filename)
        if not match:
            raise ValueError(f"Unexpected RGB crop name: {rgb_path}")
        base = match.group("base")
        if base != base_dir:
            raise ValueError(f"Base folder/name mismatch: folder={base_dir}, file={base}")
        nir_path = rgb_path.with_name(filename.replace("_RGB.png", "_NIR.png"))
        if not nir_path.exists():
            raise FileNotFoundError(f"Missing NIR pair for {rgb_path}")
        y = int(match.group("y"))
        x = int(match.group("x"))
        pair_id = f"{base}_y{y:04d}_x{x:04d}"
        pairs.append(
            CropPair(
                experiment_split=split_map[participant],
                participant=participant,
                source_split="selected",
                category="clean",
                base=base,
                hand=parse_hand(base),
                x=x,
                y=y,
                pair_id=pair_id,
                rgb_path=str(rgb_path.resolve()),
                nir_path=str(nir_path.resolve()),
            )
        )
    return pairs


def write_manifest_files(
    pairs: list[CropPair],
    manifest_csv: Path,
    train_csv: Path,
    test_csv: Path,
    participant_split_csv: Path,
    summary_md: Path,
) -> dict[str, int | dict[str, int]]:
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    train_pairs = [p for p in pairs if p.experiment_split == "train"]
    test_pairs = [p for p in pairs if p.experiment_split == "test"]

    def write_rows(path: Path, rows: list[CropPair]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for pair in rows:
                writer.writerow(pair.as_row())

    write_rows(manifest_csv, pairs)
    write_rows(train_csv, train_pairs)
    write_rows(test_csv, test_pairs)

    participant_split = {}
    for pair in pairs:
        participant_split[pair.participant] = pair.experiment_split
    with participant_split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["participant", "experiment_split"])
        writer.writeheader()
        for participant in sorted(participant_split, key=lambda p: int(p) if p.isdigit() else p):
            writer.writerow(
                {"participant": participant, "experiment_split": participant_split[participant]}
            )

    hand_counts = Counter(p.hand for p in pairs)
    participant_counts = Counter(p.participant for p in pairs)
    split_counts = Counter(p.experiment_split for p in pairs)
    summary = {
        "crop_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "participants": len(participant_counts),
        "train_participants": len({p.participant for p in train_pairs}),
        "test_participants": len({p.participant for p in test_pairs}),
        "right_hand_pairs": hand_counts.get("R", 0),
        "left_hand_pairs": hand_counts.get("L", 0),
        "split_counts": dict(split_counts),
    }

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Selected/Clean Manifest Summary\n\n")
        f.write("Scope: `data/crops/selected/clean/` only.\n\n")
        f.write("Excluded by design: `extra`, `challenging`, and `unsynced`.\n\n")
        for key, value in summary.items():
            f.write(f"- {key}: `{value}`\n")
        f.write("\nOutput files:\n\n")
        for path in [manifest_csv, train_csv, test_csv, participant_split_csv]:
            f.write(f"- `{path.name}`\n")
    return summary


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_manifest(path: Path, expected_crop_root: Path, image_size: int) -> dict[str, object]:
    rows = read_rows(path)
    leakage_guard: dict[str, str] = {}
    status_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    hand_counts: Counter[str] = Counter()
    duplicate_keys: Counter[str] = Counter()
    participants_by_split: dict[str, set[str]] = {"train": set(), "test": set()}

    expected_root = expected_crop_root.resolve()
    for row in rows:
        for mod_key in ["rgb_path", "nir_path"]:
            image_path = Path(row[mod_key]).resolve()
            if not image_path.is_relative_to(expected_root):
                raise ValueError(f"Manifest path outside selected/clean root: {image_path}")
            if "/extra/" in str(image_path) or "/challenging/" in str(image_path) or "/unsynced/" in str(
                image_path
            ):
                raise ValueError(f"Out-of-scope path in manifest: {image_path}")
            if not image_path.exists():
                status_counts[f"{mod_key}:missing"] += 1
                continue
            status_counts[f"{mod_key}:ok"] += 1
            with Image.open(image_path) as img:
                size_counts[f"{mod_key}:{img.width}x{img.height}"] += 1
                mode_counts[f"{mod_key}:{img.mode}"] += 1
                if img.width != image_size or img.height != image_size:
                    raise ValueError(f"Unexpected image size for {image_path}: {img.size}")

        split = row["experiment_split"]
        participant = row["participant"]
        if split not in participants_by_split:
            raise ValueError(f"Unexpected split: {split}")
        participants_by_split[split].add(participant)
        previous = leakage_guard.get(participant)
        if previous is None:
            leakage_guard[participant] = split
        elif previous != split:
            raise ValueError(f"Participant leakage for {participant}: {previous} and {split}")
        if row["source_split"] != "selected" or row["category"] != "clean":
            raise ValueError(f"Out-of-scope source/category row: {row}")
        duplicate_keys[
            "|".join([row["participant"], row["base"], row["pair_id"], row["rgb_path"], row["nir_path"]])
        ] += 1
        hand_counts[row["hand"]] += 1

    duplicate_count = sum(1 for count in duplicate_keys.values() if count > 1)
    if duplicate_count:
        raise ValueError(f"Duplicate manifest rows: {duplicate_count}")
    overlap = participants_by_split["train"] & participants_by_split["test"]
    if overlap:
        raise ValueError(f"Participant leakage detected: {sorted(overlap)[:10]}")

    return {
        "rows": len(rows),
        "status_counts": dict(status_counts),
        "size_counts": dict(size_counts),
        "mode_counts": dict(mode_counts),
        "hand_counts": dict(hand_counts),
        "train_participants": len(participants_by_split["train"]),
        "test_participants": len(participants_by_split["test"]),
        "participant_leakage": 0,
        "duplicate_rows": 0,
    }
