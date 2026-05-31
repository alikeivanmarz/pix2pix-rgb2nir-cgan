from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rgb2nir_pix2pix.dataset import RGBNIRCropDataset
from rgb2nir_pix2pix.inference import load_generator, predict_image
from rgb2nir_pix2pix.losses import DarkLineVesselLoss, HighPassDetailLoss, Pix2PixLoss, SobelGradientLoss
from rgb2nir_pix2pix.manifest import discover_pairs, validate_manifest, write_manifest_files
from rgb2nir_pix2pix.models import build_models


def create_pair(root: Path, participant: str, base: str, x: int, y: int) -> None:
    pair_dir = root / participant / base
    pair_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    nir = np.full((128, 128), 90, dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(pair_dir / f"{base}_y{y:04d}_x{x:04d}_RGB.png")
    Image.fromarray(nir, mode="L").save(pair_dir / f"{base}_y{y:04d}_x{x:04d}_NIR.png")


def test_manifest_and_dataset(tmp_path: Path) -> None:
    crop_root = tmp_path / "data" / "crops" / "selected" / "clean"
    create_pair(crop_root, "100", "100-01-R", 10, 20)
    create_pair(crop_root, "101", "101-02-L", 30, 40)
    pairs = discover_pairs(crop_root, train_fraction=0.5, seed=1)
    assert len(pairs) == 2

    manifest = tmp_path / "manifest.csv"
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    split = tmp_path / "split.csv"
    summary = tmp_path / "summary.md"
    write_manifest_files(pairs, manifest, train, test, split, summary)
    audit = validate_manifest(manifest, crop_root, image_size=128)
    assert audit["rows"] == 2
    assert audit["participant_leakage"] == 0

    rows = list(csv.DictReader(train.open()))
    csv_path = train if rows else test
    ds = RGBNIRCropDataset(csv_path, image_size=128)
    sample = ds[0]
    assert sample["rgb"].shape == (3, 128, 128)
    assert sample["nir"].shape == (1, 128, 128)
    assert torch.isfinite(sample["rgb"]).all()
    assert torch.isfinite(sample["nir"]).all()


def test_dataset_rejects_out_of_scope_rows(tmp_path: Path) -> None:
    crop_root = tmp_path / "data" / "crops" / "selected" / "clean"
    extra_root = tmp_path / "data" / "crops" / "extra" / "clean"
    create_pair(crop_root, "100", "100-01-R", 10, 20)
    create_pair(extra_root, "100", "100-01-R", 10, 20)
    bad_csv = tmp_path / "bad.csv"
    bad_rgb = extra_root / "100" / "100-01-R" / "100-01-R_y0020_x0010_RGB.png"
    bad_nir = extra_root / "100" / "100-01-R" / "100-01-R_y0020_x0010_NIR.png"
    with bad_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_split": "train",
                "participant": "100",
                "source_split": "extra",
                "category": "clean",
                "base": "100-01-R",
                "hand": "R",
                "x": 10,
                "y": 20,
                "pair_id": "100-01-R_y0020_x0010",
                "rgb_path": str(bad_rgb),
                "nir_path": str(bad_nir),
            }
        )
    try:
        RGBNIRCropDataset(
            bad_csv,
            image_size=128,
            allowed_root=crop_root,
            expected_source_split="selected",
            expected_category="clean",
        )
    except ValueError as exc:
        assert "Out-of-scope" in str(exc)
    else:
        raise AssertionError("Dataset accepted an out-of-scope manifest row")


def test_models_losses_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    generator, discriminator = build_models(generator_base=8, discriminator_base=8, dropout=0.0)
    rgb = torch.randn(2, 3, 128, 128)
    nir = torch.randn(2, 1, 128, 128)
    fake = generator(rgb)
    assert fake.shape == nir.shape
    logits = discriminator(rgb, fake)
    assert logits.shape[0] == 2
    loss = Pix2PixLoss(lambda_l1=100.0)
    total, parts = loss.generator_loss(logits, fake, nir)
    assert torch.isfinite(total)
    assert parts["g_total"] > 0

    ckpt = tmp_path / "generator.pt"
    torch.save(generator.state_dict(), ckpt)
    clone, _ = build_models(generator_base=8, discriminator_base=8, dropout=0.0)
    clone.load_state_dict(torch.load(ckpt, map_location="cpu"))
    with torch.no_grad():
        out1 = generator(rgb)
        out2 = clone(rgb)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_gradient_and_vessel_losses_are_finite() -> None:
    pred = torch.randn(2, 1, 128, 128)
    target = torch.randn(2, 1, 128, 128)
    grad_loss = SobelGradientLoss()(pred, target)
    vessel_loss = DarkLineVesselLoss()(pred, target)
    highpass_loss = HighPassDetailLoss()(pred, target)
    combined = Pix2PixLoss(
        lambda_l1=100.0,
        lambda_gradient=10.0,
        lambda_vessel=5.0,
        lambda_highpass=25.0,
    )
    fake_logits = torch.randn(2, 1, 14, 14)
    total, parts = combined.generator_loss(fake_logits, pred, target)
    assert torch.isfinite(grad_loss)
    assert torch.isfinite(vessel_loss)
    assert torch.isfinite(highpass_loss)
    assert torch.isfinite(total)
    assert parts["g_gradient"] >= 0
    assert parts["g_vessel"] >= 0
    assert parts["g_highpass"] >= 0


def test_resizeconv_generator_shape() -> None:
    rgb = torch.randn(2, 3, 128, 128)
    for generator_type in ("resizeconv", "nearestconv"):
        generator, discriminator = build_models(
            generator_base=8,
            discriminator_base=8,
            dropout=0.0,
            generator_type=generator_type,
        )
        fake = generator(rgb)
        assert fake.shape == (2, 1, 128, 128)
        logits = discriminator(rgb, fake)
        assert logits.shape[0] == 2


def test_inference_safetensors_roundtrip(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    generator, _ = build_models(generator_base=8, discriminator_base=8, dropout=0.0)
    weights = tmp_path / "generator.safetensors"
    save_file(generator.state_dict(), str(weights))

    loaded = load_generator(
        weights,
        generator_base=8,
        discriminator_base=8,
        dropout=0.0,
        device="cpu",
    )
    rgb = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = loaded(rgb)
    assert out.shape == (1, 1, 128, 128)

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[..., 0] = 180
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(image, mode="RGB").save(input_path)
    predict_image(loaded, input_path, output_path, image_size=128)
    assert output_path.exists()
    assert Image.open(output_path).size == (128, 128)
