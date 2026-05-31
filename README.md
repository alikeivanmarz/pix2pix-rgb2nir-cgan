# Pix2Pix RGB-to-NIR cGAN

PyTorch implementation of a pix2pix conditional GAN for translating 128x128 RGB
skin crops into NIR-like grayscale images.

The repository includes training, evaluation, and single-image inference tools.
The trained generator is distributed through GitHub Releases so the git
repository stays lightweight.

## Publication

This implementation accompanies:

Keivanmarz, A., and Sharifzadeh, H. (2024). Vein pattern visualisation for
biometric identification with cGAN on a New Zealand dataset. Forensic Science
International, 359, 112050.

- DOI: https://doi.org/10.1016/j.forsciint.2024.112050
- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0379073824001312

## Install

```bash
git clone https://github.com/alikeivanmarz/pix2pix-rgb2nir-cgan.git
cd pix2pix-rgb2nir-cgan
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Apple Silicon, install a PyTorch build with MPS support before running
training or inference.

## Download Weights

```bash
python scripts/download_weights.py --variant fp32
```

This downloads:

```text
weights/rgb2nir-pix2pix-generator-fp32.safetensors
```

An fp16 release asset is also provided for lighter local testing:

```bash
python scripts/download_weights.py --variant fp16
```

## Predict One Image

Input images are expected to be RGB skin crops. Images are resized to 128x128 by
default before inference.

```bash
python scripts/predict_image.py \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/rgb_crop.png \
  --output outputs/nir_prediction.png
```

Functional test with the included synthetic crop:

```bash
python scripts/predict_image.py \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input examples/rgb_synthetic_skin_crop.png \
  --output outputs/nir_prediction.png
```

Use Apple Silicon MPS explicitly:

```bash
python scripts/predict_image.py \
  --device mps \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/rgb_crop.png \
  --output outputs/nir_prediction.png
```

## Model

- Generator: U-Net, RGB input to single-channel NIR-like output
- Discriminator: 70x70 PatchGAN
- Objective: reconstruction-dominant pix2pix loss, `100 * L1 + adversarial`
- Input size: 128x128
- Tensor range: `[-1, 1]`
- Vein visualisation: Sato vesselness maps from predicted and paired NIR crops

## Results

Paired image metrics on the participant-disjoint test split:

| Metric | Value |
| --- | ---: |
| MAE | 0.0464 |
| RMSE | 0.0519 |
| PSNR | 27.7760 |
| SSIM | 0.9009 |

Sato vein-map evaluation check on 512 test crops:

| Metric | Value |
| --- | ---: |
| Vessel correlation | 0.6466 |
| Vessel Dice | 0.5955 |
| Vessel IoU | 0.4458 |
| Skeleton F1 | 0.1438 |

Example qualitative grid:

![RGB to NIR prediction grid with Sato vein maps](assets/sample_grids/provided_samples_compact_page_01.png)

## Train

Prepare a CSV manifest with these columns:

```text
participant,source_split,category,base,hand,x,y,pair_id,rgb_path,nir_path
```

Then edit `configs/pix2pix_mps.yaml` so `crop_root`, `train_csv`, `val_csv`,
and `test_csv` point to your local dataset.

```bash
python scripts/train_pix2pix.py \
  --config configs/pix2pix_mps.yaml \
  --run-name pix2pix_train
```

## Evaluate

```bash
python scripts/evaluate_pix2pix.py \
  --config configs/pix2pix_mps.yaml \
  --checkpoint runs/pix2pix_train/checkpoints/best.pt
```

## Export Release Weights

Training checkpoints include optimizer and discriminator state. For public
distribution, export generator-only safetensors:

```bash
python scripts/export_inference_weights.py \
  --checkpoint runs/pix2pix_train/checkpoints/best.pt \
  --output-dir release_assets \
  --include-fp16
```

Upload the generated `.safetensors` files and `checksums.sha256` to a GitHub
Release.
