# Inference Guide

This guide explains how to run the trained pix2pix RGB-to-NIR generator on a
new image.

## Quick Start

Install the package and download the generator weights:

```bash
python -m pip install -e .
python scripts/download_weights.py --variant fp32
```

Run inference on one RGB skin crop:

```bash
python scripts/predict_image.py \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/rgb_crop.png \
  --output outputs/nir_prediction.png
```

The output is an 8-bit grayscale PNG containing the predicted NIR-like image.

## Input Images

The model is designed for 128x128 RGB skin crops. For best results, prepare the
input image as a close crop of skin from the hand or forearm.

Recommended input:

- RGB image, or any image that can be converted to RGB.
- Square crop before inference, so the skin region is not stretched.
- Hand or forearm skin filling most of the crop.
- Even lighting where vein-related contrast is visible in the original image.

Avoid using full scene images directly. Crop the skin region first, then run
inference on that crop.

## Supported Formats

The inference script uses Pillow to load images. Common formats such as PNG,
JPEG, TIFF, BMP, and WebP are supported when the local Pillow installation
supports them.

The script converts the input to RGB internally:

```python
Image.open(input_path).convert("RGB")
```

The saved prediction is always a grayscale image.

## Image Size Handling

By default, any input size is accepted and resized to 128x128:

```bash
python scripts/predict_image.py \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/skin_crop.jpg \
  --output outputs/nir_prediction.png
```

For a pre-resized 128x128 crop, disable resizing:

```bash
python scripts/predict_image.py \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/skin_crop_128.png \
  --output outputs/nir_prediction.png \
  --no-resize
```

With `--no-resize`, the script raises an error if the input is not exactly
128x128.

The `--image-size` option exists for compatibility with models trained at other
sizes. Keep the default `128` for the released model.

## Device Selection

The script uses automatic device selection by default. On Apple Silicon, MPS is
used when available; otherwise CPU is used.

Force a specific device:

```bash
python scripts/predict_image.py \
  --device cpu \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/skin_crop.png \
  --output outputs/nir_prediction.png
```

```bash
python scripts/predict_image.py \
  --device mps \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/skin_crop.png \
  --output outputs/nir_prediction.png
```

CUDA can also be selected on systems with a compatible NVIDIA GPU:

```bash
python scripts/predict_image.py \
  --device cuda \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input path/to/skin_crop.png \
  --output outputs/nir_prediction.png
```

## Functional Test

The repository includes a synthetic RGB skin crop for checking that the install,
weights, and inference path are working:

```bash
python scripts/predict_image.py \
  --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors \
  --input examples/rgb_synthetic_skin_crop.png \
  --output outputs/synthetic_nir_prediction.png
```

## Troubleshooting

If weight download fails, check that the `v0.1.0` GitHub Release contains the
published `.safetensors` files.

If the input cannot be opened, convert it to PNG or JPEG and try again.

If `--no-resize` fails, check that the input image dimensions are exactly
128x128 pixels.

If MPS or CUDA is unavailable, run with `--device cpu`.
