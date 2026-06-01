# Reproducible Evaluation

Evaluation uses the released fp32 generator weights and the participant-disjoint selected-clean test split.

## Provenance

- Weights SHA256: `f0928bb058e9323dbdfe552ed0efa83f0cee335727a0ee76190759164bd26c4e`
- Test manifest SHA256: `8f1159bd2ed527888758965d066962c1e609af9eafa2caf016675e550596c172`
- Train manifest SHA256: `bb839a792c3b3c6fe962ab0d9f584ea01baea9df965baac660928ecb524e692b`
- Test crops: 19323
- Test subjects: 45
- Device: mps
- Evaluation generated: 2026-06-01T00:05:56.949906+00:00

Reproduce with:

```bash
python scripts/evaluate_benchmarks.py --weights weights/rgb2nir-pix2pix-generator-fp32.safetensors --train-csv manifests/train_fit_selected_clean.csv --test-csv manifests/test_selected_clean.csv --crop-root data/crops/selected/clean --output-dir reports/reproducible_evaluation --include-vein
```

The training history CSV reports validation metrics during training. The tables below are independent test-set metrics.

## Crop-Level Mean Metrics

| method | n | mae | rmse | psnr | ssim |
| --- | ---: | ---: | ---: | ---: | ---: |
| pix2pix_cgan | 19323 | 0.0464 | 0.0519 | 27.7760 | 0.9009 |
| mean_std_luminance | 19323 | 0.0744 | 0.0829 | 23.0867 | 0.8884 |
| rgb_luminance | 19323 | 0.2159 | 0.2204 | 14.5618 | 0.8043 |
| green_channel | 19323 | 0.2689 | 0.2733 | 12.1283 | 0.7556 |
| train_mean_nir | 19323 | 0.0885 | 0.0993 | 21.1980 | 0.8890 |

## Subject-Averaged Metrics

| method | n_subjects | n_crops | mae | rmse | psnr | ssim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pix2pix_cgan | 45 | 19323 | 0.0460 | 0.0510 | 27.7686 | 0.9032 |
| mean_std_luminance | 45 | 19323 | 0.0730 | 0.0813 | 23.2546 | 0.8885 |
| rgb_luminance | 45 | 19323 | 0.2114 | 0.2159 | 14.7263 | 0.8087 |
| green_channel | 45 | 19323 | 0.2645 | 0.2688 | 12.2677 | 0.7607 |
| train_mean_nir | 45 | 19323 | 0.0853 | 0.0960 | 21.4508 | 0.8910 |

## Full-Test Sato Vein-Map Metrics

| method | n | vessel_corr | vessel_dice | vessel_iou | skeleton_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pix2pix_cgan | 19323 | 0.5074 | 0.5026 | 0.3509 | 0.1187 |
| mean_std_luminance | 19323 | 0.3983 | 0.4304 | 0.2846 | 0.0936 |
| rgb_luminance | 19323 | 0.3983 | 0.4304 | 0.2846 | 0.0936 |
| green_channel | 19323 | 0.3219 | 0.3858 | 0.2490 | 0.0824 |
| train_mean_nir | 19323 | 0.1084 | 0.1872 | 0.1085 | 0.0737 |

## Subject-Averaged Sato Vein-Map Metrics

| method | n_subjects | vessel_corr | vessel_dice | vessel_iou | skeleton_f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pix2pix_cgan | 45 | 0.5218 | 0.5121 | 0.3580 | 0.1216 |
| mean_std_luminance | 45 | 0.4011 | 0.4291 | 0.2823 | 0.0922 |
| rgb_luminance | 45 | 0.4011 | 0.4290 | 0.2823 | 0.0922 |
| green_channel | 45 | 0.3212 | 0.3825 | 0.2453 | 0.0807 |
| train_mean_nir | 45 | 0.1179 | 0.1932 | 0.1126 | 0.0779 |

## Baselines

- `rgb_luminance`: standard RGB luminance converted directly to grayscale.
- `green_channel`: the RGB green channel used directly as a grayscale estimate.
- `mean_std_luminance`: RGB luminance matched to train-set NIR mean and standard deviation.
- `train_mean_nir`: the train-set pixelwise mean NIR image.
