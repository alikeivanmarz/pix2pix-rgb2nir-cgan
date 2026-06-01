# Final Model Metrics

Evaluation uses the released fp32 generator weights and the full
participant-disjoint selected-clean test split: 19,323 crops from 45 subjects.

Full provenance, manifest hashes, baseline definitions, and per-subject tables
are available in
[`reports/reproducible_evaluation/evaluation_summary.md`](reproducible_evaluation/evaluation_summary.md).

The training history CSV reports validation metrics during training. The tables
below are independent test-set metrics.

## Crop-Level Mean Metrics

| Method | MAE | RMSE | PSNR | SSIM | Vessel Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| pix2pix cGAN | 0.0464 | 0.0519 | 27.7760 | 0.9009 | 0.5026 |
| mean/std luminance baseline | 0.0744 | 0.0829 | 23.0867 | 0.8884 | 0.4304 |
| RGB luminance baseline | 0.2159 | 0.2204 | 14.5618 | 0.8043 | 0.4304 |
| green-channel baseline | 0.2689 | 0.2733 | 12.1283 | 0.7556 | 0.3858 |
| train-mean NIR baseline | 0.0885 | 0.0993 | 21.1980 | 0.8890 | 0.1872 |

## Subject-Averaged Metrics

| Method | Subjects | MAE | RMSE | PSNR | SSIM | Vessel Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pix2pix cGAN | 45 | 0.0460 | 0.0510 | 27.7686 | 0.9032 | 0.5121 |
| mean/std luminance baseline | 45 | 0.0730 | 0.0813 | 23.2546 | 0.8885 | 0.4291 |
| RGB luminance baseline | 45 | 0.2114 | 0.2159 | 14.7263 | 0.8087 | 0.4290 |
| green-channel baseline | 45 | 0.2645 | 0.2688 | 12.2677 | 0.7607 | 0.3825 |
| train-mean NIR baseline | 45 | 0.0853 | 0.0960 | 21.4508 | 0.8910 | 0.1932 |
