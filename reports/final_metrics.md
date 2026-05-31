# Final Model Metrics

Paired image evaluation was run on the participant-disjoint test split with
19,323 paired RGB/NIR crops.

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
| Vessel length ratio | 0.9191 |
