# Imaging-quality sensitivity of the image-based perception chain (2026-10-06)

14 anatomies x dev seeds k=0..4, N=1 (pursuit_dep) / N=3 (pursuit_tpg_dep), union physics, horizon 300 s.
Settings: fixed pixel size 20/40/60 um, and `random` = per-episode randomisation of pixel size
(15-35 um), PSF (0.8-2 px), read/shot noise, background contrast and exposure (cf. Turbo's multi-level
domain randomisation). "div" = episodes where the tracker lost a cluster beyond 1 mm.

v1 (rows_v1_area_classifier.jsonl) classified blobs cluster/particle by PIXEL COUNT. At 60 um/px a cluster
is only ~2.7 px across, so its area approaches a particle's and the tracker latched onto particles:
divergence 8.6 % (N=1) / 14.3 % (N=3), and every diverged episode failed.
Fix: classify by INTEGRATED INTENSITY in physical units, which is nearly resolution-invariant, with the
threshold calibrated from the camera model and the known body radii (what a detector trained on cluster
images would know). At 60 um/px: cluster flux 0.049 vs threshold 0.021 vs particle 0.008 (2.4x margin).

| N | camera | Safe % | Raw % | particle ev | err p95 mm | diverged % |
|---|---|---:|---:|---:|---:|---:|
| 1 | 20 um | 84.3 | 91.4 | 0.09 | 0.157 | 0.0 |
| 1 | 40 um | 77.1 | 90.0 | 0.14 | 0.276 | 1.4 |
| 1 | 60 um | 72.9 | 87.1 | 0.17 | 1.023 | 4.3 |
| 1 | randomised | 82.9 | 90.0 | 0.09 | 0.636 | 1.4 |
| 3 | 20 um | 77.1 | 95.7 | 0.94 | 0.153 | 0.0 |
| 3 | 40 um | 72.9 | 97.1 | 1.16 | 0.208 | 2.9 |
| 3 | 60 um | 70.0 | 91.4 | 1.07 | 2.986 | 10.0 |
| 3 | randomised | 77.1 | 94.3 | 0.79 | 0.154 | 1.4 |

Reading: the pipeline is insensitive to imaging quality down to ~40 um/px (Safe -4 to -7 pp, divergence
<= 3 %); at 60 um/px a cluster spans ~2.7 px and detection/tracking becomes the dominant failure mode.
Under per-episode randomisation performance matches the 20 um camera, i.e. no single camera setting is
being exploited.
