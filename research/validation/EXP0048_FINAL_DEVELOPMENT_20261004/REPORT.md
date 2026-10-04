# EXP0048 tracked local-learning analysis

Audit: PASS; 132 recorded attempts; 12 scenes.

Rates below are developmental, not sealed results. Three training seeds on one scene are correlated repeats. Intervals resample scenes, not individual robot samples. Secondary-endpoint intervals are exploratory and not multiplicity adjusted. A degenerate all-zero bootstrap difference interval is NOT evidence of equivalence. Per-arm primary-rate intervals below are exact binomial intervals for fixed policies across scenes, widened over unknown failures if necessary.

| Arm | Complete/requested | Cluster-safe % | Raw % | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Cluster-safe 95% CI % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 12/12 | 0.000 | 0.000 | 46.504 | 38.236 | 170.296 | 8.252 | [0.000, 26.465] |
| joint_n3 | 12/12 | 0.000 | 0.000 | 48.318 | 38.170 | 157.384 | 11.582 | [0.000, 26.465] |
| memory_n3 | 12/12 | 0.000 | 33.333 | 72.653 | 53.590 | 25.962 | 0.842 | [0.000, 26.465] |
| joint_n1 | 12/12 | 0.000 | 0.000 | 16.667 | 14.156 | 81.972 | 0.000 | [0.000, 26.465] |
| memory_n1 | 12/12 | 0.000 | 0.000 | 14.725 | 11.829 | 5.762 | 0.000 | [0.000, 26.465] |
| learned_s42_t32768_n3 | 12/12 | 0.000 | 0.000 | 11.586 | 10.061 | 1.177 | 11.174 | [0.000, 26.465] |
| learned_s43_t32768_n3 | 12/12 | 0.000 | 0.000 | 46.630 | 37.680 | 100.344 | 0.986 | [0.000, 26.465] |
| learned_s44_t32768_n3 | 12/12 | 0.000 | 0.000 | 13.432 | 11.761 | 29.585 | 0.815 | [0.000, 26.465] |
| learned_s42_t65536_n3 | 12/12 | 0.000 | 0.000 | 19.275 | 16.648 | 0.275 | 7.714 | [0.000, 26.465] |
| learned_s43_t65536_n3 | 12/12 | 0.000 | 0.000 | 13.455 | 11.991 | 0.217 | 11.492 | [0.000, 26.465] |
| learned_s44_t65536_n3 | 12/12 | 0.000 | 0.000 | 8.507 | 7.671 | 49.947 | 0.000 | [0.000, 26.465] |

Primary success requires physical clearance AND measured completion, with safety evaluated through observed stopping time. All sensing values are engineered stress assumptions; this is not a validated imaging or hardware study.

| Arm | Visual completion % | False visual completion % | Physically safe % | T90, failure=180 s | Observed stop s | Squared command s |
|---|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 493.898 |
| joint_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 493.291 |
| memory_n3 | 33.333 | 0.000 | 0.000 | 157.442 | 159.267 | 405.944 |
| joint_n1 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 179.452 |
| memory_n1 | 0.000 | 0.000 | 0.000 | 180.000 | 155.300 | 151.513 |
| learned_s42_t32768_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 461.757 |
| learned_s43_t32768_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 377.737 |
| learned_s44_t32768_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 477.796 |
| learned_s42_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 525.408 |
| learned_s43_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 394.105 |
| learned_s44_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 417.155 |

## Paired effects (learning minus heuristic)

| Budget / comparator | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] | Spacing pair-s [95% CI] |
|---|---:|---:|---:|---:|---:|
| 32768 / joint_n3 | +0.000 [+0.000, +0.000] | -24.435 [-33.371, -15.620] | -18.336 [-25.070, -11.772] | -113.681 [-154.548, -73.390] | -7.257 [-18.055, +1.848] |
| 32768 / memory_n3 | +0.000 [+0.000, +0.000] | -48.770 [-60.052, -37.055] | -33.756 [-42.384, -25.457] | +17.740 [-7.294, +42.789] | +3.483 [-0.296, +8.046] |
| 32768 / v2_spacing_n3 | +0.000 [+0.000, +0.000] | -22.621 [-31.522, -14.052] | -18.402 [-24.482, -12.366] | -126.594 [-171.971, -84.859] | -3.927 [-10.156, +1.695] |
| 65536 / joint_n3 | +0.000 [+0.000, +0.000] | -34.572 [-44.730, -23.512] | -26.066 [-33.963, -17.809] | -140.571 [-195.128, -89.149] | -5.180 [-18.828, +7.982] |
| 65536 / memory_n3 | +0.000 [+0.000, +0.000] | -58.907 [-71.212, -44.646] | -41.487 [-49.227, -32.106] | -9.149 [-31.299, +13.028] | +5.560 [+0.298, +13.221] |
| 65536 / v2_spacing_n3 | +0.000 [+0.000, +0.000] | -32.758 [-43.781, -21.761] | -26.133 [-33.888, -18.157] | -153.483 [-210.317, -100.149] | -1.850 [-10.374, +7.307] |

## Method 2 versus method 1, same conventional control

These paired effects isolate the stated cluster-count comparison, not learning. Aggregate catalytic-rate proxy is matched; magnetic power, material and deployment are not.

| Controller | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] |
|---|---:|---:|---:|---:|
| joint | +0.000 [+0.000, +0.000] | +31.651 [+19.151, +44.552] | +24.014 [+15.947, +31.717] | +75.412 [+19.994, +130.796] |
| memory | +0.000 [+0.000, +0.000] | +57.928 [+39.442, +75.000] | +41.761 [+29.561, +54.009] | +20.200 [+7.617, +36.903] |

Joint filtering is shared with the strong conventional comparator. Its improvement over old v2 spacing is not a learning contribution. No claim of calibrated magnetic independence, hardware transfer or safety certification is made.
