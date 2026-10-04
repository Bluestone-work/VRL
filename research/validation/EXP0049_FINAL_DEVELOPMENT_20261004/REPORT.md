# EXP0049 temporal-candidate local-learning analysis

Audit: PASS; 132 recorded attempts; 12 scenes.

Rates below are developmental, not sealed results. Three training seeds on one scene are correlated repeats. Intervals resample scenes, not individual robot samples. Secondary-endpoint intervals are exploratory and not multiplicity adjusted. A degenerate all-zero bootstrap difference interval is NOT evidence of equivalence. Per-arm primary-rate intervals below are exact binomial intervals for fixed policies across scenes, widened over unknown failures if necessary.

| Arm | Complete/requested | Cluster-safe % | Raw % | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Cluster-safe 95% CI % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 12/12 | 0.000 | 16.667 | 54.611 | 44.762 | 150.506 | 19.143 | [0.000, 26.465] |
| joint_n3 | 12/12 | 0.000 | 33.333 | 59.529 | 48.388 | 120.720 | 11.907 | [0.000, 26.465] |
| memory_n3 | 12/12 | 0.000 | 58.333 | 83.333 | 63.307 | 30.316 | 1.367 | [0.000, 26.465] |
| joint_n1 | 12/12 | 0.000 | 0.000 | 27.083 | 20.420 | 121.916 | 0.000 | [0.000, 26.465] |
| memory_n1 | 12/12 | 0.000 | 8.333 | 35.417 | 26.224 | 18.858 | 0.000 | [0.000, 26.465] |
| learned_s42_t32768_n3 | 12/12 | 0.000 | 0.000 | 0.695 | 0.654 | 33.309 | 2.106 | [0.000, 26.465] |
| learned_s43_t32768_n3 | 12/12 | 0.000 | 25.000 | 63.073 | 51.513 | 129.714 | 15.078 | [0.000, 26.465] |
| learned_s44_t32768_n3 | 12/12 | 0.000 | 0.000 | 38.528 | 35.311 | 62.666 | 4.224 | [0.000, 26.465] |
| learned_s42_t65536_n3 | 12/12 | 0.000 | 0.000 | 23.600 | 20.126 | 70.580 | 6.088 | [0.000, 26.465] |
| learned_s43_t65536_n3 | 12/12 | 0.000 | 0.000 | 5.989 | 4.464 | 26.722 | 0.013 | [0.000, 26.465] |
| learned_s44_t65536_n3 | 12/12 | 0.000 | 0.000 | 9.906 | 8.737 | 39.198 | 0.000 | [0.000, 26.465] |

Primary success requires physical clearance AND measured completion, with safety evaluated through observed stopping time. All sensing values are engineered stress assumptions; this is not a validated imaging or hardware study.

| Arm | Visual completion % | False visual completion % | Physically safe % | T90, failure=180 s | Observed stop s | Squared command s |
|---|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 16.667 | 0.000 | 0.000 | 168.025 | 169.000 | 441.295 |
| joint_n3 | 33.333 | 0.000 | 0.000 | 155.983 | 162.075 | 424.501 |
| memory_n3 | 58.333 | 0.000 | 0.000 | 134.617 | 137.850 | 380.331 |
| joint_n1 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 177.635 |
| memory_n1 | 8.333 | 0.000 | 0.000 | 179.533 | 178.358 | 173.425 |
| learned_s42_t32768_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 99.261 |
| learned_s43_t32768_n3 | 25.000 | 0.000 | 0.000 | 163.367 | 164.783 | 436.652 |
| learned_s44_t32768_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 366.078 |
| learned_s42_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 196.563 |
| learned_s43_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 34.955 |
| learned_s44_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 52.940 |

## Paired effects (learning minus heuristic)

| Budget / comparator | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] | Spacing pair-s [95% CI] |
|---|---:|---:|---:|---:|---:|
| 32768 / joint_n3 | +0.000 [+0.000, +0.000] | -25.431 [-37.984, -12.331] | -19.229 [-26.786, -10.756] | -45.491 [-92.360, -2.323] | -4.771 [-15.691, +4.467] |
| 32768 / memory_n3 | +0.000 [+0.000, +0.000] | -49.235 [-57.987, -40.313] | -34.147 [-39.638, -28.519] | +44.914 [+23.106, +66.998] | +5.769 [+1.853, +10.346] |
| 32768 / v2_spacing_n3 | +0.000 [+0.000, +0.000] | -20.512 [-30.090, -10.630] | -15.603 [-22.797, -8.073] | -75.276 [-140.485, -13.390] | -12.007 [-19.721, -4.701] |
| 65536 / joint_n3 | +0.000 [+0.000, +0.000] | -46.364 [-62.915, -29.472] | -37.279 [-48.135, -25.527] | -75.221 [-137.297, -19.153] | -9.873 [-20.074, -1.323] |
| 65536 / memory_n3 | +0.000 [+0.000, +0.000] | -70.168 [-80.103, -59.802] | -52.198 [-58.886, -45.086] | +15.184 [-13.743, +43.462] | +0.667 [-1.393, +3.958] |
| 65536 / v2_spacing_n3 | +0.000 [+0.000, +0.000] | -41.445 [-53.407, -28.737] | -33.653 [-42.794, -23.676] | -105.006 [-174.379, -35.287] | -17.109 [-25.968, -8.259] |

## Method 2 versus method 1, same conventional control

These paired effects isolate the stated cluster-count comparison, not learning. Aggregate catalytic-rate proxy is matched; magnetic power, material and deployment are not.

| Controller | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] |
|---|---:|---:|---:|---:|
| joint | +0.000 [+0.000, +0.000] | +32.446 [+12.612, +53.706] | +27.968 [+14.014, +42.882] | -1.196 [-46.080, +44.702] |
| memory | +0.000 [+0.000, +0.000] | +47.917 [+27.083, +66.667] | +37.083 [+22.856, +51.764] | +11.458 [-6.077, +31.001] |

Joint filtering is shared with the strong conventional comparator. Its improvement over old v2 spacing is not a learning contribution. No claim of calibrated magnetic independence, hardware transfer or safety certification is made.
