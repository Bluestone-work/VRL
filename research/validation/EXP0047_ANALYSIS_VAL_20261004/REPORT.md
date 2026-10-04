# EXP0047 local-learning diagnostic analysis

Audit: PASS; 84 recorded attempts; 12 scenes.

Rates below are developmental, not sealed results. Three training seeds on one scene are correlated repeats. Intervals resample scenes, not individual robot samples. Secondary-endpoint intervals are exploratory and not multiplicity adjusted.

| Arm | Complete/requested | Cluster-safe % | Raw % | Removal % | AUC % | Wall cluster-s | Spacing pair-s |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 12/12 | 8.333 | 33.333 | 70.833 | 58.225 | 15.230 | 1.275 |
| joint_n3 | 12/12 | 25.000 | 41.667 | 76.761 | 61.296 | 1.029 | 0.373 |
| memory_n3 | 12/12 | 16.667 | 58.333 | 85.417 | 66.829 | 0.773 | 0.233 |
| joint_n1 | 12/12 | 0.000 | 0.000 | 20.833 | 18.088 | 0.820 | 0.000 |
| learned_s42_t32768_n3 | 12/12 | 25.000 | 41.667 | 75.090 | 60.238 | 1.138 | 0.336 |
| learned_s43_t32768_n3 | 12/12 | 25.000 | 41.667 | 72.917 | 59.093 | 1.009 | 0.180 |
| learned_s44_t32768_n3 | 12/12 | 25.000 | 41.667 | 77.083 | 61.544 | 1.044 | 0.382 |

## Paired effects (learning minus heuristic)

| Budget / comparator | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] | Spacing pair-s [95% CI] |
|---|---:|---:|---:|---:|---:|
| 32768 / joint_n3 | -0.000 [-11.111, +11.111] | -1.731 [-8.312, +4.167] | -1.004 [-4.062, +1.951] | +0.034 [-0.068, +0.154] | -0.074 [-0.422, +0.191] |
| 32768 / memory_n3 | +8.333 [-16.667, +30.556] | -10.387 [-20.743, -2.083] | -6.537 [-13.479, -1.707] | +0.291 [-0.746, +1.689] | +0.067 [-0.188, +0.317] |
| 32768 / v2_spacing_n3 | +16.667 [-2.778, +38.889] | +4.197 [-7.579, +19.474] | +2.067 [-4.215, +9.119] | -14.167 [-28.102, -4.264] | -0.976 [-3.095, +0.145] |

Joint filtering is shared with the strong conventional comparator. Its improvement over old v2 spacing is not a learning contribution. No claim of calibrated magnetic independence, hardware transfer or safety certification is made.
