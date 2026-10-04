# EXP0048 tracked local-learning analysis

Audit: PASS; 48 recorded attempts; 12 scenes.

Rates below are developmental, not sealed results. Three training seeds on one scene are correlated repeats. Intervals resample scenes, not individual robot samples. Secondary-endpoint intervals are exploratory and not multiplicity adjusted.

| Arm | Complete/requested | Cluster-safe % | Raw % | Removal % | AUC % | Wall cluster-s | Spacing pair-s |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 12/12 | 0.000 | 0.000 | 46.504 | 38.236 | 170.296 | 8.252 |
| joint_n3 | 12/12 | 0.000 | 0.000 | 48.318 | 38.170 | 157.384 | 11.582 |
| memory_n3 | 12/12 | 0.000 | 33.333 | 72.653 | 53.590 | 25.962 | 0.842 |
| joint_n1 | 12/12 | 0.000 | 0.000 | 16.667 | 14.156 | 81.972 | 0.000 |

Primary success requires physical clearance AND measured completion, with safety evaluated through observed stopping time. All sensing values are engineered stress assumptions; this is not a validated imaging or hardware study.

| Arm | Visual completion % | False visual completion % | Physically safe % | T90, failure=180 s | Observed stop s | Squared command s |
|---|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 493.898 |
| joint_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 493.291 |
| memory_n3 | 33.333 | 0.000 | 0.000 | 157.442 | 159.267 | 405.944 |
| joint_n1 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 179.452 |

## Paired effects (learning minus heuristic)

| Budget / comparator | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] | Spacing pair-s [95% CI] |
|---|---:|---:|---:|---:|---:|

Joint filtering is shared with the strong conventional comparator. Its improvement over old v2 spacing is not a learning contribution. No claim of calibrated magnetic independence, hardware transfer or safety certification is made.
