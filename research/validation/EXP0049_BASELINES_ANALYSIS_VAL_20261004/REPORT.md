# EXP0049 temporal-candidate local-learning analysis

Audit: PASS; 60 recorded attempts; 12 scenes.

Rates below are developmental, not sealed results. Three training seeds on one scene are correlated repeats. Intervals resample scenes, not individual robot samples. Secondary-endpoint intervals are exploratory and not multiplicity adjusted.

| Arm | Complete/requested | Cluster-safe % | Raw % | Removal % | AUC % | Wall cluster-s | Spacing pair-s |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 12/12 | 0.000 | 16.667 | 54.611 | 44.762 | 150.506 | 19.143 |
| joint_n3 | 12/12 | 0.000 | 33.333 | 59.529 | 48.388 | 120.720 | 11.907 |
| memory_n3 | 12/12 | 0.000 | 58.333 | 83.333 | 63.307 | 30.316 | 1.367 |
| joint_n1 | 12/12 | 0.000 | 0.000 | 27.083 | 20.420 | 121.916 | 0.000 |
| memory_n1 | 12/12 | 0.000 | 8.333 | 35.417 | 26.224 | 18.858 | 0.000 |

Primary success requires physical clearance AND measured completion, with safety evaluated through observed stopping time. All sensing values are engineered stress assumptions; this is not a validated imaging or hardware study.

| Arm | Visual completion % | False visual completion % | Physically safe % | T90, failure=180 s | Observed stop s | Squared command s |
|---|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 16.667 | 0.000 | 0.000 | 168.025 | 169.000 | 441.295 |
| joint_n3 | 33.333 | 0.000 | 0.000 | 155.983 | 162.075 | 424.501 |
| memory_n3 | 58.333 | 0.000 | 0.000 | 134.617 | 137.850 | 380.331 |
| joint_n1 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 177.635 |
| memory_n1 | 8.333 | 0.000 | 0.000 | 179.533 | 178.358 | 173.425 |

## Paired effects (learning minus heuristic)

| Budget / comparator | Cluster-safe pp [95% CI] | Removal pp [95% CI] | AUC pp [95% CI] | Wall seconds [95% CI] | Spacing pair-s [95% CI] |
|---|---:|---:|---:|---:|---:|

Joint filtering is shared with the strong conventional comparator. Its improvement over old v2 spacing is not a learning contribution. No claim of calibrated magnetic independence, hardware transfer or safety certification is made.
