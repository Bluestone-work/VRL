# EXP0050 memory-prior local-learning analysis

Audit: BLOCKED; 132 recorded attempts; 12 scenes.

Rates below are developmental, not sealed results. Three training seeds on one scene are correlated repeats. Intervals resample scenes, not individual robot samples. Secondary-endpoint intervals are exploratory and not multiplicity adjusted. A degenerate all-zero bootstrap difference interval is NOT evidence of equivalence. Per-arm primary-rate intervals below are exact binomial intervals for fixed policies across scenes, widened over unknown failures if necessary.

| Arm | Complete/requested | Cluster-safe % | Raw % | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Cluster-safe 95% CI % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 12/12 | 0.000 | 16.667 | 51.578 | 43.634 | 172.022 | 14.562 | [0.000, 26.465] |
| joint_n3 | 12/12 | 0.000 | 8.333 | 54.363 | 42.349 | 134.772 | 29.832 | [0.000, 26.465] |
| memory_n3 | 12/12 | 0.000 | 33.333 | 75.296 | 52.815 | 32.022 | 0.879 | [0.000, 26.465] |
| joint_n1 | 12/12 | 0.000 | 0.000 | 12.500 | 11.678 | 91.962 | 0.000 | [0.000, 26.465] |
| memory_n1 | 12/12 | 8.333 | 16.667 | 37.500 | 29.665 | 20.662 | 0.000 | [0.211, 38.480] |
| learned_s42_t32768_n3 | 12/12 | 0.000 | 0.000 | 26.654 | 23.983 | 3.639 | 0.692 | [0.000, 26.465] |
| learned_s43_t32768_n3 | 12/12 | 0.000 | 58.333 | 87.500 | 63.314 | 24.819 | 2.813 | [0.000, 26.465] |
| learned_s44_t32768_n3 | 12/12 | 0.000 | 33.333 | 75.296 | 52.815 | 32.022 | 0.879 | [0.000, 26.465] |
| learned_s42_t65536_n3 | 12/12 | 0.000 | 0.000 | 0.574 | 0.426 | 88.522 | 0.174 | [0.000, 26.465] |
| learned_s43_t65536_n3 | 12/12 | 0.000 | 0.000 | 2.477 | 1.571 | 0.005 | 0.000 | [0.000, 26.465] |
| learned_s44_t65536_n3 | 11/12 | 0.000 | 0.000 | 10.965 | 8.800 | 64.372 | 0.000 | [0.000, 38.480] |

Primary success requires physical clearance AND measured completion, with safety evaluated through observed stopping time. All sensing values are engineered stress assumptions; this is not a validated imaging or hardware study.

| Arm | Visual completion % | False visual completion % | Physically safe % | T90, failure=180 s | Observed stop s | Squared command s |
|---|---:|---:|---:|---:|---:|---:|
| v2_spacing_n3 | 16.667 | 0.000 | 0.000 | 151.983 | 163.917 | 436.465 |
| joint_n3 | 8.333 | 0.000 | 0.000 | 171.825 | 178.808 | 462.703 |
| memory_n3 | 33.333 | 0.000 | 0.000 | 161.550 | 159.917 | 405.848 |
| joint_n1 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 179.414 |
| memory_n1 | 16.667 | 0.000 | 8.333 | 166.058 | 165.333 | 160.706 |
| learned_s42_t32768_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 170.933 | 411.406 |
| learned_s43_t32768_n3 | 58.333 | 0.000 | 0.000 | 132.825 | 138.158 | 371.489 |
| learned_s44_t32768_n3 | 33.333 | 0.000 | 0.000 | 161.550 | 159.917 | 405.848 |
| learned_s42_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 72.338 |
| learned_s43_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 0.117 |
| learned_s44_t65536_n3 | 0.000 | 0.000 | 0.000 | 180.000 | 180.000 | 138.839 |

**Comparisons blocked:**
- Failed learned_s44_t65536_n3 scene 1620000011: process_exit

Joint filtering is shared with the strong conventional comparator. Its improvement over old v2 spacing is not a learning contribution. No claim of calibrated magnetic independence, hardware transfer or safety certification is made.
