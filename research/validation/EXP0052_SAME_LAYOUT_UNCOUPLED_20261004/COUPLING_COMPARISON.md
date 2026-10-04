# Same-layout synthetic coupling diagnosis

Post-hoc development check. Same policies and all four layouts; no hardware calibration or safety proof.

Admission: PASS.

| Arm | Spacing-violating scenes, alpha=0 | Spacing-violating scenes, alpha=0.1 | Mean spacing pair-s, alpha=0 | Mean spacing pair-s, alpha=0.1 |
|---|---:|---:|---:|---:|
| balanced_fast_n3 | 0 | 0 | 0.000000 | 0.000000 |
| balanced_flat_n3 | 0 | 0 | 0.000000 | 0.000000 |
| balanced_n3 | 0 | 1 | 0.000000 | 15.382301 |
| flat_s42 | 1 | 0 | 0.166679 | 0.000000 |
| flat_s43 | 0 | 0 | 0.000000 | 0.000000 |
| flat_s44 | 0 | 0 | 0.000000 | 0.000000 |
| hierarchical_s42 | 0 | 0 | 0.000000 | 0.000000 |
| hierarchical_s43 | 0 | 1 | 0.000000 | 15.382301 |
| hierarchical_s44 | 0 | 2 | 0.000000 | 15.411271 |
| memory_n1 | 0 | 0 | 0.000000 | 0.000000 |
| memory_n3 | 0 | 0 | 0.000000 | 0.000000 |
| priority_n3 | 0 | 0 | 0.000000 | 0.000000 |
| untrained_balanced_n3 | 0 | 1 | 0.000000 | 15.382301 |
| untrained_flat_n3 | 0 | 0 | 0.000000 | 0.000000 |

Intervals are exploratory over only four scenes; repeated training seeds are not independent layouts.
