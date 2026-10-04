# EXP0051 paired_validation

Audit: PASS; 144/144 attempts complete; 12 paired scenes.

Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.

| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| memory_n3 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |
| balanced_n3 | 12/12 | 0/12 | 83.33 | 60.72 | 19.913 | 0.097 | 5 |
| priority_n3 | 12/12 | 0/12 | 76.36 | 58.28 | 22.763 | 0.249 | 3 |
| memory_n1 | 12/12 | 0/12 | 33.33 | 24.74 | 7.876 | 0.000 | 0 |
| legacy_memory_n3 | 12/12 | 0/12 | 78.13 | 54.94 | 16.588 | 0.715 | 6 |
| untrained_n3 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |
| hierarchical_s42 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |
| hierarchical_s43 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |
| hierarchical_s44 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |
| flat_s42 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |
| flat_s43 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |
| flat_s44 | 12/12 | 0/12 | 73.75 | 54.66 | 25.253 | 0.188 | 6 |

## Gates

Untrained equals memory: True
All failures are retained; incomplete attempts block formal comparisons.
The legacy-memory arm has a different safety layer, so any difference alone is not a learning gain.

## hierarchical minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal_auc_180: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].
- wall_contact_s: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].
- spacing_violation_pair_s: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].

## hierarchical minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.095879, paired scene bootstrap 95% [-0.200046, -0.020833].
- removal_auc_180: -0.060614, paired scene bootstrap 95% [-0.129916, -0.009456].
- wall_contact_s: 5.339547, paired scene bootstrap 95% [0.566013, 12.043099].
- spacing_violation_pair_s: 0.091605, paired scene bootstrap 95% [0.000000, 0.215855].

## hierarchical minus priority_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.026192, paired scene bootstrap 95% [-0.159480, 0.105994].
- removal_auc_180: -0.036225, paired scene bootstrap 95% [-0.118993, 0.035821].
- wall_contact_s: 2.489785, paired scene bootstrap 95% [-7.834334, 12.710415].
- spacing_violation_pair_s: -0.060942, paired scene bootstrap 95% [-0.509106, 0.268165].

## flat minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal_auc_180: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].
- wall_contact_s: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].
- spacing_violation_pair_s: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].

## flat minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.095879, paired scene bootstrap 95% [-0.200046, -0.020833].
- removal_auc_180: -0.060614, paired scene bootstrap 95% [-0.129916, -0.009456].
- wall_contact_s: 5.339547, paired scene bootstrap 95% [0.566013, 12.043099].
- spacing_violation_pair_s: 0.091605, paired scene bootstrap 95% [0.000000, 0.215855].

## flat minus priority_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.026192, paired scene bootstrap 95% [-0.159480, 0.105994].
- removal_auc_180: -0.036225, paired scene bootstrap 95% [-0.118993, 0.035821].
- wall_contact_s: 2.489785, paired scene bootstrap 95% [-7.834334, 12.710415].
- spacing_violation_pair_s: -0.060942, paired scene bootstrap 95% [-0.509106, 0.268165].
