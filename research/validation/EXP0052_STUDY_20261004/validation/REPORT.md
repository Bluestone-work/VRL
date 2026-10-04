# EXP0052 paired_validation

Audit: PASS; 168/168 attempts complete; 12 paired scenes.

Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.

| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| memory_n3 | 12/12 | 0/12 | 86.35 | 67.42 | 18.558 | 0.000 | 0 |
| balanced_n3 | 12/12 | 0/12 | 83.41 | 67.73 | 27.268 | 0.000 | 0 |
| balanced_fast_n3 | 12/12 | 0/12 | 89.66 | 71.76 | 20.824 | 0.000 | 0 |
| balanced_flat_n3 | 12/12 | 0/12 | 87.58 | 71.34 | 19.969 | 0.000 | 0 |
| priority_n3 | 12/12 | 1/12 | 81.33 | 67.51 | 30.252 | 0.000 | 0 |
| memory_n1 | 12/12 | 0/12 | 27.08 | 20.37 | 6.079 | 0.000 | 0 |
| untrained_balanced_n3 | 12/12 | 0/12 | 83.41 | 67.73 | 27.268 | 0.000 | 0 |
| untrained_flat_n3 | 12/12 | 0/12 | 87.58 | 71.34 | 19.969 | 0.000 | 0 |
| hierarchical_s42 | 12/12 | 0/12 | 86.35 | 67.42 | 18.558 | 0.000 | 0 |
| hierarchical_s43 | 12/12 | 0/12 | 82.18 | 65.37 | 23.377 | 0.000 | 0 |
| hierarchical_s44 | 12/12 | 0/12 | 82.18 | 64.97 | 23.091 | 0.000 | 0 |
| flat_s42 | 12/12 | 0/12 | 85.42 | 69.51 | 20.475 | 0.000 | 0 |
| flat_s43 | 12/12 | 0/12 | 89.58 | 69.81 | 25.494 | 0.000 | 0 |
| flat_s44 | 12/12 | 0/12 | 87.58 | 70.87 | 24.141 | 0.000 | 0 |

## Gates

All failures are retained; incomplete attempts block formal comparisons.

## hierarchical minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.027778, paired scene bootstrap 95% [-0.090278, 0.013889].
- removal_auc_180: -0.015021, paired scene bootstrap 95% [-0.047332, 0.006404].
- wall_contact_s: 3.117500, paired scene bootstrap 95% [-0.150892, 8.957151].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## hierarchical minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.001646, paired scene bootstrap 95% [-0.036858, 0.041815].
- removal_auc_180: -0.018134, paired scene bootstrap 95% [-0.067593, 0.029448].
- wall_contact_s: -5.592658, paired scene bootstrap 95% [-12.405642, 0.073762].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## hierarchical minus priority_n3

- cluster_safe_success: -0.083333, paired scene bootstrap 95% [-0.250000, 0.000000].
- removal: 0.022479, paired scene bootstrap 95% [-0.070222, 0.108837].
- removal_auc_180: -0.015905, paired scene bootstrap 95% [-0.080054, 0.043731].
- wall_contact_s: -8.576984, paired scene bootstrap 95% [-17.429575, -0.135796].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## flat minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.011737, paired scene bootstrap 95% [-0.030182, 0.053910].
- removal_auc_180: 0.026412, paired scene bootstrap 95% [-0.021121, 0.075195].
- wall_contact_s: 4.812398, paired scene bootstrap 95% [0.715406, 9.832552].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## flat minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.041161, paired scene bootstrap 95% [0.000000, 0.110100].
- removal_auc_180: 0.023299, paired scene bootstrap 95% [-0.003188, 0.054073].
- wall_contact_s: -3.897760, paired scene bootstrap 95% [-13.201432, 3.697435].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## flat minus priority_n3

- cluster_safe_success: -0.083333, paired scene bootstrap 95% [-0.250000, 0.000000].
- removal: 0.061995, paired scene bootstrap 95% [-0.028283, 0.151261].
- removal_auc_180: 0.025529, paired scene bootstrap 95% [-0.024142, 0.075115].
- wall_contact_s: -6.882086, paired scene bootstrap 95% [-14.456246, 0.531368].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## Matched observer and prior

All arms use command-aligned measured tracking. Untrained policies must match measured allocation at the same decision interval. Observer/prior upgrades alone are not learning gains.
Exact untrained allocation equivalence: {"untrained_balanced_n3": true, "untrained_flat_n3": true}
Classical allocation at 0.1 s, 1 s and 5 s remains in the strong comparator set.

### hierarchical minus balanced_fast_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.060854, scene bootstrap 95% [-0.138889, 0.006186].
- wall_contact_s: 0.851567, scene bootstrap 95% [-5.308338, 8.329992].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].

### hierarchical minus balanced_flat_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.040021, scene bootstrap 95% [-0.104925, 0.006186].
- wall_contact_s: 1.706084, scene bootstrap 95% [-5.197692, 10.672335].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].

### flat minus balanced_fast_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.021339, scene bootstrap 95% [-0.076894, 0.026767].
- wall_contact_s: 2.546465, scene bootstrap 95% [-2.459632, 8.149397].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].

### flat minus balanced_flat_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.000505, scene bootstrap 95% [-0.041667, 0.040150].
- wall_contact_s: 3.400981, scene bootstrap 95% [-2.343959, 10.166268].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
