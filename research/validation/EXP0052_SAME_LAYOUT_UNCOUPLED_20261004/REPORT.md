# EXP0052 posthoc_same_layout_uncoupled

Audit: PASS; 56/56 attempts complete; 4 paired scenes.

Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.

| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| memory_n3 | 4/4 | 0/4 | 100.00 | 83.51 | 9.545 | 0.000 | 0 |
| balanced_n3 | 4/4 | 0/4 | 90.80 | 64.84 | 37.802 | 0.000 | 0 |
| balanced_fast_n3 | 4/4 | 1/4 | 100.00 | 76.59 | 19.048 | 0.000 | 0 |
| balanced_flat_n3 | 4/4 | 0/4 | 93.75 | 71.18 | 31.033 | 0.000 | 0 |
| priority_n3 | 4/4 | 1/4 | 100.00 | 75.02 | 18.745 | 0.000 | 0 |
| memory_n1 | 4/4 | 1/4 | 43.75 | 34.34 | 10.797 | 0.000 | 0 |
| untrained_balanced_n3 | 4/4 | 0/4 | 90.80 | 64.84 | 37.802 | 0.000 | 0 |
| untrained_flat_n3 | 4/4 | 0/4 | 93.75 | 71.18 | 31.033 | 0.000 | 0 |
| hierarchical_s42 | 4/4 | 0/4 | 100.00 | 83.51 | 9.545 | 0.000 | 0 |
| hierarchical_s43 | 4/4 | 0/4 | 100.00 | 79.27 | 13.597 | 0.000 | 0 |
| hierarchical_s44 | 4/4 | 0/4 | 100.00 | 83.51 | 9.545 | 0.000 | 0 |
| flat_s42 | 4/4 | 0/4 | 87.50 | 71.23 | 38.672 | 0.167 | 1 |
| flat_s43 | 4/4 | 0/4 | 93.75 | 72.74 | 35.814 | 0.000 | 0 |
| flat_s44 | 4/4 | 0/4 | 93.75 | 74.76 | 26.929 | 0.000 | 0 |

## Gates

All failures are retained; incomplete attempts block formal comparisons.

## hierarchical minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal_auc_180: -0.014138, paired scene bootstrap 95% [-0.038541, 0.000000].
- wall_contact_s: 1.350781, paired scene bootstrap 95% [0.000000, 2.701562].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## hierarchical minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.092019, paired scene bootstrap 95% [0.000000, 0.184037].
- removal_auc_180: 0.172541, paired scene bootstrap 95% [0.059210, 0.266018].
- wall_contact_s: -26.906501, paired scene bootstrap 95% [-52.431409, -1.381593].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## hierarchical minus priority_n3

- cluster_safe_success: -0.250000, paired scene bootstrap 95% [-0.750000, 0.000000].
- removal: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal_auc_180: 0.070802, paired scene bootstrap 95% [-0.021584, 0.199763].
- wall_contact_s: -7.849194, paired scene bootstrap 95% [-21.352163, 2.292107].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## flat minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.083333, paired scene bootstrap 95% [-0.166667, 0.000000].
- removal_auc_180: -0.106000, paired scene bootstrap 95% [-0.192401, -0.029527].
- wall_contact_s: 24.260572, paired scene bootstrap 95% [7.683742, 36.964637].
- spacing_violation_pair_s: 0.055560, paired scene bootstrap 95% [0.000000, 0.166679].

## flat minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.008685, paired scene bootstrap 95% [-0.010834, 0.036890].
- removal_auc_180: 0.080679, paired scene bootstrap 95% [0.017597, 0.143760].
- wall_contact_s: -3.996710, paired scene bootstrap 95% [-24.186593, 14.534758].
- spacing_violation_pair_s: 0.055560, paired scene bootstrap 95% [0.000000, 0.166679].

## flat minus priority_n3

- cluster_safe_success: -0.250000, paired scene bootstrap 95% [-0.750000, 0.000000].
- removal: -0.083333, paired scene bootstrap 95% [-0.166667, 0.000000].
- removal_auc_180: -0.021060, paired scene bootstrap 95% [-0.108214, 0.066093].
- wall_contact_s: 15.060597, paired scene bootstrap 95% [6.656592, 25.236432].
- spacing_violation_pair_s: 0.055560, paired scene bootstrap 95% [0.000000, 0.166679].

## Matched observer and prior

All arms use command-aligned measured tracking. Untrained policies must match measured allocation at the same decision interval. Observer/prior upgrades alone are not learning gains.
Exact untrained allocation equivalence: {"untrained_balanced_n3": true, "untrained_flat_n3": true}
Classical allocation at 0.1 s, 1 s and 5 s remains in the strong comparator set.

### hierarchical minus balanced_fast_n3

- cluster_safe_success: -0.250000, scene bootstrap 95% [-0.750000, 0.000000].
- removal: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- wall_contact_s: -8.152221, scene bootstrap 95% [-19.518141, 0.001482].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].

### hierarchical minus balanced_flat_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.062500, scene bootstrap 95% [0.000000, 0.187500].
- wall_contact_s: -20.137650, scene bootstrap 95% [-32.292584, -6.183696].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].

### flat minus balanced_fast_n3

- cluster_safe_success: -0.250000, scene bootstrap 95% [-0.750000, 0.000000].
- removal: -0.083333, scene bootstrap 95% [-0.166667, 0.000000].
- wall_contact_s: 14.757570, scene bootstrap 95% [4.278516, 25.848019].
- spacing_violation_pair_s: 0.055560, scene bootstrap 95% [0.000000, 0.166679].

### flat minus balanced_flat_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.020833, scene bootstrap 95% [-0.125000, 0.062500].
- wall_contact_s: 2.772141, scene bootstrap 95% [-8.231605, 13.547939].
- spacing_violation_pair_s: 0.055560, scene bootstrap 95% [0.000000, 0.166679].
