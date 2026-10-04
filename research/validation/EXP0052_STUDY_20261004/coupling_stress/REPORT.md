# EXP0052 synthetic_coupling_stress

Audit: PASS; 56/56 attempts complete; 4 paired scenes.

Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.

| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced_n3 | 4/4 | 0/4 | 87.50 | 70.54 | 62.256 | 15.382 | 1 |
| memory_n3 | 4/4 | 0/4 | 86.92 | 68.77 | 17.126 | 0.000 | 0 |
| balanced_fast_n3 | 4/4 | 0/4 | 100.00 | 81.60 | 14.481 | 0.000 | 0 |
| balanced_flat_n3 | 4/4 | 0/4 | 93.75 | 76.07 | 20.079 | 0.000 | 0 |
| priority_n3 | 4/4 | 0/4 | 100.00 | 82.95 | 10.662 | 0.000 | 0 |
| memory_n1 | 4/4 | 1/4 | 43.75 | 34.34 | 10.797 | 0.000 | 0 |
| untrained_balanced_n3 | 4/4 | 0/4 | 87.50 | 70.54 | 62.256 | 15.382 | 1 |
| untrained_flat_n3 | 4/4 | 0/4 | 93.75 | 76.07 | 20.079 | 0.000 | 0 |
| hierarchical_s42 | 4/4 | 0/4 | 86.92 | 68.77 | 17.126 | 0.000 | 0 |
| hierarchical_s43 | 4/4 | 0/4 | 81.25 | 63.03 | 71.335 | 15.382 | 1 |
| hierarchical_s44 | 4/4 | 0/4 | 87.50 | 61.67 | 77.230 | 15.411 | 2 |
| flat_s42 | 4/4 | 0/4 | 93.75 | 74.11 | 19.925 | 0.000 | 0 |
| flat_s43 | 4/4 | 0/4 | 86.00 | 64.78 | 57.941 | 0.000 | 0 |
| flat_s44 | 4/4 | 0/4 | 93.75 | 76.05 | 18.740 | 0.000 | 0 |

## Gates

All failures are retained; incomplete attempts block formal comparisons.

## hierarchical minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.016943, paired scene bootstrap 95% [-0.062500, 0.011670].
- removal_auc_180: -0.042777, paired scene bootstrap 95% [-0.091159, 0.005606].
- wall_contact_s: 38.104298, paired scene bootstrap 95% [-2.948869, 89.908377].
- spacing_violation_pair_s: 10.264524, paired scene bootstrap 95% [0.000000, 30.764602].

## hierarchical minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.022778, paired scene bootstrap 95% [-0.062500, 0.000000].
- removal_auc_180: -0.060540, paired scene bootstrap 95% [-0.123023, 0.001943].
- wall_contact_s: -7.025975, paired scene bootstrap 95% [-41.076441, 17.298648].
- spacing_violation_pair_s: -5.117777, paired scene bootstrap 95% [-15.382301, 0.028970].

## hierarchical minus priority_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.147778, paired scene bootstrap 95% [-0.380835, 0.000000].
- removal_auc_180: -0.184563, paired scene bootstrap 95% [-0.273884, -0.114806].
- wall_contact_s: 44.567921, paired scene bootstrap 95% [10.250441, 98.214706].
- spacing_violation_pair_s: 10.264524, paired scene bootstrap 95% [0.000000, 30.764602].

## flat minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.042510, paired scene bootstrap 95% [-0.125000, 0.252530].
- removal_auc_180: 0.028780, paired scene bootstrap 95% [-0.025727, 0.118202].
- wall_contact_s: 15.076491, paired scene bootstrap 95% [-4.649980, 36.783349].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## flat minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.036675, paired scene bootstrap 95% [-0.125000, 0.235025].
- removal_auc_180: 0.011017, paired scene bootstrap 95% [-0.100910, 0.114528].
- wall_contact_s: -30.053782, paired scene bootstrap 95% [-98.901613, 13.046745].
- spacing_violation_pair_s: -15.382301, paired scene bootstrap 95% [-46.146903, 0.000000].

## flat minus priority_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.088325, paired scene bootstrap 95% [-0.176650, 0.000000].
- removal_auc_180: -0.113006, paired scene bootstrap 95% [-0.146038, -0.079973].
- wall_contact_s: 21.540114, paired scene bootstrap 95% [4.297707, 44.274166].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## Matched observer and prior

All arms use command-aligned measured tracking. Untrained policies must match measured allocation at the same decision interval. Observer/prior upgrades alone are not learning gains.
Exact untrained allocation equivalence: {"untrained_balanced_n3": true, "untrained_flat_n3": true}
Classical allocation at 0.1 s, 1 s and 5 s remains in the strong comparator set.

### hierarchical minus balanced_fast_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.147778, scene bootstrap 95% [-0.380835, 0.000000].
- wall_contact_s: 40.749348, scene bootstrap 95% [10.250441, 86.758988].
- spacing_violation_pair_s: 10.264524, scene bootstrap 95% [0.000000, 30.764602].

### hierarchical minus balanced_flat_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.085278, scene bootstrap 95% [-0.380835, 0.166667].
- wall_contact_s: 35.150792, scene bootstrap 95% [-2.324912, 86.682351].
- spacing_violation_pair_s: 10.264524, scene bootstrap 95% [0.000000, 30.764602].

### flat minus balanced_fast_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.088325, scene bootstrap 95% [-0.176650, 0.000000].
- wall_contact_s: 17.721542, scene bootstrap 95% [4.297707, 32.818449].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].

### flat minus balanced_flat_n3

- cluster_safe_success: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
- removal: -0.025825, scene bootstrap 95% [-0.139975, 0.062500].
- wall_contact_s: 12.122985, scene bootstrap 95% [-2.588889, 30.225602].
- spacing_violation_pair_s: 0.000000, scene bootstrap 95% [0.000000, 0.000000].
