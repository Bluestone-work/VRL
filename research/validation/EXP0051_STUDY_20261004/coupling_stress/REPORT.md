# EXP0051 synthetic_coupling_stress

Audit: PASS; 40/40 attempts complete; 4 paired scenes.

Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.

| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| memory_n3 | 4/4 | 0/4 | 87.50 | 61.51 | 31.220 | 0.488 | 3 |
| balanced_n3 | 4/4 | 0/4 | 81.25 | 64.63 | 31.297 | 0.277 | 3 |
| priority_n3 | 4/4 | 0/4 | 77.26 | 61.68 | 24.546 | 0.038 | 1 |
| memory_n1 | 4/4 | 0/4 | 43.75 | 32.37 | 1.454 | 0.000 | 0 |
| hierarchical_s42 | 4/4 | 0/4 | 87.50 | 61.51 | 31.220 | 0.488 | 3 |
| hierarchical_s43 | 4/4 | 0/4 | 87.50 | 61.51 | 31.220 | 0.488 | 3 |
| hierarchical_s44 | 4/4 | 0/4 | 87.50 | 61.51 | 31.220 | 0.488 | 3 |
| flat_s42 | 4/4 | 0/4 | 87.50 | 61.51 | 31.220 | 0.488 | 3 |
| flat_s43 | 4/4 | 0/4 | 87.50 | 61.51 | 31.220 | 0.488 | 3 |
| flat_s44 | 4/4 | 0/4 | 87.50 | 61.51 | 31.220 | 0.488 | 3 |

## Gates

Untrained equals memory: None
All failures are retained; incomplete attempts block formal comparisons.
The legacy-memory arm has a different safety layer, so any difference alone is not a learning gain.

## hierarchical minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal_auc_180: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].
- wall_contact_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## hierarchical minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.062500, paired scene bootstrap 95% [0.000000, 0.187500].
- removal_auc_180: -0.031168, paired scene bootstrap 95% [-0.097844, 0.007794].
- wall_contact_s: -0.077096, paired scene bootstrap 95% [-9.310196, 11.251256].
- spacing_violation_pair_s: 0.211311, paired scene bootstrap 95% [-0.051011, 0.563294].

## hierarchical minus priority_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.102441, paired scene bootstrap 95% [0.000000, 0.307322].
- removal_auc_180: -0.001663, paired scene bootstrap 95% [-0.102873, 0.108151].
- wall_contact_s: 6.673625, paired scene bootstrap 95% [-13.571005, 25.937800].
- spacing_violation_pair_s: 0.449506, paired scene bootstrap 95% [-0.069997, 0.969009].

## flat minus memory_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal_auc_180: -0.000000, paired scene bootstrap 95% [-0.000000, 0.000000].
- wall_contact_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- spacing_violation_pair_s: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].

## flat minus balanced_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.062500, paired scene bootstrap 95% [0.000000, 0.187500].
- removal_auc_180: -0.031168, paired scene bootstrap 95% [-0.097844, 0.007794].
- wall_contact_s: -0.077096, paired scene bootstrap 95% [-9.310196, 11.251256].
- spacing_violation_pair_s: 0.211311, paired scene bootstrap 95% [-0.051011, 0.563294].

## flat minus priority_n3

- cluster_safe_success: 0.000000, paired scene bootstrap 95% [0.000000, 0.000000].
- removal: 0.102441, paired scene bootstrap 95% [0.000000, 0.307322].
- removal_auc_180: -0.001663, paired scene bootstrap 95% [-0.102873, 0.108151].
- wall_contact_s: 6.673625, paired scene bootstrap 95% [-13.571005, 25.937800].
- spacing_violation_pair_s: 0.449506, paired scene bootstrap 95% [-0.069997, 0.969009].
