# EXP0051 preflight

Audit: PASS; 12/12 attempts complete; 2 paired scenes.

Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.

| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| memory_n3 | 2/2 | 0/2 | 50.00 | 36.11 | 41.054 | 0.077 | 2 |
| balanced_n3 | 2/2 | 0/2 | 50.00 | 37.08 | 25.244 | 0.007 | 1 |
| priority_n3 | 2/2 | 0/2 | 44.29 | 35.13 | 42.147 | 0.069 | 1 |
| memory_n1 | 2/2 | 0/2 | 12.50 | 1.44 | 11.897 | 0.000 | 0 |
| legacy_memory_n3 | 2/2 | 0/2 | 57.45 | 40.55 | 27.826 | 0.131 | 1 |
| untrained_n3 | 2/2 | 0/2 | 50.00 | 36.11 | 41.054 | 0.077 | 2 |

## Gates

Untrained equals memory: True
All failures are retained; incomplete attempts block formal comparisons.
The legacy-memory arm has a different safety layer, so any difference alone is not a learning gain.
