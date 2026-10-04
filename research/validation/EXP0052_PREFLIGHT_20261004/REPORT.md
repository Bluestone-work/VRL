# EXP0052 preflight

Audit: PASS; 16/16 attempts complete; 2 paired scenes.

Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.

| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced_n3 | 2/2 | 0/2 | 100.00 | 87.06 | 11.292 | 0.000 | 0 |
| balanced_fast_n3 | 2/2 | 0/2 | 100.00 | 87.06 | 11.292 | 0.000 | 0 |
| memory_n3 | 2/2 | 0/2 | 100.00 | 87.06 | 11.292 | 0.000 | 0 |
| memory_n1 | 2/2 | 0/2 | 37.50 | 25.04 | 14.244 | 0.000 | 0 |
| balanced_flat_n3 | 2/2 | 0/2 | 100.00 | 87.06 | 11.292 | 0.000 | 0 |
| priority_n3 | 2/2 | 0/2 | 100.00 | 87.06 | 11.292 | 0.000 | 0 |
| untrained_balanced_n3 | 2/2 | 0/2 | 100.00 | 87.06 | 11.292 | 0.000 | 0 |
| untrained_flat_n3 | 2/2 | 0/2 | 100.00 | 87.06 | 11.292 | 0.000 | 0 |

## Gates

All failures are retained; incomplete attempts block formal comparisons.

## Matched observer and prior

All arms use command-aligned measured tracking. Untrained policies must match measured allocation at the same decision interval. Observer/prior upgrades alone are not learning gains.
Exact untrained allocation equivalence: {"untrained_balanced_n3": true, "untrained_flat_n3": true}
Classical allocation at 0.1 s, 1 s and 5 s remains in the strong comparator set.
