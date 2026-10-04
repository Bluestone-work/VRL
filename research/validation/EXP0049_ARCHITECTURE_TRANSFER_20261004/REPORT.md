# EXP0049 explicit architecture transfer comparison

Audit: PASS.
Training budgets: feedforward [65536]; temporal [65536].

Different model/checkpoint hashes are retained. Every common tracked sensing, physics and conventional-control source must match. This is development evidence only.

| Metric | Temporal minus feedforward | Scene-bootstrap 95% CI |
|---|---:|---:|
| cluster_safe_success | +0.000000 | [+0.000000, +0.000000] |
| task_success | +0.000000 | [+0.000000, +0.000000] |
| removal | -0.013119 | [-0.124079, +0.086199] |
| removal_auc_180 | -0.010473 | [-0.111679, +0.079707] |
| wall_contact_s | +31.873233 | [+11.316563, +54.325684] |
| spacing_violation_pair_s | -9.840624 | [-16.513344, -3.869894] |

Comparisons are not evidence of hardware transfer or publication-level novelty. Secondary intervals are exploratory and not multiplicity-adjusted. An all-zero paired bootstrap interval does not establish equivalence.
