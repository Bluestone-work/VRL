# EXP_0014 Adaptive Edge-GAT MAPPO

The experiment compares the new `adaptive_edge_gat` actor with the matched
Direct Local GAT-MAPPO reference from EXP_0005. Both arms use the same
geometric36 observations, geodesic contact, reward, simulator, PPO settings,
three training seeds, and 1M real transitions per seed. The sealed test split
was not accessed.

| Metric | Direct Local GAT reference | Adaptive Edge-GAT | Difference |
|---|---:|---:|---:|
| Success | 71.43% +/- 9.12pp | 74.76% +/- 7.84pp | +3.33pp |
| Mass removal | 87.86% +/- 3.89pp | 91.17% +/- 4.39pp | +3.32pp |
| Episode steps | 155.41 +/- 21.09 | 140.68 +/- 22.90 | -14.73 |
| Wall contact rate | 61.27% +/- 3.41pp | 49.48% +/- 6.99pp | -11.79pp |
| Pair collision rate | 12.40% +/- 3.39pp | 8.56% +/- 3.83pp | -3.84pp |

The adaptive actor replaces plain robot-to-robot attention with an
edge-conditioned per-head logit bias and a sigmoid value-message gate. The
result is consistent with improved communication selectivity and safer local
control, but the mechanism is not causally isolated beyond the matched actor
comparison and no literature novelty claim is made here.
