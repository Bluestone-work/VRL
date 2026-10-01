# EXP_0042 sealed-test result

Selection on the registered validation split (200 layouts); one sealed-test evaluation (500 layouts) per selected checkpoint.

| arm | seed 42 / 43 / 44 (%) | mean |
|---|---|---:|
| route_avoid | 95.4 / 94.8 / 92.8 | 94.3% |
| baseline EXP35 500K | 47.0 / 78.2 / 60.6 | 61.9% |
| EXP41_route_prior (reference) | 88.0 / 89.2 / 94.2 | 90.5% |

- route_avoid selected steps: seed 42: 500000, seed 43: 500000, seed 44: 400000
- paired route_avoid - EXP41_route_prior on identical test layouts: +7.4 / +5.6 / -1.4 pp, mean +3.9 pp

## Interpretation (2026-10-02)

- Collision-free success per seed: 84.2 / 84.2 / 83.0%.
- Paired against the route+avoid prior alone (94.4% / 84.8%) on identical test layouts: complete +1.0 / +0.4 / −1.6 pp, collision-free −0.6 / −0.6 / −1.8 pp. Layouts won/lost per seed: 18/13, 16/14, 17/25.
- 500K steps of RL residual on top of the prior gives no measurable gain. The hand-written prior is the result; RL fine-tuning is not justified at this budget.
