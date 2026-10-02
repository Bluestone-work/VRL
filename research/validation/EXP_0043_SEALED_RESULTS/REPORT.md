# EXP_0043 sealed-test result

Selection on the registered validation split (200 layouts); one sealed-test evaluation (500 layouts) per selected checkpoint.

| arm | seed 42 / 43 / 44 (%) | mean |
|---|---|---:|
| gated | 98.4 / 98.6 / 99.6 | 98.9% |
| baseline EXP35 500K | 47.0 / 78.2 / 60.6 | 61.9% |
| EXP42_route_avoid (reference) | 95.4 / 94.8 / 92.8 | 94.3% |

- gated selected steps: seed 42: 1000000, seed 43: 300000, seed 44: 800000
- paired gated - EXP42_route_avoid on identical test layouts: +3.0 / +3.8 / +6.8 pp, mean +4.5 pp

## Interpretation (2026-10-02)

Collision-free per seed: 88.4 / 88.2 / 90.0% (mean 88.9%). Paired on identical sealed layouts:

| vs | complete clearance | collision-free |
|---|---|---|
| route+avoid prior alone (94.4 / 84.8) | +4.0 / +4.2 / +5.2 pp | +3.6 / +3.4 / +5.2 pp |
| hand-written wait rule (92.6 / 91.4) | +5.8 / +6.0 / +7.0 pp | −3.0 / −3.2 / −1.4 pp |

Deep RL with a stop action beats every traditional controller on complete clearance on all three seeds, and beats the prior it starts from on both metrics. It does not beat the hand-written wait rule on collision-free clearance. Neither dominates the other: RL clears more, the wait rule collides less.
