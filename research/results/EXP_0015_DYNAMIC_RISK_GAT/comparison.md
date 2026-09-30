# EXP_0015 Dynamic Risk-Aware Validation

Protocol: three training seeds, each trained for 1,000,000 real transitions;
140 fresh episodes per seed on the anatomical pool with 24 dynamic particles.
The sealed test split was not accessed. Path length is the sum of all robot
step displacements. `path / mass` is reported only over episodes with positive
removed mass and uses the episode-total removed mass.

| Seed | Success | Mass removal | Path length | Path / mass | Wall hits | Steps |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 34.3% | 79.5% | 6.8104 | 3.0828 | 903.8 | 231 |
| 43 | 52.1% | 87.8% | 7.6240 | 3.5261 | 569.0 | 197 |
| 44 | 73.6% | 94.5% | 8.4144 | 3.0571 | 418.3 | 169 |
| Mean +/- sample SD | **53.3% +/- 19.7pp** | 87.3% +/- 7.5pp | 7.6163 +/- 0.8020 | 3.2220 +/- 0.2637 | 630.4 +/- 248.5 | 199 +/- 31 |

## Gate

The pre-registered 85% three-seed mean-success gate **failed** (53.3%). Seed
44 reaches 73.6% on its own, but it is not valid to call the method optimal or
to select that seed after seeing the results. The result is retained as an
honest dynamic-obstacle experiment, not upgraded by changing particle count,
trajectory access, evaluation seeds, or target allocation after the fact.

The intervention is therefore an algorithmic research candidate, not a solved
85% navigation system. Further gains require a new registered experiment and
should include a dynamic-obstacle baseline and ablations for the observation
and scheduler separately.
