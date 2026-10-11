# EXP0100 short report

The implementation smoke test completed 12 PPO updates. The actor changed
the parent proposal on 55--88% of sampled live decisions across updates,
showing that the policy is no longer an alias for the parent argmin. The
training log and raw traces are in
`research/runs/EXP0100_PPO_CANDIDATE_DYNAMIC_20261011/seed0/`.

The partial paired dynamic-obstacle smoke evaluation has three completed
episodes per method:

| method | Strict | task success | collision-free | removal | particle contact (s) | wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| fixed Settle | 66.7% | 66.7% | 66.7% | 0.667 | 0 | 0 |
| Adaptive Settle (stateful) | 100% | 100% | 100% | 1.000 | 0 | 0 |
| Switch Settle | 100% | 100% | 100% | 1.000 | 0 | 0 |
| PPO candidate | 100% | 100% | 100% | 1.000 | 0 | 0 |

These counts are too small to support a performance claim, and no dynamic
particle collision occurred in this subset. A three-seed, matched-budget
validation is still required. Existing definitions of Strict and task success
were not changed; dynamic contact and collision-free status are additional
diagnostics.
