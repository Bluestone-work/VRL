# Delay-aware position extrapolation candidate

This candidate was selected after the fixed 0.05-flow delay trace diagnosis.
It extrapolates the deployable estimated position by `latency_steps ×
control_dt × measured_velocity` before route target allocation and command
generation. It uses no simulator position, true flow, response gain, anatomy
identifier or environment identifier. Physics still advances from the actual
environment state. No network, reward, action range or response
randomization was changed.

Development-only screen: 9 training anatomies × N=1/2/3 × delay 1/2/3, one
scene per cell, nominal inlet 0.05 mm/s, 243 episodes total. No errors.

| Method | delay | Strict success | Mean removal | Mean target departures |
|---|---:|---:|---:|---:|
| predictive extrapolation | 1 | 22/27 | 0.935 | 0.0 |
| predictive extrapolation | 2 | 1/27 | 0.478 | 0.7 |
| predictive extrapolation | 3 | 0/27 | 0.274 | 4.1 |
| fixed Settle | 1 | 10/27 | 0.699 | 0.0 |
| fixed Settle | 2 | 12/27 | 0.769 | 0.7 |
| fixed Settle | 3 | 17/27 | 0.870 | 2.0 |
| no deceleration | 1 | 23/27 | 0.982 | 0.5 |
| no deceleration | 2 | 0/27 | 0.179 | 3.5 |
| no deceleration | 3 | 0/27 | 0.121 | 44.7 |

The candidate is rejected. Position extrapolation helps neither delayed
target persistence nor the near-target dwell problem; at delays 2 and 3 it is
worse than fixed Settle. This is a causal negative result, not evidence that
history is unnecessary in general. It does show that a simple kinematic
extrapolation is insufficient under the current route allocator and image
latency model.

The corrected rerun uses persistent Settle state and corrected entry/departure
bookkeeping. Its fixed Settle strict-success counts are 16/42, 20/42, and
21/42 for delays 1/2/3 in the candidate screen; the candidate remains
negative and these corrected baseline counts supersede the earlier
reconstructed-controller numbers.

Reproduction:

```bash
env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.diagnose_delay_candidate \
  --out research/validation/FLOW_DELAY_AUDIT/candidate_delay_005 --workers 20
```
