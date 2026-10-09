# EXP0076 fair observable-history comparison

This is a paired single-seed screening on the nine training anatomies, N=1/2/3,
27 episodes per condition. Both policies were freshly initialized and trained
for 120 updates with no privileged auxiliary labels, no ECG, and latency reset
sampling 1--2 steps. The reference uses the original 39-token history; the
candidate adds the final sent world-frame command and camera frame age. Both
use the same fixed action map, speed prior, optimizer budget and scene registry.
The candidate is `observable_history`; it is not the earlier privileged flow_aux.

| Condition | Reference Strict | Candidate Strict | Reference removal | Candidate removal | Reference AUC | Candidate AUC | Reference wall s | Candidate wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Low delay (0.05/1/0) | 48.1% | **66.7%** | .754 | **.843** | .598 | **.674** | 1.99 | 3.69 |
| Moderate delay (0.05/2/0) | 55.6% | **63.0%** | .759 | **.843** | .615 | **.697** | .42 | .08 |
| Strong flow (0.1/2/0.625) | 11.1% | **18.5%** | .404 | **.471** | .320 | **.372** | 6.76 | 6.99 |
| Composite dynamics (0.05/2/1.25) | **29.6%** | 25.9% | .633 | .601 | .449 | .433 | 7.41 | 7.40 |

The first three conditions show a positive paired screening signal: Strict changes
are +18.5, +7.4 and +7.4 percentage points. Composite dynamics remains negative
(-3.7 points), so the proposed input repair is not yet a solution to the full
randomized dynamics problem. Wall exposure rises in the low-delay condition and
is similar in the strong-flow condition; safety must remain a co-primary metric.

This comparison does not use the earlier flow_aux checkpoint and does not use
true flow, response, position or environment labels. It also does not replace
the idealized activity/clot-visible status channels already shared by the
benchmark sensor. One training seed and 27 paired scenes per cell are screening
evidence only; no multi-seed stability claim is made.

Artifacts:

- Reference run: `research/runs/EXP0076_FAIR_REFERENCE_s7601`
- Candidate run: `research/runs/EXP0076_OBSERVABLE_HISTORY_s7601`
- Raw paired episodes: `episodes.jsonl` in this directory and the reference directory
- Smoke and interface protocol: `research/validation/EXP0076_OBSERVABLE_PROTOCOL.md`

The primary scope now treats one and two-step delay as the learning target. Three-
step delay remains a stress test and is not used to select this candidate.
