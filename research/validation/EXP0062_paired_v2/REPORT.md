# EXP0062 paired development screening

14 anatomies × N=1/2/3 × one paired scene per suite; training seed 6201 only. 300 s horizon and existing success definitions unchanged.

| Suite | Arm | n | All-clear % | Strict % | Removal % | AUC | T90 s (observed n) | Wall s | Exit episodes % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| in025 | A | 42 | 97.6 | 97.6 | 99.4 | 0.811 | 92.351 (41) | 0.000 | 0.0 |
| in025 | A_legacy | 42 | 97.6 | 97.6 | 99.4 | 0.811 | 92.393 (41) | 0.000 | 0.0 |
| in025 | B | 42 | 90.5 | 90.5 | 94.5 | 0.758 | 98.076 (38) | 0.000 | 0.0 |
| in025 | C | 42 | 97.6 | 97.6 | 99.4 | 0.793 | 101.349 (41) | 0.000 | 0.0 |
| in025 | D | 42 | 90.5 | 90.5 | 93.0 | 0.748 | 97.216 (38) | 0.000 | 0.0 |
| in025 | no_settle | 42 | 97.6 | 92.9 | 99.4 | 0.786 | 104.763 (41) | 0.066 | 0.0 |
| in025 | settle | 42 | 97.6 | 97.6 | 99.4 | 0.814 | 89.715 (41) | 0.000 | 0.0 |
| in050 | A | 42 | 78.6 | 73.8 | 91.8 | 0.749 | 90.812 (33) | 0.000 | 4.8 |
| in050 | A_legacy | 42 | 76.2 | 71.4 | 90.6 | 0.732 | 97.000 (32) | 0.000 | 4.8 |
| in050 | B | 42 | 69.0 | 64.3 | 87.1 | 0.701 | 93.193 (29) | 0.000 | 4.8 |
| in050 | C | 42 | 52.4 | 50.0 | 80.5 | 0.643 | 107.074 (23) | 0.000 | 4.8 |
| in050 | D | 42 | 59.5 | 57.1 | 79.3 | 0.638 | 96.780 (25) | 0.000 | 4.8 |
| in050 | no_settle | 42 | 95.2 | 85.7 | 98.8 | 0.780 | 107.247 (40) | 4.465 | 4.8 |
| in050 | settle | 42 | 57.1 | 54.8 | 81.2 | 0.668 | 101.487 (24) | 0.000 | 4.8 |
| stress100 | A | 42 | 21.4 | 21.4 | 49.8 | 0.413 | 84.011 (9) | 0.000 | 4.8 |
| stress100 | A_legacy | 42 | 21.4 | 21.4 | 48.6 | 0.404 | 86.600 (9) | 0.000 | 4.8 |
| stress100 | B | 42 | 19.0 | 19.0 | 52.6 | 0.409 | 101.637 (8) | 0.000 | 4.8 |
| stress100 | C | 42 | 21.4 | 21.4 | 46.0 | 0.368 | 105.211 (9) | 0.000 | 4.8 |
| stress100 | D | 42 | 19.0 | 19.0 | 43.8 | 0.357 | 100.962 (8) | 0.000 | 4.8 |
| stress100 | no_settle | 42 | 31.0 | 31.0 | 69.3 | 0.563 | 92.777 (13) | 6.303 | 7.1 |
| stress100 | settle | 42 | 21.4 | 21.4 | 43.9 | 0.369 | 90.400 (9) | 0.000 | 4.8 |
| unseen0375 | A | 42 | 92.9 | 92.9 | 97.3 | 0.791 | 95.482 (39) | 0.000 | 0.0 |
| unseen0375 | A_legacy | 42 | 83.3 | 83.3 | 94.4 | 0.772 | 91.429 (35) | 0.000 | 0.0 |
| unseen0375 | B | 42 | 90.5 | 85.7 | 94.7 | 0.746 | 107.629 (38) | 0.000 | 4.8 |
| unseen0375 | C | 42 | 81.0 | 81.0 | 93.1 | 0.750 | 97.709 (34) | 0.000 | 0.0 |
| unseen0375 | D | 42 | 73.8 | 71.4 | 86.7 | 0.706 | 91.719 (31) | 0.000 | 2.4 |
| unseen0375 | no_settle | 42 | 97.6 | 88.1 | 99.4 | 0.787 | 104.654 (41) | 0.071 | 4.8 |
| unseen0375 | settle | 42 | 78.6 | 78.6 | 92.1 | 0.764 | 84.473 (33) | 0.000 | 0.0 |
| unseen_combination | A | 42 | 85.7 | 83.3 | 97.5 | 0.769 | 109.444 (39) | 0.000 | 4.8 |
| unseen_combination | A_legacy | 42 | 90.5 | 88.1 | 99.2 | 0.794 | 100.871 (41) | 0.000 | 2.4 |
| unseen_combination | B | 42 | 21.4 | 21.4 | 69.8 | 0.533 | 167.838 (13) | 0.000 | 4.8 |
| unseen_combination | C | 42 | 85.7 | 81.0 | 93.6 | 0.730 | 109.254 (37) | 0.000 | 4.8 |
| unseen_combination | D | 42 | 76.2 | 73.8 | 86.9 | 0.673 | 113.128 (32) | 0.000 | 4.8 |
| unseen_combination | no_settle | 42 | 0.0 | 0.0 | 8.1 | 0.074 | NA (0) | 19.156 | 7.1 |
| unseen_combination | settle | 42 | 97.6 | 97.6 | 99.4 | 0.801 | 97.137 (41) | 0.000 | 0.0 |

T90 is averaged only over episodes reaching 90%; missing values remain censored and counts are shown. See summary.json for paired deltas and trajectory diagnostics.

## Scope and limitations

- A/B already have a 32-step Transformer; C/D add an 8-step (0.8 s) GRU context. C−B measures added GRU context, not history versus no history.
- Flow values are nominal healthy inlet calibration parameters. The pre-existing per-scene multiplier is retained (actual mean = nominal × multiplier/2); nominal intermediate flow does not guarantee disjoint actual flow support.
- Training used 120 updates × 4 workers × 256 environment steps = 122880 environment steps per arm. Retained active-agent samples differ (~259–262k). The earlier 90k estimate was incorrect.
- The existing GAE treats time limits as terminal and omits held-agent steps. This inherited training limitation is shared by all arms and prevents claiming fully corrected truncation handling.
- D predicts next measured normalized velocity, not velocity displacement; the actual auxiliary weight is 0.01. Earlier EXP0062 module/config descriptions (delta velocity, 0.05) do not describe these trained checkpoints.
- Old residual scale .5 allows up to .5 speed at zero prior; it does not force zero commands. Scale 1 restores scalar speed range, but route aiming still restricts direction. At exactly zero aim displacement and zero lateral action, the command remains zero.
- A−A_legacy is a same-checkpoint inference intervention, not an equal-budget old/new mapping retraining comparison.
- Neighborhood diagnostics use estimated position and a 0.3 mm preoperative-target neighborhood; observed motion includes command response and sensing noise, so it is not an isolated estimate of blood flow.
- Single training seed and one scene per anatomy/N/suite: exploratory screening, no robust performance claim or three-seed confirmation.
- The abandoned EXP0062_*_eval.jsonl files used uncalibrated flow and mismatched history actions. They are invalid, retained for audit, and excluded here.

## Reproduction

```bash
env PYTHONPATH=. OMP_NUM_THREADS=1 /home/wj/miniconda3/envs/v/bin/python -m scripts.evaluate_lysis_abcd --out research/validation/EXP0062_paired_reproduction --workers 20
env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.report_lysis_abcd research/validation/EXP0062_paired_v2
```
