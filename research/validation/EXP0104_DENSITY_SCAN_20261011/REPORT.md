# EXP0104 dynamic obstacle comparison (8 particles)

All methods use the same screen scene pool (24 scenes). Raw rows are retained. The second paired run exposed small numerical replay sensitivity: the same scenario hash can differ by about 0.01 mm in path and cross the all-clots-cleared threshold. Therefore Strict/task success are reported, but removal, T90, wall, spacing and particle diagnostics are the primary mechanism comparison until deterministic replay is fixed.

| method | N | Strict | full clear | removal | AUC | T90 (s) | wall (s) | spacing (s) | lost | particle contact (s) | collision events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_settle | 24 | 0.000 | 0.000 | 0.446 | 0.378 | 89.7 | 12.65 | 0.00 | 0.042 | 0.314 | 0.875 |
| fixed_settle | 24 | 0.000 | 0.000 | 0.615 | 0.503 | 112.6 | 12.00 | 0.00 | 0.083 | 0.113 | 0.042 |
| parent | 24 | 0.000 | 0.000 | 0.790 | 0.651 | 96.4 | 1.60 | 0.00 | 0.083 | 0.216 | 0.167 |
| sac_finite=research/runs/EXP0102_SAC_OBSFEATURE_DYNAMIC_20261011/seed4/policy.pt | 24 | 0.000 | 0.000 | 0.791 | 0.650 | 98.7 | 1.64 | 0.00 | 0.083 | 0.196 | 0.167 |
| switch_settle | 24 | 0.000 | 0.000 | 0.697 | 0.565 | 105.8 | 12.00 | 0.00 | 0.042 | 0.113 | 0.042 |
| td3=research/runs/EXP0103_TD3_OBSFEATURE_DYNAMIC_20261011/seed4/policy.pt | 24 | 0.000 | 0.000 | 0.719 | 0.589 | 97.3 | 3.38 | 0.00 | 0.083 | 0.191 | 0.750 |

The current seed4 checkpoints do not show a reliable learning gain over the Parent selector. SAC is close to Parent on removal and collision diagnostics; TD3 has lower removal and more wall/collision exposure. Adaptive/Fixed/Switch results in this batch should not be treated as definitive success comparisons because the all-clots-cleared threshold is replay-sensitive; rerun under deterministic single-worker settings before selecting a classical baseline.
