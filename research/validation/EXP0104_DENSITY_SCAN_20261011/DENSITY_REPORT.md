# EXP0104 dynamic obstacle density comparison

| density | method | removal | AUC | wall (s) | particle contact (s) | collision events | lost |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | parent | 0.793 | 0.645 | 1.67 | 0.000 | 0.000 | 0.083 |
| 0 | sac_finite | 0.790 | 0.642 | 1.70 | 0.000 | 0.000 | 0.083 |
| 0 | td3 | 0.769 | 0.621 | 3.83 | 0.000 | 0.000 | 0.083 |
| 8 | parent | 0.793 | 0.652 | 1.60 | 0.216 | 0.167 | 0.083 |
| 8 | sac_finite | 0.793 | 0.651 | 1.64 | 0.196 | 0.167 | 0.083 |
| 8 | td3 | 0.720 | 0.590 | 3.38 | 0.191 | 0.750 | 0.083 |
| 16 | parent | 0.791 | 0.649 | 1.63 | 0.296 | 0.417 | 0.083 |
| 16 | sac_finite | 0.791 | 0.650 | 1.58 | 0.286 | 0.375 | 0.083 |
| 16 | td3 | 0.719 | 0.591 | 3.27 | 0.229 | 1.125 | 0.083 |
| 32 | parent | 0.790 | 0.641 | 1.68 | 0.606 | 0.667 | 0.083 |
| 32 | sac_finite | 0.790 | 0.645 | 1.76 | 0.560 | 0.667 | 0.083 |
| 32 | td3 | 0.736 | 0.602 | 5.25 | 0.488 | 2.042 | 0.083 |

Strict/full-clear is retained with the original definition. The deterministic 8-particle replay showed that tiny residual-mass numerical differences can change the termination flag, so continuous metrics are the primary cross-density comparison until replay determinism is fixed.

Result: SAC remains near Parent across 0/8/16/32 particles, with slightly lower particle contact at 16/32 but no removal/AUC gain. TD3 has lower removal and higher wall/collision exposure, worsening with density.
