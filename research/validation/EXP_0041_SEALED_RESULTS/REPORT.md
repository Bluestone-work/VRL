# EXP_0041 sealed-test result

Selection on the registered validation split (200 layouts); one sealed-test evaluation (500 layouts) per selected checkpoint.

| arm | seed 42 / 43 / 44 (%) | mean |
|---|---|---:|
| route_prior | 88.0 / 89.2 / 94.2 | 90.5% |
| baseline EXP35 500K | 47.0 / 78.2 / 60.6 | 61.9% |
| EXP40_own_bearing (reference) | 59.2 / 75.2 / 70.2 | 68.2% |

- route_prior selected steps: seed 42: 500000, seed 43: 300000, seed 44: 200000
- paired route_prior - EXP40_own_bearing on identical test layouts: +28.8 / +14.0 / +24.0 pp, mean +22.3 pp
