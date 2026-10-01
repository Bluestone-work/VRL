# EXP_0040 sealed-test result

Selection on the registered validation split (200 layouts); one sealed-test evaluation (500 layouts) per selected checkpoint.

| arm | seed 42 / 43 / 44 (%) | mean |
|---|---|---:|
| own_bearing | 59.2 / 75.2 / 70.2 | 68.2% |
| baseline EXP35 500K | 47.0 / 78.2 / 60.6 | 61.9% |
| EXP39_assigned (reference) | 49.4 / 78.2 / 70.6 | 66.1% |

- own_bearing selected steps: seed 42: 300000, seed 43: 300000, seed 44: 300000
- paired own_bearing - EXP39_assigned on identical test layouts: +9.8 / -3.0 / -0.4 pp, mean +2.1 pp
