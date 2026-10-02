# EXP_0045 sealed-test result

Selection on the registered validation split (200 layouts); one sealed-test evaluation (500 layouts) per selected checkpoint.

| arm | seed 42 / 43 / 44 (%) | mean |
|---|---|---:|
| shield | 98.0 / 97.0 / 98.8 | 97.9% |
| baseline EXP35 500K | 47.0 / 78.2 / 60.6 | 61.9% |

- shield selected steps: seed 42: 100000, seed 43: 300000, seed 44: 0

## Interpretation (2026-10-02)

Collision-free per seed: 96.0 / 95.8 / 95.0%. Selected steps 100K / 300K / 0 (seed 44 kept the untrained parent). Paired on identical sealed layouts vs EXP43 + shield without further training: complete −0.4 / −0.8 / 0.0 pp, collision-free −0.4 / −0.6 / 0.0 pp. Vs the strongest traditional controller (wait rule): complete +5.4 / +4.4 / +6.2, collision-free +4.6 / +4.4 / +3.6.

500K more steps with the shield in the loop and collision penalty 20 add nothing over EXP43 + shield. The reported method stays EXP43 + shield (98.3% / 95.9%).
