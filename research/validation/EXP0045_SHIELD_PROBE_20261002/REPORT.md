# EXP43 + post-policy safety shield: sealed test

The three EXP43 checkpoints selected on the validation split (seed 42 @1M, 43 @300K, 44 @800K) are executed unchanged, with one addition after the policy: a robot's whole command (prior + RL residual) becomes an exact stop when an observed predicted-clearance slot forecasts clearance < 0.3 within 0.5 s (same thresholds as the wait rule, chosen on the diagnostic split). No retraining, no reselection. One sealed evaluation per checkpoint (study EXP_0045_SHIELD_EVAL, 3 declared).

| method | complete (42 / 43 / 44) | collision-free (42 / 43 / 44) |
|---|---|---|
| route+avoid prior (traditional) | 94.4 | 84.8 |
| route+avoid+wait rule (strongest traditional) | 92.6 | 91.4 |
| EXP43 deep RL residual with stop | 98.4 / 98.6 / 99.6 | 88.4 / 88.2 / 90.0 |
| **EXP43 + shield** | **98.4 / 97.8 / 98.8 (98.3)** | **96.4 / 96.4 / 95.0 (95.9)** |

Paired on identical sealed layouts vs the strongest traditional controller (wait rule): complete +5.8 / +5.2 / +6.2 pp, collision-free +5.0 / +5.0 / +3.6 pp, all seeds positive on both metrics. Vs the same-seed EXP43 policy without the shield: complete 0.0 / −0.8 / −0.8, collision-free +8.0 / +8.2 / +5.0.

The shield alone (on the traditional prior) is the wait rule, 92.6 / 91.4; the RL policy alone is 98.9 / 88.9; combined they reach 98.3 / 95.9. The learned policy supplies the clearance rate and the shield supplies the last collisions.

Caveats: structured, not pure RL; synthetic in-vitro simulation; thresholds from the diagnostic split; MCA only.
