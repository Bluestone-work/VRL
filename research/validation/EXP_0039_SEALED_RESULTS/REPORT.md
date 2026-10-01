# EXP_0039 sealed-test result

Selection on the registered validation split (200 layouts); one sealed-test evaluation (500 layouts) per selected checkpoint.

| arm | seed 42 / 43 / 44 (%) | mean |
|---|---|---:|
| assigned | 49.4 / 78.2 / 70.6 | 66.1% |
| control | 55.8 / 78.2 / 60.6 | 64.9% |
| baseline EXP35 500K | 47.0 / 78.2 / 60.6 | 61.9% |

- assigned selected steps: seed 42: 200000, seed 43: 0, seed 44: 200000
- control selected steps: seed 42: 100000, seed 43: 0, seed 44: 0
- paired assigned - control on identical test layouts: -6.4 / +0.0 / +10.0 pp, mean +1.2 pp

## Interpretation (2026-10-01 21:50)

- **The sealed test shows no reliable gain.** assigned 66.1% vs control 64.9%, paired +1.2 pp. By seed: -6.4 (SE 2.4) / 0 / +10.0 (SE 1.8). The two non-zero seeds go in opposite directions, and for seed 43 both arms selected the parent, so the difference is 0. Against the EXP35 baseline (61.9%) the gain is +4.2 pp, but part of that comes from model selection itself: the control arm also reaches 64.9%. The 80% target is not reached.
- **Training stability is clearly better.** Mean over the five milestones on the validation split: assigned 61.7%, control 43.4%. At the final 500K: assigned 59.5 / 73.0 / 67.0 (66.5%), control 36.5 / 45.0 / 38.5 (40.0%). The control arm still degrades as it continues training. Its sealed-test result relies on selecting the parent (2 seeds) or 100K, so selection is effectively a rollback. The own-target potential removes the continued-training degradation but does not raise the ceiling.
- **Selection noise:** for seed 42, assigned selected 200K (validation 61.0%) and scored 49.4% on test, while control selected 100K (validation 58.5%) and scored 55.8% on test. With 200 layouts the validation SE is about 3.5 pp, which cannot separate candidates closer than a few points.
- **Conclusion:** the potential fix is a necessary correction (it lets training keep improving instead of degrading), but it does not by itself solve the navigation bottleneck: route-following cosine stays at 0.25. Next we need a change that acts directly on route following.
