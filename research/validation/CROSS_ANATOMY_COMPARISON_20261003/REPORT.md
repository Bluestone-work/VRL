# Cross-anatomy sealed comparison: pure RL vs traditional vs residual RL

All 14 anatomies, 500 registered sealed-test layouts each, one evaluation per (method, seed, anatomy) in study CROSS_ANATOMY_20261003. Every method was trained, tuned and selected on MCA only; its checkpoint was fixed before this run, so the other 13 anatomies are zero-shot for all methods. Learned methods: 3 seeds each (the MCA-validation-selected checkpoint per seed); traditional controllers are deterministic (1 run). Values are complete-clearance / collision-free-clearance percent; learned methods show the seed mean.

## Overall (mean over 14 anatomies)

| method | type | complete | collision-free | worst anatomy (complete) | seed range of 14-anatomy mean (complete) |
|---|---|---:|---:|---|---|
| Pure RL (EXP40) | pure RL | 27.0 | 15.7 | pulmonary_saddle 0.0 | 17.1–33.2 |
| Route only | traditional | 97.8 | 76.4 | cerebral_venous_sinus 87.6 | — (deterministic) |
| Route + avoid | traditional | 98.3 | 97.4 | cerebral_venous_sinus 88.0 | — (deterministic) |
| Route + avoid + wait (strongest traditional) | traditional | 98.1 | 98.0 | cerebral_venous_sinus 88.2 | — (deterministic) |
| Residual RL (EXP43) | residual RL | 98.0 | 97.1 | sma_embolism 88.9 | 97.3–98.5 |
| Residual RL + shield | residual RL | 98.0 | 97.7 | sma_embolism 89.1 | 97.3–98.5 |

## Per anatomy (complete / collision-free, %)

| anatomy | Pure RL (EXP40) | Route only | Route + avoid | Route + avoid + wait (strongest traditional) | Residual RL (EXP43) | Residual RL + shield |
|---|---:|---:|---:|---:|---:|---:|
| mca_m1_lvo | 68.2 / 22.2 | 88.4 / 22.8 | 94.4 / 84.8 | 92.6 / 91.4 | 98.9 / 88.9 | 98.3 / 95.9 |
| pulmonary_saddle | 0.0 / 0.0 | 100.0 / 99.2 | 100.0 / 100.0 | 100.0 / 100.0 | 95.1 / 95.1 | 95.2 / 95.2 |
| coronary_lm_bifurcation | 48.0 / 30.9 | 100.0 / 71.2 | 100.0 / 99.8 | 100.0 / 100.0 | 99.7 / 99.5 | 99.7 / 99.5 |
| coronary_rca | 62.3 / 33.8 | 100.0 / 69.4 | 100.0 / 99.8 | 100.0 / 99.8 | 100.0 / 99.7 | 100.0 / 99.7 |
| iliac_may_thurner | 0.0 / 0.0 | 100.0 / 98.8 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 99.9 | 100.0 / 99.9 |
| popliteal_calf_dvt | 1.2 / 0.8 | 100.0 / 92.0 | 100.0 / 99.8 | 100.0 / 100.0 | 96.3 / 96.3 | 96.2 / 96.2 |
| ica_siphon | 27.7 / 19.0 | 100.0 / 67.4 | 100.0 / 99.4 | 100.0 / 100.0 | 100.0 / 99.3 | 100.0 / 99.9 |
| ica_terminus_t | 74.9 / 47.6 | 100.0 / 64.8 | 100.0 / 99.8 | 100.0 / 99.8 | 100.0 / 99.7 | 100.0 / 99.8 |
| carotid_bifurcation | 4.3 / 2.5 | 100.0 / 78.6 | 100.0 / 100.0 | 100.0 / 100.0 | 98.9 / 98.7 | 99.0 / 98.9 |
| basilar_vertebral | 45.8 / 29.3 | 100.0 / 65.2 | 100.0 / 99.0 | 100.0 / 100.0 | 99.6 / 98.7 | 99.7 / 99.6 |
| cerebral_venous_sinus | 0.0 / 0.0 | 87.6 / 86.0 | 88.0 / 88.0 | 88.2 / 88.2 | 99.9 / 99.9 | 99.9 / 99.9 |
| sma_embolism | 6.4 / 5.1 | 92.8 / 83.4 | 93.2 / 93.2 | 93.0 / 93.0 | 88.9 / 88.9 | 89.1 / 89.1 |
| femoropopliteal_pad | 1.9 / 1.3 | 100.0 / 92.8 | 100.0 / 100.0 | 100.0 / 100.0 | 94.4 / 94.4 | 94.4 / 94.4 |
| renal_artery | 36.9 / 26.7 | 100.0 / 77.4 | 100.0 / 99.8 | 100.0 / 99.8 | 100.0 / 99.9 | 100.0 / 99.9 |

## Paired differences on identical layouts (14-anatomy mean, pp; per learned seed)

| comparison | complete | collision-free | anatomies better / worse (complete, >0.5 pp) |
|---|---|---|---|
| Residual RL + shield − Route + avoid + wait (strongest traditional) | +0.4 / -0.9 / +0.0 | +0.3 / -0.9 / -0.3 | 2 / 5 |
| Residual RL (EXP43) − Route + avoid | +0.2 / -1.0 / -0.1 | +0.2 / -1.0 / -0.2 | 2 / 5 |
| Residual RL + shield − Pure RL (EXP40) | +67.9 / +80.2 / +64.9 | +79.8 / +88.2 / +78.2 | 14 / 0 |
| Route + avoid + wait (strongest traditional) − Pure RL (EXP40) | +67.5 | +79.4 | 14 / 0 |
| Residual RL + shield − Residual RL (EXP43) | +0.0 / -0.0 / -0.1 | +0.7 / +0.7 / +0.5 | 0 / 1 |

Caveats: synthetic in-vitro simulation, not calibrated physiology or hardware; residual RL is structured (rule prior + learned residual + rule shield), not pure RL; all tuning on MCA only.
