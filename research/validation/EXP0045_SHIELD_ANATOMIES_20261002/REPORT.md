# EXP43 + shield vs the traditional route+avoid prior on all 14 anatomies

Sealed test, 500 registered layouts per anatomy, one evaluation each. Both methods were tuned only on MCA (prior gain on the MCA diagnostic split; EXP43 trained and selected on MCA; shield thresholds from the MCA diagnostic split). The checkpoint (EXP43 seed 42, selected at 1M) was fixed before any non-MCA evaluation. The other 13 anatomies are zero-shot for both. MCA row: seed 42 sealed result.

| anatomy | prior complete | prior collision-free | EXP43+shield complete | EXP43+shield collision-free | paired Δ complete | paired Δ collision-free |
|---|---:|---:|---:|---:|---:|---:|
| mca_m1_lvo | 94.4 | 84.8 | 98.4 | 96.4 | +4.0 | +11.6 |
| pulmonary_saddle | 100.0 | 100.0 | 98.2 | 98.2 | -1.8 | -1.8 |
| coronary_lm_bifurcation | 100.0 | 99.8 | 99.8 | 99.6 | -0.2 | -0.2 |
| coronary_rca | 100.0 | 99.8 | 100.0 | 99.8 | +0.0 | +0.0 |
| iliac_may_thurner | 100.0 | 100.0 | 100.0 | 100.0 | +0.0 | +0.0 |
| popliteal_calf_dvt | 100.0 | 99.8 | 99.4 | 99.4 | -0.6 | -0.4 |
| ica_siphon | 100.0 | 99.4 | 100.0 | 100.0 | +0.0 | +0.6 |
| ica_terminus_t | 100.0 | 99.8 | 100.0 | 100.0 | +0.0 | +0.2 |
| carotid_bifurcation | 100.0 | 100.0 | 99.6 | 99.6 | -0.4 | -0.4 |
| basilar_vertebral | 100.0 | 99.0 | 99.8 | 99.8 | -0.2 | +0.8 |
| cerebral_venous_sinus | 88.0 | 88.0 | 100.0 | 100.0 | +12.0 | +12.0 |
| sma_embolism | 93.2 | 93.2 | 89.2 | 89.2 | -4.0 | -4.0 |
| femoropopliteal_pad | 100.0 | 100.0 | 94.8 | 94.8 | -5.2 | -5.2 |
| renal_artery | 100.0 | 99.8 | 100.0 | 100.0 | +0.0 | +0.2 |
| **mean of 14** | 98.3 | 97.4 | 98.5 | 98.3 | +0.3 | +1.0 |
