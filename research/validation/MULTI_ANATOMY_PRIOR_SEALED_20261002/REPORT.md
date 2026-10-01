# Route + avoid prior (gain 6, no learning): sealed test on all 14 anatomies

Controller: observed own route bearing + 6 x repulsion from the observed predicted particle slots. Gain chosen on the MCA diagnostic split only; the other 13 anatomies are zero-shot (never tuned). Each row is one sealed-test evaluation of 500 registered test layouts (configs/evaluation_splits.json), study PRIOR_ZERO_SHOT_ANATOMIES (13 declared) plus BASELINE_ROUTE_AVOID_PRIOR_ONLY for MCA.

| anatomy | complete | collision-free | mean removal |
|---|---:|---:|---:|
| mca_m1_lvo | 94.4% | 84.8% | 98.6% |
| pulmonary_saddle | 100.0% | 100.0% | 100.0% |
| coronary_lm_bifurcation | 100.0% | 99.8% | 100.0% |
| coronary_rca | 100.0% | 99.8% | 100.0% |
| iliac_may_thurner | 100.0% | 100.0% | 100.0% |
| popliteal_calf_dvt | 100.0% | 99.8% | 100.0% |
| ica_siphon | 100.0% | 99.4% | 100.0% |
| ica_terminus_t | 100.0% | 99.8% | 100.0% |
| carotid_bifurcation | 100.0% | 100.0% | 100.0% |
| basilar_vertebral | 100.0% | 99.0% | 100.0% |
| cerebral_venous_sinus | 88.0% | 88.0% | 97.0% |
| sma_embolism | 93.2% | 93.2% | 98.3% |
| femoropopliteal_pad | 100.0% | 100.0% | 100.0% |
| renal_artery | 100.0% | 99.8% | 100.0% |

Mean over 14: complete 98.3%, collision-free 97.4%.

Caveats: this is a structured (non-learned) controller; the sim is a synthetic engineering model, not calibrated physiology. Weakest: cerebral_venous_sinus (retrograde, 88%), sma_embolism (93%), mca_m1_lvo (94%).
