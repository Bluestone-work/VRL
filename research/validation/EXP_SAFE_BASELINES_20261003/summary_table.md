
### All 14 anatomies

| method | info | Raw | Safe | Gap | wall mean s | wall median s | wall ratio % | max run s | timeout % | removal % | route cos | wrong half | Safe seed range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_rl | privileged | 28.3 | 21.9 | 6.4 | 20.3 | 5.7 | 2.5 | 10.6 | 71.3 | 60.4 | 0.19 | 35% | 12.1–27.1 |
| residual | privileged | 98.1 | 95.4 | 2.7 | 0.1 | 0.0 | 0.1 | 0.1 | 1.9 | 99.5 | 0.97 | 0% | 94.6–96.4 |
| residual_shield | privileged | 98.3 | 95.6 | 2.7 | 0.2 | 0.0 | 0.1 | 0.1 | 1.7 | 99.6 | 0.97 | 0% | 95.0–96.8 |
| teacher | privileged | 97.9 | 96.1 | 1.8 | 0.2 | 0.0 | 0.2 | 0.1 | 2.1 | 99.4 | 1.00 | 0% | — |
| reactive_bearing | fair | 83.2 | 1.1 | 82.1 | 219.1 | 116.4 | 52.0 | 54.2 | 16.8 | 94.9 | -0.05 | 52% | — |
| reactive_path | fair | 66.1 | 64.6 | 1.4 | 0.3 | 0.0 | 0.1 | 0.1 | 33.9 | 89.5 | 0.61 | 19% | — |

### Train anatomies (anatomy_holdout_v1)

| method | info | Raw | Safe | Gap | wall mean s | wall median s | wall ratio % | max run s | timeout % | removal % | route cos | wrong half | Safe seed range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_rl | privileged | 28.3 | 22.4 | 5.9 | 11.1 | 1.4 | 1.5 | 5.4 | 71.1 | 59.7 | 0.19 | 35% | 13.3–27.8 |
| residual | privileged | 99.4 | 95.4 | 4.1 | 0.2 | 0.0 | 0.2 | 0.1 | 0.6 | 99.9 | 0.97 | 0% | 94.4–96.7 |
| residual_shield | privileged | 99.4 | 95.4 | 4.1 | 0.3 | 0.0 | 0.1 | 0.1 | 0.6 | 99.8 | 0.97 | 0% | 94.4–96.7 |
| teacher | privileged | 98.9 | 96.1 | 2.8 | 0.3 | 0.0 | 0.2 | 0.2 | 1.1 | 99.6 | 1.00 | 0% | — |
| reactive_bearing | fair | 82.2 | 1.7 | 80.6 | 212.7 | 102.2 | 51.4 | 53.0 | 17.8 | 94.3 | -0.06 | 53% | — |
| reactive_path | fair | 66.7 | 64.4 | 2.2 | 0.4 | 0.0 | 0.2 | 0.1 | 33.3 | 89.9 | 0.60 | 20% | — |

### Held-out anatomies

| method | info | Raw | Safe | Gap | wall mean s | wall median s | wall ratio % | max run s | timeout % | removal % | route cos | wrong half | Safe seed range |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_rl | privileged | 28.3 | 21.0 | 7.3 | 36.9 | 13.4 | 4.5 | 20.0 | 71.7 | 61.7 | 0.19 | 34% | 10.0–29.0 |
| residual | privileged | 95.7 | 95.3 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 4.3 | 98.9 | 0.97 | 0% | 95.0–96.0 |
| residual_shield | privileged | 96.3 | 96.0 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 3.7 | 99.1 | 0.97 | 0% | 95.0–97.0 |
| teacher | privileged | 96.0 | 96.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 99.0 | 1.00 | 0% | — |
| reactive_bearing | fair | 85.0 | 0.0 | 85.0 | 230.4 | 141.8 | 53.1 | 56.4 | 15.0 | 96.0 | -0.03 | 50% | — |
| reactive_path | fair | 65.0 | 65.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 35.0 | 88.8 | 0.63 | 18% | — |

### Safe success per anatomy (%; learned methods: seed mean)

| anatomy | pure_rl | residual | residual_shield | teacher | reactive_bearing | reactive_path |
|---|---:|---:|---:|---:|---:|---:|
| mca_m1_lvo | 37 | 62 | 62 | 70 | 0 | 45 |
| pulmonary_saddle | 0 | 97 | 97 | 100 | 5 | 75 |
| coronary_lm_bifurcation | 38 | 100 | 100 | 100 | 0 | 60 |
| coronary_rca | 67 | 98 | 98 | 100 | 0 | 95 |
| iliac_may_thurner | 0 | 100 | 100 | 100 | 5 | 90 |
| popliteal_calf_dvt | 0 | 100 | 100 | 100 | 0 | 55 |
| ica_siphon | 20 | 100 | 100 | 100 | 5 | 55 |
| ica_terminus_t | 75 | 100 | 100 | 100 | 0 | 45 |
| carotid_bifurcation | 3 | 100 | 100 | 95 | 0 | 85 |
| basilar_vertebral | 38 | 100 | 100 | 100 | 0 | 45 |
| cerebral_venous_sinus | 0 | 100 | 100 | 95 | 0 | 80 |
| sma_embolism | 0 | 83 | 87 | 85 | 0 | 40 |
| femoropopliteal_pad | 0 | 95 | 95 | 100 | 0 | 65 |
| renal_artery | 28 | 100 | 100 | 100 | 0 | 70 |
