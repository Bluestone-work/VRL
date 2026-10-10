# EXP0090 第一轮困难矩阵配对 pilot（开发集，非密封测试）

共同配对场景 252 个；scenario_hash 不一致 0 个；臂：switch_settle, G_sel_inv

T90_300 = 未达 90 % 记 300 s；reached90 = 达到比例；cond_t90 = 仅对达到者求均值。PAC-NMPC、STPG 为本地适配实现。

## low_delay / all (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 88.1 | 90.5 | 96.5 | 76.2 | 132.0 | 90.5 | 114.3 | 0.002 | 0.002 | 0.0 | 0.019 | 4.473 | 0.000 | 156.1 | 8.014 |
| G_sel_inv | 81.0 | 83.3 | 92.8 | 73.6 | 140.4 | 85.7 | 113.8 | 9.608 | 0.416 | 4.8 | 0.233 | 1.872 | 0.000 | 136.2 | 22.8 |

## low_delay / seen_topology (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 88.1 | 90.5 | 96.5 | 76.2 | 132.0 | 90.5 | 114.3 | 0.002 | 0.002 | 0.0 | 0.019 | 4.473 | 0.000 | 156.1 | 8.014 |
| G_sel_inv | 81.0 | 83.3 | 92.8 | 73.6 | 140.4 | 85.7 | 113.8 | 9.608 | 0.416 | 4.8 | 0.233 | 1.872 | 0.000 | 136.2 | 22.8 |

## low_delay / unseen_topology (n=15)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 86.7 | 86.7 | 96.7 | 72.9 | 160.0 | 86.7 | 138.4 | 0.000 | 0.000 | 0.0 | 0.000 | 0.416 | 0.000 | 175.4 | 8.049 |
| G_sel_inv | 73.3 | 73.3 | 91.5 | 69.3 | 170.8 | 80.0 | 138.5 | 0.353 | 0.340 | 6.7 | 0.000 | 2.286 | 0.000 | 148.0 | 20.7 |

## moderate_delay / all (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 50.0 | 50.0 | 78.1 | 65.1 | 195.8 | 50.0 | 91.7 | 4.893 | 0.199 | 4.8 | 0.000 | 0.253 | 0.000 | 147.8 | 8.010 |
| G_sel_inv | 76.2 | 81.0 | 92.9 | 76.1 | 135.4 | 81.0 | 96.7 | 7.317 | 2.225 | 9.5 | 0.000 | 0.323 | 0.000 | 121.1 | 20.2 |

## moderate_delay / seen_topology (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 50.0 | 50.0 | 78.1 | 65.1 | 195.8 | 50.0 | 91.7 | 4.893 | 0.199 | 4.8 | 0.000 | 0.253 | 0.000 | 147.8 | 8.010 |
| G_sel_inv | 76.2 | 81.0 | 92.9 | 76.1 | 135.4 | 81.0 | 96.7 | 7.317 | 2.225 | 9.5 | 0.000 | 0.323 | 0.000 | 121.1 | 20.2 |

## moderate_delay / unseen_topology (n=15)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 33.3 | 33.3 | 68.5 | 55.2 | 236.0 | 33.3 | 107.9 | 0.000 | 0.000 | 0.0 | 0.000 | 0.000 | 0.000 | 173.6 | 8.141 |
| G_sel_inv | 60.0 | 60.0 | 85.0 | 68.4 | 184.1 | 60.0 | 106.8 | 5.760 | 5.760 | 13.3 | 0.000 | 0.000 | 0.000 | 129.1 | 19.4 |

## high_delay / all (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 66.7 | 71.4 | 86.5 | 69.6 | 160.2 | 71.4 | 104.3 | 0.000 | 0.000 | 0.0 | 0.124 | 0.800 | 0.000 | 146.7 | 8.433 |
| G_sel_inv | 78.6 | 81.0 | 93.5 | 75.3 | 139.6 | 81.0 | 101.9 | 8.903 | 4.222 | 7.1 | 0.000 | 2.937 | 0.000 | 140.8 | 21.6 |

## high_delay / seen_topology (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 66.7 | 71.4 | 86.5 | 69.6 | 160.2 | 71.4 | 104.3 | 0.000 | 0.000 | 0.0 | 0.124 | 0.800 | 0.000 | 146.7 | 8.433 |
| G_sel_inv | 78.6 | 81.0 | 93.5 | 75.3 | 139.6 | 81.0 | 101.9 | 8.903 | 4.222 | 7.1 | 0.000 | 2.937 | 0.000 | 140.8 | 21.6 |

## high_delay / unseen_topology (n=15)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 46.7 | 53.3 | 75.1 | 58.4 | 195.6 | 53.3 | 104.3 | 0.000 | 0.000 | 0.0 | 0.348 | 2.070 | 0.000 | 166.0 | 8.195 |
| G_sel_inv | 66.7 | 66.7 | 86.8 | 67.7 | 171.4 | 66.7 | 107.1 | 0.304 | 0.073 | 6.7 | 0.001 | 7.997 | 0.000 | 173.2 | 21.2 |

## strong_flow / all (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 11.9 | 14.3 | 43.8 | 34.5 | 274.7 | 14.3 | 122.7 | 7.472 | 0.545 | 7.1 | 0.082 | 4.231 | 0.000 | 188.7 | 8.641 |
| G_sel_inv | 28.6 | 31.0 | 60.4 | 48.6 | 241.5 | 31.0 | 110.9 | 5.485 | 0.365 | 7.1 | 0.004 | 4.156 | 0.000 | 163.8 | 20.7 |

## strong_flow / seen_topology (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 11.9 | 14.3 | 43.8 | 34.5 | 274.7 | 14.3 | 122.7 | 7.472 | 0.545 | 7.1 | 0.082 | 4.231 | 0.000 | 188.7 | 8.641 |
| G_sel_inv | 28.6 | 31.0 | 60.4 | 48.6 | 241.5 | 31.0 | 110.9 | 5.485 | 0.365 | 7.1 | 0.004 | 4.156 | 0.000 | 163.8 | 20.7 |

## strong_flow / unseen_topology (n=15)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 13.3 | 13.3 | 39.4 | 30.3 | 278.7 | 13.3 | 140.0 | 8.630 | 1.185 | 6.7 | 0.000 | 0.000 | 0.000 | 196.1 | 8.875 |
| G_sel_inv | 20.0 | 20.0 | 50.2 | 40.3 | 260.7 | 20.0 | 103.3 | 15.2 | 0.853 | 13.3 | 0.000 | 0.000 | 0.000 | 162.2 | 21.3 |

## variable_response / all (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 28.6 | 42.9 | 65.4 | 49.5 | 224.0 | 42.9 | 122.6 | 1.657 | 0.232 | 16.7 | 0.105 | 7.237 | 0.000 | 213.9 | 7.803 |
| G_sel_inv | 45.2 | 61.9 | 78.7 | 61.1 | 191.1 | 61.9 | 124.1 | 6.108 | 0.405 | 21.4 | 0.069 | 7.212 | 0.000 | 146.6 | 21.2 |

## variable_response / seen_topology (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 28.6 | 42.9 | 65.4 | 49.5 | 224.0 | 42.9 | 122.6 | 1.657 | 0.232 | 16.7 | 0.105 | 7.237 | 0.000 | 213.9 | 7.803 |
| G_sel_inv | 45.2 | 61.9 | 78.7 | 61.1 | 191.1 | 61.9 | 124.1 | 6.108 | 0.405 | 21.4 | 0.069 | 7.212 | 0.000 | 146.6 | 21.2 |

## variable_response / unseen_topology (n=15)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 13.3 | 26.7 | 52.2 | 40.9 | 250.4 | 26.7 | 114.0 | 0.085 | 0.026 | 0.0 | 0.213 | 1.556 | 0.000 | 221.5 | 8.160 |
| G_sel_inv | 26.7 | 33.3 | 67.0 | 52.3 | 256.5 | 33.3 | 169.6 | 0.168 | 0.077 | 6.7 | 0.000 | 0.796 | 0.000 | 183.0 | 21.7 |

## ood_flow_delay / all (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 35.7 | 40.5 | 65.6 | 51.1 | 226.7 | 40.5 | 118.8 | 0.679 | 0.189 | 9.5 | 0.003 | 0.351 | 0.000 | 216.8 | 8.280 |
| G_sel_inv | 45.2 | 57.1 | 79.9 | 64.0 | 186.7 | 59.5 | 109.7 | 1.892 | 0.830 | 9.5 | 0.000 | 0.262 | 0.000 | 146.7 | 19.4 |

## ood_flow_delay / seen_topology (n=42)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 35.7 | 40.5 | 65.6 | 51.1 | 226.7 | 40.5 | 118.8 | 0.679 | 0.189 | 9.5 | 0.003 | 0.351 | 0.000 | 216.8 | 8.280 |
| G_sel_inv | 45.2 | 57.1 | 79.9 | 64.0 | 186.7 | 59.5 | 109.7 | 1.892 | 0.830 | 9.5 | 0.000 | 0.262 | 0.000 | 146.7 | 19.4 |

## ood_flow_delay / unseen_topology (n=15)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 33.3 | 40.0 | 67.9 | 51.3 | 230.1 | 40.0 | 125.1 | 1.505 | 0.362 | 13.3 | 0.000 | 0.022 | 0.000 | 257.0 | 7.663 |
| G_sel_inv | 53.3 | 60.0 | 81.9 | 62.8 | 193.2 | 60.0 | 122.1 | 0.000 | 0.000 | 0.0 | 0.000 | 0.021 | 0.000 | 172.3 | 18.5 |

## ALL / all (n=252)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 46.8 | 51.6 | 72.7 | 57.7 | 202.2 | 51.6 | 110.5 | 2.451 | 0.195 | 6.3 | 0.056 | 2.891 | 0.000 | 178.3 | 8.197 |
| G_sel_inv | 59.1 | 65.9 | 83.0 | 66.5 | 172.5 | 66.7 | 108.7 | 6.552 | 1.411 | 9.9 | 0.051 | 2.794 | 0.000 | 142.5 | 21.0 |

## ALL / seen_topology (n=252)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 46.8 | 51.6 | 72.7 | 57.7 | 202.2 | 51.6 | 110.5 | 2.451 | 0.195 | 6.3 | 0.056 | 2.891 | 0.000 | 178.3 | 8.197 |
| G_sel_inv | 59.1 | 65.9 | 83.0 | 66.5 | 172.5 | 66.7 | 108.7 | 6.552 | 1.411 | 9.9 | 0.051 | 2.794 | 0.000 | 142.5 | 21.0 |

## ALL / unseen_topology (n=90)

| arm | strict_success | task_success | removal | removal_auc | t90_300 | reached90 | cond_t90 | wall_contact_s | max_continuous_wall_contact_s | wall_ge_1s | spacing_violation_pair_s | coupling_exposure | tpg_hold_s | path_mm | ctrl_ms_per_step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | 37.8 | 42.2 | 66.7 | 51.5 | 225.1 | 42.2 | 122.6 | 1.703 | 0.262 | 3.3 | 0.093 | 0.677 | 0.000 | 198.3 | 8.181 |
| G_sel_inv | 50.0 | 52.2 | 77.1 | 60.1 | 206.1 | 53.3 | 124.0 | 3.624 | 1.184 | 7.8 | 0.000 | 1.850 | 0.000 | 161.3 | 20.5 |

## 种子统计（均值 ± 标准差；逐种子值）


## 配对差值（场景 bootstrap 95% CI；多种子臂先按场景对种子取均值）

### 相对 A_nav_off

| panel | arm | Δstrict pp | ΔT90_300 s | Δwall s | ΔAUC pp |
|---|---|---|---|---|---|

### 相对 switch_settle

| panel | arm | Δstrict pp | ΔT90_300 s | Δwall s | ΔAUC pp |
|---|---|---|---|---|---|
| low_delay | G_sel_inv | -7.1 [-16.7, +2.4] | +8.4 [-6.2, +25.7] | +9.6 [+0.0, +28.7] | -2.6 [-5.8, -0.0] |
| moderate_delay | G_sel_inv | +26.2 [+11.9, +40.5] | -60.5 [-93.8, -27.1] | +2.4 [+0.1, +6.7] | +11.0 [+5.3, +17.5] |
| high_delay | G_sel_inv | +11.9 [+0.0, +23.8] | -20.6 [-49.2, +6.9] | +8.9 [+0.0, +22.4] | +5.7 [+0.4, +11.9] |
| strong_flow | G_sel_inv | +16.7 [+4.8, +31.0] | -33.2 [-59.5, -9.4] | -2.0 [-12.9, +6.5] | +14.0 [+6.9, +21.2] |
| variable_response | G_sel_inv | +16.7 [+2.4, +31.0] | -32.9 [-61.4, -5.0] | +4.5 [-2.9, +16.2] | +11.6 [+4.2, +19.4] |
| ood_flow_delay | G_sel_inv | +9.5 [-4.8, +26.2] | -39.9 [-69.4, -10.9] | +1.2 [-1.0, +4.1] | +12.9 [+6.5, +20.2] |
| ALL | G_sel_inv | +12.3 [+6.7, +17.9] | -29.8 [-41.0, -18.6] | +4.1 [+0.3, +9.0] | +8.8 [+6.2, +11.5] |

## 执行诊断（ALL / all）

| arm | guard override % | shield override % | TPG hold % | learn dev % | learn dev mag | takeover % | fallback % | stop % | infer ms | ctrl ms |
|---|---|---|---|---|---|---|---|---|---|---|
| switch_settle | - | - | - | - | - | - | - | - | - | 8.197 |
| G_sel_inv | - | - | - | - | - | 27.6 | 0.0 | 5.0 | 13.0 | 21.0 |

## 逐解剖 Strict %（ALL panels）

| anatomy | switch_settle | G_sel_inv |
|---|---|---|
| basilar_vertebral | 61 | 44 |
| carotid_bifurcation | 50 | 72 |
| cerebral_venous_sinus | 56 | 72 |
| coronary_lm_bifurcation | 78 | 67 |
| coronary_rca | 22 | 61 |
| femoropopliteal_pad | 22 | 28 |
| ica_siphon | 44 | 78 |
| ica_terminus_t | 67 | 83 |
| iliac_may_thurner | 22 | 39 |
| mca_m1_lvo | 44 | 39 |
| popliteal_calf_dvt | 78 | 78 |
| pulmonary_saddle | 33 | 50 |
| renal_artery | 50 | 72 |
| sma_embolism | 28 | 44 |
