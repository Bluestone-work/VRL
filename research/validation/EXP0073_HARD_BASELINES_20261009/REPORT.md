# EXP0073 paired hard-condition screening

Ours (flow_aux) denotes the evaluated Transformer with flow auxiliary training; nav_tf_v3 is the original Transformer reference. Ours here does not include belief or rolling-TPG.
Single learned training seed 7101. Classical defaults unchanged. PAC-NMPC and STPG are repository adaptations, not exact author reproductions.
Infrastructure failures remain in failures.jsonl. Tables use only common completed scenes; missing cases are not counted as successes or task failures.
All episodes contribute to AUC and T90_300 (unreached = 300 s). Conditional T90 is recorded separately. No sealed test access.
Variation strength changes flow/pulsatility, actuation, lag, wall effects and lysis rate jointly, not response gain alone. Flow 0.1 is in the EXP0073 training support.

## low_delay: flow/delay/variation = (0.05, 1, 0.0)

Common completed cases: 42/42. Available by method: {'adaptive_settle': 42, 'flow_aux': 42, 'nav_tf_v3': 42, 'no_settle': 42, 'pac_nmpc': 42, 'settle': 42, 'stpg': 42}

| Method | n | Full % | Strict % | Removal % | AUC | T90_300 s | T90 reached | conditional T90 s | Wall s | Lost total | Spacing pair-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_settle | 42 | 90.5 | 88.1 | 95.5 | 0.753 | 131.9 | 38/42 | 114.2 | 9.76 | 1 | 4.274 |
| Ours (flow_aux) | 42 | 66.7 | 61.9 | 85.3 | 0.681 | 171.0 | 28/42 | 106.5 | 2.31 | 1 | 0.000 |
| nav_tf_v3 | 42 | 66.7 | 64.3 | 88.1 | 0.705 | 168.3 | 28/42 | 102.4 | 7.98 | 1 | 0.180 |
| no_settle | 42 | 90.5 | 85.7 | 97.5 | 0.747 | 133.7 | 40/42 | 125.4 | 0.13 | 1 | 0.000 |
| pac_nmpc | 42 | 50.0 | 47.6 | 75.0 | 0.607 | 205.3 | 21/42 | 110.6 | 0.22 | 1 | 0.000 |
| settle | 42 | 47.6 | 42.9 | 70.6 | 0.574 | 207.7 | 20/42 | 106.1 | 1.36 | 1 | 0.000 |
| stpg | 42 | 47.6 | 42.9 | 70.6 | 0.574 | 207.7 | 20/42 | 106.1 | 1.36 | 1 | 0.000 |

## high_delay: flow/delay/variation = (0.05, 3, 0.0)

Common completed cases: 42/42. Available by method: {'adaptive_settle': 42, 'flow_aux': 42, 'nav_tf_v3': 42, 'no_settle': 42, 'pac_nmpc': 42, 'settle': 42, 'stpg': 42}

| Method | n | Full % | Strict % | Removal % | AUC | T90_300 s | T90 reached | conditional T90 s | Wall s | Lost total | Spacing pair-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_settle | 42 | 0.0 | 0.0 | 15.3 | 0.138 | 300.0 | 0/42 | NA | 9.87 | 1 | 0.000 |
| Ours (flow_aux) | 42 | 54.8 | 54.8 | 80.0 | 0.601 | 202.2 | 24/42 | 128.9 | 0.03 | 1 | 0.000 |
| nav_tf_v3 | 42 | 16.7 | 16.7 | 65.9 | 0.461 | 264.1 | 13/42 | 184.1 | 5.04 | 1 | 0.000 |
| no_settle | 42 | 0.0 | 0.0 | 11.6 | 0.105 | 300.0 | 0/42 | NA | 13.56 | 1 | 0.000 |
| pac_nmpc | 42 | 42.9 | 35.7 | 70.7 | 0.548 | 218.6 | 18/42 | 110.0 | 7.30 | 1 | 0.000 |
| settle | 42 | 71.4 | 69.0 | 86.5 | 0.695 | 160.6 | 30/42 | 104.9 | 0.00 | 1 | 0.000 |
| stpg | 42 | 71.4 | 69.0 | 86.5 | 0.695 | 160.2 | 30/42 | 104.3 | 0.00 | 1 | 0.000 |

## strong_flow: flow/delay/variation = (0.1, 2, 0.625)

Common completed cases: 42/42. Available by method: {'adaptive_settle': 42, 'flow_aux': 42, 'nav_tf_v3': 42, 'no_settle': 42, 'pac_nmpc': 42, 'settle': 42, 'stpg': 42}

| Method | n | Full % | Strict % | Removal % | AUC | T90_300 s | T90 reached | conditional T90 s | Wall s | Lost total | Spacing pair-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_settle | 42 | 4.8 | 4.8 | 35.3 | 0.287 | 293.3 | 2/42 | 159.5 | 7.48 | 2 | 0.000 |
| Ours (flow_aux) | 42 | 26.2 | 23.8 | 49.7 | 0.382 | 255.4 | 11/42 | 129.7 | 16.12 | 1 | 0.033 |
| nav_tf_v3 | 42 | 28.6 | 26.2 | 50.9 | 0.387 | 255.7 | 12/42 | 144.9 | 16.14 | 1 | 0.064 |
| no_settle | 42 | 0.0 | 0.0 | 17.1 | 0.138 | 300.0 | 0/42 | NA | 7.53 | 2 | 0.000 |
| pac_nmpc | 42 | 7.1 | 7.1 | 35.8 | 0.276 | 285.1 | 3/42 | 91.1 | 0.25 | 1 | 0.000 |
| settle | 42 | 14.3 | 11.9 | 42.3 | 0.338 | 274.5 | 6/42 | 121.6 | 7.47 | 2 | 0.015 |
| stpg | 42 | 14.3 | 11.9 | 42.3 | 0.338 | 274.5 | 6/42 | 121.6 | 7.47 | 2 | 0.015 |

## variable_response: flow/delay/variation = (0.05, 2, 1.25)

Common completed cases: 42/42. Available by method: {'adaptive_settle': 42, 'flow_aux': 42, 'nav_tf_v3': 42, 'no_settle': 42, 'pac_nmpc': 42, 'settle': 42, 'stpg': 42}

| Method | n | Full % | Strict % | Removal % | AUC | T90_300 s | T90 reached | conditional T90 s | Wall s | Lost total | Spacing pair-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_settle | 42 | 0.0 | 0.0 | 26.5 | 0.200 | 299.0 | 2/42 | 278.4 | 5.30 | 2 | 0.077 |
| Ours (flow_aux) | 42 | 40.5 | 26.2 | 59.0 | 0.428 | 241.2 | 18/42 | 162.7 | 4.84 | 3 | 0.030 |
| nav_tf_v3 | 42 | 35.7 | 23.8 | 58.0 | 0.415 | 246.2 | 17/42 | 167.0 | 4.84 | 3 | 0.046 |
| no_settle | 42 | 0.0 | 0.0 | 23.0 | 0.155 | 299.3 | 1/42 | 270.4 | 5.72 | 2 | 0.058 |
| pac_nmpc | 42 | 42.9 | 35.7 | 62.4 | 0.470 | 234.7 | 19/42 | 155.6 | 9.98 | 2 | 0.000 |
| settle | 42 | 42.9 | 31.0 | 65.4 | 0.492 | 225.3 | 18/42 | 125.8 | 4.54 | 2 | 0.040 |
| stpg | 42 | 42.9 | 31.0 | 65.4 | 0.493 | 224.9 | 18/42 | 124.8 | 4.54 | 2 | 0.040 |

