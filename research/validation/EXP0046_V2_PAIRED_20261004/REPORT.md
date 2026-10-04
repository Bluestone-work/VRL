# EXP0046 revision 2 paired diagnostic results — 2026-10-04

Audit: **BLOCKED**; 259/260 episodes completed; 1 failed processes.

The predeclared numerical gate failed. **No method ranking or paired performance claim is admissible from this batch.** Original failures remain in the requested denominator; later debugging reproductions cannot replace them.

- multi_unshielded_n3_d2_s42_fixed_total, seed 1302010005: process_exit
- Runner reports an incomplete or failed cell

## Descriptive outcomes only

Rows below are for debugging and next-study design. Metrics are conditional on completed processes; bounds count every requested episode and treat missing results as failure/success at the lower/upper endpoints. No intervals, ranking or superiority claim is made. There are 10 scenes; repeated controller seeds are not independent scenes.

| Arm | Complete / requested | Failed | Cluster-safe bounds % | Removal % among complete | AUC % among complete | Spacing violations % among complete |
|---|---:|---:|---:|---:|---:|---:|
| single_sequential_n1_d2_fixed_total | 30/30 | 0 | [16.7, 16.7] | 34.2 | 26.5 | 0.0 |
| multi_parallel_n2_d1_fixed_total | 30/30 | 0 | [33.3, 33.3] | 61.7 | 49.8 | 13.3 |
| multi_parallel_n2_d2_fixed_total | 30/30 | 0 | [30.0, 30.0] | 56.7 | 48.6 | 13.3 |
| multi_parallel_n2_d4_fixed_total | 30/30 | 0 | [30.0, 30.0] | 59.2 | 49.5 | 3.3 |
| multi_parallel_n3_d1_fixed_total | 30/30 | 0 | [30.0, 30.0] | 82.5 | 67.6 | 20.0 |
| multi_parallel_n3_d2_fixed_total | 30/30 | 0 | [16.7, 16.7] | 69.9 | 59.8 | 33.3 |
| multi_parallel_n3_d4_fixed_total | 30/30 | 0 | [13.3, 13.3] | 73.0 | 60.6 | 50.0 |
| single_route_n1_d2_fixed_total | 10/10 | 0 | [40.0, 40.0] | 53.2 | 35.5 | 0.0 |
| multi_unshielded_n2_d2_fixed_total | 10/10 | 0 | [30.0, 30.0] | 67.5 | 56.1 | 50.0 |
| multi_parallel_n2_d2_per_cluster | 10/10 | 0 | [30.0, 30.0] | 60.0 | 50.3 | 10.0 |
| multi_unshielded_n3_d2_fixed_total | 9/10 | 1 | [10.0, 20.0] | 83.3 | 70.5 | 88.9 |
| multi_parallel_n3_d2_per_cluster | 10/10 | 0 | [20.0, 20.0] | 67.5 | 60.2 | 20.0 |

## Definitions and next gate

- Primary cluster-safe success: full clearing, <1 total cluster-s wall contact, no particle/pair contact, no cluster loss, no separation violation.
- AUC holds terminal removal to a common 180 s horizon. Unreached clearance times receive a 180 s penalty in the detailed JSON, not a fictitious completion time.
- Fixed total catalytic rate is 0.36 mass/s; per-cluster arms have N times that capacity. Neither total magnetic power nor material volume is matched.
- The policy excludes route/true-flow/edge queries. Local geometry and tracking, including 6 mm peer imaging, remain simulated sensor assumptions.
- Separation control is heuristic; ideal independent actuators are uncalibrated. These results do not establish magnetic independence or hardware transfer.
- Capture and diagnose the failing process first. Keep any subsequent runtime repair and new performance batch separately registered; do not silently rerun the missing row.

All raw checksums, missing-data bounds and conditional metrics are in `analysis.json`.
