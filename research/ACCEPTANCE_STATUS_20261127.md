# 2026-11-27 acceptance status

## Completed real runs

- 2026-11-27 completion records are in `research/NAVIGATION_COMPLETION_20261127.md`. This file supersedes the earlier pending list.

- Corrected timeout accounting was rerun for the six-scene, 18-episode-per-file fixed-three-target matrix. The boundary case `linear_sum`, `N=3`, seed `2944200005` now records `termination_class=timeout` at `elapsed_s=299.9999999999997`.
- S43 MAPPO-V2 and IPPO paired evaluation is complete for train 180 and holdout 100. MAPPO-V2 is 153/180 and 80/100 for navigation success. IPPO is 153/180 and 80/100. Every holdout comparison is a per-scene tie on navigation success.
- A separate 100,000-physical-step MAPPO-V2 seed-41 run is complete. Its train result is 152/180 and holdout result is 80/100. The paired long-run summary is in `research/paired_navigation_corrected/mappo_v2_long_s41/paired_results.json`.
- Corrected fixed-target outputs are in `research/three_targets_corrected_*.json`. Geodesic N=3 reaches all three in all six scenes. Linear-sum N=3 reaches all three in five of six scenes. The 30-second synchronized rate is 2/6 for both in this regression set.
- CCBS fixed-route planning has a real single-episode smoke run and unit coverage. The full baseline matrix remains a long run.

## Still pending

- The S42 and S43 500,000-physical-step MAPPO-V2 jobs are still running in separate directories. Their latest progress is recorded in each `status.json`. The S41 attempt stopped at 249,088 physical steps without a completion record. No final 500k performance claim is made yet.
- The requested patient-specific HCC liver-artery centerline/radius/three-target export is unavailable. The completed HCC transfer uses the procedural portal-hepatic model and is labeled synthetic.
