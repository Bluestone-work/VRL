# 2026-11-27 acceptance status

## Completed real runs

- Corrected timeout accounting was rerun for the six-scene, 18-episode-per-file fixed-three-target matrix. The boundary case `linear_sum`, `N=3`, seed `2944200005` now records `termination_class=timeout` at `elapsed_s=299.9999999999997`.
- S43 MAPPO-V2 and IPPO paired evaluation is complete for train 180 and holdout 100. MAPPO-V2 is 153/180 and 80/100 for navigation success. IPPO is 153/180 and 80/100. Every holdout comparison is a per-scene tie on navigation success.
- A separate 100,000-physical-step MAPPO-V2 seed-41 run is complete. Its train result is 152/180 and holdout result is 80/100. The paired long-run summary is in `research/paired_navigation_corrected/mappo_v2_long_s41/paired_results.json`.
- Corrected fixed-target outputs are in `research/three_targets_corrected_*.json`. Geodesic N=3 reaches all three in all six scenes. Linear-sum N=3 reaches all three in five of six scenes. The 30-second synchronized rate is 2/6 for both in this regression set.
- CCBS fixed-route planning has a real single-episode smoke run and unit coverage. The full baseline matrix remains a long run.

## Still pending

- 500,000-step MAPPO-V2 runs for seeds 42 and 43 were not started because the available time was used to complete S43 and the seed-41 long run. Existing 50,000-step S42/S43 runs remain intact.
- The mr-nav upstream interface and license were audited at commit `c129da1447d9854e31b1d7b970b7f48b8a553ea7`, but its dataset and exported policy are not present. No external mr-nav score is reported.
- Full CCBS and long-run all-seed baseline matrices are implemented and resumable, but not counted as completed results.
