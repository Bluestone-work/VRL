# V3 experiment plan

Status: Phase A and Phase B registered 2026-11-28 **before** the validation and test runs.
Phases C–I are planned, not run (see `ACCEPTANCE_REPORT.md`).

## Phase B: autonomous configuration with a fixed local controller

**Question.** Does the graph-based global layer, choosing N, release stations, allocation and release
schedule by itself, improve the four metrics over fixed configurations when the local controller is
held fixed?

**Fixed for every arm.** Local controller = pure pursuit + SettleGuard + wall SDF projection + spacing
shield (d_safe 2 mm) + release-aware TPG; image sensing with 1-step latency; patient-level variation
s = 0.75 (pulsatility only in the pulsatile regime); 8 dynamic obstacles; catheter model and release
candidates; n_max = 4; deadline 300 s from catheter insertion.

**Task sets** (`CONFIGURATION_RESULTS/tasks_*.json`, seeds disjoint from all earlier VRL sets)

| set | tasks | design | use |
|---|---:|---|---|
| validation | 45 | 15 territories × m ∈ {3,4,5}, pattern alternating, flow cycling | choose the best fixed N for the fixed-N baselines; nothing else |
| test (frozen) | 120 | 15 territories × m ∈ {2,3,4,5} × {dispersed, clustered}, flow cycling | all reported Phase B results; set hash `3882f0f2bba18e2d…` |

The 15 territories are the 9 RL-training anatomies, the 5 RL-holdout anatomies and the HCC portal tree.
For the global layer every territory is a known preoperative map; the train/holdout label matters only
for the later RL phases.

**Arms**

| arm | N | deployment | objective | search | purpose |
|---|---|---|---|---|---|
| fixedN_root | best fixed N from validation (root release) | all at the root | full J | local | conventional fixed configuration |
| fixedN_random | same N | random feasible release stations | full J | local | deployment ablation |
| fixedN_opt | best fixed N from validation (optimised release) | optimised | full J | local | isolates adaptive N |
| adaptiveN_root | ε-rule | all at the root | full J | local | isolates deployment optimisation |
| adaptiveN_opt_time | ε-rule | optimised | time only (T_free) | local | decoupled, interaction-unaware planning |
| **proposed** | ε-rule | optimised | full J | local | joint configuration and task planning |
| exact | ε-rule | optimised | full J | exact best-first | small-scale exact reference (all test tasks have m ≤ 5) |
| abl_euclid | ε-rule | optimised | full J, Euclidean allocation costs | local | w/o topology-aware assignment |
| abl_no_tpg | proposed configuration | optimised | full J | local | w/o temporal coordination (TPG off at execution) |

Fixed N is capped at m (a cluster without a target is not deployed).

**Pre-registered hypotheses** (test set, paired by base_task_hash; reference = fixedN_root)

| id | metric | hypothesis | minimal meaningful effect |
|---|---|---|---|
| H1 | success | proposed ≥ fixedN_root | +5 percentage points |
| H2 | T_last (deadline-scored) | proposed < fixedN_root | −10 s paired mean |
| H3 | conflict rate | proposed ≤ fixedN_root | −5 percentage points |
| H4 | safe_success | proposed ≥ fixedN_root | +5 percentage points |
| H5 | planner | local-search J within 5% of exact on ≥ 95% of tasks | — |

Verdicts: PASS if the 95% interval of the paired difference excludes zero in the hypothesised direction
and the point estimate reaches the minimal effect; FAIL if the interval excludes zero in the opposite
direction; otherwise INCONCLUSIVE. Secondary comparisons (each ablation against proposed) are reported
with the same intervals and are not used for verdicts.

**Statistics.** Paired differences per base task; 95% intervals from a two-level bootstrap (resample
territories, then tasks within territory; 10,000 replicates). Binary outcomes also get an exact McNemar
test. One execution per (task, arm): the local controller is deterministic given the task seed, so
repeated runs are identical; variability comes from tasks.

**Not tuned.** w_wait = 1, w_risk = 0.5, ε = 0.10 and all deployment parameters are fixed in
`configs/v3/assumptions.json` before the validation run. Sensitivity to these values is reported on
the validation set only.

## Known limits of Phase B

* The local controller is the rule-based pursuit, so Phase B measures the global layer only.
* A TPG hold is a zero command, not station keeping: held clusters drift with flow (Phase C).
* The catheter does not interact with clusters, and its speed (2 mm/s) is an assumption.
* No magnetic interaction model: conflicts are distance-based (layer D not modelled).
