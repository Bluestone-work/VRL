# Phase B results: autonomous configuration with a fixed local controller

Frozen test set: 120 base tasks (15 territories × m ∈ {2,3,4,5} × {dispersed, clustered}, flow
cycling weak/strong/pulsatile), set hash `3882f0f2bba18e2d…`. 1,080 executions (9 arms), 0 errors.
Every arm: pure pursuit + SettleGuard + wall SDF projection + spacing shield + release-aware TPG,
1-step image latency, patient variation s = 0.75, 8 dynamic obstacles, catheter deployment, deadline
300 s from catheter insertion. Fixed N = 3 for both fixed baselines, selected on the 45 validation tasks
before the test run (`validation_selection.json`). Raw rows: `test_runs.jsonl`; statistics:
`test_summary.json` (`scripts/v3/analyze_configuration.py`).

## Four core metrics (mean, 95% two-level bootstrap interval)

| arm | success | safe (contact-free) success | last arrival, deadline-scored (s) | tasks with conflict |
|---|---|---|---|---|
| fixedN_root | 58.3 [40.0, 75.8]% | 42.5 [25.8, 60.0]% | 193.8 [152.0, 234.1] | 20.8 [10.8, 31.7]% |
| fixedN_random | 70.0 [55.8, 83.3]% | 46.7 [30.8, 63.3]% | 172.0 [135.6, 207.7] | 9.2 [3.3, 16.7]% |
| fixedN_opt | 79.2 [66.7, 90.0]% | 56.7 [40.8, 72.5]% | 145.1 [113.6, 177.6] | 1.7 [0.0, 5.8]% |
| adaptiveN_root | 73.3 [57.5, 87.5]% | 59.2 [41.7, 75.8]% | 172.0 [135.6, 207.9] | 3.3 [0.0, 9.2]% |
| adaptiveN_opt_time | 81.7 [68.3, 92.5]% | 62.5 [45.0, 79.2]% | 140.2 [107.8, 174.8] | 1.7 [0.0, 6.7]% |
| proposed | 81.7 [70.0, 91.7]% | 62.5 [45.8, 78.3]% | 139.1 [108.3, 171.5] | 0.0 [0.0, 0.0]% |
| exact | 81.7 [70.0, 91.7]% | 64.2 [47.5, 79.2]% | 138.6 [108.7, 169.8] | 0.0 [0.0, 0.0]% |
| abl_euclid | 81.7 [70.0, 91.7]% | 61.7 [45.0, 77.5]% | 139.5 [108.5, 172.1] | 0.0 [0.0, 0.0]% |
| abl_no_tpg | 80.8 [69.2, 90.8]% | 62.5 [45.8, 78.3]% | 140.2 [109.5, 172.5] | 0.0 [0.0, 0.0]% |

## Pre-registered hypotheses (proposed − fixedN_root, paired, n = 120)

| id | metric | difference [95% CI] | McNemar p (binary) | minimal effect | verdict |
|---|---|---|---|---|---|
| H1 | success | +23.3 pp [+10.0, +38.3] | 5.8e-08 (29 vs 1 discordant) | +5 pp | PASS |
| H2 | T_last_deadline_scored_s | -54.7 s [-79.2, -32.5] | — | −10 s | PASS |
| H3 | conflict | -20.8 pp [-31.7, -10.8] | 6e-08 (0 vs 25 discordant) | −5 pp | PASS |
| H4 | safe_success | +20.0 pp [+6.7, +34.2] | 8.4e-06 (27 vs 3 discordant) | +5 pp | PASS |
| H5 | local-search J within 5% of exact | 100% of tasks (max gap 4.2%, median 0.0%) | — | ≥ 95% | PASS |

## Ablations and baselines against the proposed configuration (arm − proposed, paired)

| arm | Δ success (pp) | Δ safe success (pp) | Δ last arrival (s) | Δ conflict tasks (pp) |
|---|---|---|---|---|
| fixedN_root | -23.3 [-38.3, -10.0] | -20.0 [-34.2, -6.7] | +54.7 [+32.5, +79.2] | +20.8 [+10.8, +31.7] |
| fixedN_random | -11.7 [-21.7, -3.3] | -15.8 [-27.5, -5.0] | +32.9 [+15.0, +52.2] | +9.2 [+3.3, +16.7] |
| fixedN_opt | -2.5 [-8.3, +2.5] | -5.8 [-13.3, +0.8] | +6.1 [-3.0, +17.0] | +1.7 [+0.0, +5.0] |
| adaptiveN_root | -8.3 [-18.3, -0.8] | -3.3 [-10.0, +2.5] | +32.9 [+18.5, +49.1] | +3.3 [+0.0, +9.2] |
| adaptiveN_opt_time | +0.0 [-3.3, +3.3] | +0.0 [-3.3, +3.3] | +1.1 [-3.3, +7.5] | +1.7 [+0.0, +6.7] |
| exact | +0.0 [-3.3, +3.3] | +1.7 [-2.5, +6.7] | -0.5 [-5.6, +3.7] | +0.0 [+0.0, +0.0] |
| abl_euclid | +0.0 [+0.0, +0.0] | -0.8 [-4.2, +0.0] | +0.5 [-0.3, +1.9] | +0.0 [+0.0, +0.0] |
| abl_no_tpg | -0.8 [-3.3, +0.0] | +0.0 [+0.0, +0.0] | +1.1 [-0.7, +5.1] | +0.0 [+0.0, +0.0] |

## Planner

* Local search vs exact best-first: J gap median 0.0%, max 4.2%, same N* on 98.3% of tasks; all 120 exact runs certified `exact_within_model`.
* Median planning time: exact 0.06 s, local search 0.80 s. At m ≤ 5 the exact search is cheaper than the local search because the bound prunes almost everything; the local search is only needed for larger instances, which this test set does not contain.
* Predicted vs executed last arrival on the 98 successful proposed tasks: Pearson r = 0.41, median executed/predicted = 1.00, mean executed − predicted = +26.2 s (a minority of tasks take much longer than planned).

## Resource use, failures and interventions

| arm | mean N | N dist | cluster-time (s) | lost-cluster tasks | timeout tasks | TPG hold (robot·s) | shield steps | planned potential-conflict tasks | HCC success | m=5 success |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixedN_root | 2.75 | {2: 30, 3: 90} | 454 | 20.8% | 24.2% | 0.13 | 2006 | 5.0% | 2/8 | 13/30 |
| fixedN_random | 2.75 | {2: 30, 3: 90} | 344 | 22.5% | 14.2% | 13.12 | 1121 | 5.0% | 3/8 | 18/30 |
| fixedN_opt | 2.75 | {2: 30, 3: 90} | 237 | 14.2% | 14.2% | 3.09 | 650 | 0.8% | 3/8 | 20/30 |
| adaptiveN_root | 1.52 | {1: 69, 2: 40, 3: 10, 4: 1} | 242 | 5.0% | 23.3% | 0.45 | 537 | 0.0% | 3/8 | 18/30 |
| adaptiveN_opt_time | 1.97 | {1: 45, 2: 42, 3: 25, 4: 8} | 164 | 7.5% | 14.2% | 0.71 | 332 | 5.0% | 4/8 | 24/30 |
| proposed | 1.97 | {1: 45, 2: 42, 3: 25, 4: 8} | 163 | 7.5% | 14.2% | 0.57 | 310 | 1.7% | 4/8 | 23/30 |
| exact | 1.95 | {1: 45, 2: 42, 3: 27, 4: 6} | 165 | 8.3% | 13.3% | 0.72 | 304 | 2.5% | 4/8 | 23/30 |
| abl_euclid | 1.97 | {1: 45, 2: 42, 3: 25, 4: 8} | 169 | 9.2% | 13.3% | 1.00 | 315 | 2.5% | 4/8 | 23/30 |
| abl_no_tpg | 1.97 | {1: 45, 2: 42, 3: 25, 4: 8} | 165 | 7.5% | 15.0% | 0.00 | 330 | 1.7% | 4/8 | 22/30 |

"Planned potential-conflict tasks" = tasks whose plan, executed without coordination on the planning
model, would violate d_safe (layer A). "Shield steps" = cluster-steps in which the spacing shield changed
the command (layer C). Lost-cluster and timeout tasks can overlap with success (a cluster lost after
delivering its targets).

* proposed N* by m: {2: {1: 15, 2: 15}, 3: {1: 10, 2: 14, 3: 6}, 4: {1: 13, 2: 8, 3: 5, 4: 4}, 5: {1: 7, 2: 5, 3: 14, 4: 4}}
* proposed success by group: {'rl_train_anatomy': '59/72', 'rl_holdout_anatomy': '35/40', 'hcc': '4/8'}
* proposed success by flow: {'weak': '28/40', 'strong': '35/40', 'pulsatile': '35/40'}
* proposed safe-success failures (success but unsafe): wall 20 obstacle 1

## Planner-only sensitivity (validation set, exact search)

| setting | tasks | mean N* | N* distribution | same N* as registered | same release stations | mean predicted T (s) | mean deployment time (s) |
|---|---:|---:|---|---:|---:|---:|---:|
| registered | 45 | 1.96 | {1: 19, 2: 14, 3: 7, 4: 5} | 100% | 100% | 83.9 | 35.0 |
| eps_0.0 | 45 | 2.02 | {1: 18, 2: 14, 3: 7, 4: 6} | 93% | 93% | 83.5 | 35.2 |
| eps_0.05 | 45 | 2.00 | {1: 19, 2: 13, 3: 7, 4: 6} | 96% | 96% | 83.5 | 35.2 |
| eps_0.2 | 45 | 1.84 | {1: 20, 2: 14, 3: 9, 4: 2} | 89% | 89% | 85.5 | 34.3 |
| vcath_0.5 | 45 | 1.56 | {1: 27, 2: 11, 3: 7} | 64% | 7% | 118.3 | 1.7 |
| vcath_1.0 | 45 | 1.58 | {1: 26, 2: 12, 3: 7} | 64% | 49% | 115.7 | 54.4 |
| vcath_4.0 | 45 | 1.93 | {1: 19, 2: 14, 3: 8, 4: 4} | 93% | 80% | 69.5 | 18.8 |
| release_interval_1.5 | 45 | 1.96 | {1: 19, 2: 14, 3: 7, 4: 5} | 100% | 100% | 83.9 | 35.0 |
| release_interval_6.0 | 45 | 1.98 | {1: 19, 2: 13, 3: 8, 4: 5} | 98% | 98% | 83.7 | 35.9 |
| w_wait_0.0 | 45 | 1.96 | {1: 19, 2: 14, 3: 7, 4: 5} | 100% | 100% | 83.9 | 35.0 |
| w_wait_3.0 | 45 | 1.96 | {1: 19, 2: 14, 3: 7, 4: 5} | 100% | 100% | 83.9 | 35.0 |

## Reading

* Against the conventional fixed configuration (3 clusters released from the root), the proposed
  global layer improves all four core metrics on the frozen test set, meeting every pre-registered
  minimal effect (H1–H4 PASS); the local search matches the exact reference (H5 PASS).
* The decomposition matters more than the headline. **Release optimisation is the largest single
  contribution** (removing it: −8.3 pp success, +32.9 s). **Adaptive N** does not change success or time
  relative to the best fixed N with optimised release (intervals include zero) but achieves it with
  1.97 instead of 2.75 clusters and 31% less cluster-time; its benefit is resource use, not outcome.
* Topology-aware allocation, the interaction (waiting) cost and temporal coordination at execution
  have **no measurable effect** once release points are optimised: the chosen configurations already
  keep clusters apart (0/120 tasks with a violation; mean TPG hold 0.6 robot·s). The interaction cost
  removes violations in 2 of 120 tasks relative to the time-only plan.
* Remaining failures are local: 22/120 proposed tasks fail (lost clusters, timeouts), 20 successful
  tasks still touch the wall, HCC succeeds in 4/8 and weak flow is the hardest regime (28/40). This is
  the ground for the local layer (Phase D–G), which is not run yet.
* The catheter speed (2 mm/s, assumed) drives the release decisions: at 0.5 mm/s the planner releases
  from the root in almost every task. Platform measurement M3 is needed before any physical claim.
