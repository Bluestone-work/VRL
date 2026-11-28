# Acceptance report (status after Phase A and Phase B, 2026-11-28)

Verdict scale: PASS / FAIL / INCONCLUSIVE / NOT RUN. Every verdict refers to simulation unless stated.
Evidence: `CONFIGURATION_RESULTS/REPORT.md` (frozen test set, 120 base tasks, 9 arms, fixed local controller).

## Phase status

| phase | content | status |
|---|---|---|
| A | literature (local evidence only; web blocked), code audit, protocol fixes, four-metric evaluator | DONE — `LITERATURE_AND_TRANSFER.md`, `CODE_AUDIT.md`, `FOUR_METRIC_DEFINITIONS.md`, `marl/v3/metrics.py` |
| B | graph configuration (N, release, allocation, schedule), exact reference, configuration arms | DONE — validation (45) + frozen test (120) |
| C | scheduling baselines (STPG, CCBS, BTPG-max, uncertainty-aware), high-conflict topologies, station keeping during holds | NOT RUN |
| D–G | variable-size interaction-aware RL, predictor, ablations, multi-seed comparison incl. PAC-NMPC | NOT RUN (design in `LOCAL_INTERACTION_RL.md`) |
| H | Fig. 1–6 and Extended Data | Fig. 2 drafted from Phase B (`research/figures/NMI_v3/`); others NOT RUN |
| I | physical magnetic validation | NOT RUN — protocol only (`MAGNETIC_VALIDATION/PROTOCOL.md`) |

## The twelve questions

| # | question | verdict | evidence and limits |
|---|---|---|---|
| 1 | Can the global layer choose the number of clusters autonomously? | **PASS** (mechanism) · **INCONCLUSIVE** (outcome benefit over the best fixed N) | N* is computed for every task (1: 45, 2: 42, 3: 25, 4: 8 tasks); local search and exact search agree on N* in 98.3% of tasks. Against fixed N = 3 with optimised release, success −2.5 pp [−8.3, +2.5] and time +6.1 s [−3.0, +17.0] for the fixed arm: no significant outcome difference, but the adaptive planner uses 1.97 vs 2.75 clusters and 163 vs 237 cluster-seconds. |
| 2 | Does it choose feasible deployment positions sensibly? | **PASS** | Release optimisation is the largest single effect: adaptive N with root release vs proposed −8.3 pp success [−18.3, −0.8], +32.9 s [+18.5, +49.1]; random feasible release vs proposed −11.7 pp, +32.9 s. Positions depend on the assumed catheter speed (sensitivity table). |
| 3 | Does it reduce mission time and potential conflicts? | **PASS** vs the fixed root-release configuration | −54.7 s [−79.2, −32.5] (H2); tasks with an executed violation 25 → 0 (H3); planned potential-conflict tasks 5.0% → 1.7%. Against the best fixed-N optimised configuration: INCONCLUSIVE. |
| 4 | Does local RL handle flow and dynamic interaction? | **NOT RUN** | Phase D–G. Legacy evidence: IPPO equals pure pursuit on success and picks the classical candidate in 91.5% of steps. |
| 5 | Are executed spacing violations reduced? | **PASS** (global layer) · NOT RUN (local layer) | 0/120 tasks with a violation (proposed) vs 25/120 (fixed root). Temporal coordination at execution adds nothing measurable once release is optimised (abl_no_tpg: 0/120). |
| 6 | Is safety improved? | **PASS** vs fixed root (H4: +20.0 pp contact-free success) · **INCONCLUSIVE** vs fixed N optimised (−5.8 pp [−13.3, +0.8] for the fixed arm) | 20 of 98 successful proposed tasks still have wall contact; safety beyond the global layer depends on the local controller. |
| 7 | Is task success improved? | **PASS** vs fixed root (H1: +23.3 pp) · INCONCLUSIVE vs fixed N optimised | 22/120 proposed tasks still fail; HCC 4/8; weak flow 28/40. |
| 8 | Is navigation / delivery-proxy efficiency improved? | **PASS** vs fixed root · INCONCLUSIVE vs fixed N optimised | Arrival time is a proxy; drug release and dose are not modelled. Planned vs executed arrival: r = 0.41. |
| 9 | Does every module have an independent ablation? | **PASS** for the global layer · NOT RUN for the local layer and the safety layer | Global ablations: adaptive N (resource effect only), release optimisation (large), topology-aware allocation (no effect), interaction cost (2/120 tasks), temporal coordination (no effect). Safety-layer ablations (w/o shield, w/o wall projection) NOT RUN. |
| 10 | Can one policy serve different N? | **NOT RUN** | The global layer and executor already run N = 1–4 on identical base tasks; the RL checkpoint does not yet (fixed-width neighbour input). |
| 11 | Which results have a physical magnetic basis? | **None** | No platform data. n_max = 4, d_safe = 2 mm, catheter speed 2 mm/s, release interval 3 s and the actuation variation are assumptions (`configs/v3/assumptions.json`). |
| 12 | Which results remain simulation-only or hypotheses? | — | All of Phase B. In particular: that 2 mm separation prevents magnetic interference; that clusters can be steered independently; the catheter model; flow and actuation models; drug delivery itself. |

## Negative and null results kept on record

* Topology-aware (geodesic/flow) allocation is not better than Euclidean-cost allocation on these
  trees (Δ ≤ 1 pp, ≤ 0.5 s): routes on a tree are unique and the two cost models rank allocations
  almost identically.
* The waiting (interaction) cost in the objective changes the configuration in few tasks; the
  time-only plan has the same success and time.
* Execution-time TPG has no measurable effect under the proposed configuration.
* Adaptive N gives no outcome gain over a well-chosen fixed N; the gain is fewer clusters.
* First validation run (`*_v0_deadlock_bug.*`) used a planner that treated catheter transit > 60 s as a
  deadlock; it was discarded before the test run and kept for the record. A later fix (catheter speed
  read from the planner instance) does not change results at the registered speed; re-execution of
  four test rows reproduced their configuration hashes and metrics exactly.

## What the next phase must establish

The global layer's effect is now measured. The open scientific question is the local layer: whether a
variable-size, interaction-aware learned policy reduces the remaining failures (lost clusters, wall
contact, weak-flow timeouts) relative to pure pursuit **and PAC-NMPC** on these same base tasks and
configurations, with image latency > 0 and ≥ 5 training seeds.
