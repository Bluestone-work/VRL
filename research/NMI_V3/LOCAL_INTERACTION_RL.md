# Local layer: interaction-aware reinforcement learning (design; Phases D–G NOT RUN)

Nothing in this file is a result. It records what the audit found in the existing code and the
order in which the local layer will be built and tested.

## What the existing IPPO already has (audit)

* `scripts/train_sb3_ippo.py:535-576` `IPPOExtractor`: GRU over an 11-d motion/command history, CNN on a
  4×32×32 local image, route/map context, masked mean pooling over obstacle tracks **and over
  neighbours** (`peers`, `peer_valid`), candidate descriptors, optional zero-initialised flow branch.
  The neighbour encoder is already permutation-invariant and tolerates empty neighbour sets.
* The only reason a checkpoint is tied to one N is the observation-space shape
  (`peers: Box(..., (n, 8))`, `peer_valid: (n,)`, line 235). Padding both to n_max = 4 with
  `peer_valid = 0` makes one checkpoint executable for every N ≤ 4 with no architecture change.
* Frozen action-conditioned predictor checkpoints exist (`~/桌面/VRL_clean_20261101/checkpoints/predictor/predictor_s{0,1,2}.pt`,
  code `marl/dynamic_predictor.py`) but the IPPO used in v2 did not consume them; it used the online
  least-squares estimate (`ObservedMotion`). Their auxiliary accuracy on held-out data: flow magnitude
  r = 0.913, direction cosine 0.708 (`manuscript/PREDICTOR_AUX_20261101.md` in the clean copy).
* All RL training and the frozen RL evaluation forced zero image latency (CODE_AUDIT #3).

## Planned build order

| step | change | test that must pass before the next step |
|---|---|---|
| D1 | variable-size observation (pad to n_max, mask) | the same checkpoint runs on N = 1..4 V3 tasks; outputs for N = 3 equal the fixed-size model when the padding is masked |
| D2 | executor hook `local='rl'` in `marl/v3/execute.py` (same release gate, TPG, shield, wall projection) | pursuit-candidate-only policy reproduces the pursuit arm exactly on 10 tasks |
| D3 | training distribution from the Phase B planner (configurations of the proposed arm on the dev set, N = 1..4, latency 1–3 steps, variation s ≤ 1.0) | training scenes disjoint from validation/test base tasks (hash check) |
| D4 | candidate features from the frozen predictor ensemble: μ, σ of the 1 s displacement, predicted wall clearance, predicted distance to every neighbour, predicted route progress | predictor RMSE, calibration (coverage of μ ± 2σ) and inference time measured on held-out rollouts; ensemble compared with the LS estimate |
| D5 | interaction-aware reward term: −w_sep · max(0, 1 − d_ij/(1.5·d_safe))² per pair; unnecessary-wait penalty; weights fixed on the dev set | reward sensitivity on validation only |
| E | pilots: 3 seeds × small budget for full, w/o prediction, w/o σ, w/o neighbours, w/o interaction reward, w/o history, w/o randomisation, fixed-size | decide the final architecture from validation; record which variants are dropped and why |
| F | freeze algorithm, test set, budget (≥ 5 seeds) and minimal effects | written before Phase G |
| G | full comparison on V3 tasks with the same global configuration: pure pursuit, adaptive pursuit, conventional NMPC (to implement), PAC-NMPC, original IPPO, variable-size IPPO, MAPPO, graph-IPPO, proposed | four metrics, paired by base_task_hash and configuration_hash |

## Claims that are not allowed until Phase G

* that the learned policy beats PAC-NMPC or pure pursuit on any core metric;
* that prediction or uncertainty helps (requires D4 calibration and the E ablations);
* that one checkpoint generalises across N (requires D1 + G on N = 1..4).

The legacy result stands: on the v2 manifest IPPO equals pure pursuit in success and PAC-NMPC is better
on held-out anatomies (84 vs 80 of 100, less wall contact).
