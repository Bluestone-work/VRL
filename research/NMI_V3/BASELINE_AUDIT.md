# Baseline audit (Phase A)

Legacy numbers below come from the frozen `navigation_dataset_v1` manifest (N = 3, nested pre-deployed
starts, zero image latency for the RL arms, 1-step latency for the classical arms; see CODE_AUDIT #3).
They are kept as the **legacy benchmark** and are not mixed with V3 results.

## Held-out anatomies, 100 paired scenes (seed 41 for learned methods)

| method | navigation success | scenes with wall contact | mean wall contact (robot·s) | 90th pct. |
|---|---:|---:|---:|---:|
| IPPO | 80/100 | 12% | 7.08 | 1.07 |
| MAPPO-V2 | 80/100 | 12% | 3.21 | 1.19 |
| Pursuit | 80/100 | 12% | 3.21 | 1.19 |
| STPG | 80/100 | 12% | 3.14 | 1.19 |
| PAC-NMPC | 84/100 | 3% | 0.29 | 0.00 |

PAC-NMPC is the strongest legacy local baseline: higher held-out success and far less wall contact.
It is retained in every later local-control comparison and must not be weakened.

## Per-baseline status

| baseline | source | implementation | fidelity notes | V3 use |
|---|---|---|---|---|
| Pure pursuit + SettleGuard | in-house rule | `marl/deployable_sensing.DeployablePursuit`, `scripts/benchmark_lysis.SettleGuard` | deterministic; same wall projection and spacing shield as every arm | fixed local controller of Phase B; local baseline later |
| Adaptive pursuit | in-house | `AdaptiveSettleGuard` | releases the settle slow-down when progress stalls | Phase G local baseline |
| PAC-NMPC (Gonzales et al., CASE 2025, CODE-CITED) | adapted, no author code | `marl/lysis_baselines.PacNMPC`: horizon 10 steps, 96 samples, 3 cross-entropy iterations, Hoeffding PAC bounds on cost and violation, shared plans between clusters | the time-varying LQR feedback term of the paper is replaced by a cross-entropy update (docstring) | keep; Phase G must run it on the V3 tasks with the same deployment and latency |
| Conventional NMPC | not implemented | — | needed by brief §9 | NOT RUN |
| Switchable TPG (Jiang et al., AAAI 2025, CODE-CITED) | exhaustive switchable search for N ≤ 3 | `marl/lysis_baselines.SwitchableTPG` | on the legacy manifest it is identical to pursuit in 280/280 scenes because few conflict zones are active | Phase C |
| CCBS (Andreychuk et al., CODE-CITED) | fixed-route precedence search | `marl/ccbs_adapter.py` | cannot change spatial routes; equal to TPG on the six three-lesion scenes | Phase C |
| MAPPO-V2 | Yu et al. (L-VERIFIED) | `scripts/train_sb3_ippo.py` with centralised critic | 50k physical steps; 500k runs incomplete | Phase G |
| IPPO (original) | de Witt et al. (L-VERIFIED) | `scripts/train_sb3_ippo.py`, K = 13 semantic candidates | fixed-size neighbour branch: a checkpoint only runs at the N it was trained for; picks the classical route candidate in 91.5% of replay steps | Phase D replaces it with a variable-size policy |
| CSN adapter, Yang hierarchical adapter | uncited in code | `scripts/evaluate_classical_manifest.py` | sources not identified | excluded until verified |
| mr-nav (Sun et al.) | author code, BSD-3 | `scripts/evaluate_mr_nav_2d.py` | 2-D single robot on the authors' maps only | Extended Data only |

## Global-layer baselines added in V3 (Phase B)

Fixed N with root release, random feasible release and optimised release; adaptive N with root
release; time-only (interaction-unaware) joint planning; exact reference; Euclidean-cost allocation;
no temporal coordination. Definitions: `EXPERIMENT_PLAN.md`.
