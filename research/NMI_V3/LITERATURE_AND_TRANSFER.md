# Literature and cross-domain transfer

**Verification status.** Web search and web fetch were blocked in the session that wrote this file
(2026-11-28). Each entry is therefore marked:

* **L-VERIFIED**: title, authors, venue and DOI read from a page or repository saved in this repository
  (path given), fetched in an earlier session;
* **CODE-CITED**: citation copied from a docstring in this repository, not re-checked against the paper;
* **UNVERIFIED**: from memory. Do not cite until author, year, venue, DOI, code and licence are checked.

## 1. Microrobot navigation with learning

| Work | Status | What we take | What we do not claim |
|---|---|---|---|
| An X., Luo S., Zhang H., Yang Q., Ma Y., Wang B., Du J., Wang Q. *Autonomous navigation of intelligent microrobotic swarms in unknown environments.* Nat. Mach. Intell. 8(6) (2026), doi:10.1038/s42256-026-01252-6 ("Turbo"; Southeast University group of Q. Wang) | L-VERIFIED: `research/literature/VCTPG_COMPARATORS_20261004/turbo_fig2.html` (Fig. 2 caption: randomised training simulator, three-axis Helmholtz coils, transformer-based PPO, visual detection; comparison against GTrXL/GRU/MLP; obstacle number and velocity sweeps) | Narrative and validation pattern: a randomised simulator, an architecture comparison, sweeps over disturbance strength, then physical trials. Turbo controls one swarm in an unknown environment; our setting is several clusters on a known vessel map. | No figure, code or result of Turbo is reused. Turbo was not re-run here; the archived `turbo` HCC baseline (`scripts/evaluate_hcc_navigation_matrix.py:44`) is a local re-implementation whose fidelity is not established. |
| Medany M. et al. *Model-based reinforcement learning for ultrasound-driven autonomous microrobots.* Nat. Mach. Intell. (2025), doi:10.1038/s42256-025-01054-2 | L-VERIFIED: `research/literature/Medany2025_ultrasound_mbrl/sources.json` (author repository listed there; environment code not audited) | Model-based RL with a learned dynamics model as context for the action-conditioned predictor of Phase D. | — |
| Sun Y., Zhu A., Ji X., Li Y., Zhao J., Wang Y., Zhang L., Gao H., Yang L. *Minute-scale training for microrobot navigation* (code: github.com/yinghansun/mr-nav, commit c129da14, BSD-3-Clause) | L-VERIFIED for authors, code and licence: `research/mr_nav_adapter_manifest.json`, `/home/wj/mr-nav-upstream/README.md`. Venue UNVERIFIED. | 2-D single-robot vessel navigation with virtual LiDAR; already evaluated on its own maps (`research/mr_nav_2d_paired_20261127.json`). | Not a multi-cluster or 3-D result. |
| Yang L. et al., microrobot swarm navigation with real-time distribution planning, Nat. Mach. Intell. (2022) | UNVERIFIED | Possible source of the `yang_hier` adapter in `scripts/evaluate_classical_manifest.py`, which cites no paper. | `yang_hier` stays out of every figure until its source is confirmed. |

## 2. Multi-agent reinforcement learning

| Work | Status |
|---|---|
| de Witt C. S. et al. *Is independent learning all you need in the StarCraft multi-agent challenge?* arXiv:2011.09533 (IPPO) | L-VERIFIED: `research/literature/VCTPG_MARL_20261004/ippo.html` |
| Yu C. et al. *The surprising effectiveness of PPO in cooperative multi-agent games* arXiv:2103.01955 (MAPPO) | L-VERIFIED: `research/literature/VCTPG_MARL_20261004/mappo.html` |
| Wen M. et al. *Multi-agent reinforcement learning is a sequence modeling problem* arXiv:2205.14953 (MAT) | L-VERIFIED: `research/literature/VCTPG_MARL_20261004/mat.html` |
| Graph-based MARL / Deep Sets neighbour encoders; GCBF+ (graph control barrier functions, Zhang S. et al., IEEE T-RO 2025) | UNVERIFIED |
| Domain randomisation: Tobin J. et al. IROS 2017; Peng X. B. et al. ICRA 2018 | UNVERIFIED |

## 3. Multi-robot planning and coordination

| Work | Status | Used in |
|---|---|---|
| Jiang H., Lin, Li J. *Speedup techniques for switchable temporal plan graph optimization.* AAAI 39 (2025), doi:10.1609/aaai.v39i22.34487 (STPG; builds on Berndt A. et al., switchable action dependency graphs) | CODE-CITED: `marl/lysis_baselines.py:14` | `SwitchableTPG` baseline |
| Gonzales et al. *Multi-agent feedback motion planning using probably approximately correct nonlinear model predictive control.* IEEE CASE (2025), doi:10.1109/CASE58245.2025.11164063, arXiv:2501.12234 | CODE-CITED: `marl/lysis_baselines.py:3` | `PacNMPC` baseline (adapted, no author code) |
| Andreychuk A. et al. *Multi-agent pathfinding with continuous time* (IJCAI 2019; AIJ 2022) | CODE-CITED: `research/COMPARISON_ALGORITHMS.md` | `marl/ccbs_adapter.py` (fixed-route precedence search, not the authors' implementation) |
| Hönig W. et al. *Multi-agent path finding with kinematic constraints*, ICAPS 2016 (temporal plan graph) | UNVERIFIED | `marl/tpg_coordinator.py` |
| Bidirectional TPG (BTPG / BTPG-max) | UNVERIFIED (author list and venue not checked) | Phase C candidate |
| Kuhn H. W. Hungarian method (1955); consensus-based bundle algorithm (Choi, Brunet, How, T-RO 2009) | UNVERIFIED | allocation background |
| Fleet-size optimisation, facility location, multi-depot routing | UNVERIFIED (no specific paper chosen) | framing of N and release selection |

## 4. Transfer table

| Macro-scale idea | New constraint in the vasculature | V3 adaptation | Code | Validating experiment |
|---|---|---|---|---|
| Fleet sizing (choose how many vehicles) | Every extra cluster costs a catheter release and one more independently actuated channel; n_max is set by the actuation system | N chosen by the ε-rule: smallest N whose best J is within 10% of the best over N ≤ n_max | `marl/v3/configuration.py::Planner.solve` | Phase B arms fixedN_* vs proposed; N distribution |
| Depot / facility location | Clusters can only be released where a catheter can reach (lumen ≥ 0.6 mm along the root path); deployment takes catheter travel time | Release stations = root + accessible bifurcations on the root path of the cluster's first target; catheter time inside the mission | `marl/v3/vessel_graph.py::release_candidates`, `Planner.release_times` | fixedN_root vs fixedN_opt; adaptiveN_root vs proposed; fixedN_random |
| Multi-robot task allocation / multi-depot routing | Motion is confined to a tree: routes are unique, upstream travel is slowed by flow, backtracking is expensive | Ordered allocation on geodesic, flow-dependent edge times; exact best-first search for m ≤ 5 and local search beyond | `Planner.exact`, `Planner.local` | H5 (gap to exact), abl_euclid |
| MAPF + temporal plan graphs | Clusters cannot pass each other within d_safe in a lumen; a held cluster drifts in flow | Prioritised schedule on the planning model (planner); release-aware TPG at execution | `Planner.simulate`, `marl/v3/execute.py::release_aware_tpg` | abl_no_tpg; Phase C scheduling baselines |
| Decentralised learned control with local sensing | Delayed image feedback, flow disturbance, wall contact is harm, shared external field | Phase D (not run yet) | — | Phases D–G |

This is transfer of algorithmic ideas; no weights, data or code from aerial-swarm systems are used.
