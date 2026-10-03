# EXP_TEACHER_BC / EXP_ONLINE_IMITATION (round 1) — 2026-10-03

Branch `research/graph-teacher-distillation`, code commit dc42d2e (+ trainer schedule fix in this commit).
Diagnostic split only (20 layouts × 14 anatomies, 5 robots). No validation selection, no sealed test: these are development results, not claims.

## Question
Can a Graph Transformer student, reading only the vessel topology / physical state scene graph, learn the structured navigation of the route+avoid+wait teacher, and does it generalise to held-out anatomies?

## Setup
- Teacher: route + avoid + wait (EXP_0044_WAIT_PRIOR_DYNAMICS, residual 0, stop deadzone 0.35). Training only; never in the student's inference path (`tests/test_scene_graph_student.py` makes route/allocation accessors raise while the scene graph is built; replaying teacher labels reproduces the teacher trajectory exactly).
- Student: edge-aware Graph Transformer, 4 layers, dim 128, 4 heads, 610K parameters. Nodes: vessel stations (≈2 mm), robots, clots, particles; relations: relative position/velocity in the receiver's Frenet frame, closest approach, geodesic distance, vessel adjacency, downstream/upstream, type pair. Output: Frenet direction + stop logit (+ auxiliary subgoal head over clots).
- Anatomy split `anatomy_holdout_v1` (registered before training): 9 train, 5 held-out (coronary_rca, basilar_vertebral, cerebral_venous_sinus, sma_embolism, femoropopliteal_pad).
- Dataset TEACHER_BC_V1: 540 train episodes (198,759 scenes, 945,250 robot samples, robots ∈ {4,5,6}, stop 0.97%) + 90 val episodes; DAgger round 1: 270 episodes (102,429 scenes), β = 0.5, student = BC epoch 7.
- BC: 8 epochs, AdamW 3e-4 one-cycle, batch 64, seed 0. DAgger r1 and its control both start from BC epoch 7, lr 1e-4, 14,121 optimizer steps each, identical schedule; only the data differ.

## Results (complete / collision-free %, mean over anatomies)
| policy | train (9) | held-out (5) | wall contact s train / held-out | episode s train / held-out | self-rollout cos to teacher train / held-out |
|---|---|---|---|---|---|
| teacher | 98.9 / 98.9 | 96.0 / 96.0 | 0.3 / 0.0 | 35.0 / 46.1 | — |
| pure RL EXP40, seeds 42/43/44 (MCA-trained GAT) | 32.8 / 21.1 / 30.0 complete | 34.0 / 16.0 / 31.0 complete | 10.8 / 46.4 (s42) | 138 / 135 (s42) | 0.17 / 0.17 |
| BC epoch 0 | 78.3 / 75.0 | 80.0 / 69.0 | 5.7 / 19.9 | 71.0 / 73.2 | 0.62 / 0.57 |
| **BC epoch 7** | **97.8 / 97.2** | **91.0 / 90.0** | 4.6 / 12.3 | 43.5 / 55.2 | 0.88 / 0.81 |
| BC continued (control) | 96.1 / 96.1 | 91.0 / 90.0 | 1.1 / 15.7 | 43.3 / 56.8 | 0.90 / 0.82 |
| DAgger round 1 | 97.8 / 97.2 | 89.0 / 88.0 | 0.9 / 10.4 | 38.9 / 55.0 | 0.92 / 0.84 |

Per-anatomy table: summary_table.md and `summarize.py`.

Paired per layout, DAgger r1 − control (bootstrap 95% CI):
- train: complete +1.7 pp [−1.7, +5.0], episode length −4.4 s [−8.3, −0.6], self-rollout cos +0.025 [+0.010, +0.040]
- held-out: complete −2.0 pp [−6.0, +2.0], collision-free −2.0 pp [−7.0, +3.0], wall contact −5.3 s [−16.9, +5.4]

## Interpretation
1. The student learns the teacher's structured navigation from topology alone: 97.8% on training anatomies vs 98.9% for the teacher, and 91.0% zero-shot on held-out anatomies where pure RL reaches 34%.
2. The epoch-0 gap was mostly under-training, not distribution shift: offline cosine moved only 0.980 → 0.988, but closed-loop completion moved +20 pp. Offline imitation metrics are not a usable selection signal; closed-loop rollouts are.
3. DAgger round 1 is a null result on completion. It makes the student track the teacher more closely on its own states (cos +0.025, CI excludes 0) and shortens train episodes, but completion differences are inside noise on both sets. Not reported as a gain.
4. The remaining held-out gap is concentrated in femoropopliteal_pad (75–85 vs 100) and sma_embolism (75–80 vs 85), with large wall contact. These are the longest/largest anatomies; none of the training anatomies is as long.

## Failure modes
- Wall contact on held-out anatomies 10–16 s vs 0.01 s for the teacher.
- Stop under-learning: offline stop recall 27–28% at stop frequency ~1%.
- Out-of-range scale: femoropopliteal_pad (241 mm extent; largest training anatomy iliac 181 mm).
- Single seed; 20 layouts per anatomy gives ~±5 pp per-anatomy resolution.

## Not yet done
- GAT vs Graph Transformer under the same BC data (the pure-RL GAT here differs in algorithm and training anatomies; it is not that comparison).
- Multiple seeds, validation-split selection, sealed test.
