# EXP0046 v2 retrospective failure diagnostics

The numerical gate failed (one native crash in the full matrix). These are symptom counts for debugging, not a performance ranking. Counts overlap. Each listed arm has 30 completed episodes but only 10 independent scenes; controller seeds 42/43/44 repeat each scene.

| Arm, fixed total catalytic rate | Raw completions | Cluster-safe completions | Wall threshold failures | Separation violations | Late stagnation proxy |
|---|---:|---:|---:|---:|---:|
| single_sequential, N=1, d=2 mm | 5/30 | 5/30 | 3/30 | 0/30 | 22/30 |
| multi_parallel, N=2, d=2 mm | 9/30 | 9/30 | 6/30 | 4/30 | 21/30 |
| multi_parallel, N=3, d=2 mm | 13/30 | 5/30 | 19/30 | 10/30 | 15/30 |

No particle contacts or cluster losses occur in these three arms. Wall/spacing violations and incomplete progress therefore deserve separate investigation. Stagnation means <0.001 removal gain over at least the final 60 s of an uncleared episode; it can reflect inaccessible targets, local navigation traps, waiting, or other causes. The trace does not establish which one. Prolonged-yield counters also do not prove deadlock.

## Two selected action replays

Selection is retrospective: the largest wall-contact case and a second case whose minimum separation remained >6 mm, from N=3, d=2 mm, controller seed 42. The read-only action tap reproduces each original final-state hash exactly.

- Scene 1302010003: 24.3785 cluster-s wall contact. Of 248 wall-contact agent-steps, 244 also changed the nominal command through the spacing filter; 224 changed a nominal outward component <=0.05 to >0.05 in the measured radial direction. This motivates joint wall/separation constraints, but does not by itself quantify their causal benefit.
- Scene 1302010005: 7.9323 cluster-s wall contact, 50% removal. The spacing filter never changed a command; all 165 wall-contact agent-steps already had a nominal outward component >0.05. Improving coordination alone cannot explain away this navigation symptom.

Replay evidence: `../EXP0046_REPAIR_20261004/trace_seed1302010003/summary.json` and `../EXP0046_REPAIR_20261004/trace_seed1302010005/summary.json`; complete measured commands, clearances and evaluation-only contact increments are in their `trace.jsonl` files.

## Next study decisions

1. Keep the native fault open and preserve process-level failure accounting. No retry may overwrite the failed matrix row. Obtain a native stack/core if it recurs before asserting a repair.
2. Prototype joint wall/separation action constraints using the same local observation. Test contradictory constraints, image noise and actuator saturation explicitly; report infeasibility instead of calling a stop command a safety guarantee.
3. Improve the shared local navigation baseline separately (including N=1). Then compare coordination changes at fixed navigation, resources, starts, and sensing on fresh scenes. Any advantage that disappears under this stronger baseline is not a coordination contribution.
4. Before learning, compare a conventional scheduling/memory baseline and identify actual conflict states from trajectories. Do not infer a need for RL merely from a timeout counter.
5. Keep measured magnetic coupling, imaging error and deployment feasibility as explicit physical gates. No magnetic-field independence claim is supported by separation-only simulation.
