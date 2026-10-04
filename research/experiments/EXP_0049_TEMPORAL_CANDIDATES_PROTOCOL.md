# EXP0049 — causal history and executable-candidate scoring

Registered 2026-10-04 after EXP0048 32k validation, before this model is trained.
EXP0048 64k runs continue as an unchanged-source budget check.

## Hypothesis and attribution

The EXP0048 actor chooses fixed maneuver indices without explicitly seeing
what the shared projection will execute. Delayed, noisy measurements also
leave single-frame velocity/action estimates temporally misaligned. We test
whether a learned six-frame GRU and shared candidate scorer improve the
safety/clearing tradeoff. Candidate descriptors are actual proposed actuator
commands, projection residual, goal alignment, command norm, a nominal-action
indicator, approximate measured-peer clearance, and radial command component.
There is no true flow, route, state, future observation, or exact mass input.

The value network sees the same measurements as the actor. PPO, the reward,
physics, tracked sensors, action set, projection and observed completion rule
are held fixed. This is a mechanism experiment, not an established novelty
claim. A gain cannot be attributed separately to recurrence and candidate
scoring without further ablations.

## Controls and outcomes

- Re-run joint, memory, old spacing and N=1 joint conventional controls on
  fresh paired validation scenes. Conventional outputs must be bit-identical
  to EXP0048 when run on the same preflight scene.
- Also run the EXP0048 64k learned models on these same fresh validation scenes
  using their original entry point and frozen checkpoint hashes. Preserve their
  distinct model source hashes and explicit common sensing/physics provenance.
  These rows are a separate cross-revision comparison, never silently pooled.
- Train from scratch, seeds 42/43/44, 32,768 then at most 65,536 environment
  steps per seed. Primary endpoint remains observed-and-physical cluster-safe
  success; removal/AUC, wall contact, spacing, T90, energy proxy and false
  visual completion are reported jointly. Do not claim victory from reducing
  treatment activity, dropping failures, or choosing a favorable seed.
- Training bases 1510000000 / 1510100000 / 1510200000; continuation adds 10000.
  Validation 1520000000–1520000011; confirmation 1530000000–1530000039 remains
  unopened until the method is frozen. Preflight uses 1540000000 onward.

Known limits remain: engineered sensor accuracy, ideal identity association,
synthetic geometry, independent magnetic actuators, and an unresolved native
crash observed in earlier preflight. Preserve all attempts and source snapshots.

### Prospective baseline strengthening, 2026-10-04

Before examining EXP0049 validation results, additionally register N=1 memory
navigation as a separate 12-scene arm. Method 1 must not be represented only by
the weaker joint-without-memory controller when the multicluster comparisons
include memory. This adds a comparator; it changes no existing results, sources,
training conditions or primary outcome. The same strengthening is also reported
as a separate diagnostic for EXP0048.
