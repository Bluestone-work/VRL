# EXP0046 protocol repair and native-crash audit — 2026-10-04

## Final experiment outcome

The frozen 260-episode matrix finished with **259 completed and one SIGSEGV**.
The failure was the unshielded N=3, d=2 mm, fixed-total arm, scene 1302010005,
controller seed 42. The predeclared numerical gate therefore failed; the batch
does not support a method-ranking claim. The descriptive report preserves the
failed denominator and provides missing-data bounds. Source/scene/configuration
audits found no additional mismatch.

The Python fault trace ends at `mca_physical_dynamics.py:307`, the path-length
NumPy expression inside `advance`. This identifies an active Python frame, not
the defective native instruction or causal root. One immediate gdb repeat and
a separately registered 16-attempt diagnostic stress batch (12 plain processes,
four under gdb, four workers) all completed. The 16 stress outputs have identical
final-state hashes. No failed performance row was replaced. Evidence is in
`native_repro_v2/`; the native fault remains intermittent and unresolved.

## What was established

Revision 2 implements the requested multi-cluster study with an observation-only
controller boundary, a comparable local single-cluster baseline, paired scenes,
an explicit catalytic-rate budget, independently measured separation, and a
process-isolated evaluation that retains every requested attempt.

The old statement that boundary exits or junction oscillations caused the native
SIGSEGV/SIGILL failures was not established. No native crash root cause has been
identified or fixed. Before the v2 matrix, reproductions on unchanged source completed: three 30 s
reference episodes and ten 180 s N=3 episodes under `gdb`, plus five isolated
180 s episodes in the previously problematic N=2 seed range. These are failed
reproductions of an intermittent defect, not proof that it cannot recur.

Evidence: `gdb_before.txt`, `gdb_full_before.txt`, `tests_before.txt`,
`unchanged_isolated_full.jsonl`, `unchanged_isolated_full_summary.json`.
Before-edit sources and hashes are retained in `before_source/` and
`before_hashes.json`. All older v1 raw outputs are preserved separately.

## Confirmed implementation and protocol corrections

1. `MultiClusterController` no longer has an environment handle. It consumes
   explicit sensor arrays; the policy cannot query true flow, route, edge IDs,
   or absolute simulator positions. The edge-stall guard was removed: it read
   privileged state and could disrupt legitimate clot dwell.
2. `single_sequential` is a local-observation method with sticky tracked clot
   identity. `single_route` remains a separate privileged comparator. The
   environment prior is disabled, so the old single-baseline double application
   of its teacher action is eliminated.
3. Each scene has the same geometry, clots, masses, flow, particles and nested
   start pool for every arm. Full scene hashes, accepted/rejected seeds, actual
   configurations and source hashes are stored per episode. Diagnostic resets
   reject all registered evaluation seeds before accessing their environments.
4. The primary budget divides 0.36 catalytic-rate units/s by N. Secondary arms
   retain 0.36 per cluster. This controls one resource proxy only; equal volume,
   magnetic power or physical controllability is not claimed.
5. Realized minimum spacing and violation pair-seconds come from accepted
   integrator substeps with interpolated crossings and exit timing. Predicted
   conflicts remain separate controller diagnostics. The measurement tap is
   tested to leave final physics state bit-identical.
6. Failed/invalid/timed-out child processes remain in all-request denominators.
   Any such failure blocks comparisons. Outputs use exclusive creation and
   flush each episode; failures are never silently retried or dropped.
7. Metrics include cluster-safe success, legacy safe/raw success, removal,
   T50/T90/T100 with nulls for unreached milestones, clearance AUC, wall/particle/
   pair contact, exits, path, active cluster-seconds and squared command integral.
   Analysis extends terminal removal to the common 180 s AUC horizon and reports
   unreached milestones with a clearly labeled horizon penalty, avoiding bias
   from averaging completion time only among successful episodes.

## Verification

- Broad regression: **92 passed**, covering multi-cluster behavior, reference and
  compiled dynamics, scene-graph students, partial observations and safety
  metrics (`regression_v2.txt`).
- After adding the reserved-seed-before-reset check: **27 multi-cluster tests
  passed**. The extra test was added after the broad regression.
- Final multi-cluster plus statistical-analysis checks: **32 passed**
  (`analysis_and_final_tests.txt`), including common-horizon AUC, unreached-time
  penalties, scene-level pairing/bootstrap independence, and enforcement of the
  missing-outcome gate after a native failure.
- Tests cover invisible-peer masking, clot identity after slot reordering,
  executed-command frame history, poisoned route accessors, paired scene and
  resource hashes, crossing/exit spacing integration, repeatability, and failed
  process denominators.
- Three full N=2, d=2 mm preflight episodes completed without native failure.
  Removal was 0.5, 0.5 and 1.0, but cluster-safe success was zero. One run violated
  separation (about 0.229 pair-s; minimum about 1.923 mm). This explicitly
  falsifies any claim that the current filter guarantees the requested spacing.
- Frozen paired evaluation: 26 cells × 10 scenes = 260 requested episodes,
  180 s horizon, source snapshot and manifest written before launch. Final
  results and identity/hash audit are in
  `../EXP0046_V2_PAIRED_20261004/REPORT.md` and `analysis.json`. The one native
  failure blocks paired performance claims, despite passing identity checks.

## Remaining transfer and research constraints

The sensor adapter still synthesizes a local centerline frame, lumen geometry,
clot tracking/completion status and 6 mm peer imaging from simulation. Noise
(2.5% multiplicative features, 0.02 mm centroid error) is an engineering
assumption, not measured sensing accuracy. Particle sensing remains 1.5 mm and
lumen view 4 mm. These requirements must be validated on an imaging pipeline.

The simulator still assumes independently commanded actuators. Distance alone
cannot establish independent magnetic control. No calibrated field-coupling
model or hardware experiment has been added. The present comparison assumes
pre-deployed clusters, rather than common-inlet insertion.

The controller is a conventional heuristic. Prolonged yielding is only a
deadlock proxy. These experiments cannot yet establish a learning contribution,
clinical effectiveness, or a journal acceptance probability. A subsequent
method change needs a separate frozen protocol and fresh diagnostic scenes;
this batch is not a training-seed comparison or a sealed evaluation.
