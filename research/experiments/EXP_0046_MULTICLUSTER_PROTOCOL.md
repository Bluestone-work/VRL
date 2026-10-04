# EXP_0046: multi-cluster parallel thrombus removal

**Status:** registered development / feasibility protocol (2026-10-04). No
sealed split, clinical claim, or hardware claim is authorized by this document.

## Question

Can independently actuated magnetic clusters clear multiple thrombi in
parallel while a local spacing protocol prevents command-field conflict? The
primary comparison is method 2 against method 1, with the same vessel and
lysis model:

* **Method 1, sequential baseline:** `N=1`; one cluster visits a fixed
  pre-operative clot order and uses the existing route + avoid + wait teacher.
  This is the conventional, privileged planning baseline.
* **Method 2, parallel controller:** `N=2` and `N=3`; one controlled entity is
  one cluster, each follows its nearest clot from the fair partial observation,
  and a short-horizon right-of-way shield enforces `d_min`.

The parallel observation contains noisy ego motion, straight-line clot vectors,
local lumen paths, nearby particles and nearby teammates. It does not read a
route table, geodesic assignment, future state, or global map. The spacing
shield is a local safety abstraction; it is not yet a calibrated magnetic
field model.

## Registered factors

* `d_min = {1, 2, 4} mm`.
* `N = 1` for method 1 and `N = {2, 3}` for method 2.
* Three controller seeds, fixed before each run. Development seed bases are
  separate from the sealed-test ledger.
* MCA first, then the fixed 14-anatomy diagnostic matrix if the numerical
  integration gate passes.
* The reference physical environment is used until variable-`N` behavior is
  certified in the compiled kernel.

## Outcomes

Primary: Safe Success (all clots removed and total wall contact `< 1`
robot-second). Secondary outcomes are raw task success, clot removal fraction,
elapsed time, wall contact, particle contact/collision, minimum observed
cluster spacing, spacing-violation control steps, yield events, deadlock
events, and active robot loss.

The main comparison is paired by seed and scene. Report mean, bootstrap 95%
confidence interval, and the unsafe-success gap (`raw - safe`). Do not select
`d_min`, scene, or checkpoint using the sealed split.

## Feasibility gate and limitations

Before formal episodes, run short-horizon smoke tests at `0.5`, `1`, `2`, and
`5 s`. A run is blocked if the integrator produces a native crash, invalid
state, or an active body outside the lumen. The current runner therefore
defaults to a 5-second development horizon. Long-horizon runs are not valid
until the boundary-exit crash in the reference transport is repaired and
covered by a regression test.

The controller includes a local junction-stall guard: repeated edge returns or
25 consecutive frames on one edge temporarily issue a zero command. This is a
numerical robustness measure, not a performance feature, and it does not make
long-horizon runs valid by itself.

The simulator currently has no magnetic dipole or coil-array coupling. The
minimum-spacing rule is an operational surrogate motivated by the expected
distance decay of local magnetic fields; it must be replaced or calibrated
with measured field-interference curves before a hardware or physical
independence claim. A fair single-cluster controller using the same local
observation is an additional sensitivity baseline whenever the privileged
method-1 comparison would confound information access with cluster count.

## Reproducibility

The implementation is in `marl/multicluster.py` and
`scripts/evaluate_multicluster.py`. For long or repeated diagnostics, use
`scripts/evaluate_multicluster_isolated.py`, which runs one episode per child
process and records native crashes instead of losing the whole batch. Every
JSONL episode records the seed,
configuration, local-observation flag, spacing trace, wall metrics, and an
explicit `sealed_test_used=false` field. Smoke outputs belong under
`research/validation/EXP0046_MULTICLUSTER_SMOKE_20261004/` and are diagnostic
only.

## Revision 2 — 2026-10-04: corrections before new paired runs

This section supersedes the earlier implementation description. Old raw files
remain historical diagnostics and must not be pooled with revision 2.

* **Native crash:** not yet causally explained. Today's unchanged-source
  reproductions (including previously failing seeds) and native-debugger runs
  completed. This is non-reproduction, not proof of repair. No physics change
  or junction guard is credited with fixing a native crash. The previous
  edge-stall guard was removed because it reads simulator edge IDs and can
  interrupt legitimate clot dwell.
* **Local baseline:** `single_sequential` now uses the same observation as
  parallel methods, with a sticky tracked target until its clearance. It is
  not a pre-operative optimal touring solver. The route+avoid+wait baseline is
  retained separately as `single_route` and explicitly privileged. The old
  baseline's prior was applied twice; revision 2 disables the environment
  prior and executes each controller command once.
* **Observation boundary:** controller has no environment reference. It gets
  noisy ego observations, Euclidean clot vectors and tracked identities, local
  lumen features, and relative peer positions/velocities inside 6 mm. Particle
  sensing stays 1.5 mm and local lumen view stays 4 mm. Peer sensing is an
  additional declared imaging requirement (needed for the 4 mm exclusion
  experiment), not an invisible extension of the old sensor. Relative peer
  data are zeroed outside range. Multiplicative noise is 2.5%; additive centroid
  error is 0.02 mm, an engineering sensitivity, not measured hardware accuracy.
  Exact local centreline/Frenet reconstruction and clot completion sensing
  remain simulation sensor assumptions; hardware portability is unverified.
* **Shared physics:** bounded commands, same speed/radius/flow/contact law and
  180 s horizon. Same geometry, clots, flow, particles and three-position start
  pool are hashed for each scene. N=1/2 use nested starts, cyclically rotated by
  scene seed; starts have at least 4 mm separation independent of d_min. Every
  rejected initialization is logged. This compares pre-deployed clusters, not
  insertion from one common inlet.
* **Resources:** primary `fixed_total` divides catalytic rate 0.36 mass/s by N.
  Secondary `per_cluster` retains 0.36 per cluster, exposing the added-capacity
  effect. This matches a catalytic-rate proxy only: total material volume,
  magnetic power, and hardware controllability are not matched or validated.
* **Safety:** legacy Safe Success is retained. Primary cluster-safe success
  additionally requires no particle contact, all clusters retained, no robot
  pair contact and no d_min violation. Actual separation and violation pair-s
  come from accepted integrator substeps (linear interpolation with exit times),
  separate from predictions. The filter is heuristic, not a safety certificate.
  Prolonged low-speed yielding is a deadlock *proxy*, not proven deadlock.
* **Efficiency:** clearance fraction/time curve AUC; times to 50/90/100%
  clearance (null if unreached); total path; removed mass per active cluster-s;
  squared-command integral (not magnetic energy); wall/particle/pair contact.
* **Missing outcomes:** keep every requested attempt including native crashes
  and timeouts. Report completed-only aggregates explicitly as conditional and
  success bounds over all requested episodes. Any native failure blocks a
  method-performance claim; no silent rerun or removal of failed seeds.

Registered development matrix before execution: MCA, scene seeds
1302010000–1302010009, 180 s, controller/noise seeds 42/43/44 (not training
seeds). For each seed: local N=1 and parallel N=2/3 at d_min=1/2/4 mm, fixed
aggregate catalytic rate. Additional seed-42 arms: privileged N=1; unshielded
N=2/3 at d_min=2 mm; per-cluster-rate N=2/3 at d_min=2 mm. Total 26 cells × 10
scenes = 260 requested episodes. Frozen source hashes and the manifest must be
written before launch. Bootstrap comparisons resample scenes, averaging across
controller seeds inside each scene; do not treat repeated controller seeds as
independent anatomies. This is diagnostic evidence, not validation or sealed
results. No checkpoint selection or RL training in this batch.

### Revision-2 outcome and analysis conventions — 2026-10-04

The frozen matrix recorded all 260 attempts: 259 completed and one native
SIGSEGV in the N=3 unshielded, d=2 mm, fixed-total arm (scene 1302010005,
controller seed 42). The predeclared numerical gate failed. Completed-only
descriptive metrics and all-request success bounds are reported; method ranking
and inferential paired comparisons are withheld. Debug reproductions are
separate data and never replace this failed row.

For comparable efficiency reporting, analysis holds terminal removal through
the common 180 s horizon before normalizing AUC. Raw per-episode AUC remains
unchanged. Unreached T50/T90/T100 milestones remain null in raw data and receive
a labeled 180 s penalty only in analysis scores. These are not estimated future
completion times. The analysis code tests early-termination ranking, failure
penalties, scene-level repeat averaging and missing-outcome blocking.

The final report is `research/validation/EXP0046_V2_PAIRED_20261004/REPORT.md`.
Retrospective symptom counts and action replays are explicitly exploratory and
documented in that directory's `FAILURE_ANALYSIS.md`. No new policy is selected
or trained from this batch.
