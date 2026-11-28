# Statistical analysis

Implementation: `scripts/v3/analyze_configuration.py`.

* **Unit of analysis**: base task (`base_task_hash`). All arms of an experiment run on the same base
  tasks; comparisons are paired by base task.
* **Intervals**: two-level bootstrap, 10,000 replicates (4,000 in figures): resample territories with
  replacement, then tasks within each sampled territory. This respects that tasks within a territory
  share geometry.
* **Binary metrics**: additionally an exact two-sided McNemar test on discordant pairs.
* **Verdicts** (pre-registered in `EXPERIMENT_PLAN.md`): PASS = 95% interval of the paired difference
  excludes zero in the hypothesised direction and the point estimate reaches the minimal meaningful
  effect; FAIL = interval excludes zero in the opposite direction; otherwise INCONCLUSIVE.
* **Repeats**: Phase B uses one execution per (task, arm). The pipeline is deterministic given the task
  seed, so a repeat reproduces the row; variation comes from tasks. Learning phases will add training
  seeds (≥ 5) as a further level (seeds nested in arm, crossed with tasks).
* **Multiplicity**: four primary hypotheses (H1–H4) plus the planner check H5; secondary ablation
  comparisons are descriptive and carry no verdict.
* **Selection**: the fixed-N baselines' N was chosen on the 45 validation tasks before the test run
  (`CONFIGURATION_RESULTS/validation_selection.json`). The objective weights and ε were registered
  before validation and not changed.
