# Fixed hard-condition comparison

The panel was fixed before these baseline evaluations: nominal flow/delay/variation
(.05, 1, 0), (.05, 3, 0), (.1, 2, .625), (.05, 2, 1.25).
Each cell contains all 14 anatomies and N=1/2/3, using exact jobs from the
EXP0073 nav_tf_v3_s7101 manifest. Learned outputs are reused from that experiment;
there is no retraining or checkpoint selection here. Panel contrasts are not
controlled interventions on flow alone because the original protocol assigns
different seeds to different conditions. Method contrasts within a panel are paired.

No parameter tuning was performed. Classical controllers retain their state for
the entire episode. Geometry, image sensing, physics, allocation and metrics follow
the shared LysisEpisode pipeline. No oracle is evaluated.

## Paper-related implementations

- PAC-NMPC-style: repository `marl/lysis_baselines.py:PacNMPC`, citing Gonzales et al.,
  *Multi-Agent Feedback Motion Planning using Probably Approximately Correct
  Nonlinear Model Predictive Control*, CASE 2025,
  https://doi.org/10.1109/CASE58245.2025.11164063 . The repository uses CEM sampling,
  a first-order estimated-drift model, no time-varying LQR term, and sample-based
  Hoeffding-style penalties. This is a simplified adaptation, not author-code
  reproduction or a validated PAC safety guarantee. Defaults H=10, K=96, iterations=3.
  The registered wrapper replaces TPG and local control; the common spacing shield
  remains. Thus this is a system-level baseline, not a single-module ablation.
- Switchable TPG / STPG-objective adaptation: repository `SwitchableTPG`, citing
  Jiang et al., *Speedup Techniques for Switchable Temporal Plan Graph Optimization*,
  AAAI 2025, https://doi.org/10.1609/aaai.v39i22.34487 . It enumerates precedence
  choices with its own completion-time predictor instead of reproducing IGSES.
  Fixed Settle and the wall guard remain; update period 2 s, max_free=10.
  Do not interpret weak or null differences as disproving the source paper.

Methods use image estimates and preoperative map information; local true flow,
true response and true robot position are not controller inputs. The previously
trained flow_aux does use privileged simulation supervision in training and is
not the originally requested observable-only auxiliary predictor.

## Robust execution and reporting

Each episode runs in its own process. Native crashes cannot silently lose a pool
job. A 900 s wall-clock timeout is an infrastructure failure, not a clinical/task
failure. All such failures are retained in failures.jsonl. Successful raw episodes
are retained in episodes.jsonl. Tables require a common completed scene set, and
scenario hashes are checked across all methods. Infrastructure missingness must
be reported if present. The existing EXP0073 crash is outside this four-cell panel.

Metrics preserve the benchmark definitions. T90_300 includes unreached episodes at
300 s; AUC includes all episodes. Conditional T90 and its denominator are separate.
Confidence intervals resample whole anatomies with all their N cases. These
exploratory intervals do not estimate variability across learned training seeds.
aggregate.json also stratifies by anatomy, topology split and N.

Variation jointly changes blood flow, pulsatility, actuation gain/direction/lag,
wall interaction, execution noise and lysis rate; it is not a gain-only experiment.
Nominal flow .1 was included in EXP0073 training and is not out-of-range testing.
These are four selected development conditions, not the full 1638-condition grid
or sealed testing. No claim of broad or stable superiority is warranted.

## Reproduction

From the repository root, using the existing `v` environment:

```bash
python -m scripts.evaluate_hard_baselines --out research/validation/EXP0073_HARD_BASELINES_20261009 --workers 20
python -m scripts.report_hard_baselines --out research/validation/EXP0073_HARD_BASELINES_20261009
```

Use a fresh output directory when rerunning; the evaluator refuses to overwrite.
Source hashes and all jobs are recorded in manifest.json.
