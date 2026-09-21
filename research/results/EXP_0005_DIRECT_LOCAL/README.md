# EXP_0005_DIRECT_LOCAL Results

These compact artifacts are generated from the three 1M-transition Direct Local checkpoints and the matched EXP_0004 prospective validation manifest. The sealed test split was not opened.

- `direct_local_summary.json`: 420 Direct Local episodes, per-seed and per-territory metrics, failure labels, and flow bins.
- `flow_guided_baseline_summary.json`: existing `geodesic_v` flow-guided checkpoints evaluated on the same 420 records.
- `comparison.{json,csv,md}`: three-seed mean/sample-SD comparison with absolute and relative differences.
- `direct_local_per_scenario.csv`: Direct Local territory breakdown.
- `flow_action_diagnostics.png`: flow magnitude and action/flow alignment diagnostics.

The full training checkpoints, transition shards, and per-episode NPZ traces remain under the local ignored path `research/runs/EXP_0005_DIRECT_LOCAL/`.
