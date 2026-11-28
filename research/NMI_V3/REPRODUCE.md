# Reproduce V3 Phase A/B

```bash
PY=/home/wj/miniconda3/envs/v/bin/python
cd ~/桌面/vascular_marl_local.tar.
$PY -m pytest -q tests/test_v3_phase_ab.py

O=research/NMI_V3/CONFIGURATION_RESULTS
# task sets (deterministic; hashes must match the committed files)
$PY scripts/v3/build_task_sets.py --split validation --out $O/tasks_validation.json
$PY scripts/v3/build_task_sets.py --split test       --out $O/tasks_test.json

# validation sweep -> fixed-N selection
$PY scripts/v3/run_configuration_experiments.py --tasks $O/tasks_validation.json \
  --arms fixedN1_root,fixedN2_root,fixedN3_root,fixedN4_root,fixedN1_opt,fixedN2_opt,fixedN3_opt,fixedN4_opt,proposed \
  --out $O/validation_runs.jsonl --workers 16
$PY scripts/v3/analyze_configuration.py --runs $O/validation_runs.jsonl --out $O/validation_summary.json --reference fixedN3_root

# frozen test set (N = 3 from validation_selection.json)
$PY scripts/v3/run_configuration_experiments.py --tasks $O/tasks_test.json \
  --arms fixedN_root,fixedN_random,fixedN_opt,adaptiveN_root,adaptiveN_opt_time,proposed,exact,abl_euclid,abl_no_tpg \
  --fixed-n-root 3 --fixed-n-opt 3 --out $O/test_runs.jsonl --workers 18 --timeout 2400
$PY scripts/v3/analyze_configuration.py --runs $O/test_runs.jsonl --out $O/test_summary.json --reference fixedN_root

# planner-only sensitivity (validation set)
$PY scripts/v3/planner_sensitivity.py --tasks $O/tasks_validation.json --out $O/planner_sensitivity_validation.jsonl

# figure
$PY scripts/make_nmi_figures_v3.py --out research/figures/NMI_v3 --only fig2
```

The runner is resumable: it skips (base_task_hash, arm) pairs already present without an error.
Every row stores the git commit, `assumptions_version`, `base_task_hash` and `configuration_hash`.
Runtime on 18 worker processes: validation ≈ 35 min, test ≈ 2 h.
