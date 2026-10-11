# EXP0100: candidate PPO with dynamic obstacles

This is a short implementation and mechanism smoke test for the first
candidate-action PPO round. The global allocation, route/TPG, frozen motion
ensemble, candidate generator, WallGuard, and spacing shield are unchanged.
The actor selects a deduplicated candidate index every control step. No
epistemic-uncertainty hard fallback is enabled; the rule command remains a
candidate and the safety layers may still alter the final command.

Dynamic obstacles are created at reset (`particle_count=8`) and then move in
the physical environment. The actor receives only sensed particle-relative
features when available; simulator particle positions are used for metrics,
never for actor input. The predictor ensemble is frozen.

Training used two workers, seed 0, 12 PPO updates, 64-step rollouts, flows
sampled from 0.025/0.05/0.1, latency 1--3, and training anatomies from the
existing split. This is a smoke run, not a stability claim or a sealed-test
evaluation. The exact command is recorded below.

```bash
/home/wj/miniconda3/envs/v/bin/python -m scripts.train_predictive_nav_rl \
  --out research/runs/EXP0100_PPO_CANDIDATE_DYNAMIC_20261011/seed0 \
  --seed 0 --updates 12 --workers 2 --rollout 64 --device cpu \
  --scene-seed-base 2130000000 --dynamic-obstacles --dynamic-obstacle-count 8
```

The paired smoke evaluation uses the same generated scenes for fixed Settle,
stateful Adaptive Settle, Switch Settle, and the PPO checkpoint. It was
interrupted after three completed episodes per method because the requested
`--limit` applies per panel rather than globally; all completed raw episodes
remain in `smoke/episodes.jsonl`.
