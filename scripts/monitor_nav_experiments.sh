#!/usr/bin/env bash
set -u
repo="$(cd "$(dirname "$0")/.." && pwd)"
py=/home/wj/miniconda3/envs/v/bin/python
cd "$repo"
base=research/runs/EXP0081_MULTI_SEED_AUX_20261010
for name in nav_tf_v3_s7102 nav_tf_v3_s7103 belief_s7102 belief_s7103; do
  run="$base/$name"
  # policy.pt is saved at early milestones; wait for the registered 500-update
  # endpoint and for the corresponding trainer to exit before evaluating.
  while true; do
    it="$($py -c 'import json,sys; p=sys.argv[1];
try:
 r=json.loads(open(p).read().splitlines()[-1]); print(r.get("it",0))
except Exception: print(0)' "$run/log.jsonl" 2>/dev/null || echo 0)"
    running="$(pgrep -f "train_lysis_nav --out $run" || true)"
    if [ "${it:-0}" -ge 500 ] && [ -z "$running" ]; then break; fi
    sleep 60
  done
  out="$run/v5_eval_full_final"
  if [ ! -f "$out/aggregate.json" ]; then
    "$py" -m scripts.evaluate_nav_v5_decomposition --policy "$run/policy.pt" --out "$out" --variants full --workers 18
    "$py" -m scripts.report_nav_decomposition --root "$out"
    "$py" -m scripts.plot_nav_decomposition --root "$out"
  fi
done
echo finished
