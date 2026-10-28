#!/bin/bash
# HCC paired-seed expansion: 12 repeats x 4 difficulties x 9 methods = 432 episodes.
# Four shards cover disjoint repeat blocks (offset 0/3/6/9); every method inside a
# shard sees the identical scene seed, so the comparison stays paired.
set -u
ROOT="/home/wj/桌面/vascular_marl_local.tar."
OUT="$ROOT/research/runs/HCC_SEEDS12_20261026"
PY=/home/wj/miniconda3/envs/v/bin/python
METHODS=ours_hcc,online,pursuit,pac,mppi,cbf,turbo,stpg,IPPO_S41
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
pids=()
for off in 0 3 6 9; do
  "$PY" scripts/evaluate_hcc_navigation_matrix.py \
    --out "$OUT/shard_off${off}.json" --repeats 3 --repeat-offset "$off" \
    --methods "$METHODS" --ippo-root research/runs \
    > "$OUT/shard_off${off}.log" 2>&1 &
  pids+=($!)
  echo "launched shard offset=$off pid=${pids[-1]}"
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "shard $i ok"; else echo "shard $i FAILED rc=$?"; fail=1; fi
done
echo "ALL SHARDS DONE fail=$fail"
exit $fail
