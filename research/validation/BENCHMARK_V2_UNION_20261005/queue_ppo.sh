#!/bin/bash
# Step 2: pure RL (parameter-shared PPO from scratch), deployable observations, union junction model.
# 3 seeds x 150 min, starts when the v2 baseline sweep finishes.
cd "/home/wj/桌面/vascular_marl_local.tar."
until [ -f research/validation/BENCHMARK_V2_UNION_20261005/status.txt ]; do sleep 30; done
mkdir -p research/runs/BENCH_PPO_UNION_20261005
for s in 0 1 2; do
  cpus=$(echo "0-5 8-15 16-23" | cut -d' ' -f$((s+1))); w=$( [ $s = 0 ] && echo 6 || echo 8 )
  PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c $cpus $HOME/miniconda3/envs/v/bin/python -u scripts/train_cluster_ppo.py --out research/runs/BENCH_PPO_UNION_20261005/seed$s --minutes 150 --workers $w --seed $s --obs deployable --junction union --device cuda:$((s%2)) > research/runs/BENCH_PPO_UNION_20261005/seed$s.log 2>&1 &
done
wait; echo DONE >> research/runs/BENCH_PPO_UNION_20261005/status.txt
