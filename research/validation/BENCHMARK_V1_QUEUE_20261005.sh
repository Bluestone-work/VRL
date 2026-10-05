#!/bin/bash
# Queue: (1) Codex HRL eval finishes -> (2) pure RL PPO, 3 seeds x 150 min in parallel (6 workers each)
#        + (3) route_tpg development evaluation (N=2,3, 14 anatomies x 30 scenes) on cores 20-23.
cd "/home/wj/桌面/vascular_marl_local.tar."
until [ -f research/validation/BENCHMARK_V1_CODEX_HRL_20261005/status.txt ]; do sleep 30; done
mkdir -p research/runs/BENCH_PPO_LONG_20261005
for s in 0 1 2; do
  cpus=$(echo "0-5 8-13 14-19" | cut -d' ' -f$((s+1)))
  PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c $cpus $HOME/miniconda3/envs/v/bin/python -u scripts/train_cluster_ppo.py --out research/runs/BENCH_PPO_LONG_20261005/seed$s --minutes 150 --workers 6 --seed $s --device cuda:$((s%2)) > research/runs/BENCH_PPO_LONG_20261005/seed$s.log 2>&1 &
done
: > research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_tpg.txt
while read a n s0 s1; do [ $n -gt 1 ] && echo "$a $n $s0 $s1" >> research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_tpg.txt; done < research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_tabu.txt
cat research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_tpg.txt | xargs -P 4 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 20-23 $HOME/miniconda3/envs/v/bin/python scripts/benchmark_multicluster.py --method route_tpg --clusters $1 --anatomy $0 --seeds $2:$3 --out research/validation/BENCHMARK_V1_INNOVATION_20261005/route_tpg_N$1_$0.jsonl 2>>research/validation/BENCHMARK_V1_INNOVATION_20261005/errors.log'
echo DONE >> research/validation/BENCHMARK_V1_INNOVATION_20261005/status_tpg.txt
wait
echo DONE >> research/runs/BENCH_PPO_LONG_20261005/status.txt
