#!/bin/bash
cd "/home/wj/桌面/vascular_marl_local.tar."
until ! pgrep -f "train_cluster_ppo.py --out research/runs/BENCH_PPO_20261005/seed0" > /dev/null; do sleep 20; done
: > research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_final.txt
while read a n s0 s1; do
  echo "rl_local $a $n $s0 $s1 research/runs/BENCH_PPO_20261005/seed0/policy.pt rl_ppo_s0 research/validation/BENCHMARK_V1_LEARNING_20261005" >> research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_final.txt
  for s in 1 2; do echo "local_learned $a $n $s0 $s1 research/runs/BENCH_BC_20261005/frontier_selector_s$s.pt local_learned_s$s research/validation/BENCHMARK_V1_INNOVATION_20261005" >> research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_final.txt; done
done < research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_tabu.txt
cat research/validation/BENCHMARK_V1_INNOVATION_20261005/jobs_final.txt | xargs -P 22 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 0-5,8-23 $HOME/miniconda3/envs/v/bin/python scripts/benchmark_multicluster.py --method $0 --tag $6 --checkpoint $5 --clusters $2 --anatomy $1 --seeds $3:$4 --out $7/$6_N$2_$1.jsonl 2>>research/validation/BENCHMARK_V1_INNOVATION_20261005/errors.log'
echo DONE >> research/validation/BENCHMARK_V1_INNOVATION_20261005/status_final.txt
