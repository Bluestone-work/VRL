#!/bin/bash
# DAgger round 1 for the local student: same 9 training anatomies x N, 16 episodes each, beta 0.3.
# Seeds 1105000000 + k*10000 + N*1000 + 600 (disjoint from BC train/val offsets 0-23 and 500-503).
cd "/home/wj/桌面/vascular_marl_local.tar."
PY=$HOME/miniconda3/envs/v/bin/python
train=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']))")
: > research/datasets/CLUSTER_BC_V1_20261005/dagger_jobs.txt
for a in $train; do
  k=$($PY -c "import json;print(json.load(open('configs/evaluation_splits.json'))['anatomy_order'].index('$a'))")
  for n in 1 2 3; do s=$((1105000000+k*10000+n*1000+600)); echo "$a $n $s 8" >> research/datasets/CLUSTER_BC_V1_20261005/dagger_jobs.txt; echo "$a $n $((s+8)) 8" >> research/datasets/CLUSTER_BC_V1_20261005/dagger_jobs.txt; done
done
cat research/datasets/CLUSTER_BC_V1_20261005/dagger_jobs.txt | xargs -P 18 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 4-5,8-23 '$PY' scripts/collect_cluster_bc.py --anatomy $0 --clusters $1 --first-seed $2 --episodes $3 --out research/datasets/CLUSTER_BC_V1_20261005/dagger_local_r1 --student-local research/runs/BENCH_BC_20261005/local_s0/student.pt --beta 0.3 >> research/datasets/CLUSTER_BC_V1_20261005/dagger_local_r1.log 2>&1'
echo DONE >> research/datasets/CLUSTER_BC_V1_20261005/dagger_local_r1.log
