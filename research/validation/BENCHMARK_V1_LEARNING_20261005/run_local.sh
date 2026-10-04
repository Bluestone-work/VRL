#!/bin/bash
cd "/home/wj/桌面/vascular_marl_local.tar."
cat research/validation/BENCHMARK_V1_LEARNING_20261005/jobs_local.txt | xargs -P 18 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 4-5,8-23 /home/wj/miniconda3/envs/v/bin/python scripts/benchmark_multicluster.py --method bc_local --checkpoint research/runs/BENCH_BC_20261005/local_s0/student.pt --clusters $1 --anatomy $0 --seeds $2:$3 --out research/validation/BENCHMARK_V1_LEARNING_20261005/bc_local_N$1_$0.jsonl 2>>research/validation/BENCHMARK_V1_LEARNING_20261005/errors.log'
echo DONE >> research/validation/BENCHMARK_V1_LEARNING_20261005/status_local.txt
