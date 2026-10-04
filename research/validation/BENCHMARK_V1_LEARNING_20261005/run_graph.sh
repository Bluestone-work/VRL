#!/bin/bash
cd "/home/wj/桌面/vascular_marl_local.tar."
until [ -f research/validation/BENCHMARK_V1_INNOVATION_20261005/status_tabu.txt ]; do sleep 20; done
cat research/validation/BENCHMARK_V1_LEARNING_20261005/jobs_graph.txt | xargs -P 22 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 0-5,8-23 $HOME/miniconda3/envs/v/bin/python scripts/benchmark_multicluster.py --method bc_graph --checkpoint research/runs/BENCH_BC_20261005/graph_s0/student_epoch9.pt --clusters $1 --anatomy $0 --seeds $2:$3 --out research/validation/BENCHMARK_V1_LEARNING_20261005/bc_graph_N$1_$0.jsonl 2>>research/validation/BENCHMARK_V1_LEARNING_20261005/errors.log'
echo DONE >> research/validation/BENCHMARK_V1_LEARNING_20261005/status_graph.txt
