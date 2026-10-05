#!/bin/bash
# One-time test-set evaluation (multicluster_benchmark_v1.test, 100 scenes/anatomy) of the frozen methods at commit b2e6dcf.
cd "/home/wj/桌面/vascular_marl_local.tar."
cat research/validation/BENCHMARK_V1_TEST_20261005/jobs.txt | xargs -P 22 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 0-5,8-23 $HOME/miniconda3/envs/v/bin/python scripts/benchmark_multicluster.py --method $0 --tag $5 --clusters $2 --anatomy $1 --seeds $3:$4 --out research/validation/BENCHMARK_V1_TEST_20261005/$5_N$2_$1_$3.jsonl 2>>research/validation/BENCHMARK_V1_TEST_20261005/errors.log'
echo DONE >> research/validation/BENCHMARK_V1_TEST_20261005/status.txt
