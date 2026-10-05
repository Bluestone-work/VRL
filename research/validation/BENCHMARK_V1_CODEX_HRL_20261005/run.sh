#!/bin/bash
cd "/home/wj/桌面/vascular_marl_local.tar."
cat research/validation/BENCHMARK_V1_CODEX_HRL_20261005/jobs.txt | xargs -P 22 -L 1 bash -c 'ck=""; [ "$1" != none ] && ck="--checkpoint $1"; PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 0-5,8-23 $HOME/miniconda3/envs/v/bin/python scripts/codex_hrl_benchmark.py --variant $0 $ck --clusters $2 --anatomy $3 --seeds $4:$5 --tag $6 --out research/validation/BENCHMARK_V1_CODEX_HRL_20261005/$6_N$2_$3.jsonl 2>>research/validation/BENCHMARK_V1_CODEX_HRL_20261005/errors.log'
echo DONE >> research/validation/BENCHMARK_V1_CODEX_HRL_20261005/status.txt
