#!/bin/bash
# Frontier supervision: 9 training anatomies x N={1,2,3}, 12 episodes each. Seeds 1108000000 + k*10000 + N*1000.
cd "/home/wj/桌面/vascular_marl_local.tar."
D=research/datasets/FRONTIER_V1_20261005; PY=$HOME/miniconda3/envs/v/bin/python
train=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']))")
: > $D/jobs.txt
for a in $train; do k=$($PY -c "import json;print(json.load(open('configs/evaluation_splits.json'))['anatomy_order'].index('$a'))")
  for n in 1 2 3; do echo "$a $n $((1108000000+k*10000+n*1000)) 12" >> $D/jobs.txt; done; done
cat $D/jobs.txt | xargs -P 4 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 0-3 '$PY' scripts/collect_frontier_data.py --anatomy $0 --clusters $1 --first-seed $2 --episodes $3 --out '$D'/$0_N$1.npz >> '$D'/collect.log 2>&1'
echo DONE >> $D/collect.log
