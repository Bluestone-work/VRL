#!/bin/bash
# route_follow+shield demonstrations, 9 training anatomies x N={1,2,3}; 24 train + 4 val episodes each.
# Seeds: 1105000000 + k*10000 + N*1000 (+500 for val); k = index in anatomy_order. Inside the teacher-dataset range.
cd "/home/wj/桌面/vascular_marl_local.tar."
D=research/datasets/CLUSTER_BC_V1_20261005; PY=$HOME/miniconda3/envs/v/bin/python
train=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']))")
: > $D/jobs.txt
for a in $train; do
  k=$($PY -c "import json;print(json.load(open('configs/evaluation_splits.json'))['anatomy_order'].index('$a'))")
  for n in 1 2 3; do
    s=$((1105000000+k*10000+n*1000))
    for h in 0 1 2; do echo "$a $n $((s+h*8)) 8 train" >> $D/jobs.txt; done
    echo "$a $n $((s+500)) 4 val" >> $D/jobs.txt
  done
done
cat $D/jobs.txt | xargs -P 22 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 '$PY' scripts/collect_cluster_bc.py --anatomy $0 --clusters $1 --first-seed $2 --episodes $3 --out '$D'/$4 >> '$D'/collect.log 2>&1'
echo DONE >> $D/collect.log
