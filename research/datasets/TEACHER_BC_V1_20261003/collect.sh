#!/bin/bash
# Teacher BC dataset v1: 9 training anatomies (anatomy_holdout_v1.train), 60 train + 10 val episodes each, robots {4,5,6}.
# Seeds: 1100000000 + k*100000 (train), +50000 (val); k = index in anatomy_order.
cd "/home/wj/桌面/vascular_marl_local.tar."
D=research/datasets/TEACHER_BC_V1_20261003
PY=$HOME/miniconda3/envs/v/bin/python
train=$($PY -c "import json;r=json.load(open('configs/evaluation_splits.json'));print(' '.join(r['anatomy_holdout_v1']['train']))")
cpus=(0 1 2 3 4 5 8 9 10 11 12 13 14 15 16 17 18 19); i=0
for a in $train; do
  k=$($PY -c "import json;print(json.load(open('configs/evaluation_splits.json'))['anatomy_order'].index('$a'))")
  for split in train val; do
    if [ $split = train ]; then s=$((1100000000+k*100000)); n=60; else s=$((1100000000+k*100000+50000)); n=10; fi
    PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c ${cpus[$((i%18))]} $PY scripts/collect_teacher_dataset.py --anatomy $a --first-seed $s --episodes $n --out $D/$split >> $D/collect_$split.log 2>&1 &
    i=$((i+1))
  done
done
wait; echo DONE >> $D/collect_train.log
