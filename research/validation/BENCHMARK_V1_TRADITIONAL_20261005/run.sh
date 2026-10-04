#!/bin/bash
# Benchmark v1, traditional family: 4 methods x N={1,2,3} x 14 anatomies x 30 development scenes.
cd "/home/wj/桌面/vascular_marl_local.tar."
E=research/validation/BENCHMARK_V1_TRADITIONAL_20261005; PY=$HOME/miniconda3/envs/v/bin/python
anat=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_order']))")
k=0; : > $E/jobs.txt
for a in $anat; do
  s0=$((2600000000+k*100000)); s1=$((s0+29))
  for m in route_follow local_follow plan_reactive nearest_reactive; do for n in 1 2 3; do echo "$m $n $a $s0 $s1" >> $E/jobs.txt; done; done
  k=$((k+1))
done
cat $E/jobs.txt | xargs -P 22 -L 1 bash -c 'PYTHONPATH=. OMP_NUM_THREADS=1 '$PY' scripts/benchmark_multicluster.py --method $0 --clusters $1 --anatomy $2 --seeds $3:$4 --out '$E'/$0_N$1_$2.jsonl 2>>'$E'/errors.log'
echo DONE >> $E/status.txt
