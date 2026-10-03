#!/bin/bash
# Fair-observation measurement: privileged teacher (upper bound) vs reactive controllers that read only
# the Turbo-style partial observation. Diagnostic split, 20 layouts x 14 anatomies, 5 robots, noise 2.5%.
cd "/home/wj/桌面/vascular_marl_local.tar."
E=research/validation/EXP_FAIR_OBS_REACTIVE_20261003; PY=$HOME/miniconda3/envs/v/bin/python
anat=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_order']))")
cpus=(0 1 2 3 4 5 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)
jobs=(); for a in $anat; do for p in teacher reactive_bearing reactive_path; do jobs+=("$p:$a"); done; done
i=0; for j in "${jobs[@]}"; do
  p=${j%%:*}; a=${j#*:}
  PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c ${cpus[$((i%22))]} $PY scripts/evaluate_fair_policies.py --policy $p --anatomy $a --count 20 --out $E/diag_${p}_$a.jsonl >> $E/run.log 2>&1 &
  i=$((i+1)); [ $((i%22)) = 0 ] && wait
done; wait; echo DONE >> $E/run.log
