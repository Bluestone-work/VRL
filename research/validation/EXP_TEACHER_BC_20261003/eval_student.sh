#!/bin/bash
# usage: eval_student.sh CKPT TAG [ROBOTS] [COUNT]  -- diagnostic split, all 14 anatomies
cd "/home/wj/桌面/vascular_marl_local.tar."
E=research/validation/EXP_TEACHER_BC_20261003; PY=$HOME/miniconda3/envs/v/bin/python
ck=$1; tag=$2; robots=${3:-5}; count=${4:-20}
anat=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_order']))")
cpus=(0 1 2 3 4 5 8 9 10 11 12 13 14 15 16 17 18 19); i=0
for a in $anat; do
  PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c ${cpus[$((i%18))]} $PY scripts/evaluate_student_policies.py --policy student:$ck --anatomy $a --count $count --robots $robots --out $E/diag_${tag}_r${robots}_$a.jsonl >> $E/eval_$tag.log 2>&1 &
  i=$((i+1))
done; wait; echo DONE >> $E/eval_$tag.log
