#!/bin/bash
cd "/home/wj/桌面/vascular_marl_local.tar."
R=research/runs/EXP_ONLINE_IMITATION_20261003; E=research/validation/EXP_TEACHER_BC_20261003; PY=$HOME/miniconda3/envs/v/bin/python
anat=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_order']))")
cpus=(0 1 2 3 4 5 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23); i=0
for tag in dagger_r1:$R/dagger_r1/student_epoch2.pt bc_cont:$R/bc_continued/student_epoch4.pt; do
  name=${tag%%:*}; ck=${tag#*:}
  for a in $anat; do
    PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c ${cpus[$((i%22))]} $PY scripts/evaluate_student_policies.py --policy student:$ck --anatomy $a --count 20 --robots 5 --out $E/diag_${name}_r5_$a.jsonl >> $E/eval_$name.log 2>&1 &
    i=$((i+1))
  done
done; wait; echo DONE >> $E/eval_dagger_r1.log
