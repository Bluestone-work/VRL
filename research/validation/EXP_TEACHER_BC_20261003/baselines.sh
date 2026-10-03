#!/bin/bash
# Diagnostic-split baselines (20 layouts per anatomy, 5 robots): teacher and EXP40 pure RL (3 seeds).
cd "/home/wj/桌面/vascular_marl_local.tar."
E=research/validation/EXP_TEACHER_BC_20261003; PY=$HOME/miniconda3/envs/v/bin/python
anat=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_order']))")
cpus=(0 1 2 3 4 5 8 9 10 11 12 13 14 15 16 17 18 19); i=0
run(){ PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c ${cpus[$((i%18))]} $PY scripts/evaluate_student_policies.py --policy $1 --anatomy $2 --count 20 --out $E/diag_$3_$2.jsonl >> $E/baselines.log 2>&1; }
for a in $anat; do
  run teacher $a teacher & i=$((i+1))
  for s in 42 43 44; do run pure_rl:research/initializations/CROSS_ANATOMY_20261003/pure_rl_exp40_seed_$s.pt $a pure_rl_s$s & i=$((i+1)); [ $((i%18)) = 0 ] && wait; done
done; wait; echo DONE >> $E/baselines.log
