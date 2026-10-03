#!/bin/bash
# Safe-navigation baseline table: every existing controller, 14 anatomies x 20 diagnostic layouts, 5 robots.
# Privileged: teacher, pure RL EXP40 (x3), residual EXP43 (x3), residual+shield (x3). Fair: reactive_bearing, reactive_path.
cd "/home/wj/桌面/vascular_marl_local.tar."
E=research/validation/EXP_SAFE_BASELINES_20261003; PY=$HOME/miniconda3/envs/v/bin/python
I=research/initializations/CROSS_ANATOMY_20261003
anat=$($PY -c "import json;print(' '.join(json.load(open('configs/evaluation_splits.json'))['anatomy_order']))")
specs="teacher|teacher reactive_bearing|reactive_bearing reactive_path|reactive_path"
for s in 42 43 44; do specs="$specs pure_rl:$I/pure_rl_exp40_seed_$s.pt|pure_rl_s$s residual:$I/residual_exp43_seed_$s.pt|residual_s$s residual_shield:$I/residual_shield_exp43_seed_$s.pt|residual_shield_s$s"; done
# pure RL first: its episodes run to the time limit, so it is the long pole
jobs=(); for sp in $specs; do case $sp in pure_rl*) for a in $anat; do jobs+=("$sp@$a"); done;; esac; done
for sp in $specs; do case $sp in pure_rl*) ;; *) for a in $anat; do jobs+=("$sp@$a"); done;; esac; done
printf '%s\n' "${jobs[@]}" > $E/jobs.txt
cat $E/jobs.txt | xargs -P 22 -I{} bash -c '
  j="{}"; sp=${j%@*}; a=${j##*@}; pol=${sp%|*}; tag=${sp#*|}
  PYTHONPATH=. OMP_NUM_THREADS=1 '$PY' scripts/evaluate_fair_policies.py --policy $pol --tag $tag --anatomy $a --count 20 --out '$E'/diag_${tag}_$a.jsonl >> '$E'/run.log 2>&1'
echo DONE >> $E/run.log
