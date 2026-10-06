#!/bin/bash
# after v2 training: evaluate the 3 seeds; then the no-image ablation with identical v2 settings, then its evaluation
cd /home/wj/桌面/vascular_marl_local.tar.
sleep 120
while ps -eo args | grep -q "[t]rain_drl_local.py --out research/runs/DRL_IRPPO_V2_20261006"; do sleep 30; done
R=research/runs/DRL_IRPPO_V2_20261006; R2=research/runs/DRL_IRPPO_V2_NOIMG_20261006; D=research/validation/DRL_EVAL_20261006
C=(0-5 8-13 14-19)
for s in 0 1 2; do PYTHONPATH=. taskset -c ${C[$s]} ~/miniconda3/envs/v/bin/python $D/eval.py irppo_v2_s$s 6 $R/seed$s/policy.pt > $D/irppo_v2_s$s.log 2>&1 & done
wait
echo done > $D/V2_EVAL_DONE
mkdir -p $R2
for s in 0 1 2; do PYTHONPATH=. taskset -c ${C[$s]} ~/miniconda3/envs/v/bin/python scripts/train_drl_local.py --out $R2/seed$s --minutes 180 --workers 6 --seed $s --reward v2 --res-scale 1.0 --init-std -1.0 --no-image --device cuda:$((s%2)) > $R2/seed$s.log 2>&1 & done
wait
for s in 0 1 2; do PYTHONPATH=. taskset -c ${C[$s]} ~/miniconda3/envs/v/bin/python $D/eval.py irppo_v2_noimg_s$s 6 $R2/seed$s/policy.pt > $D/irppo_v2_noimg_s$s.log 2>&1 & done
wait
echo done > $D/V2_NOIMG_DONE
