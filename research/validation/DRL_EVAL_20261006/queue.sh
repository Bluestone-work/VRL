#!/bin/bash
# wait for the three training runs, then evaluate each seed on the 14x30x3 dev scenes (image perception)
cd /home/wj/桌面/vascular_marl_local.tar.
while ps -eo args | grep -q "[t]rain_drl_local.py"; do sleep 30; done
R=research/runs/DRL_IRPPO_20261006; D=research/validation/DRL_EVAL_20261006
PYTHONPATH=. taskset -c 0-5 ~/miniconda3/envs/v/bin/python $D/eval.py irppo_s0 6 $R/seed0/policy.pt > $D/irppo_s0.log 2>&1 &
PYTHONPATH=. taskset -c 8-13 ~/miniconda3/envs/v/bin/python $D/eval.py irppo_s1 6 $R/seed1/policy.pt > $D/irppo_s1.log 2>&1 &
PYTHONPATH=. taskset -c 14-19 ~/miniconda3/envs/v/bin/python $D/eval.py irppo_s2 6 $R/seed2/policy.pt > $D/irppo_s2.log 2>&1 &
wait
echo done > $D/QUEUE_DONE
# ablation: IR-PPO without the image input (identical budget), then its evaluation
R2=research/runs/DRL_IRPPO_NOIMG_20261006; mkdir -p $R2
for s in 0 1 2; do c=$([ $s = 0 ] && echo 0-5 || ([ $s = 1 ] && echo 8-13 || echo 14-19))
  PYTHONPATH=. taskset -c $c ~/miniconda3/envs/v/bin/python scripts/train_drl_local.py --out $R2/seed$s --minutes 180 --workers 6 --seed $s --no-image --device cuda:$((s%2)) > $R2/seed$s.log 2>&1 & done
wait
for s in 0 1 2; do c=$([ $s = 0 ] && echo 0-5 || ([ $s = 1 ] && echo 8-13 || echo 14-19))
  PYTHONPATH=. taskset -c $c ~/miniconda3/envs/v/bin/python $D/eval.py irppo_noimg_s$s 6 $R2/seed$s/policy.pt > $D/irppo_noimg_s$s.log 2>&1 & done
wait
echo done > $D/QUEUE2_DONE
