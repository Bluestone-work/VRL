#!/bin/bash
cd /home/wj/桌面/vascular_marl_local.tar.
D=research/validation/OBST_BENCH_20261006
until [ -f research/runs/OBST_DRL_20261006/ALL_TRAINED ]; do sleep 60; done
while ps -eo args | grep -q "[e]val.py rule_"; do sleep 30; done
PYTHONPATH=. taskset -c 0-5,8-23 ~/miniconda3/envs/v/bin/python $D/eval_all.py 22 > $D/eval_all.log 2>&1
PYTHONPATH=. ~/miniconda3/envs/v/bin/python scripts/analyze_obstacles.py > $D/analyze.log 2>&1
echo done > $D/EVAL_DONE
