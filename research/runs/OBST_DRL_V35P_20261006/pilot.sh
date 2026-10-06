#!/bin/bash
cd /home/wj/桌面/vascular_marl_local.tar.
R=research/runs/OBST_DRL_V35P_20261006
t() { PYTHONPATH=. taskset -c $2 ~/miniconda3/envs/v/bin/python scripts/train_obstacle_drl.py --out $R/$1 --minutes 40 --workers 5 --device cuda:$3 "${@:4}" > $R/$1.log 2>&1; }
t tres_s0 0-4 0 --seed 0 & t tres_s1 5,8-11 1 --seed 1 & t tres_s2 12-16 0 --seed 2 & t mlp_s0 17-21 1 --seed 0 --arch mlp & wait
PYTHONPATH=. taskset -c 0-5,8-23 ~/miniconda3/envs/v/bin/python $R/check.py > $R/CHECK.txt 2>&1
