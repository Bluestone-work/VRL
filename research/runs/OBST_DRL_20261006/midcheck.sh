#!/bin/bash
# 40 min into round 1: paired check of tres_s0 vs APF on 84 dev episodes (2 spare cores)
cd /home/wj/桌面/vascular_marl_local.tar.
R=research/runs/OBST_DRL_20261006
sleep 2460
mkdir -p $R/midcheck/tres_s0 && cp $R/tres_s0/policy.pt $R/midcheck/tres_s0/policy.pt
sed "s#Pool(22)#Pool(2)#" research/runs/OBST_DRL_V33P_20261006/check.py > $R/midcheck/check.py
PYTHONPATH=. taskset -c 22-23 ~/miniconda3/envs/v/bin/python $R/midcheck/check.py > $R/midcheck/CHECK.txt 2>&1
