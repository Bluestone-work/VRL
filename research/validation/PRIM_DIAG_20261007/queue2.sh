run(){ env PYTHONPATH=. taskset -c 0-5,8-23 /home/wj/miniconda3/envs/v/bin/python scripts/run_primitive_eval.py "$@" --workers 5; }
run --controller vp --ckpt research/runs/OBST_VPPPO_20261007/vp_wm_s0/step01049k.pt --frame carrot --safety sdf --out research/validation/PRIM_DIAG_20261007/vp_carrot_sdf.jsonl &
run --controller advance --frame carrot --safety sdf --out research/validation/PRIM_DIAG_20261007/advance_carrot_sdf.jsonl &
wait; run --controller switch --safety sdf --out research/validation/PRIM_DIAG_20261007/switch_sdf.jsonl; echo QUEUE2_DONE
