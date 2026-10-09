run(){ env PYTHONPATH=. taskset -c 0-5,8-23 /home/wj/miniconda3/envs/v/bin/python scripts/run_primitive_eval.py "$@" --workers 9; }
for fr in carrot route velocity fused; do run --controller vp --ckpt research/runs/OBST_VPPPO_20261007/vp_wm_s0/step01049k.pt --frame $fr --out research/validation/PRIM_DIAG_20261007/vp_${fr}_none.jsonl; done
run --controller vp --ckpt research/runs/OBST_VPPPO_20261007/vp_wm_s0/step01049k.pt --frame carrot --safety sdf --out research/validation/PRIM_DIAG_20261007/vp_carrot_sdf.jsonl
run --controller switch --safety sdf --out research/validation/PRIM_DIAG_20261007/switch_sdf.jsonl
for fr in carrot route velocity fused; do run --controller advance --frame $fr --out research/validation/PRIM_DIAG_20261007/advance_${fr}_none.jsonl; done
run --controller advance --frame carrot --safety sdf --out research/validation/PRIM_DIAG_20261007/advance_carrot_sdf.jsonl
echo QUEUE_DONE
