"""Diagnostic only: an EXP43 checkpoint executed with a post-policy safety shield
(command stops when an observed slot forecasts clearance < C within H s). Diagnostic layouts only.
usage: probe.py CHECKPOINT C H SEED_FIRST SEED_LAST OUT_JSONL"""
import json, sys
from dataclasses import replace
import numpy as np, torch
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_compiled import reset_with_valid_particles
torch.set_num_threads(1)
ck, c, h = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
cfg = replace(DynamicsConfig.from_json('configs/experiments/EXP_0043_GATED_PRIOR_DYNAMICS.json'), action_shield_clearance=c, action_shield_horizon_s=h)
p = torch.load(ck, map_location='cpu', weights_only=False)
agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=p['meta']['training_seed'], hidden_dim=128, device='cpu'); agent.load(ck, load_optimizers=False)
with open(sys.argv[6], 'a') as out:
    for seed in range(int(sys.argv[4]), int(sys.argv[5])+1):
        env = CompiledMCAPhysicalEnv(cfg); obs, _ = reset_with_valid_particles(env, seed)
        while True:
            _, _, _, ex, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
            obs, _, t, tr, info = env.step(ex)
            if t or tr: break
        out.write(json.dumps(dict(ck=ck.split('/')[-2]+'/'+ck.split('/')[-1], c=c, h=h, seed=seed, success=bool(info['success']), cf=bool(info['collision_free_success']), elapsed=env.elapsed_s))+'\n'); out.flush(); env.close()
