"""Eval-only probe: how far does following the OBSERVED own route bearing go, with no learning?

Action (robot Frenet frame) = own route bearing from observation columns 112:115
(schema mca_point_routed_115_own_v9), optionally plus a scaled policy proposal.
Same execution path as the policy (direct_local_action). Diagnostic layouts only.
usage: probe.py SEED_FIRST SEED_LAST OUT_JSONL [CHECKPOINT RESIDUAL_SCALE]
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.geometric_control import direct_local_action
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_compiled import reset_with_valid_particles

torch.set_num_threads(1)
cfg = DynamicsConfig.from_json(Path('configs/experiments/EXP_0040_OWN_BEARING_DYNAMICS.json'))
agent = None
if len(sys.argv) > 4:
    payload = torch.load(sys.argv[4], map_location='cpu', weights_only=False)
    agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'], hidden_dim=128, device='cpu')
    agent.load(sys.argv[4], load_optimizers=False); agent.actor.eval()
    scale = float(sys.argv[5])
with open(sys.argv[3], 'a') as out:
    for seed in range(int(sys.argv[1]), int(sys.argv[2])+1):
        env = CompiledMCAPhysicalEnv(cfg)
        obs, _ = reset_with_valid_particles(env, seed)
        while True:
            bearing = obs['nodes'][:, 112:115].astype(np.float64)
            local = bearing
            if agent is not None:
                proposal, *_ = physical_policy_action(agent, env, obs, deterministic=True)
                local = bearing + scale*np.asarray(proposal, np.float64).reshape(bearing.shape)
            ex = direct_local_action(local, env)
            ex[~env.active[:env.num_robots]] = 0
            obs, r, term, trunc, info = env.step(ex)
            if term or trunc:
                break
        out.write(json.dumps(dict(seed=seed, success=bool(info['success']), elapsed=env.elapsed_s,
                                  removal=1-info['remaining_mass']/env.initial_mass.sum(), lost=info['lost_robots'],
                                  collision_free=bool(info['collision_free_success'])))+'\n'); out.flush()
        env.close()
