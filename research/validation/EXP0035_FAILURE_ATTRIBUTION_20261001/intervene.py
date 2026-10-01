"""Eval-only counterfactual substitutions on EXP35 checkpoints (diagnostic, never used for training).

Modes (per robot, per step; the policy runs unchanged otherwise):
  policy     baseline, identical to trace.py
  far_map    known-map follower when geodesic distance to own assigned target > 0.7 mm
  near_map   known-map follower when geodesic distance to own assigned target <= 0.7 mm
  last_map   known-map follower for every robot once only one target remains
  gate0.3    no oracle: zero the command when the pre-normalisation policy command norm < 0.3
usage: intervene.py PROTOCOL CHECKPOINT MODE OUT_JSONL SEED_FIRST SEED_LAST
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_compiled import reset_with_valid_particles
from scripts.validate_mca_surface_task import witness_action

torch.set_num_threads(1)
protocol = json.loads(Path(sys.argv[1]).read_text())
cfg = DynamicsConfig.from_json(Path(protocol['physics_config']))
payload = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'],
                            hidden_dim=protocol['hidden_dim'], device='cpu', ppo=protocol.get('ppo'))
agent.load(sys.argv[2], load_optimizers=False); agent.actor.eval(); agent.critic.eval()
mode = sys.argv[3]
with open(sys.argv[4], 'a') as out:
    for seed in range(int(sys.argv[5]), int(sys.argv[6])+1):
        env = CompiledMCAPhysicalEnv(cfg)
        obs, _ = reset_with_valid_particles(env, seed)
        n = env.num_robots; swapped = 0; steps = 0
        while True:
            _, _, _, ex, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
            ex = np.asarray(ex, np.float64).copy()
            live = np.flatnonzero(env.masses > 0)
            if mode != 'policy' and len(live):
                assign = env._assigned_targets().copy()
                if mode == 'gate0.3':
                    use = np.linalg.norm(np.clip(ex, -1, 1), axis=1) < .3
                    ex[use] = 0.
                else:
                    d = env._target_distances()
                    own = d[np.arange(n), np.clip(assign, 0, env.num_clots-1)]
                    use = {'far_map': own > .7, 'near_map': own <= .7,
                           'last_map': np.full(n, len(live) == 1)}[mode] & env.active[:n] & (assign >= 0)
                    if use.any():
                        w = witness_action(env, assign.copy())
                        ex[use] = w[use]
                swapped += int(use.sum()); steps += int(env.active[:n].sum())
            obs, r, term, trunc, info = env.step(ex)
            if term or trunc:
                break
        out.write(json.dumps(dict(seed=seed, mode=mode, success=bool(info['success']), elapsed=env.elapsed_s,
                                  removal=1-info['remaining_mass']/env.initial_mass.sum(), lost=info['lost_robots'],
                                  collision_free=bool(info['collision_free_success']),
                                  swapped_frac=swapped/max(steps, 1)))+'\n'); out.flush()
        env.close()
