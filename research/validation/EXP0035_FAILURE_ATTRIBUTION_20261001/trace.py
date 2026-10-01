"""Read-only per-step trace of EXP35 evaluation episodes (diagnostic only).

Replays exactly the evaluate() loop of scripts/train_mca_compiled.py
(compiled env, deterministic actions, same validation seeds) and records
per-step state. The policy action is never altered.
usage: trace.py PROTOCOL CHECKPOINT OUT_DIR SEED_FIRST SEED_LAST
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.mca_physical_policy import make_physical_agent, physical_policy_action
from scripts.train_mca_compiled import reset_with_valid_particles

torch.set_num_threads(1)
protocol = json.loads(Path(sys.argv[1]).read_text())
cfg = DynamicsConfig.from_json(Path(protocol['physics_config']))
payload = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'],
                            hidden_dim=protocol['hidden_dim'], device='cpu', ppo=protocol.get('ppo'))
agent.load(sys.argv[2], load_optimizers=False); agent.actor.eval(); agent.critic.eval()
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
n = cfg.num_robots
for seed in range(int(sys.argv[4]), int(sys.argv[5])+1):
    env = CompiledMCAPhysicalEnv(cfg)
    obs, reset_info = reset_with_valid_particles(env, seed)
    edge_branch = env.tree.branch_ids[env.transport.ends[:, 0]]
    rec = {k: [] for k in ('t', 'mass', 'pos', 'active', 'geo', 'euc', 'assign', 'act', 'raw',
                           'route', 'vel', 'flow', 'contact', 'wall', 'pcontact', 'branch', 'value', 'radial', 'lumen')}
    while True:
        proposal, _, value, ex, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
        axis, lumen, radial, _ = env.transport.coordinates(env.positions_mm[:n], env.edges[:n], env.solution)
        geo = env._target_distances()
        rec['t'].append(env.elapsed_s); rec['mass'].append(env.masses.copy())
        rec['pos'].append(env.positions_mm[:n].copy()); rec['active'].append(env.active[:n].copy())
        rec['geo'].append(np.where(np.isfinite(geo), geo, -1.))
        rec['euc'].append(np.linalg.norm(env.positions_mm[:n, None]-env.clot_positions_mm, axis=-1))
        rec['assign'].append(env._assigned_targets().copy())
        rec['route'].append(env._route_directions(axis))
        rec['flow'].append(env.transport.velocity_mm_s(env.positions_mm[:n], env.edges[:n], env.solution))
        rec['branch'].append(edge_branch[env.edges[:n]]); rec['radial'].append(radial); rec['lumen'].append(lumen)
        rec['act'].append(np.asarray(ex, np.float64)); rec['raw'].append(np.asarray(proposal, np.float64).reshape(n, -1))
        rec['value'].append(np.asarray(value, np.float64).reshape(-1))
        obs, r, term, trunc, info = env.step(ex)
        rec['vel'].append(env.velocity_mm_s.copy()); rec['contact'].append(np.asarray(info['contact_s']).copy())
        rec['wall'].append(np.asarray(info['wall_contact_s']).copy())
        rec['pcontact'].append(np.asarray(info['particle_contact_s']).copy())
        if term or trunc:
            break
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(out/f'ep_{seed}.npz', **arrays, clot_pos=env.clot_positions_mm,
                        clot_branch=env.tree.branch_ids[env.clot_stations],
                        final_mass=env.masses, success=info['success'], reason=str(info['termination_reason']),
                        lost=info['lost_robots'], elapsed=info['elapsed_s'],
                        removal=1-info['remaining_mass']/env.initial_mass.sum())
    print(json.dumps(dict(seed=seed, success=bool(info['success']), t=round(env.elapsed_s, 1),
                          reason=info['termination_reason'])), flush=True)
    env.close()
