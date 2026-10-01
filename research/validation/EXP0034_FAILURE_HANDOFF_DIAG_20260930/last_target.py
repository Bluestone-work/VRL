"""Diagnose time-limit failures: what happens to the final remaining clot."""
import json, sys, numpy as np, torch
from pathlib import Path
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.mca_physical_policy import make_physical_agent
from scripts.train_mca_compiled import reset_with_valid_particles, physical_policy_action
proto_path, ckpt, seeds, out = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(',')], sys.argv[4]
torch.set_num_threads(1)
protocol = json.loads(Path(proto_path).read_text())
cfg = DynamicsConfig.from_json(Path(protocol['physics_config']))
payload = torch.load(ckpt, map_location='cpu', weights_only=False)
agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'],
                            hidden_dim=protocol['hidden_dim'], device='cpu', ppo=protocol.get('ppo'))
agent.load(ckpt, load_optimizers=False); agent.actor.eval(); agent.critic.eval()
rows = []
for seed in seeds:
    env = MCAPhysicalEnv(cfg); obs, _ = reset_with_valid_particles(env, seed)
    clear_t = [None]*env.num_clots; trace = []; pos_hist = []
    while True:
        _, _, _, ex, _, _ = physical_policy_action(agent, env, obs, deterministic=True)
        obs, r, term, trunc, info = env.step(ex)
        for k in range(env.num_clots):
            if clear_t[k] is None and env.masses[k] <= 0: clear_t[k] = env.elapsed_s
        d = env._target_distances(); act = env.active[:env.num_robots]
        live = np.flatnonzero(env.masses > 0)
        if len(live):
            eu = np.linalg.norm(env.positions_mm[:env.num_robots, None]-env.clot_positions_mm[live], axis=-1)
            trace.append((env.elapsed_s, float(np.min(d[act][:, live])) if act.any() else np.inf,
                          float(np.min(eu[act])) if act.any() else np.inf))
        pos_hist.append(env.positions_mm[:env.num_robots].copy())
        if term or trunc: break
    live = np.flatnonzero(env.masses > 0)
    P = np.array(pos_hist); last60 = P[-600:] if len(P) > 600 else P
    rec = dict(seed=seed, success=bool(info['success']), reason=info['termination_reason'], lost=int(info['lost_robots']),
               clear_times=clear_t, remaining=live.tolist(), elapsed=env.elapsed_s)
    if len(live):
        d = env._target_distances()[:, live]
        rec['final_geodesic_mm_per_robot'] = np.round(np.min(d, axis=1), 2).tolist()
        rec['min_geodesic_ever_mm'] = round(min(t[1] for t in trace), 3)
        rec['init_geodesic_to_last_mm'] = round(trace[0][1], 2)
        rec['min_euclid_ever_mm'] = round(min(t[2] for t in trace), 3)
        eu = np.linalg.norm(env.positions_mm[:env.num_robots, None]-env.clot_positions_mm[live], axis=-1)
        rec['final_euclid_mm_per_robot'] = np.round(eu.min(axis=1), 3).tolist()
        _, radius, radial, _ = env.transport.coordinates(env.clot_positions_mm[live], env.transport.nearest_edges(env.clot_positions_mm[live]), env.solution)
        rec['target_lumen_radius_mm'] = np.round(radius, 3).tolist(); rec['target_radial_mm'] = np.round(radial, 3).tolist()
        rec['robot_radius_mm'] = env.config.robot_radius_mm
        # robots motion over final 60 s: net displacement vs path
        rec['last60_net_disp_mm'] = np.round(np.linalg.norm(last60[-1]-last60[0], axis=1), 2).tolist()
        rec['last60_path_mm'] = np.round(np.abs(np.diff(last60, axis=0)).sum(axis=(0, 2)), 1).tolist()
        rec['remaining_mass'] = env.masses[live].round(3).tolist()
    rows.append(rec); print(json.dumps(rec), flush=True)
Path(out).write_text(json.dumps(rows, indent=1))
