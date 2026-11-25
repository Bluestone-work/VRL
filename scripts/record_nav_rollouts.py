"""Record IPPO navigation rollouts with per-step geometry, for Fig. 3c.

Fig. 3c has to make success, obstacle-detour and failure visually separable, so a
trajectory alone is not enough: we also log the distance of each cluster to its
assigned target at every control step. Success falls monotonically to the reach
tolerance, a detour shows a plateau or a rise before descending, and a failure
stalls at a finite distance. Episodes come from the frozen manifest, so a
recorded rollout is reproducible from its row index.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
if hasattr(os, 'sched_getaffinity'):
    _a = os.sched_getaffinity(0) - {6, 7}
    if _a: os.sched_setaffinity(0, _a)
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REACH_MM = .30


def rollout(model, env, row):
    phys = env.physical
    obs = env.reset(options=dict(row))
    ep = phys.ep
    n = phys.clusters
    traj, dist, wall, near = [], [], [], []
    while True:
        traj.append(np.asarray(ep.env.positions_mm[:n], float).copy())
        dist.append(np.asarray(phys._navigation_distances(), float).copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = env.step(action)
        info = infos[0]
        wall.append(float(np.asarray(info.get('agent_wall_contact_s', 0.)).sum()))
        near.append(float(np.asarray(info.get('agent_particle_near_s', 0.)).sum()))
        if bool(done[0]):
            m = dict(info.get('episode_metrics', {}))
            traj.append(np.asarray(ep.env.positions_mm[:n], float).copy())
            dist.append(np.asarray(phys._navigation_distances(), float).copy())
            return dict(
                success=bool(info.get('navigation_success', False)),
                reached=int(info.get('navigation_reached', 0)),
                clusters=n,
                elapsed_s=float(m.get('elapsed_s', 0.)),
                wall_contact_s=float(m.get('wall_contact_s', sum(wall))),
                lost=int(m.get('lost', 0)),
                dt_s=float(ep.env.config.control_dt_s),
                anatomy=row.get('anatomy'),
                flow_mode=row.get('flow_mode'),
                obstacle_density=row.get('obstacle_density'),
                trajectory=np.asarray(traj).tolist(),
                target_distance=np.asarray(dist).tolist(),
                targets=np.asarray(ep.env.clot_positions_mm, float).tolist(),
                assigned=np.asarray(phys.nav_targets, int).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--clusters', type=int, default=3)
    ap.add_argument('--max-episodes', type=int, default=24)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.train_sb3_ippo import IPPOAgentVecEnv
    from scripts.navigation_dataset import iter_rows, MANIFEST
    from stable_baselines3 import PPO
    rows = list(iter_rows(split=a.split, clusters=a.clusters))[:a.max_episodes]
    phys = CurriculumVascularEnv(seed=1041, clusters=a.clusters, curriculum=False,
                                 reward_mode='navigation', zero_latency=True,
                                 direct_action=True)
    env = IPPOAgentVecEnv(phys, action_set='semantic_reduced')
    model = PPO.load(str(a.model), env=env, device='cpu')
    out = []
    try:
        for i, row in enumerate(rows):
            try:
                r = rollout(model, env, row)
            except Exception as exc:
                print(json.dumps({'row': i, 'error': repr(exc)[:160]}), flush=True)
                continue
            r['row_index'] = i
            out.append(r)
            print(json.dumps({'row': i, 'anatomy': r['anatomy'], 'success': r['success'],
                              'reached': r['reached'], 'elapsed_s': round(r['elapsed_s'], 1),
                              'wall_s': round(r['wall_contact_s'], 2)}), flush=True)
    finally:
        env.close()
    a.out.write_text(json.dumps(dict(model=str(a.model), split=a.split,
                                     clusters=a.clusters, reach_mm=REACH_MM,
                                     manifest=str(MANIFEST.relative_to(ROOT)),
                                     episodes=out), ensure_ascii=False) + '\n')
    print(f'recorded {len(out)} rollouts -> {a.out}')


if __name__ == '__main__':
    main()
