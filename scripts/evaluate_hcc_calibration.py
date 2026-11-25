"""Paired frozen-policy and pursuit rollouts on calibrated fixed HCC layouts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, load_navigation_ppo


def rollout(task):
    model_path, scene, method, action_set, flags, fraction = task
    torch.set_num_threads(1)
    physical = CurriculumVascularEnv(seed=1041, clusters=3, curriculum=False, reward_mode="navigation",
                                     zero_latency=True, direct_action=True, centerline_match=flags["centerline_match"],
                                     fixed_scenes=[scene])
    physical.train_anatomies = [scene["anatomy"]]
    env = IPPOAgentVecEnv(physical, action_set=action_set, flow_feature=flags["flow_feature"],
                          safety_parity=flags["safety_parity"], exit_mask=flags["exit_mask"],
                          settle_radius=flags["settle_radius"])
    try:
        obs = env.reset()
        sim = physical.ep.env
        transport = sim.transport
        if fraction is not None:
            transport.fraction = fraction
        actual_fraction = transport.fraction
        if method == "IPPO":
            policy = load_navigation_ppo(model_path, env=env)
        while True:
            if method == "IPPO":
                obs, _, done, infos = env.step(policy.predict(obs, deterministic=True)[0])
                if not done[0]:
                    continue
                info = infos[0]
            else:
                cmd = env._apply_guards(np.asarray(physical.current_rule).copy())
                _, _, done, truncated, info = physical.step(cmd)
                if not (done or truncated):
                    continue
            row = info["episode_metrics"]
            keys = ("navigation_success", "navigation_reached", "termination", "lost", "wall_contact_s",
                    "particle_collision_events", "spacing_violation_pair_s", "elapsed_s")
            exits = [dict(cluster=int(i+1), station=int(sim.exit_node[i]), t_s=float(sim.exit_time_s[i]),
                          boundary="inlet" if sim.transport.groups[sim.exit_node[i]] == sim.transport.root_group else "outlet")
                     for i in np.flatnonzero(sim.exit_node[:physical.clusters] >= 0)]
            return dict(exits=exits, method=method, seed=scene["seed"], spatial_fraction=actual_fraction, **{k: row.get(k) for k in keys},
                        integration_failure=bool(info.get("integration_failure")))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("scenes", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--spatial-fraction", type=float, help="maximum substep displacement as a fraction of lumen radius")
    args = p.parse_args()
    if args.spatial_fraction is not None and not 0 < args.spatial_fraction <= .25:
        p.error("spatial fraction must lie in (0, .25]")
    rows = json.loads(args.scenes.read_text())
    cfg = json.loads(args.model.with_name("config.json").read_text())
    tasks = [(args.model, row, method, cfg["action_set"], cfg, args.spatial_fraction) for row in rows for method in ("IPPO", "pursuit")]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(rollout, tasks))
    args.out.write_text(json.dumps(dict(model=str(args.model), scene_file=str(args.scenes), rows=results),
                                   indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
