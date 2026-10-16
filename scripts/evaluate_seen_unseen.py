"""Evaluate a trained local policy on registered seen and unseen anatomies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--kind", choices=("ppo", "dqn"), required=True)
    p.add_argument("--action-set", choices=("semantic", "semantic_reduced", "atomic"), default="semantic")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes-per-anatomy", type=int, default=2)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(); torch.set_num_threads(1); a.out.parent.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "configs/evaluation_splits.json").read_text())["anatomy_holdout_v1"]
    env = IPPOAgentVecEnv(CurriculumVascularEnv(seed=a.seed, clusters=3, curriculum=False, reward_mode="navigation", zero_latency=True, direct_action=True), action_set=a.action_set)
    model = PPO.load(a.model, env=env, device=a.device) if a.kind == "ppo" else DQN.load(a.model, env=env, device=a.device)
    rows = []
    # The paired runner rejects registered validation/test seeds unless a
    # sealed-test ledger is explicitly enabled.  This transfer diagnostic
    # therefore uses a separate, unregistered scene namespace while keeping
    # the anatomy split fixed.  It must not be reported as a sealed test.
    scene_base = 1400000000 + int(a.seed) * 100000
    for group, anatomies in (("seen_train_anatomy", split["train"]), ("unseen_holdout_anatomy", split["held_out"])):
        # CurriculumVascularEnv normally accepts only training anatomies. For
        # the zero-shot holdout diagnostic, widen that allow-list explicitly;
        # no anatomy identifier is exposed to the policy observation.
        env.physical.train_anatomies = list(anatomies)
        for j, anatomy in enumerate(anatomies):
            for rep in range(a.episodes_per_anatomy):
                options = dict(anatomy=anatomy, scene_seed=scene_base + j * 10000 + rep,
                               flow=[.01, .025, .05, .1][(j + rep) % 4], latency=0, difficulty=(j + rep) % 4)
                obs = env.reset(options=options)
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, infos = env.step(action)
                    if bool(done[0]):
                        info = infos[0]; m = info.get("episode_metrics", {})
                        rows.append(dict(group=group, anatomy=anatomy, scene_seed=options["scene_seed"],
                            flow=options["flow"], difficulty=options["difficulty"],
                            navigation_success=bool(info.get("navigation_success", m.get("navigation_success", False))),
                            strict_success=bool(m.get("strict_success", False)),
                            wall_contact_s=float(m.get("wall_contact_s", np.asarray(info.get("agent_wall_contact_s", 0.)).sum())),
                            lost=int(m.get("lost", np.asarray(info.get("agent_lost", 0.)).sum())),
                            particle_collision_events=float(m.get("particle_collision_events", np.asarray(info.get("agent_particle_collision_events", 0.)).sum()))))
                        break
    def agg(part):
        return dict(episodes=len(part), navigation_success=float(np.mean([r["navigation_success"] for r in part])), strict_success=float(np.mean([r["strict_success"] for r in part])), wall_contact_s=float(np.mean([r["wall_contact_s"] for r in part])), lost_rate=float(np.mean([r["lost"] > 0 for r in part])), collision_rate=float(np.mean([r["particle_collision_events"] > 0 for r in part])))
    result = dict(model=str(a.model), action_set=a.action_set,
                  episodes_per_anatomy=a.episodes_per_anatomy,
                  protocol="unregistered_development_scene_namespace",
                  scene_seed_base=scene_base,
                  seen=agg([r for r in rows if r["group"] == "seen_train_anatomy"]),
                  unseen=agg([r for r in rows if r["group"] == "unseen_holdout_anatomy"]), episodes=rows)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)); env.close(); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
