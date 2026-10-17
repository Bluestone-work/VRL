"""Resume an interrupted SB3 3M run from its latest agent-step checkpoint."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import PPO, SAC

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, IPPOCallback, IPPOExtractor, evaluate


def latest_checkpoint(out: Path) -> Path:
    candidates = list(out.glob("policy_agent_steps*.zip"))
    if not candidates:
        raise FileNotFoundError(f"no policy checkpoint in {out}")
    return max(candidates, key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    p.add_argument("--action-set", choices=("semantic", "atomic", "continuous"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--target-steps", type=int, default=3_000_000)
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-count", type=int, default=36)
    a = p.parse_args()
    torch.set_num_threads(1)
    ckpt = latest_checkpoint(a.out)
    loaded_agents = int(re.search(r"(\d+)$", ckpt.stem).group(1))
    target_agents = int(a.target_steps) * int(a.clusters)
    if loaded_agents >= target_agents:
        print(json.dumps(dict(status="already_at_target", checkpoint=str(ckpt), agent_steps=loaded_agents)))
        return
    physical = CurriculumVascularEnv(
        seed=a.seed, clusters=a.clusters, budget=a.target_steps,
        curriculum=True, prediction=True, shaping=True, reward_mode="navigation",
        zero_latency=True, direct_action=True,
        episode_log=a.out / "episodes_resume.jsonl")
    env = IPPOAgentVecEnv(physical, action_set=a.action_set)
    cls = PPO if a.algorithm == "ppo" else SAC
    model = cls.load(ckpt, env=env, device=a.device)
    model.set_env(env)
    callback = IPPOCallback(a.out, a.target_steps, a.clusters, save_every=8192)
    callback.last_saved = loaded_agents
    remaining = target_agents - loaded_agents
    started = time.monotonic()
    try:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callback)
        model.save(a.out / "final_model")
        (a.out / "TRAINING_COMPLETE.json").write_text(json.dumps(dict(
            environment_steps=a.target_steps, agent_steps=model.num_timesteps,
            resumed_from=str(ckpt), resumed_agent_steps=loaded_agents,
            training_seconds=time.monotonic() - started), indent=2))
        result = evaluate(model, a.out, a.clusters, a.eval_count, a.action_set)
        (a.out / "RUN_COMPLETE.json").write_text(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
