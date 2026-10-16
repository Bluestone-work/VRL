"""Finish an interrupted 1M pilot from its latest SB3 checkpoint.

The simulator state/RNG is not restored; the SB3 policy, optimizer and replay
state (for SAC) are restored.  This is recorded in TRAINING_COMPLETE.json so
the resulting run is not confused with an exact bitwise continuation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sb3_ippo import (
    CurriculumVascularEnv,
    IPPOAgentVecEnv,
    IPPOCallback,
    evaluate,
)
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure


def latest_checkpoint(out: Path) -> Path:
    files = list(out.glob("policy_agent_steps*.zip"))
    if not files:
        raise FileNotFoundError(f"no policy checkpoints in {out}")
    return max(files, key=lambda p: int(re.search(r"(\d+)(?=\.zip$)", p.name).group(1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--algorithm", choices=("ippo", "isac"), required=True)
    p.add_argument("--action-set", choices=("semantic", "atomic", "continuous"), default="continuous")
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--target-physical", type=int, default=1_000_000)
    p.add_argument("--eval-count", type=int, default=36)
    a = p.parse_args()
    torch.set_num_threads(1)
    out = a.out.resolve()
    ckpt = latest_checkpoint(out)
    model_cls = PPO if a.algorithm == "ippo" else SAC
    env = IPPOAgentVecEnv(CurriculumVascularEnv(
        seed=a.seed, clusters=a.clusters, budget=a.target_physical,
        curriculum=True, prediction=True, shaping=True, reward_mode="navigation",
        zero_latency=True, direct_action=True,
        episode_log=out / "episodes_resume.jsonl"), action_set=a.action_set)
    model = model_cls.load(ckpt, env=env, device="cuda")
    old_agent_steps = int(model.num_timesteps)
    target_agent_steps = int(a.target_physical * a.clusters)
    remaining = target_agent_steps - old_agent_steps
    if remaining <= 0:
        raise SystemExit(f"checkpoint already at {old_agent_steps}; target is {target_agent_steps}")
    model.set_logger(configure(str(out), ["csv", "json", "tensorboard"]))
    callback = IPPOCallback(out, a.target_physical, a.clusters)
    started = time.monotonic()
    try:
        model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
        model.save(out / "final_model")
        completion = dict(
            environment_steps=int(model.num_timesteps // a.clusters),
            agent_steps=int(model.num_timesteps),
            target_environment_steps=a.target_physical,
            resumed_from=str(ckpt),
            initial_agent_steps=old_agent_steps,
            exact_environment_state_restored=False,
            training_seconds=time.monotonic() - started,
        )
        (out / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2))
        result = evaluate(model, out, a.clusters, a.eval_count, a.action_set)
        (out / "RUN_COMPLETE.json").write_text(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
