"""Resume an interrupted SB3 3M run from its latest agent-step checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, IPPOCallback, evaluate, load_navigation_ppo
from scripts.navigation_dataset import iter_rows


class BudgetedPPO(PPO):
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        # SB3 resets the buffer inside collect_rollouts. A shorter final buffer
        # lets PPO train on the last samples without going past the physical budget.
        remaining = (self._total_timesteps - self.num_timesteps) // env.num_envs
        steps = min(n_rollout_steps, remaining)
        rollout_buffer.buffer_size = steps
        return super().collect_rollouts(env, callback, rollout_buffer, steps)


def latest_checkpoint(out: Path) -> Path:
    candidates = list(out.glob("policy_agent_steps*.zip"))
    if not candidates:
        raise FileNotFoundError(f"no policy checkpoint in {out}")
    return max(candidates, key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--segment-out", type=Path, help="write this continuation to a separate directory")
    p.add_argument("--centerline-match", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--eval-anatomy")
    p.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    p.add_argument("--action-set", choices=("semantic", "semantic_reduced", "semantic_minimal", "atomic", "continuous"), required=True)
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
    cfg_file = a.out / "config.json"
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    if cfg.get("clusters", a.clusters) != a.clusters or loaded_agents % a.clusters:
        p.error("checkpoint cluster count conflicts with --clusters")
    if cfg.get("action_set", a.action_set) != a.action_set:
        p.error("--action-set differs from the checkpoint configuration")
    offset = loaded_agents // a.clusters
    dest = a.segment_out or a.out
    if a.segment_out:
        dest.mkdir(parents=True, exist_ok=False)
    log_dir = dest if a.segment_out else dest / f"resume_{loaded_agents}"
    centerline = cfg.get("centerline_match", False) if a.centerline_match is None else a.centerline_match
    weights = cfg.get("reward_weights", {})
    rows = None
    if cfg.get("dataset_split"):
        rows = list(iter_rows(split=cfg["dataset_split"], clusters=a.clusters))[:cfg["dataset_rows"]]
    physical = CurriculumVascularEnv(
        seed=a.seed, clusters=a.clusters, budget=a.target_steps, curriculum=cfg.get("curriculum", True),
        reward_mode="navigation", zero_latency=True, direct_action=True, dataset_rows=rows,
        centerline_match=centerline, flow_curriculum=cfg.get("flow_curriculum", False),
        flow_boundaries=cfg.get("flow_boundaries", (.2, .45, .7)),
        flow_values=cfg.get("flow_values", (.025, .05, .1, .15)),
        **{k: weights[k] for k in ("time_cost", "success_bonus", "lost_penalty") if k in weights},
        episode_log=dest / ("episodes.jsonl" if a.segment_out else "episodes_resume.jsonl"))
    if cfg.get("training_anatomies"):
        physical.train_anatomies = list(cfg["training_anatomies"])
    physical.total_environment_steps = offset
    physical.total_agent_steps = loaded_agents
    # The checkpoint has no simulator state. Start at a fresh scene index
    # without replaying the first episodes or restarting the flow schedule.
    physical.episode_index = offset
    physical.rng = np.random.default_rng([a.seed, loaded_agents])
    status_path = a.out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get("agent_steps") == loaded_agents:
        physical.frontier = int(status.get("frontier", 0))
    if rows:
        completed = sum(1 for line in (a.out / "episodes.jsonl").open() if line.strip())
        physical.dataset_cursor = cfg.get("dataset_cursor_at_restart", 0) + completed + 1

    guards = dict(flow_feature=cfg.get("flow_feature", False), safety_parity=cfg.get("safety_parity", False),
                  exit_mask=cfg.get("exit_mask", False), settle_radius=cfg.get("settle_radius", .3))
    env = IPPOAgentVecEnv(physical, action_set=a.action_set, failure_log=dest / "integration_failures.json", **guards)
    if a.algorithm == "ppo":
        model = load_navigation_ppo(ckpt, device=a.device, algorithm=BudgetedPPO)
    else:
        model = SAC.load(ckpt, device=a.device)
    if model.num_timesteps != loaded_agents or model.n_envs != a.clusters:
        raise ValueError("checkpoint counters disagree with filename or cluster count")
    model.set_env(env)
    model.set_logger(configure(str(log_dir), ["csv", "json", "tensorboard"]))
    eval_anatomy = a.eval_anatomy or cfg.get("eval_anatomy") or (cfg.get("anatomy_repeat") or "").partition(":")[0] or None
    settings = dict(cfg, action_set=a.action_set, centerline_match=centerline,
                    eval_anatomy=eval_anatomy, seed=a.seed, clusters=a.clusters,
                    physical_steps=a.target_steps, agent_steps=target_agents,
                    resumed_environment_steps=offset, resumed_agent_steps=loaded_agents,
                    resumed_from=str(ckpt.resolve()), resumed_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                    dataset_cursor_at_restart=physical.dataset_cursor, exact_environment_state_restored=False)
    (log_dir / "config.json").write_text(json.dumps(settings, indent=2))
    callback = IPPOCallback(dest, a.target_steps, a.clusters, save_every=8192)
    callback.last_saved = loaded_agents
    remaining = target_agents - loaded_agents
    started = time.monotonic()
    try:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callback)
        model.save(dest / "final_model")
        (dest / "TRAINING_COMPLETE.json").write_text(json.dumps(dict(
            environment_steps=physical.total_environment_steps, target_environment_steps=a.target_steps, agent_steps=model.num_timesteps,
            resumed_from=str(ckpt), resumed_agent_steps=loaded_agents,
            training_seconds=time.monotonic() - started), indent=2))
        result = evaluate(model, dest, a.clusters, a.eval_count, a.action_set, eval_anatomy,
                          centerline_match=centerline, **guards)
        (dest / "RUN_COMPLETE.json").write_text(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
