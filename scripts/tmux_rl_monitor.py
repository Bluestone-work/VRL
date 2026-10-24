"""Compact live monitor for a single SB3 run inside tmux."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default if default is not None else {}


def last_jsonl(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return {}


def jsonl_rows(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                out.append(value)
        except ValueError:
            pass
    return out


def main():
    run = Path(os.environ.get("VRL_RUN", "research/runs/EXP0115_DRONE_REWARD_TB_20261015"))
    target = int(os.environ.get("VRL_TARGET", "131072"))
    while True:
        status = read_json(run / "status.json")
        progress = last_jsonl(run / "progress.json")
        episode = last_jsonl(run / "episodes.jsonl")
        episodes = jsonl_rows(run / "episodes.jsonl")
        recent = episodes[-6:]
        steps = int(status.get("environment_steps", progress.get("time/total_timesteps", 0)))
        pct = 100 * steps / target
        print("\033[2J\033[H", end="")
        print(f"VRL SB3 live monitor  |  {run}")
        print(f"steps      {steps:>7}/{target} ({pct:5.1f}%)   agent_steps {status.get('agent_steps', '—')}")
        scene = status.get("scene", {})
        print(f"frontier   {status.get('frontier', '—')}   difficulty {status.get('difficulty', '—')}   "
              f"scene {scene.get('anatomy', '—')}   flow {scene.get('flow', '—')}   "
              f"dyn_obs {scene.get('dynamic_obstacles', '—')}   latency {scene.get('latency', '—')}")
        print(f"fps        {progress.get('time/fps', '—')}   updates {progress.get('train/n_updates', '—')}   "
              f"ep_reward {progress.get('rollout/ep_rew_mean', '—')}   success {progress.get('rollout/success_rate', '—')}")
        print(f"value_loss {progress.get('train/value_loss', '—')}   approx_kl {progress.get('train/approx_kl', '—')}   "
              f"action_sat {status.get('action_saturation', '—')}")
        if episode:
            collision_rate = sum(float(x.get("particle_collision_events", 0)) > 0 for x in recent) / max(len(recent), 1)
            lost_rate = sum(float(x.get("lost", 0)) > 0 for x in recent) / max(len(recent), 1)
            success_rate = sum(bool(x.get("task_success", False)) for x in recent) / max(len(recent), 1)
            strict_rate = sum(bool(x.get("strict_success", False)) for x in recent) / max(len(recent), 1)
            print(f"last ep    removal {episode.get('removal', '—')}   wall_s {episode.get('wall_contact_s', '—')}   "
                  f"lost {episode.get('lost', '—')}   collisions {episode.get('particle_collision_events', '—')}   "
                  f"strict {episode.get('strict_success', '—')}")
            print(f"recent6    removal_mean {sum(float(x.get('removal', 0)) for x in recent)/max(len(recent),1):.4f}   "
                  f"task_success_rate {success_rate:.1%}   strict_success_rate {strict_rate:.1%}")
            print(f"recent6    collision_rate {collision_rate:.1%}   lost_rate {lost_rate:.1%}   "
                  f"wall_mean_s {sum(float(x.get('wall_contact_s', 0)) for x in recent)/max(len(recent),1):.2f}")
        diag = status.get("predictor_diagnostics", {})
        if diag:
            print("predictor  " + " | ".join(
                f"{h}s rmse={v.get('forecast_rmse_mm', '—')} n={v.get('matched_detection_targets', 0)}"
                for h, v in diag.items()))
        tb_url = os.environ.get("VRL_TB_URL", "http://127.0.0.1:6006")
        tb_session = os.environ.get("VRL_TB_SESSION", "vrl_tb")
        print(f"\nTensorBoard: {tb_url}   attach: tmux attach -t {tb_session}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
