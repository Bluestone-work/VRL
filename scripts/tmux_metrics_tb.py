"""Bridge episode JSONL metrics into TensorBoard while SB3 is training."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def rows(path):
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


def mean(items, key):
    values = [float(x[key]) for x in items if key in x and x[key] is not None]
    return sum(values) / len(values) if values else 0.0


def main():
    run = Path(os.environ.get("VRL_RUN", "research/runs/EXP0115_DRONE_REWARD_TB_20261015"))
    writer = SummaryWriter(log_dir=str(run), filename_suffix=".episode_metrics")
    seen = 0
    training_steps = 0
    while True:
        data = rows(run / "episodes.jsonl")
        status = {}
        try:
            status = json.loads((run / "status.json").read_text())
        except (OSError, ValueError):
            pass
        for index, row in enumerate(data[seen:], start=seen):
            training_steps += int(row.get("environment_steps", 0))
            step = max(training_steps, int(status.get("environment_steps", training_steps)))
            collision_events = float(row.get("particle_collision_events", 0))
            writer.add_scalar("episode/removal_fraction", float(row.get("removal", 0)), step)
            writer.add_scalar("episode/removal_auc", float(row.get("removal_auc", 0)), step)
            writer.add_scalar("episode/task_success", float(row.get("task_success", False)), step)
            writer.add_scalar("episode/strict_success", float(row.get("strict_success", False)), step)
            writer.add_scalar("episode/relaxed_success", float(row.get("relaxed_success", False)), step)
            writer.add_scalar("episode/collision_events", collision_events, step)
            writer.add_scalar("episode/collision_rate", float(collision_events > 0), step)
            writer.add_scalar("episode/particle_contact_s", float(row.get("particle_contact_s", 0)), step)
            writer.add_scalar("episode/wall_contact_s", float(row.get("wall_contact_s", 0)), step)
            writer.add_scalar("episode/lost_robots", float(row.get("lost", 0)), step)
            writer.add_scalar("episode/lost_rate", float(row.get("lost", 0) > 0), step)
            writer.add_scalar("episode/drone_return", float(row.get("drone_return", 0)), step)
            recent = data[max(0, index - 5):index + 1]
            writer.add_scalar("rolling6/removal_fraction", mean(recent, "removal"), step)
            writer.add_scalar("rolling6/strict_success_rate", mean(recent, "strict_success"), step)
            writer.add_scalar("rolling6/collision_rate", mean([{"v": float(float(x.get("particle_collision_events", 0)) > 0)} for x in recent], "v"), step)
            writer.add_scalar("rolling6/lost_rate", mean([{"v": float(float(x.get("lost", 0)) > 0)} for x in recent], "v"), step)
            seen = index + 1
        writer.flush()
        recent = data[-6:]
        if recent:
            removal = sum(float(x.get("removal", 0.)) for x in recent) / len(recent)
            success = sum(bool(x.get("task_success", False)) for x in recent) / len(recent)
            collision = sum(float(x.get("particle_collision_events", 0.)) > 0 for x in recent) / len(recent)
            print(f"metrics episodes={len(data)} steps={status.get('environment_steps', '—')} "
                  f"removal6={removal:.3f} success6={success:.1%} collision6={collision:.1%}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
