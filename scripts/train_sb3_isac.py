"""Independent/parameter-shared SAC for continuous local vascular actions.

The synchronized physical simulator is exposed as one logical local agent per
cluster by ``IPPOAgentVecEnv``.  SAC therefore learns from local observations
and local rewards without a centralized privileged critic; shared weights make
this the parameter-shared ISAC variant.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, IPPOCallback, IPPOExtractor, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--timesteps", type=int, default=20000, help="physical environment steps")
    p.add_argument("--clusters", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-count", type=int, default=12)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    physical = CurriculumVascularEnv(
        seed=a.seed, clusters=a.clusters, budget=a.timesteps,
        curriculum=True, prediction=True, shaping=True, reward_mode="navigation",
        zero_latency=True, direct_action=True,
        episode_log=a.out / "episodes.jsonl")
    env = IPPOAgentVecEnv(physical, action_set="continuous")
    cfg = dict(
        algorithm="ISAC_shared",
        framework="stable-baselines3 2.7.1",
        seed=a.seed,
        clusters=a.clusters,
        physical_steps=a.timesteps,
        agent_steps=a.timesteps * a.clusters,
        device=a.device,
        reward_mode="navigation",
        zero_latency=True,
        action_semantics="per-cluster continuous direct local velocity, unit-ball bounded",
        actor="parameter-shared local SAC actor; one 3-D action per cluster",
        critic="local SAC twin critics; no privileged state",
        architecture="GRU history + CNN image + map + masked obstacle/peer pooling",
        source_script=str(Path(__file__).resolve()),
    )
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    policy_kwargs = dict(features_extractor_class=IPPOExtractor, normalize_images=False,
                         net_arch=dict(pi=[128, 128], qf=[128, 128]))
    model = SAC(
        "MultiInputPolicy", env, seed=a.seed, device=a.device, verbose=0,
        policy_kwargs=policy_kwargs, learning_rate=3e-4, buffer_size=100000,
        learning_starts=1000, train_freq=1, gradient_steps=1,
        batch_size=256, gamma=.995, tau=.005, ent_coef="auto",
    )
    model.set_logger(configure(str(a.out), ["csv", "json", "tensorboard"]))
    model.save(a.out / "initial_model")
    started = time.monotonic()
    try:
        model.learn(total_timesteps=a.timesteps * a.clusters,
                    callback=IPPOCallback(a.out, a.timesteps, a.clusters))
        model.save(a.out / "final_model")
        completion = dict(environment_steps=a.timesteps,
                          agent_steps=model.num_timesteps,
                          training_seconds=time.monotonic() - started)
        (a.out / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2))
        summary = evaluate(model, a.out, a.clusters, a.eval_count, "continuous")
        (a.out / "RUN_COMPLETE.json").write_text(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
