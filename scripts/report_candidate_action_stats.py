"""Report semantic/atomic candidate selection distributions for a trained model."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.utils import obs_as_tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, IPPOExtractor, SEMANTIC_CANDIDATE_COUNT, ATOMIC_CANDIDATE_COUNT
from scripts.train_discrete_rl import DiscreteSAC


SEMANTIC_LABELS = ["zero", "rule", "route_full", "route_075", "route_050", "target_full", "target_050", "target_025", "settle", "reverse", "normal_035", "normal_-035", "binormal_035", "binormal_-035", "normal_070", "normal_-070", "binormal_070", "binormal_-070"]
ATOMIC_LABELS = ["zero", "tangent_035", "tangent_070", "tangent_100", "reverse_035", "reverse_070", "reverse_100", "normal_035", "normal_-035", "normal_070", "normal_-070", "binormal_035", "binormal_-035", "binormal_070", "binormal_-070"]


def candidate_distribution(model, obs, kind, device="cpu"):
    t = obs_as_tensor(obs, device)
    valid = t["candidate_valid"].float()
    if isinstance(model, PPO):
        logits = model.policy.get_distribution(t).distribution.logits
        raw = torch.softmax(logits, -1)
        probs = raw * valid
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-8)
    elif isinstance(model, DQN):
        q = model.policy.q_net(t)
        raw = torch.softmax(q, -1)
        probs = raw * valid
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-8)
    else:
        with torch.no_grad():
            logits = model.logits(t)
            probs, _ = model.masked_dist(logits, valid)
    return probs.detach().cpu().numpy(), valid.detach().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--kind", choices=("ppo", "dqn", "isac"), required=True)
    p.add_argument("--action-set", choices=("semantic", "atomic"), default="semantic")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--device", default="cpu")
    a = p.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True); torch.set_num_threads(1)
    k = SEMANTIC_CANDIDATE_COUNT if a.action_set == "semantic" else ATOMIC_CANDIDATE_COUNT
    labels = SEMANTIC_LABELS if a.action_set == "semantic" else ATOMIC_LABELS
    env = IPPOAgentVecEnv(CurriculumVascularEnv(seed=1041, clusters=a.clusters, curriculum=False, reward_mode="navigation", zero_latency=True, direct_action=True), action_set=a.action_set)
    if a.kind == "ppo": model = PPO.load(a.model, env=env, device=a.device)
    elif a.kind == "dqn": model = DQN.load(a.model, env=env, device=a.device)
    else:
        model = DiscreteSAC(env, k, 41, a.device)
        state = torch.load(a.model, map_location=a.device)
        for name in ("actor", "q1", "q2", "tq1", "tq2"): getattr(model, name).load_state_dict(state[name])
    prob_sum = np.zeros(k); valid_count = np.zeros(k); chosen_count = np.zeros(k); rows = []; obs = env.reset(); episode = 0
    while episode < a.episodes:
        probs, valid = candidate_distribution(model, obs, a.kind, a.device)
        actions = probs.argmax(-1)
        prob_sum += probs.sum(0); valid_count += valid.sum(0); chosen_count += np.bincount(actions, minlength=k)
        obs, _, done, infos = env.step(actions)
        if bool(done[0]):
            info = infos[0]; success = bool(info.get("navigation_success", False)); rows.append(dict(episode=episode, navigation_success=success, steps=int(info.get("environment_steps", 0))))
            episode += 1
    summary = []
    for i in range(k):
        summary.append(dict(index=i, label=labels[i], mean_probability=float(prob_sum[i] / max(1, valid_count[i])), valid_fraction=float(valid_count[i] / max(1, valid_count.max())), selection_frequency=float(chosen_count[i] / max(1, chosen_count.sum())), selections=int(chosen_count[i])))
    result = dict(model=str(a.model), kind=a.kind, action_set=a.action_set, episodes=a.episodes, total_selections=int(chosen_count.sum()), candidates=summary, episodes_detail=rows, probability_definition="PPO/ISAC masked action probability; DQN masked Q softmax, plus deterministic argmax selection frequency")
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False));
    with a.out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    env.close(); print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
