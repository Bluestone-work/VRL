"""Discrete independent local RL baselines for vascular navigation.

The ``isac`` mode is a parameter-shared independent discrete SAC learner:
each cluster contributes one local transition, while the actor and two critics
share weights.  The ``dqn`` mode uses SB3's target-network DQN with a dueling head through
SB3's replay/target-update implementation.  Both modes use the same deployable
candidate observations and protection layer as IPPO.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import create_mlp

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import (
    ATOMIC_CANDIDATE_COUNT,
    IPPOAgentVecEnv,
    IPPOCallback,
    IPPOExtractor,
    SEMANTIC_CANDIDATE_COUNT,
)


class DuelingQNetwork(QNetwork):
    """Dueling Q head while reusing the deployable IPPO feature encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.features_dim
        n = int(self.action_space.n)
        self.advantage = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, n))
        self.value = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1))
        self.q_net = nn.Identity()

    def forward(self, obs):
        f = self.extract_features(obs, self.features_extractor)
        a = self.advantage(f)
        v = self.value(f)
        return v + a - a.mean(dim=1, keepdim=True)


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


def obs_tensor(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


class DiscreteSAC:
    def __init__(self, env, action_count, seed, device="cuda", gamma=.995, tau=.005, reward_clip=10.):
        self.env = env
        self.n = int(action_count)
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.actor = nn.Sequential(IPPOExtractor(env.observation_space), nn.Linear(128, self.n)).to(self.device)
        self.q1 = nn.Sequential(IPPOExtractor(env.observation_space), nn.Linear(128, self.n)).to(self.device)
        self.q2 = nn.Sequential(IPPOExtractor(env.observation_space), nn.Linear(128, self.n)).to(self.device)
        self.tq1 = nn.Sequential(IPPOExtractor(env.observation_space), nn.Linear(128, self.n)).to(self.device)
        self.tq2 = nn.Sequential(IPPOExtractor(env.observation_space), nn.Linear(128, self.n)).to(self.device)
        self.tq1.load_state_dict(self.q1.state_dict()); self.tq2.load_state_dict(self.q2.state_dict())
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4)
        self.log_alpha = torch.tensor(np.log(.1), device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.gamma, self.tau, self.reward_clip = gamma, tau, float(reward_clip)
        # SAC target entropy is a positive entropy target.  The previous pilot
        # used a negative target with the opposite sign and drove alpha upward.
        self.target_entropy = float(np.log(max(2, self.n)) * .7)

    def logits(self, obs):
        return self.actor(obs)

    @staticmethod
    def masked_dist(logits, valid):
        logits = logits.masked_fill(valid < .5, -1e9)
        logp = torch.log_softmax(logits, dim=-1)
        p = logp.exp()
        return p, logp

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            t = obs_tensor(obs, self.device)
            p, logp = self.masked_dist(self.logits(t), t["candidate_valid"])
            return (p.argmax(-1) if deterministic else torch.multinomial(p, 1).squeeze(-1)).cpu().numpy()

    def update(self, batch):
        o, a, r, no, d = batch
        o, no = obs_tensor(o, self.device), obs_tensor(no, self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(np.clip(r, -self.reward_clip, self.reward_clip), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p, lp = self.masked_dist(self.logits(no), no["candidate_valid"])
            tq = torch.minimum(self.tq1(no), self.tq2(no))
            nv = (p * (tq - self.log_alpha.exp() * lp)).sum(-1)
            target = r + self.gamma * (1. - d) * nv
        q1 = self.q1(o).gather(1, a[:, None]).squeeze(1)
        q2 = self.q2(o).gather(1, a[:, None]).squeeze(1)
        qloss = .5 * ((q1 - target).square().mean() + (q2 - target).square().mean())
        self.opt_q.zero_grad(); qloss.backward(); nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.); self.opt_q.step()
        p, lp = self.masked_dist(self.logits(o), o["candidate_valid"])
        q = torch.minimum(self.q1(o), self.q2(o)).detach()
        aloss = (p * (self.log_alpha.exp() * lp - q)).sum(-1).mean()
        self.opt_actor.zero_grad(); aloss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 10.); self.opt_actor.step()
        entropy = -(p * lp).sum(-1).detach()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()
        with torch.no_grad():
            self.log_alpha.clamp_(-5., 1.)
        with torch.no_grad():
            for t, s in zip(self.tq1.parameters(), self.q1.parameters()): t.mul_(1-self.tau).add_(self.tau*s)
            for t, s in zip(self.tq2.parameters(), self.q2.parameters()): t.mul_(1-self.tau).add_(self.tau*s)
        return dict(q_loss=float(qloss), actor_loss=float(aloss), alpha=float(self.log_alpha.exp()), entropy=float(entropy.mean()))

    def save(self, path):
        torch.save(dict(actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(), tq1=self.tq1.state_dict(), tq2=self.tq2.state_dict(), log_alpha=self.log_alpha.detach().cpu()), path)


def evaluate_custom(agent, out, clusters, action_set, count):
    env = IPPOAgentVecEnv(CurriculumVascularEnv(seed=1041, clusters=clusters, curriculum=False,
        reward_mode="navigation", zero_latency=True, direct_action=True), action_set=action_set)
    rows = []
    obs = env.reset()
    while len(rows) < count:
        act = agent.act(obs, deterministic=True)
        obs, _, done, infos = env.step(act)
        if bool(done[0]):
            inf = infos[0]
            rows.append(dict(navigation_success=bool(inf.get("navigation_success", False)),
                wall_contact_s=float(np.asarray(inf.get("agent_wall_contact_s", 0.)).sum()),
                lost=int(np.asarray(inf.get("agent_lost", 0.)).sum()),
                particle_collision_events=int(np.asarray(inf.get("agent_particle_collision_events", 0.)).sum())))
    summary = dict(episodes=len(rows), navigation_success=float(np.mean([x["navigation_success"] for x in rows])), split="development_random_no_latency", deterministic=True)
    (Path(out) / "evaluation.json").write_text(json.dumps(summary, indent=2)); env.close(); return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", choices=("isac", "dqn"), required=True)
    p.add_argument("--action-set", choices=("semantic", "atomic"), default="semantic")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--seed", type=int, default=41)
    p.add_argument("--timesteps", type=int, default=100000); p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--device", default="cuda"); p.add_argument("--eval-count", type=int, default=12)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1); random.seed(a.seed); np.random.seed(a.seed)
    physical = CurriculumVascularEnv(seed=a.seed, clusters=a.clusters, budget=a.timesteps, curriculum=True, prediction=True, shaping=True, reward_mode="navigation", zero_latency=True, direct_action=True, episode_log=a.out/"episodes.jsonl")
    env = IPPOAgentVecEnv(physical, action_set=a.action_set); k = SEMANTIC_CANDIDATE_COUNT if a.action_set == "semantic" else ATOMIC_CANDIDATE_COUNT
    cfg = dict(algorithm=("ISAC_discrete_shared" if a.algorithm == "isac" else "DuelingDQN_shared"), action_set=a.action_set, action_count=k, physical_steps=a.timesteps, agent_steps=a.timesteps*a.clusters, seed=a.seed, no_privileged_actor=True, reward_clip=(10. if a.algorithm == "isac" else None), source_script=str(Path(__file__).resolve()))
    (a.out/"config.json").write_text(json.dumps(cfg, indent=2))
    if a.algorithm == "dqn":
        model = DQN(DuelingDQNPolicy, env, seed=a.seed, device=a.device, learning_rate=1e-4, buffer_size=100000, learning_starts=2000, batch_size=256, train_freq=4, gradient_steps=1, target_update_interval=4000, exploration_fraction=.25, exploration_final_eps=.05, gamma=.995, policy_kwargs=dict(features_extractor_class=IPPOExtractor, normalize_images=False, net_arch=[128,128]), verbose=0)
        model.set_logger(configure(str(a.out), ["csv","json","tensorboard"])); model.save(a.out/"initial_model"); started=time.monotonic(); model.learn(total_timesteps=a.timesteps*a.clusters, callback=IPPOCallback(a.out,a.timesteps,a.clusters)); model.save(a.out/"final_model"); (a.out/"TRAINING_COMPLETE.json").write_text(json.dumps(dict(environment_steps=a.timesteps, agent_steps=model.num_timesteps, training_seconds=time.monotonic()-started), indent=2))
        # Reuse the adapter's deterministic predictor for the standard summary.
        from scripts.train_sb3_ippo import evaluate
        result=evaluate(model,a.out,a.clusters,a.eval_count,a.action_set)
    else:
        agent=DiscreteSAC(env,k,a.seed,a.device); agent.save(a.out/"initial_model.pt"); replay=deque(maxlen=100000); obs=env.reset(); started=time.monotonic(); updates=[]
        for step in range(a.timesteps):
            act=agent.act(obs); no,r,d,infos=env.step(act)
            for i in range(a.clusters): replay.append(({key:obs[key][i].copy() for key in obs},int(act[i]),float(r[i]),{key:no[key][i].copy() for key in no},float(d[i])))
            obs=no
            if len(replay)>=2000:
                batch=random.sample(replay, min(256,len(replay))); keys=batch[0][0].keys(); bo={k:np.stack([x[0][k] for x in batch]) for k in keys}; bno={k:np.stack([x[3][k] for x in batch]) for k in keys}; stats=agent.update((bo,np.array([x[1] for x in batch]),np.array([x[2] for x in batch]),bno,np.array([x[4] for x in batch]))); updates.append(stats)
            if step%256==0:
                (a.out/"status.json").write_text(json.dumps(dict(environment_steps=step, agent_steps=step*a.clusters, target_environment_steps=a.timesteps, **(updates[-1] if updates else {})), indent=2))
        agent.save(a.out/"final_model.pt"); (a.out/"TRAINING_COMPLETE.json").write_text(json.dumps(dict(environment_steps=a.timesteps, agent_steps=a.timesteps*a.clusters, training_seconds=time.monotonic()-started), indent=2)); result=evaluate_custom(agent,a.out,a.clusters,a.action_set,a.eval_count)
    (a.out/"RUN_COMPLETE.json").write_text(json.dumps(result, indent=2)); env.close()


if __name__ == "__main__": main()
