"""
MADDPG for Vascular Navigation (Lowe et al. 2017)

Multi-Agent DDPG addresses non-stationarity via centralized training:
- Each agent has its own actor π_i(obs_i) and critic Q_i(obs_1:n, a_1:n)
- Critics observe all agents during training (centralized)
- Actors use only local obs during execution (decentralized)
- Deterministic policy + OU noise for exploration

Key advantages over MAPPO:
1. Continuous actions (no discretization)
2. Centralized critics stabilize learning (no moving target)
3. Replay buffer for sample efficiency
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
from collections import deque
import random


class Actor(nn.Module):
    """Per-agent actor network (decentralized, local obs only)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # action range [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, obs_dim]
        Returns:
            action: [batch, action_dim] in [-1, 1]
        """
        return self.net(obs)


class Critic(nn.Module):
    """Centralized critic Q_i(obs_all, actions_all, state).

    Permutation-invariant in the *other* agents: each peer's (obs, action) pair is
    embedded independently and the embeddings are mean-pooled. Concatenating all
    agents, as the original did, has two costs -- the input width scales with
    `n_agents` so the network must be retrained from scratch when the swarm size
    changes, and the critic has to learn separately that peer 3 and peer 7 are
    interchangeable, which they are.

    The evaluated agent's own (obs, action) is kept as a distinct input, since
    Q_i is emphatically not symmetric in the agent being evaluated.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 0,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        pair_dim = obs_dim + action_dim
        self.self_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.ReLU(),
        )
        self.peer_encoder = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.ReLU(),
        )
        head_in = 2 * hidden_dim + state_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs_all: torch.Tensor,      # [batch, n_agents, obs_dim]
        actions_all: torch.Tensor,  # [batch, n_agents, action_dim]
        agent_index: int = 0,
        state: torch.Tensor | None = None,  # [batch, state_dim]
    ) -> torch.Tensor:
        """
        Returns:
            q_value: [batch, 1]
        """
        pairs = torch.cat([obs_all, actions_all], dim=2)  # [B, N, obs+act]
        own = self.self_encoder(pairs[:, agent_index])     # [B, H]

        n = pairs.size(1)
        if n > 1:
            mask = torch.ones(n, dtype=torch.bool, device=pairs.device)
            mask[agent_index] = False
            peers = self.peer_encoder(pairs[:, mask])      # [B, N-1, H]
            pooled = peers.mean(dim=1)
        else:
            pooled = torch.zeros_like(own)

        parts = [own, pooled]
        if self.state_dim > 0:
            if state is None:
                raise ValueError("critic was built with state_dim > 0 but got no state")
            parts.append(state)
        return self.head(torch.cat(parts, dim=1))

    def forward_all(
        self,
        obs_all: torch.Tensor,      # [batch, n_agents, obs_dim]
        actions_all: torch.Tensor,  # [batch, n_agents, action_dim]
        state: torch.Tensor | None = None,  # [batch, state_dim]
    ) -> torch.Tensor:
        """Q for every agent in one pass. Returns [batch, n_agents].

        Evaluating agents in a Python loop costs N sequential forward passes per
        update, which dominated training throughput (~10 ms/update at N=3, so the
        trainer ran at ~96 steps/s no matter how fast the environment was). The
        peer pooling is a mean over all agents except self, which can be computed
        for every agent at once from the total sum:

            pooled_i = (sum_j e_j - e_i) / (N - 1)
        """
        b, n, _ = obs_all.shape
        pairs = torch.cat([obs_all, actions_all], dim=2)   # [B, N, obs+act]
        flat = pairs.reshape(b * n, -1)

        own = self.self_encoder(flat).reshape(b, n, -1)    # [B, N, H]
        peer = self.peer_encoder(flat).reshape(b, n, -1)   # [B, N, H]

        if n > 1:
            total = peer.sum(dim=1, keepdim=True)          # [B, 1, H]
            pooled = (total - peer) / (n - 1)              # [B, N, H]
        else:
            pooled = torch.zeros_like(own)

        parts = [own, pooled]
        if self.state_dim > 0:
            if state is None:
                raise ValueError("critic was built with state_dim > 0 but got no state")
            parts.append(state.unsqueeze(1).expand(b, n, self.state_dim))
        x = torch.cat(parts, dim=2).reshape(b * n, -1)
        return self.head(x).reshape(b, n)

    def forward_all_policy_grad(
        self,
        obs_all: torch.Tensor,        # [batch, n_agents, obs_dim]
        fresh_actions: torch.Tensor,  # [batch, n_agents, action_dim] (grad)
        peer_actions: torch.Tensor,   # [batch, n_agents, action_dim] (detached)
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Q_i for every i, with gradient flowing ONLY through agent i's action.

        This is the MADDPG actor objective, exactly, in O(N) work rather than the
        O(N^2) of materialising one joint-action tensor per agent. It exploits the
        pooling structure: Q_i depends on agent i's action solely through the self
        encoder, while the peers enter through a mean of peer embeddings. So the
        peer term can be built once from detached actions and reused for all i:

            pooled_i = (sum_j peer(o_j, a_j.detach()) - peer(o_i, a_i.detach()))/(N-1)

        Returns [batch, n_agents].
        """
        b, n, _ = obs_all.shape
        own_pairs = torch.cat([obs_all, fresh_actions], dim=2).reshape(b * n, -1)
        own = self.self_encoder(own_pairs).reshape(b, n, -1)

        with torch.no_grad():
            peer_pairs = torch.cat([obs_all, peer_actions], dim=2).reshape(b * n, -1)
            peer = self.peer_encoder(peer_pairs).reshape(b, n, -1)
            if n > 1:
                pooled = (peer.sum(dim=1, keepdim=True) - peer) / (n - 1)
            else:
                pooled = torch.zeros_like(peer)

        parts = [own, pooled]
        if self.state_dim > 0:
            if state is None:
                raise ValueError("critic was built with state_dim > 0 but got no state")
            parts.append(state.unsqueeze(1).expand(b, n, self.state_dim))
        x = torch.cat(parts, dim=2).reshape(b * n, -1)
        return self.head(x).reshape(b, n)


class ReplayBuffer:
    """Transition replay buffer with per-agent rewards and global state.

    Two additions over the original, both closing gaps where the environment
    computed information that training then threw away:

    * `rewards` is per-agent, shape [n_agents]. Storing a single team scalar made
      every critic regress the same target, so N critics learned N copies of one
      function -- N times the compute for zero extra information. The environment
      already decomposes the reward in `info["agent_rewards"]`.
    * `state` carries the global clot state, so the centralized critic can see
      task progress that no single agent observes. This is legitimate under CTDE:
      the critic is training-only.
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def store(
        self,
        obs: np.ndarray,           # [n_agents, obs_dim]
        actions: np.ndarray,       # [n_agents, action_dim]
        rewards: np.ndarray,       # [n_agents] per-agent reward
        next_obs: np.ndarray,      # [n_agents, obs_dim]
        done: bool,
        state: np.ndarray | None = None,       # [state_dim] global critic input
        next_state: np.ndarray | None = None,  # [state_dim]
    ):
        self.buffer.append({
            'obs': obs,
            'actions': actions,
            'rewards': np.asarray(rewards, dtype=np.float32),
            'next_obs': next_obs,
            'done': done,
            'state': state,
            'next_state': next_state,
        })

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)

        out = {
            'obs': torch.FloatTensor(np.array([t['obs'] for t in batch])),
            'actions': torch.FloatTensor(np.array([t['actions'] for t in batch])),
            'rewards': torch.FloatTensor(np.array([t['rewards'] for t in batch])),
            'next_obs': torch.FloatTensor(np.array([t['next_obs'] for t in batch])),
            'done': torch.FloatTensor([[t['done']] for t in batch]),
        }
        if batch[0]['state'] is not None:
            out['state'] = torch.FloatTensor(np.array([t['state'] for t in batch]))
            out['next_state'] = torch.FloatTensor(
                np.array([t['next_state'] for t in batch])
            )
        return out

    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration."""

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, rng: np.random.Generator | None = None):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self._rng = rng or np.random.default_rng()
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * self._rng.standard_normal(
            self.action_dim
        )
        self.state = x + dx
        return self.state.copy()


class MADDPG:
    """Multi-Agent DDPG trainer."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,  # soft update rate
        buffer_size: int = 100000,
        batch_size: int = 256,
        device: str = 'cuda',
        state_dim: int = 0,
        share_parameters: bool = True,
        seed: int | None = None,
    ):
        """
        Args:
            state_dim: width of the extra global state handed to the critic. 0
                disables it.
            share_parameters: share one actor and one critic across all agents.
                The robots here are homogeneous -- same dynamics, same action
                space, same egocentric observation layout -- so separate networks
                split the experience N ways to learn N copies of one function.
                Sharing multiplies the effective sample size by N. Agents stay
                distinguishable through their observations, not their weights.
        """
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.state_dim = state_dim
        self.share_parameters = bool(share_parameters)

        n_nets = 1 if self.share_parameters else n_agents

        def make_actor() -> Actor:
            return Actor(obs_dim, action_dim, hidden_dim).to(device)

        def make_critic() -> Critic:
            return Critic(
                n_agents, obs_dim, action_dim, hidden_dim, state_dim=state_dim
            ).to(device)

        self.actors = [make_actor() for _ in range(n_nets)]
        self.target_actors = [make_actor() for _ in range(n_nets)]
        self.critics = [make_critic() for _ in range(n_nets)]
        self.target_critics = [make_critic() for _ in range(n_nets)]

        # Copy weights to target networks
        for i in range(n_nets):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        # Optimizers
        self.actor_optimizers = [
            torch.optim.Adam(self.actors[i].parameters(), lr=actor_lr)
            for i in range(n_nets)
        ]
        self.critic_optimizers = [
            torch.optim.Adam(self.critics[i].parameters(), lr=critic_lr)
            for i in range(n_nets)
        ]

        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        rng = np.random.default_rng(seed)
        self.noises = [OUNoise(action_dim, rng=rng) for _ in range(n_agents)]

    # -- network access ------------------------------------------------------
    # With sharing there is one network serving every agent; without it there is
    # one per agent. Routing every access through these keeps the update logic
    # identical in both modes.

    def actor(self, i: int) -> Actor:
        return self.actors[0 if self.share_parameters else i]

    def target_actor(self, i: int) -> Actor:
        return self.target_actors[0 if self.share_parameters else i]

    def critic(self, i: int) -> Critic:
        return self.critics[0 if self.share_parameters else i]

    def target_critic(self, i: int) -> Critic:
        return self.target_critics[0 if self.share_parameters else i]

    def select_actions(
        self,
        obs: np.ndarray,        # [n_agents, obs_dim]
        add_noise: bool = True,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Select actions for all agents.

        Returns:
            actions: [n_agents, action_dim]
        """
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), device=self.device
        )
        with torch.no_grad():
            if self.share_parameters:
                # One batched forward for the whole swarm instead of N calls.
                actions = self.actors[0](obs_t).cpu().numpy()
            else:
                actions = np.stack([
                    self.actor(i)(obs_t[i : i + 1]).cpu().numpy()[0]
                    for i in range(self.n_agents)
                ])

        if add_noise:
            noise = np.stack([self.noises[i].sample() for i in range(self.n_agents)])
            actions = np.clip(actions + noise_scale * noise, -1.0, 1.0)
        return actions.astype(np.float32)

    def update(self) -> Dict[str, float]:
        """
        Sample batch and update all agents.

        Returns:
            dict with loss metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)

        obs = batch['obs'].to(self.device)              # [batch, n_agents, obs_dim]
        actions = batch['actions'].to(self.device)      # [batch, n_agents, action_dim]
        rewards = batch['rewards'].to(self.device)      # [batch, n_agents]
        next_obs = batch['next_obs'].to(self.device)    # [batch, n_agents, obs_dim]
        done = batch['done'].to(self.device)            # [batch, 1]
        state = batch['state'].to(self.device) if 'state' in batch else None
        next_state = batch['next_state'].to(self.device) if 'next_state' in batch else None

        # Target actions come from the target actors for EVERY agent, computed
        # once and reused across the per-agent critic updates.
        with torch.no_grad():
            next_actions = torch.stack(
                [self.target_actor(j)(next_obs[:, j]) for j in range(self.n_agents)],
                dim=1,
            )

        if self.share_parameters:
            return self._update_shared(
                obs, actions, rewards, next_obs, next_actions, done, state, next_state
            )

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_q = 0.0

        for agent_i in range(self.n_agents):
            # ========== Update Critic ==========
            with torch.no_grad():
                target_q = self.target_critic(agent_i)(
                    next_obs, next_actions, agent_index=agent_i, state=next_state
                )
                # Per-agent reward: each critic now regresses a target specific to
                # its own agent instead of N critics fitting one shared scalar.
                y = rewards[:, agent_i : agent_i + 1] + self.gamma * target_q * (1 - done)

            current_q = self.critic(agent_i)(
                obs, actions, agent_index=agent_i, state=state
            )
            critic_loss = F.mse_loss(current_q, y)

            opt_c = self.critic_optimizers[0 if self.share_parameters else agent_i]
            opt_c.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic(agent_i).parameters(), 0.5)
            opt_c.step()

            total_critic_loss += critic_loss.item()
            total_q += float(current_q.mean().item())

            # ========== Update Actor ==========
            # Peers use their CURRENT target-free actor output, detached. The
            # original substituted the *replayed* actions for peers, which
            # evaluates Q on a stale joint-action distribution -- the peers'
            # policies have moved on since those actions were stored.
            current_actions = []
            for j in range(self.n_agents):
                if j == agent_i:
                    current_actions.append(self.actor(j)(obs[:, j]))
                else:
                    with torch.no_grad():
                        current_actions.append(self.actor(j)(obs[:, j]))
            current_actions = torch.stack(current_actions, dim=1)

            actor_loss = -self.critic(agent_i)(
                obs, current_actions, agent_index=agent_i, state=state
            ).mean()

            opt_a = self.actor_optimizers[0 if self.share_parameters else agent_i]
            opt_a.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor(agent_i).parameters(), 0.5)
            opt_a.step()

            total_actor_loss += actor_loss.item()

        # Soft update target networks
        self._soft_update()

        return {
            'actor_loss': total_actor_loss / self.n_agents,
            'critic_loss': total_critic_loss / self.n_agents,
            'mean_q': total_q / self.n_agents,
        }

    def _update_shared(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_actions: torch.Tensor,
        done: torch.Tensor,
        state: torch.Tensor | None,
        next_state: torch.Tensor | None,
    ) -> Dict[str, float]:
        """One batched critic+actor update covering every agent at once.

        Mathematically the same updates as the per-agent loop when parameters are
        shared -- the shared optimiser would apply the sum of the per-agent
        gradients anyway -- but with two forward/backward passes instead of 2N.
        """
        critic, actor = self.critics[0], self.actors[0]
        b, n = obs.shape[0], self.n_agents

        # ---- critic ----
        with torch.no_grad():
            target_q = self.target_critics[0].forward_all(
                next_obs, next_actions, state=next_state
            )                                            # [B, N]
            y = rewards + self.gamma * target_q * (1 - done)

        current_q = critic.forward_all(obs, actions, state=state)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizers[0].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        self.critic_optimizers[0].step()

        # ---- actor ----
        # Every agent's action is recomputed from the current actor. Under sharing
        # the peers' actions come from the same network, so a detached copy is
        # used for the peer slots to keep the gradient for agent i flowing only
        # through its own action -- matching the per-agent formulation.
        fresh = actor(obs.reshape(b * n, -1)).reshape(b, n, -1)
        q_own = critic.forward_all_policy_grad(
            obs, fresh, fresh.detach(), state=state
        )                                                   # [B, N]
        actor_loss = -q_own.mean()

        self.actor_optimizers[0].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        self.actor_optimizers[0].step()

        self._soft_update()
        return {
            'actor_loss': float(actor_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'mean_q': float(current_q.mean().item()),
        }

    def _soft_update(self):
        """Soft update target networks: θ' ← τθ + (1-τ)θ'"""
        for i in range(len(self.actors)):
            with torch.no_grad():
                for tp, p in zip(self.target_actors[i].parameters(),
                                 self.actors[i].parameters()):
                    tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
                for tp, p in zip(self.target_critics[i].parameters(),
                                 self.critics[i].parameters()):
                    tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

    def reset_noise(self):
        """Reset OU noise for all agents (call at episode start)."""
        for noise in self.noises:
            noise.reset()

    def save(self, path: str):
        """Save all agent networks plus the shape metadata needed to reload."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'meta': {
                'n_agents': self.n_agents,
                'obs_dim': self.obs_dim,
                'action_dim': self.action_dim,
                'state_dim': self.state_dim,
                'share_parameters': self.share_parameters,
            },
        }, path)

    def load(self, path: str, strict: bool = True):
        """Load networks, tolerating checkpoints from a different agent count.

        Older checkpoints stored one network per agent with no metadata. When the
        stored count does not match, the first entry is broadcast, which is valid
        precisely because the networks are homogeneous.
        """
        checkpoint = torch.load(path, map_location=self.device)
        stored_actors = checkpoint['actors']
        stored_critics = checkpoint.get('critics', [])

        for i in range(len(self.actors)):
            sd = stored_actors[i] if i < len(stored_actors) else stored_actors[0]
            self.actors[i].load_state_dict(sd, strict=strict)
            self.target_actors[i].load_state_dict(sd, strict=strict)

        if not stored_critics:
            return
        for i in range(len(self.critics)):
            sd = stored_critics[i] if i < len(stored_critics) else stored_critics[0]
            try:
                self.critics[i].load_state_dict(sd, strict=strict)
                self.target_critics[i].load_state_dict(sd, strict=strict)
            except RuntimeError:
                # A checkpoint from the old concatenating critic has an
                # incompatible architecture. The actors are what matter for
                # evaluation, so keep those and leave the critics freshly
                # initialised rather than failing the load outright.
                break
