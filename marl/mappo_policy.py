"""
Multi-Agent Proximal Policy Optimization (MAPPO) for Vascular Navigation

MAPPO (Yu et al. 2021) is an on-policy MARL algorithm that extends PPO to
multi-agent settings with centralized training and decentralized execution (CTDE).

Key features:
- On-policy learning (no replay buffer)
- Clipped surrogate objective for stable updates
- Value function baseline for variance reduction
- GAE (Generalized Advantage Estimation) for advantage computation
- Centralized critic with full observability during training
- Decentralized actor with local observations during execution

Compared to MADDPG:
- Better sample efficiency on cooperative tasks
- More stable training (on-policy)
- No replay buffer needed (simpler)
- Works well with continuous actions via Gaussian policy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import deque

from marl.gat_policy import GATActor, GATCritic, build_adjacency_matrix


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy for a bounded continuous action space.

    Checkpoint compatibility is preserved: the same ``mean`` linear layer and
    ``log_std`` parameter are used, and deterministic actions remain
    ``tanh(mean(features))``. Stochastic actions now have a probability density
    that matches the action actually executed by the environment.
    """

    def __init__(self, feature_dim: int, action_dim: int, log_std_init: float = 0.0):
        super().__init__()
        self.action_dim = action_dim

        self.mean = nn.Linear(feature_dim, action_dim)
        # Learnable log std (independent of state)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [batch, n_agents, feature_dim]
        Returns:
            mean: [batch, n_agents, action_dim]
            std: [batch, n_agents, action_dim]
        """
        mean = torch.tanh(self.mean(features))
        std = torch.exp(self.log_std.clamp(-5.0, 2.0)).expand_as(mean)
        return mean, std

    def _distribution(self, features: torch.Tensor):
        location = self.mean(features)
        std = torch.exp(self.log_std.clamp(-5.0, 2.0)).expand_as(location)
        return torch.distributions.Normal(location, std)

    @staticmethod
    def _atanh(action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return 0.5 * (torch.log1p(action) - torch.log1p(-action))

    @staticmethod
    def _squashed_log_prob(
        dist: torch.distributions.Normal,
        raw_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)

    def get_actions(self, features: torch.Tensor, deterministic: bool = False):
        dist = self._distribution(features)
        raw_action = dist.mean if deterministic else dist.sample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        return action, log_prob, -log_prob

    def evaluate_actions(self, features: torch.Tensor, actions: torch.Tensor):
        dist = self._distribution(features)
        raw_action = self._atanh(actions)
        log_prob = self._squashed_log_prob(dist, raw_action, actions)
        return log_prob, -log_prob


class GATActorStochastic(nn.Module):
    """GAT-based stochastic actor for MAPPO."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        from marl.gat_policy import GATEncoder

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # GAT encoder
        self.gat_encoder = GATEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
        )

        # Gaussian policy head
        self.policy_head = GaussianActor(hidden_dim, action_dim, log_std_init)

    def forward(
        self,
        obs: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            adj_matrix: [batch, n_agents, n_agents]
        Returns:
            mean: [batch, n_agents, action_dim]
            std: [batch, n_agents, action_dim]
        """
        # Handle single sample case
        squeeze_batch = False
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
            squeeze_batch = True
            if adj_matrix is not None:
                adj_matrix = adj_matrix.unsqueeze(0)

        # Encode with GAT
        features = self.gat_encoder(obs, adj_matrix)

        # Get policy distribution
        mean, std = self.policy_head(features)

        if squeeze_batch:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std

    def get_actions(
        self,
        obs: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions and compute log probabilities.

        Returns:
            actions: [batch, n_agents, action_dim]
            log_probs: [batch, n_agents]
            entropy: [batch, n_agents]
        """
        features = self.gat_encoder(obs, adj_matrix)
        return self.policy_head.get_actions(features, deterministic)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Args:
            obs: [batch, n_agents, obs_dim]
            actions: [batch, n_agents, action_dim]
            adj_matrix: [batch, n_agents, n_agents]
        Returns:
            log_probs: [batch, n_agents]
            entropy: [batch, n_agents]
        """
        features = self.gat_encoder(obs, adj_matrix)
        return self.policy_head.evaluate_actions(features, actions)


class RolloutBuffer:
    """Buffer for storing on-policy rollout data."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.adj_matrices = []
        self.states = []  # Global states for critic
        self.next_obs = []
        self.next_adj_matrices = []
        self.next_states = []

    def store(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        adj_matrix: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        next_obs: Optional[np.ndarray] = None,
        next_adj_matrix: Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
    ):
        self.obs.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.adj_matrices.append(adj_matrix)
        self.states.append(state)
        self.next_obs.append(next_obs)
        self.next_adj_matrices.append(next_adj_matrix)
        self.next_states.append(next_state)

    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored data and clear buffer."""
        data = {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
        }

        if self.adj_matrices[0] is not None:
            data['adj_matrices'] = np.array(self.adj_matrices)
        if self.states[0] is not None:
            data['states'] = np.array(self.states)
        if self.next_obs[0] is not None:
            data['next_obs'] = np.array(self.next_obs)
        if self.next_adj_matrices[0] is not None:
            data['next_adj_matrices'] = np.array(self.next_adj_matrices)
        if self.next_states[0] is not None:
            data['next_states'] = np.array(self.next_states)

        self.clear()
        return data

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.adj_matrices.clear()
        self.states.clear()
        self.next_obs.clear()
        self.next_adj_matrices.clear()
        self.next_states.clear()

    def __len__(self):
        return len(self.obs)


class MAPPO:
    """Multi-Agent Proximal Policy Optimization."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 0,
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip: float = 10.0,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_gat: bool = True,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_gat = use_gat

        # Networks
        if use_gat:
            self.actor = GATActorStochastic(
                obs_dim, action_dim, hidden_dim, num_gat_layers, num_heads
            ).to(device)
            self.critic = GATCritic(
                obs_dim, action_dim, hidden_dim, num_gat_layers, num_heads, state_dim
            ).to(device)
        else:
            # Fallback to simple MLP (for ablation)
            from marl.maddpg_policy import Actor, Critic
            self.actor = Actor(obs_dim, action_dim, hidden_dim).to(device)
            self.critic = Critic(n_agents, obs_dim, action_dim, hidden_dim, state_dim).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout buffer
        self.buffer = RolloutBuffer()

    def select_action(
        self,
        obs: np.ndarray,
        positions: Optional[np.ndarray] = None,
        state: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for all agents.

        Args:
            obs: [n_agents, obs_dim]
            positions: [n_agents, 3] for building adjacency matrix
            state: [state_dim] global state for critic
            deterministic: if True, use mean action
        Returns:
            actions: [n_agents, action_dim]
            log_probs: [n_agents]
            values: [n_agents]
            adj_matrix: [n_agents, n_agents] or None
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # [1, N, obs_dim]

            # Build adjacency matrix if using GAT
            adj_matrix = None
            if self.use_gat and positions is not None:
                pos_t = torch.FloatTensor(positions).unsqueeze(0).to(self.device)
                adj_matrix = build_adjacency_matrix(pos_t, threshold=0.15)
                adj_matrix_np = adj_matrix.squeeze(0).cpu().numpy()
            else:
                adj_matrix_np = None

            # Convert state to tensor if provided
            state_t = None
            if state is not None:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, state_dim]

            # Get actions from actor
            actions, log_probs, _ = self.actor.get_actions(obs_t, adj_matrix, deterministic)

            # Get values from critic
            actions_for_critic = actions.detach()
            if self.use_gat:
                values = self.critic(obs_t, actions_for_critic, adj_matrix, state_t)
            else:
                values = self.critic(obs_t, actions_for_critic, 0, state_t)

            # Convert to numpy
            actions = actions.squeeze(0).cpu().numpy()
            log_probs = log_probs.squeeze(0).cpu().numpy()
            values = values.squeeze(0).squeeze(-1).cpu().numpy()

        return actions, log_probs, values, adj_matrix_np

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: [T, n_agents]
            values: [T, n_agents]
            dones: [T, n_agents]
            last_value: [n_agents]
        Returns:
            advantages: [T, n_agents]
            returns: [T, n_agents]
        """
        T = len(rewards)
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros_like(last_value)  # Initialize with same shape as last_value

        # Compute advantages backwards
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def update(
        self,
        n_epochs: int = 10,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """
        Update actor and critic using PPO.

        Returns:
            metrics: dict with loss values
        """
        # Get rollout data
        data = self.buffer.get()

        obs = torch.FloatTensor(data['obs']).to(self.device)  # [T, N, obs_dim]
        actions = torch.FloatTensor(data['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
        rewards = data['rewards']  # [T, N]
        dones = data['dones']
        values = data['values']

        adj_matrices = None
        if 'adj_matrices' in data:
            adj_matrices = torch.FloatTensor(data['adj_matrices']).to(self.device)

        states = None
        if 'states' in data:
            states = torch.FloatTensor(data['states']).to(self.device)

        # Compute last value for GAE
        with torch.no_grad():
            if 'next_obs' in data:
                last_obs = torch.as_tensor(
                    data['next_obs'][-1:], dtype=torch.float32, device=self.device
                )
                last_adj = (
                    torch.as_tensor(
                        data['next_adj_matrices'][-1:],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if 'next_adj_matrices' in data else None
                )
                last_state = (
                    torch.as_tensor(
                        data['next_states'][-1:],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if 'next_states' in data else None
                )
                last_actions, _, _ = self.actor.get_actions(
                    last_obs, last_adj, deterministic=True
                )
            else:
                last_obs = obs[-1:]
                last_actions = actions[-1:]
                last_adj = adj_matrices[-1:] if adj_matrices is not None else None
                last_state = states[-1:] if states is not None else None

            if self.use_gat:
                last_value = self.critic(last_obs, last_actions, last_adj, last_state)
            else:
                last_value = self.critic(last_obs, last_actions, 0, last_state)
            last_value = last_value.squeeze(0).squeeze(-1).cpu().numpy()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Mini-batches are drawn over TIMESTEPS, not over flattened agent-steps.
        # Flattening agents into the batch dimension would hand the GAT layers a
        # graph of one node per sample, so the attention block would collapse to
        # a per-agent MLP and the whole point of the architecture is lost: the
        # policy would be trained under a different model than it acts under.
        T, N = obs.shape[0], obs.shape[1]
        old_values = torch.FloatTensor(values).to(self.device)  # [T, N]

        # A "batch" is expressed in agent-steps for comparability with the flat
        # version, so convert to a whole number of timesteps (at least one).
        steps_per_batch = max(1, batch_size // N)

        metrics = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        n_updates = 0

        for epoch in range(n_epochs):
            indices = np.random.permutation(T)

            for start in range(0, T, steps_per_batch):
                batch_indices = indices[start:start + steps_per_batch]

                # Mini-batch data, keeping the [batch, n_agents, ...] layout.
                obs_batch = obs[batch_indices]                       # [B, N, obs_dim]
                actions_batch = actions[batch_indices]               # [B, N, act_dim]
                old_log_probs_batch = old_log_probs[batch_indices]   # [B, N]
                advantages_batch = advantages[batch_indices]         # [B, N]
                returns_batch = returns[batch_indices]               # [B, N]
                old_values_batch = old_values[batch_indices]         # [B, N]

                adj_batch = adj_matrices[batch_indices] if adj_matrices is not None else None
                state_batch = states[batch_indices] if states is not None else None

                # Evaluate actions under the same graph the actor acted under.
                log_probs, entropy = self.actor.evaluate_actions(
                    obs_batch, actions_batch, adj_batch
                )
                if self.use_gat:
                    values_pred = self.critic(obs_batch, actions_batch, adj_batch, state_batch)
                else:
                    values_pred = self.critic(obs_batch, actions_batch, 0, state_batch)
                values_pred = values_pred.squeeze(-1)  # [B, N]

                # Actor loss (PPO clip objective)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                # Critic loss. The clip is a trust region around the value
                # estimate that generated the rollout, so it has to reference
                # `old_values`; clipping around the fresh prediction (as an
                # earlier version did) makes the term vanish identically.
                value_pred_clipped = old_values_batch + torch.clamp(
                    values_pred - old_values_batch,
                    -self.value_clip,
                    self.value_clip,
                )
                value_loss1 = (values_pred - returns_batch).pow(2)
                value_loss2 = (value_pred_clipped - returns_batch).pow(2)
                critic_loss = self.value_loss_coef * torch.max(value_loss1, value_loss2).mean()

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Track metrics
                metrics['actor_loss'] += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy'] += entropy.mean().item()
                n_updates += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= n_updates

        return metrics

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
