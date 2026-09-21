"""
Transformer-based Multi-Agent Policy (MAPT - Multi-Agent Policy Transformer)

Replaces GAT with full Transformer architecture for richer agent interactions.
Inspired by recent work on Transformers for MARL (e.g., MAT, TarMAC).

Key advantages over GAT:
1. Self-attention captures all-to-all relationships (not just neighbors)
2. Positional encoding handles spatial structure
3. Scales better to large swarms
4. Pre-training potential on diverse tasks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding for spatial coordinates."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_agents, d_model]
            positions: [batch, n_agents, 3] spatial positions
        Returns:
            x with positional encoding added
        """
        # Simple approach: use first 3 dimensions of PE based on normalized positions
        # More sophisticated: learn a spatial embedding
        batch_size, n_agents, _ = x.shape

        # Normalize positions to [0, max_len)
        pos_normalized = (positions - positions.min()) / (positions.max() - positions.min() + 1e-8)
        pos_indices = (pos_normalized * (self.pe.size(0) - 1)).long()

        # Average PE from x, y, z coordinates
        pe_x = self.pe[pos_indices[:, :, 0]]
        pe_y = self.pe[pos_indices[:, :, 1]]
        pe_z = self.pe[pos_indices[:, :, 2]]
        pe_spatial = (pe_x + pe_y + pe_z) / 3

        return x + pe_spatial


class MultiAgentTransformerLayer(nn.Module):
    """Single Transformer layer for multi-agent communication."""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch, n_agents, d_model]
            src_mask: [n_agents, n_agents] attention mask
            src_key_padding_mask: [batch, n_agents] padding mask
        Returns:
            output: [batch, n_agents, d_model]
        """
        # Self-attention with residual
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward with residual
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Multi-layer Transformer encoder for agent observations."""

    def __init__(
        self,
        obs_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(obs_dim, d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            MultiAgentTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        obs: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            positions: [batch, n_agents, 3] for positional encoding
            mask: [batch, n_agents, n_agents] attention mask
        Returns:
            features: [batch, n_agents, d_model]
        """
        # Project input
        x = self.input_proj(obs)

        # Add positional encoding if positions provided
        if positions is not None:
            x = self.pos_encoder(x, positions)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_mask=mask)

        # Output projection
        x = self.output_proj(x)

        return x


class TransformerActor(nn.Module):
    """Transformer-based actor for MARL."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        log_std_init: float = 0.0,
    ):
        super().__init__()

        # Transformer encoder
        self.encoder = TransformerEncoder(
            obs_dim, d_model, nhead, num_layers
        )

        # Action head (Gaussian policy)
        self.mean_head = nn.Linear(d_model, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def forward(
        self,
        obs: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            positions: [batch, n_agents, 3]
        Returns:
            mean: [batch, n_agents, action_dim]
            std: [batch, n_agents, action_dim]
        """
        # Encode
        features = self.encoder(obs, positions)

        # Policy distribution
        mean = torch.tanh(self.mean_head(features))
        std = torch.exp(self.log_std).expand_as(mean)

        return mean, std

    def get_actions(
        self,
        obs: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions and compute log probabilities."""
        mean, std = self.forward(obs, positions)

        if deterministic:
            actions = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities for given actions."""
        mean, std = self.forward(obs, positions)
        dist = torch.distributions.Normal(mean, std)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, entropy


class TransformerCritic(nn.Module):
    """Transformer-based centralized critic."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        state_dim: int = 0,
        use_action_input: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim

        # Observation encoder
        self.obs_encoder = TransformerEncoder(
            obs_dim, d_model, nhead, num_layers, dropout=dropout
        )

        # Action encoder (simple MLP)
        self.use_action_input = use_action_input
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model // 2),
            nn.ReLU(),
        ) if use_action_input else None

        # State encoder if provided
        if state_dim > 0:
            self.state_encoder = nn.Linear(state_dim, d_model // 2)

        # Value head
        combine_dim = d_model + (d_model // 2 if use_action_input else 0)
        if state_dim > 0:
            combine_dim += d_model // 2

        self.value_head = nn.Sequential(
            nn.Linear(combine_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            actions: [batch, n_agents, action_dim]
            positions: [batch, n_agents, 3]
            state: [batch, state_dim]
        Returns:
            values: [batch, n_agents, 1]
        """
        batch_size, n_agents, _ = obs.shape

        # Encode observations with Transformer
        obs_features = self.obs_encoder(obs, positions)

        # Encode actions
        combined = obs_features
        if self.use_action_input:
            combined = torch.cat([obs_features, self.action_encoder(actions)], dim=-1)

        # Add global state if provided
        if self.state_dim > 0:
            if state is None:
                state = combined.new_zeros(batch_size, self.state_dim)
            state_features = self.state_encoder(state)
            state_features = state_features.unsqueeze(1).expand(-1, n_agents, -1)
            combined = torch.cat([combined, state_features], dim=-1)

        # Compute values
        values = self.value_head(combined)

        return values


class SparseAttentionMask(nn.Module):
    """Compute sparse attention mask based on spatial locality.

    For large swarms, full O(N^2) attention is expensive.
    This module creates sparse masks based on k-nearest neighbors.
    """

    def __init__(self, k_neighbors: int = 8):
        super().__init__()
        self.k_neighbors = k_neighbors

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch, n_agents, 3]
        Returns:
            mask: [batch, n_agents, n_agents] sparse attention mask
        """
        batch_size, n_agents, _ = positions.shape

        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        distances = torch.norm(pos_i - pos_j, dim=-1)

        # Find k nearest neighbors for each agent
        k = min(self.k_neighbors, n_agents)
        _, indices = torch.topk(distances, k, dim=2, largest=False)

        # Create mask (1 for attended, 0 for masked)
        mask = torch.zeros(batch_size, n_agents, n_agents, device=positions.device)
        for b in range(batch_size):
            for i in range(n_agents):
                mask[b, i, indices[b, i]] = 1

        # Add self-loops
        for b in range(batch_size):
            mask[b].fill_diagonal_(1)

        # Convert to attention mask format (0 for attend, -inf for mask)
        attn_mask = torch.zeros_like(mask)
        attn_mask[mask == 0] = float('-inf')

        return attn_mask


class CrossAttentionCritic(nn.Module):
    """Critic with cross-attention between agents and global context.

    Separate streams for individual agent reasoning and global coordination.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 8,
    ):
        super().__init__()

        # Agent stream: local processing
        self.agent_encoder = nn.Linear(obs_dim + action_dim, d_model)

        # Global stream: aggregate information
        self.global_encoder = nn.Linear(obs_dim + action_dim, d_model)

        # Cross-attention: agent queries, global keys/values
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            actions: [batch, n_agents, action_dim]
        Returns:
            values: [batch, n_agents, 1]
        """
        # Concatenate obs and actions
        x = torch.cat([obs, actions], dim=-1)

        # Agent-local features
        agent_features = self.agent_encoder(x)

        # Global features (mean-pooled)
        global_features = self.global_encoder(x)
        global_context = global_features.mean(dim=1, keepdim=True)
        global_context = global_context.expand_as(agent_features)

        # Cross-attention
        attended, _ = self.cross_attn(
            agent_features,  # query: individual agents
            global_context,  # key/value: global coordination
            global_context,
        )

        # Combine and predict values
        combined = torch.cat([agent_features, attended], dim=-1)
        values = self.value_head(combined)

        return values
