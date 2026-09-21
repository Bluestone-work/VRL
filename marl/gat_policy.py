"""
Graph Attention Network (GAT) Policy for Multi-Agent Vascular Navigation

Inspired by the DGR_VDS project's GNN-based MARL approach, this module implements
a GAT encoder that captures inter-agent relationships for the blood clot removal task.

Key features:
- Graph attention mechanism to model agent interactions
- Permutation-invariant architecture
- Scales to variable number of agents
- Combines local observations with graph-structured communication
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class GATLayer(nn.Module):
    """Single Graph Attention Layer (Veličković et al. 2018)."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Linear projections for query, key, value
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)

        # Attention mechanism
        self.attn = nn.Linear(2 * self.head_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, n_agents, in_dim] node features
            adj_matrix: [batch, n_agents, n_agents] adjacency matrix (optional)
                       If None, uses fully-connected graph
        Returns:
            out: [batch, n_agents, out_dim] updated node features
        """
        batch_size, n_agents, _ = x.shape

        # Project to query, key, value
        Q = self.W_q(x).view(batch_size, n_agents, self.num_heads, self.head_dim)  # [B, N, H, D]
        K = self.W_k(x).view(batch_size, n_agents, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, n_agents, self.num_heads, self.head_dim)

        # Compute attention scores for each head
        Q = Q.transpose(1, 2)  # [B, H, N, D]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Calculate attention weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [B, H, N, N]

        # Apply adjacency mask if provided
        if adj_matrix is not None:
            # Expand adjacency matrix for multi-head attention
            mask = adj_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [B, H, N, N]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, H, N, D]
        out = out.transpose(1, 2).contiguous().view(batch_size, n_agents, self.out_dim)  # [B, N, out_dim]

        # Residual connection and layer norm
        out = self.layer_norm(out + self.W_v(x))

        return out


class GATEncoder(nn.Module):
    """Multi-layer GAT encoder for agent observations."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial projection
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch, n_agents, obs_dim] agent observations
            adj_matrix: [batch, n_agents, n_agents] adjacency matrix
        Returns:
            features: [batch, n_agents, hidden_dim] graph-aware features
        """
        # Initial projection
        x = F.relu(self.input_proj(obs))  # [B, N, hidden_dim]

        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, adj_matrix)

        # Final MLP
        x = self.output_mlp(x)

        return x


class GATActor(nn.Module):
    """GAT-based actor for decentralized execution.

    Uses graph attention to encode agent relationships, then outputs
    per-agent actions based on graph-aware features.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # GAT encoder
        self.gat_encoder = GATEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Action head (per-agent)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

    def forward(
        self,
        obs: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch, n_agents, obs_dim] or [n_agents, obs_dim]
            adj_matrix: [batch, n_agents, n_agents] or [n_agents, n_agents]
        Returns:
            actions: [batch, n_agents, action_dim] or [n_agents, action_dim]
        """
        # Handle single sample case
        squeeze_batch = False
        if obs.ndim == 2:
            obs = obs.unsqueeze(0)
            squeeze_batch = True
            if adj_matrix is not None:
                adj_matrix = adj_matrix.unsqueeze(0)

        # Encode with GAT
        features = self.gat_encoder(obs, adj_matrix)  # [B, N, hidden_dim]

        # Generate actions
        actions = self.action_head(features)  # [B, N, action_dim]

        if squeeze_batch:
            actions = actions.squeeze(0)

        return actions


class GATCritic(nn.Module):
    """GAT-based centralized critic for CTDE (Centralized Training Decentralized Execution).

    The critic observes all agents' observations and actions during training,
    using GAT to capture complex multi-agent interactions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        state_dim: int = 0,
        use_action_input: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Encode observations with GAT
        self.obs_encoder = GATEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # V critics intentionally omit actions; q critics retain the legacy path.
        self.use_action_input = bool(use_action_input)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2), nn.ReLU(),
        ) if self.use_action_input else None

        # Combine obs features, action features, and global state
        combine_dim = hidden_dim + (hidden_dim // 2 if self.use_action_input else 0)
        if state_dim > 0:
            self.state_encoder = nn.Linear(state_dim, hidden_dim // 2)
            combine_dim += hidden_dim // 2

        # Value head (aggregates over all agents)
        self.value_head = nn.Sequential(
            nn.Linear(combine_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs_all: torch.Tensor,
        actions_all: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_all: [batch, n_agents, obs_dim]
            actions_all: [batch, n_agents, action_dim]
            adj_matrix: [batch, n_agents, n_agents]
            state: [batch, state_dim] global state (e.g., clot positions)
        Returns:
            values: [batch, n_agents, 1] per-agent value estimates
        """
        batch_size, n_agents, _ = obs_all.shape

        # Encode observations with GAT
        obs_features = self.obs_encoder(obs_all, adj_matrix)  # [B, N, hidden_dim]

        # Encode actions only for legacy Q critics.
        if self.use_action_input:
            action_features = self.action_encoder(actions_all)
            combined = torch.cat([obs_features, action_features], dim=-1)
        else:
            combined = obs_features

        # Add global state. The head's input width is fixed at construction, so
        # when state_dim > 0 the slot must be filled on every call -- zero-fill
        # when the caller has no state rather than silently changing the width.
        if self.state_dim > 0:
            if state is None:
                state = combined.new_zeros(batch_size, self.state_dim)
            state_features = self.state_encoder(state)  # [B, hidden_dim/2]
            state_features = state_features.unsqueeze(1).expand(-1, n_agents, -1)  # [B, N, hidden_dim/2]
            combined = torch.cat([combined, state_features], dim=-1)

        # Compute per-agent values
        values = self.value_head(combined)  # [B, N, 1]

        return values


def build_adjacency_matrix(
    positions: torch.Tensor,
    threshold: float = 0.15,
    include_self: bool = True,
) -> torch.Tensor:
    """
    Build adjacency matrix based on spatial proximity.

    Args:
        positions: [batch, n_agents, 3] agent positions
        threshold: distance threshold for edge creation
        include_self: whether to include self-loops
    Returns:
        adj_matrix: [batch, n_agents, n_agents] binary adjacency
    """
    batch_size, n_agents, _ = positions.shape

    # Compute pairwise distances
    pos_i = positions.unsqueeze(2)  # [B, N, 1, 3]
    pos_j = positions.unsqueeze(1)  # [B, 1, N, 3]
    distances = torch.norm(pos_i - pos_j, dim=-1)  # [B, N, N]

    # Create adjacency matrix (1 if distance < threshold, 0 otherwise)
    adj_matrix = (distances < threshold).float()

    # Add self-loops if requested
    if include_self:
        eye = torch.eye(n_agents, device=positions.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_matrix = adj_matrix + eye
        adj_matrix = (adj_matrix > 0).float()

    return adj_matrix
