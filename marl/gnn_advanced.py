"""
Advanced GNN-MAPPO Features and Optimizations

This module implements several research-driven improvements:
1. Adaptive adjacency matrix (learnable threshold)
2. Edge features (distance, relative velocity)
3. Heterogeneous attention (different edge types)
4. Communication bottleneck
5. Reward shaping with graph metrics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class AdaptiveAdjacencyBuilder(nn.Module):
    """Learnable adjacency matrix construction.

    Instead of fixed threshold, learns optimal connection radius
    and edge weights based on agent states.
    """

    def __init__(self, feature_dim: int = 3, init_threshold: float = 0.15):
        super().__init__()

        # Learnable threshold (log-space for positivity)
        self.log_threshold = nn.Parameter(torch.log(torch.tensor(init_threshold)))

        # Edge weight MLP: predicts connection strength
        self.edge_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 32),  # concat(feat_i, feat_j, distance)
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Weight in [0, 1]
        )

    def forward(
        self,
        positions: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build adaptive adjacency matrix.

        Args:
            positions: [batch, n_agents, 3]
            features: [batch, n_agents, feature_dim] optional node features
        Returns:
            adj: [batch, n_agents, n_agents] weighted adjacency
        """
        batch_size, n_agents, _ = positions.shape

        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)  # [B, N, 1, 3]
        pos_j = positions.unsqueeze(1)  # [B, 1, N, 3]
        distances = torch.norm(pos_i - pos_j, dim=-1)  # [B, N, N]

        # Learnable threshold
        threshold = torch.exp(self.log_threshold)

        # Binary connectivity based on distance
        connected = (distances < threshold).float()

        # If features provided, compute edge weights
        if features is not None:
            feat_i = features.unsqueeze(2).expand(-1, -1, n_agents, -1)  # [B, N, N, F]
            feat_j = features.unsqueeze(1).expand(-1, n_agents, -1, -1)  # [B, N, N, F]
            dist_feat = distances.unsqueeze(-1)  # [B, N, N, 1]

            edge_input = torch.cat([feat_i, feat_j, dist_feat], dim=-1)
            edge_weights = self.edge_mlp(edge_input).squeeze(-1)  # [B, N, N]

            # Combine: only connected edges have weight
            adj = connected * edge_weights
        else:
            adj = connected

        # Add self-loops
        eye = torch.eye(n_agents, device=positions.device).unsqueeze(0)
        adj = adj + eye

        return adj


class EdgeFeatureGAT(nn.Module):
    """GAT layer with edge features.

    Extends standard GAT to incorporate edge features (distance, velocity)
    into attention computation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        assert out_dim % num_heads == 0

        # Node projections
        self.W_node = nn.Linear(node_dim, out_dim)

        # Edge projection
        self.W_edge = nn.Linear(edge_dim, out_dim)

        # Attention mechanism (includes edge features)
        # Input: concatenated [node_i, node_j, edge_ij] per head
        attn_input_dim = self.head_dim * 3  # Each of node_i, node_j, edge has head_dim
        self.attn = nn.Linear(attn_input_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        adj_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            nodes: [batch, n_agents, node_dim]
            edges: [batch, n_agents, n_agents, edge_dim]
            adj_matrix: [batch, n_agents, n_agents]
        Returns:
            out: [batch, n_agents, out_dim]
        """
        batch_size, n_agents, _ = nodes.shape

        # Project nodes and edges
        h_nodes = self.W_node(nodes)  # [B, N, out_dim]
        h_edges = self.W_edge(edges)  # [B, N, N, out_dim]

        # Reshape for multi-head attention
        h_nodes = h_nodes.view(batch_size, n_agents, self.num_heads, self.head_dim)
        h_edges = h_edges.view(batch_size, n_agents, n_agents, self.num_heads, self.head_dim)

        # Compute attention scores
        h_i = h_nodes.unsqueeze(2)  # [B, N, 1, H, D]
        h_j = h_nodes.unsqueeze(1)  # [B, 1, N, H, D]

        # Concatenate node_i, node_j, edge_ij for attention
        h_i_exp = h_i.expand(-1, -1, n_agents, -1, -1)  # [B, N, N, H, D]
        h_j_exp = h_j.expand(-1, n_agents, -1, -1, -1)  # [B, N, N, H, D]

        attn_input = torch.cat([h_i_exp, h_j_exp, h_edges], dim=-1)  # [B, N, N, H, 3*D]

        # Flatten heads and features for attention MLP
        attn_input_flat = attn_input.reshape(batch_size * n_agents * n_agents, self.num_heads, -1)
        attn_scores = self.attn(attn_input_flat).squeeze(-1)  # [B*N*N, H]
        attn_scores = attn_scores.view(batch_size, n_agents, n_agents, self.num_heads)  # [B, N, N, H]

        # Mask with adjacency
        mask = adj_matrix.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax attention weights
        attn_weights = F.softmax(attn_scores, dim=2)  # [B, N, N, H]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to neighbor features (including edge info)
        h_j_with_edge = (h_j + h_edges) / 2  # Combine node and edge features
        h_j_with_edge = h_j_with_edge.expand(-1, n_agents, -1, -1, -1)

        # Weighted sum
        out = torch.einsum('bnmh,bnmhd->bnhd', attn_weights, h_j_with_edge)
        out = out.reshape(batch_size, n_agents, self.out_dim)

        # Residual and layer norm
        out = self.layer_norm(out + self.W_node(nodes))

        return out


class CommunicationBottleneck(nn.Module):
    """Communication bottleneck for bandwidth-limited scenarios.

    Agents must compress their information before sharing,
    simulating realistic communication constraints.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 16,
        output_dim: int = 64,
    ):
        super().__init__()

        # Encoder: compress local observation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            nn.Tanh(),  # Normalize message
        )

        # Decoder: reconstruct for neighbor consumption
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim * 2, output_dim),
        )

        self.bottleneck_dim = bottleneck_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [batch, n_agents, input_dim]
        Returns:
            messages: [batch, n_agents, bottleneck_dim] - compressed
            decoded: [batch, n_agents, output_dim] - for local use
        """
        messages = self.encoder(obs)
        decoded = self.decoder(messages)
        return messages, decoded


class GraphMetricRewardShaper(nn.Module):
    """Neural reward shaping based on graph metrics.

    Learns to shape rewards based on swarm connectivity,
    centrality, and coordination patterns.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Graph feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(5, hidden_dim),  # [density, avg_degree, diameter, clustering, component_size]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Shaping value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def compute_graph_metrics(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute graph-level features.

        Args:
            adj: [batch, n_agents, n_agents]
        Returns:
            features: [batch, 5]
        """
        batch_size, n_agents, _ = adj.shape

        # Remove self-loops for metric computation
        adj_no_self = adj.clone()
        for b in range(batch_size):
            adj_no_self[b].fill_diagonal_(0)

        # 1. Density: proportion of edges present
        max_edges = n_agents * (n_agents - 1)
        density = adj_no_self.sum(dim=[1, 2]) / max_edges  # [B]

        # 2. Average degree
        degree = adj_no_self.sum(dim=2)  # [B, N]
        avg_degree = degree.mean(dim=1)  # [B]

        # 3. Approximate diameter (max shortest path length)
        # Use matrix powers to find reachability
        reach = adj_no_self.clone()
        diameter = torch.zeros(batch_size, device=adj.device)
        for step in range(1, n_agents):
            reach = torch.bmm(reach, adj_no_self)
            reach = (reach > 0).float()
            if reach.sum(dim=[1, 2]).min() >= n_agents * (n_agents - 1):
                diameter = torch.full((batch_size,), step, device=adj.device)
                break

        # 4. Clustering coefficient (local)
        # Approximate: triangles / connected triples
        adj_sq = torch.bmm(adj_no_self, adj_no_self)
        triangles = (adj_no_self * adj_sq).sum(dim=[1, 2]) / 2
        triples = degree * (degree - 1) / 2
        clustering = triangles / (triples.sum(dim=1) + 1e-8)  # [B]

        # 5. Largest component size (approximation)
        # Use sum of reachability matrix
        component_size = reach.sum(dim=2).max(dim=1)[0] / n_agents  # [B]

        features = torch.stack([
            density, avg_degree / n_agents, diameter / n_agents,
            clustering, component_size
        ], dim=1)  # [B, 5]

        return features

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute reward shaping bonus.

        Args:
            adj: [batch, n_agents, n_agents]
        Returns:
            shaping: [batch] reward bonus
        """
        metrics = self.compute_graph_metrics(adj)
        features = self.feature_net(metrics)
        shaping = self.value_head(features).squeeze(-1)
        return shaping


def compute_edge_features(
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """
    Compute edge features for all agent pairs.

    Args:
        positions: [batch, n_agents, 3]
        velocities: [batch, n_agents, 3]
    Returns:
        edge_features: [batch, n_agents, n_agents, edge_dim]
    """
    batch_size, n_agents, _ = positions.shape

    # Pairwise differences
    pos_i = positions.unsqueeze(2)  # [B, N, 1, 3]
    pos_j = positions.unsqueeze(1)  # [B, 1, N, 3]
    rel_pos = pos_j - pos_i  # [B, N, N, 3]

    vel_i = velocities.unsqueeze(2)
    vel_j = velocities.unsqueeze(1)
    rel_vel = vel_j - vel_i  # [B, N, N, 3]

    # Distances
    distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # [B, N, N, 1]

    # Normalized relative position
    rel_pos_norm = rel_pos / (distances + 1e-8)

    # Relative speed (scalar)
    rel_speed = torch.norm(rel_vel, dim=-1, keepdim=True)  # [B, N, N, 1]

    # Approach/separation indicator (dot product of rel_pos and rel_vel)
    approach = (rel_pos * rel_vel).sum(dim=-1, keepdim=True) / (distances * rel_speed + 1e-8)

    # Concatenate all edge features
    edge_features = torch.cat([
        rel_pos_norm,  # 3D: direction
        distances,     # 1D: distance
        rel_vel,       # 3D: relative velocity
        approach,      # 1D: approaching/separating
    ], dim=-1)  # Total: 8D

    return edge_features


class EdgeBiasGATLayer(nn.Module):
    """`gat_policy.GATLayer` plus edge features as an additive attention bias.

    This exists to isolate a single variable. `EdgeFeatureGAT` (above) differs
    from the plain GAT layer in three ways at once: it replaces the Q/K/V triple
    with one node projection (~3x fewer parameters), swaps scaled dot-product
    attention for an additive MLP over concatenated features, AND adds edge
    features. So an EdgeFeatureGAT-vs-GAT gap cannot be attributed to the edge
    features -- capacity and attention style moved too.

    Here the layer is the dot-product GAT verbatim; the only change is a
    per-head scalar bias read off the 8-D edge features and added to the
    attention logits. That is `8 * num_heads + num_heads` extra parameters
    (~36 at 4 heads), so capacity is matched to within 0.03% and exactly one
    thing differs.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 8,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads

        # Identical to GATLayer.
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(in_dim, out_dim)
        self.W_v = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        # The only addition. Zero-initialised so the layer starts exactly as the
        # plain GAT layer and edge information has to earn its influence --
        # otherwise the two arms differ at step 0 for no reason.
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        nn.init.zeros_(self.edge_bias.weight)
        nn.init.zeros_(self.edge_bias.bias)

    def forward(self, x: torch.Tensor, edges: torch.Tensor,
                adj_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:     [batch, n_agents, in_dim]
            edges: [batch, n_agents, n_agents, edge_dim]
            adj_matrix: [batch, n_agents, n_agents]
        Returns:
            [batch, n_agents, out_dim]
        """
        B, N, _ = x.shape

        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # [B, N, N, H] -> [B, H, N, N]; bias on the logits, before masking.
        scores = scores + self.edge_bias(edges).permute(0, 3, 1, 2)

        if adj_matrix is not None:
            mask = adj_matrix.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        w = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(w, V).transpose(1, 2).contiguous().view(B, N, self.out_dim)
        return self.layer_norm(out + self.W_v(x))


class EdgeBiasGATEncoder(nn.Module):
    """Multi-layer encoder built from `EdgeBiasGATLayer`.

    Mirrors `gat_policy.GATEncoder` layer for layer (input projection, N graph
    layers, output MLP) so the only structural difference from the `gat` arm is
    the edge bias inside attention.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, edge_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EdgeBiasGATLayer(hidden_dim, hidden_dim, edge_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, obs, edges, adj_matrix=None):
        h = F.relu(self.input_proj(obs))
        for layer in self.layers:
            h = layer(h, edges, adj_matrix)
        return self.output_mlp(h)


class HierarchicalGNN(nn.Module):
    """Hierarchical GNN with robot-level and swarm-level reasoning.

    Two-level architecture:
    1. Robot level: local coordination with neighbors
    2. Swarm level: global strategy and task allocation
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        num_robot_layers: int = 2,
        num_swarm_layers: int = 1,
    ):
        super().__init__()

        # Robot-level GNN (fine-grained local coordination)
        self.robot_encoder = nn.Linear(obs_dim, hidden_dim)
        self.robot_layers = nn.ModuleList([
            EdgeFeatureGAT(hidden_dim, edge_dim=8, out_dim=hidden_dim)
            for _ in range(num_robot_layers)
        ])

        # Swarm-level GNN (coarse global strategy)
        # Operates on aggregated robot representations
        self.swarm_pooling = nn.Linear(hidden_dim, hidden_dim // 2)
        self.swarm_layers = nn.ModuleList([
            EdgeFeatureGAT(hidden_dim // 2, edge_dim=8, out_dim=hidden_dim // 2)
            for _ in range(num_swarm_layers)
        ])

        # Combine both levels
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs: [batch, n_agents, obs_dim]
            positions: [batch, n_agents, 3]
            velocities: [batch, n_agents, 3]
            adj: [batch, n_agents, n_agents]
        Returns:
            features: [batch, n_agents, hidden_dim]
        """
        # Compute edge features
        edges = compute_edge_features(positions, velocities)

        # Robot-level processing
        h_robot = F.relu(self.robot_encoder(obs))
        for layer in self.robot_layers:
            h_robot = layer(h_robot, edges, adj)

        # Swarm-level processing
        h_swarm = F.relu(self.swarm_pooling(h_robot))
        for layer in self.swarm_layers:
            h_swarm = layer(h_swarm, edges, adj)

        # Combine
        h_combined = torch.cat([h_robot, h_swarm], dim=-1)
        features = self.combine(h_combined)

        return features
