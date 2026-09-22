"""
Architecture-agnostic MAPPO.

`marl/mappo_policy.MAPPO` hardcodes the GAT actor/critic pair. The research
modules (`gnn_advanced`, `transformer_policy`) offer several other encoders,
each of which needs a different side-input: GAT wants an adjacency matrix, the
Transformer wants positions for its positional encoding, the hierarchical GNN
wants positions *and* velocities. Rather than thread three optional arguments
through every call site, everything a policy might need is bundled into a
per-timestep "graph context" dict, and each architecture is wrapped in an
adapter that picks out the entries it uses and ignores the rest.

This keeps one training loop and one update() for every architecture, which is
the point: an architecture comparison is only meaningful if the optimiser,
the advantage estimator, and the batching are held fixed across arms.

Context keys (all [n_agents, ...] per timestep, batched to [B, n_agents, ...]):
  adjacency  -- [N, N] binary/weighted agent connectivity
  positions  -- [N, 3] agent positions
  velocities -- [N, 3] agent velocities
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from marl.gat_policy import GATEncoder, GATCritic, build_adjacency_matrix
from marl.gnn_advanced import (
    EdgeBiasGATEncoder,
    EdgeFeatureGAT,
    HierarchicalGNN,
    compute_edge_features,
)
from marl.mappo_policy import GaussianActor, MAPPO
from marl.transformer_policy import (
    SparseAttentionMask,
    TransformerActor,
    TransformerCritic,
)

CONTEXT_KEYS = ("adjacency", "positions", "velocities")

# The context dict may also carry "agent_mask" ([n_agents] or [batch, n_agents]
# bool). It is NOT in CONTEXT_KEYS because it is stored in the rollout buffer as
# its own array rather than per-key lists; see ContextRolloutBuffer.store.
MASK_KEY = "agent_mask"

ARCHITECTURES = (
    "gat",                # standard GAT (matches train_gnn_mappo.py)
    "edge_gat",           # 8-D edge features, but also ~3x fewer params and
                          # additive attention -- NOT a controlled comparison
    "edge_bias_gat",      # `gat` + edge bias only; capacity-matched to `gat`
    "transformer",        # full all-to-all attention
    "sparse_transformer", # k-NN masked attention
    "hierarchical_gnn",   # robot-level + swarm-level GNN
    "mlp",                # no interaction model (ablation floor)
)


# --------------------------------------------------------------------- helpers

def _stochastic_head(features: torch.Tensor, head: GaussianActor):
    return head(features)


class _ActorBase(nn.Module):
    """Common sampling/evaluation logic; subclasses only implement encode()."""

    def __init__(self, hidden_dim: int, action_dim: int, log_std_init: float = 0.0):
        super().__init__()
        self.policy_head = GaussianActor(hidden_dim, action_dim, log_std_init)

    def encode(self, obs: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, obs, ctx):
        return self.policy_head(self.encode(obs, ctx))

    def get_actions(self, obs, ctx, deterministic: bool = False):
        return self.policy_head.get_actions(self.encode(obs, ctx), deterministic)

    def evaluate_actions(self, obs, actions, ctx):
        return self.policy_head.evaluate_actions(self.encode(obs, ctx), actions)


# ---------------------------------------------------------------- actor adapters

class GATActorAdapter(_ActorBase):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__(hidden_dim, action_dim)
        self.encoder = GATEncoder(obs_dim, hidden_dim, num_layers, num_heads)

    def encode(self, obs, ctx):
        return self.encoder(obs, ctx.get("adjacency"), ctx.get(MASK_KEY))


class EdgeGATActor(_ActorBase):
    """GAT whose attention also sees the 8-D edge features."""

    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__(hidden_dim, action_dim)
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EdgeFeatureGAT(hidden_dim, edge_dim=8, out_dim=hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        # NOTE: this arm changes three things at once relative to `gat` -- one
        # node projection instead of Q/K/V (~3x fewer parameters), additive
        # attention instead of scaled dot-product, and edge features. A gap
        # against `gat` therefore cannot be attributed to the edge features.
        # Use `edge_bias_gat` for the controlled comparison.

    def encode(self, obs, ctx):
        pos, vel = ctx["positions"], ctx["velocities"]
        edges = compute_edge_features(pos, vel)
        # The env always supplies adjacency, so no fallback construction here.
        adj = ctx.get("adjacency")
        h = torch.relu(self.input_proj(obs))
        for layer in self.layers:
            h = layer(h, edges, adj)
        return h


class EdgeBiasGATActor(_ActorBase):
    """Plain GAT + edge features as an attention bias (capacity-matched).

    The controlled counterpart to `EdgeGATActor`: identical to the `gat` arm
    except for ~36 extra parameters carrying the edge bias, so a difference is
    attributable to the edge features themselves.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__(hidden_dim, action_dim)
        self.encoder = EdgeBiasGATEncoder(
            obs_dim, hidden_dim, num_layers, num_heads, edge_dim=8
        )

    def encode(self, obs, ctx):
        edges = compute_edge_features(ctx["positions"], ctx["velocities"])
        return self.encoder(obs, edges, ctx.get("adjacency"))


class TransformerActorAdapter(_ActorBase):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads, sparse_k=0):
        super().__init__(hidden_dim, action_dim)
        # TransformerActor owns its own Gaussian head; only its encoder is used
        # here so the head stays in one place for every architecture.
        self.inner = TransformerActor(obs_dim, action_dim, hidden_dim, num_heads, num_layers)
        self.encoder = self.inner.encoder
        self.sparse = SparseAttentionMask(sparse_k) if sparse_k > 0 else None

    def encode(self, obs, ctx):
        pos = ctx.get("positions")
        mask = None
        if self.sparse is not None and pos is not None:
            # nn.MultiheadAttention wants a 2-D [N, N] float mask; the k-NN
            # neighbourhood is per-sample, so collapse the batch by keeping an
            # edge that any sample in the minibatch uses. Sparsity is retained
            # without silently dropping a sample's own neighbours.
            m = self.sparse(pos)                       # [B, N, N], 0 / -inf
            mask = torch.where(
                (m == 0).any(dim=0), 0.0, float("-inf")
            ).to(obs.dtype)
        return self.encoder(obs, pos, mask)


class HierarchicalActor(_ActorBase):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__(hidden_dim, action_dim)
        self.encoder = HierarchicalGNN(
            obs_dim, hidden_dim,
            num_robot_layers=max(1, num_layers), num_swarm_layers=1,
        )

    def encode(self, obs, ctx):
        pos, vel = ctx["positions"], ctx["velocities"]
        adj = ctx.get("adjacency")
        if adj is None:
            adj = build_adjacency_matrix(pos)
        return self.encoder(obs, pos, vel, adj)


class MLPActor(_ActorBase):
    """No interaction model at all -- the ablation floor."""

    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads):
        super().__init__(hidden_dim, action_dim)
        layers, d = [], obs_dim
        for _ in range(max(1, num_layers)):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        self.encoder = nn.Sequential(*layers, nn.LayerNorm(hidden_dim))

    def encode(self, obs, ctx):
        return self.encoder(obs)


# --------------------------------------------------------------- critic adapters

class _CriticAdapter(nn.Module):
    """Uniform critic signature: (obs, actions, ctx, state) -> [B, N, 1]."""

    def forward(self, obs, actions, ctx, state=None):
        raise NotImplementedError


class GATCriticAdapter(_CriticAdapter):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads, state_dim, critic_value_mode="v", dropout=0.0):
        super().__init__()
        self.inner = GATCritic(obs_dim, action_dim, hidden_dim, num_layers, num_heads, state_dim, use_action_input=critic_value_mode == "q", dropout=dropout)

    def forward(self, obs, actions, ctx, state=None):
        return self.inner(obs, actions, ctx.get("adjacency"), state, ctx.get(MASK_KEY))


class TransformerCriticAdapter(_CriticAdapter):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_layers, num_heads, state_dim, critic_value_mode="v", dropout=0.0):
        super().__init__()
        self.inner = TransformerCritic(
            obs_dim, action_dim, hidden_dim, num_heads, num_layers, state_dim,
            use_action_input=critic_value_mode == "q", dropout=dropout,
        )

    def forward(self, obs, actions, ctx, state=None):
        return self.inner(obs, actions, ctx.get("positions"), state)


def build_actor_critic(
    architecture: str,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    state_dim: int,
    sparse_k: int = 8,
    critic_value_mode: str = "v",
    dropout: float = 0.0,
) -> Tuple[nn.Module, nn.Module]:
    """Construct the (actor, critic) pair for one architecture name."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}; choose from {ARCHITECTURES}"
        )

    a = (obs_dim, action_dim, hidden_dim, num_layers, num_heads)

    if architecture == "gat":
        actor = GATActorAdapter(*a)
    elif architecture == "edge_gat":
        actor = EdgeGATActor(*a)
    elif architecture == "edge_bias_gat":
        actor = EdgeBiasGATActor(*a)
    elif architecture == "transformer":
        actor = TransformerActorAdapter(*a, sparse_k=0)
    elif architecture == "sparse_transformer":
        actor = TransformerActorAdapter(*a, sparse_k=sparse_k)
    elif architecture == "hierarchical_gnn":
        actor = HierarchicalActor(*a)
    else:  # mlp
        actor = MLPActor(*a)

    # The critic is deliberately NOT varied alongside the actor: a Transformer
    # critic for a Transformer actor would confound "which encoder helps the
    # policy" with "which encoder helps the value function". Only the attention
    # -free arm gets its structural counterpart removed.
    if architecture in ("transformer", "sparse_transformer"):
        critic = TransformerCriticAdapter(obs_dim, action_dim, hidden_dim,
                                          num_layers, num_heads, state_dim, critic_value_mode, dropout)
    else:
        critic = GATCriticAdapter(obs_dim, action_dim, hidden_dim,
                                  num_layers, num_heads, state_dim, critic_value_mode, dropout)
    for module in actor.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = dropout
    return actor, critic


# ------------------------------------------------------------------- the trainer

class MAPPOAdvanced(MAPPO):
    """MAPPO with a pluggable interaction model and a context-dict rollout.

    Reuses the parent's GAE, PPO clipping and optimiser settings; overrides
    only the parts that touch the actor/critic signature.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        architecture: str = "gat",
        state_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        sparse_k: int = 8,
        adjacency_threshold: float = 0.15,
        control_mode: str = "world",
        residual_scale: float = 0.2,
        guidance_speed: float = 0.65,
        critic_value_mode: str = "v",
        dropout: float = 0.0,
        log_std_init: float = 0.0,
        max_agents: int = 0,
        **kwargs,
    ):
        from marl.geometric_control import CONTROL_MODES

        if control_mode not in CONTROL_MODES:
            raise ValueError(f"unknown control mode: {control_mode}")
        if critic_value_mode not in ("v", "q"):
            raise ValueError("critic_value_mode must be v or q")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if control_mode != "world" and obs_dim not in (36, 42):
            raise ValueError("local control requires geometric observations")
        if not np.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual scale must be finite and nonnegative")
        if not np.isfinite(guidance_speed) or not 0 < guidance_speed <= 1:
            raise ValueError("guidance speed must be in (0, 1]")
        # max_agents declares the padding capacity of the interaction models.
        # 0 disables variable-N machinery entirely (legacy behaviour); a value
        # >= n_agents enables running the same parameter-shared checkpoint
        # with fewer real agents than the tensor dimension.
        max_agents = int(max_agents)
        if max_agents and max_agents < n_agents:
            raise ValueError("max_agents must be >= n_agents when nonzero")
        # Build the parent with use_gat=True so it does not construct the
        # MADDPG fallback, then replace both networks and their optimisers.
        super().__init__(
            n_agents=n_agents, obs_dim=obs_dim, action_dim=action_dim,
            state_dim=state_dim, hidden_dim=hidden_dim,
            num_gat_layers=num_layers, num_heads=num_heads,
            use_gat=True, **kwargs,
        )

        self.architecture = architecture
        self.adjacency_threshold = adjacency_threshold
        self.uses_adjacency = architecture != "mlp"

        lr_actor = self.actor_optimizer.param_groups[0]["lr"]
        lr_critic = self.critic_optimizer.param_groups[0]["lr"]

        self.actor, self.critic = build_actor_critic(
            architecture, obs_dim, action_dim, hidden_dim,
            num_layers, num_heads, state_dim, sparse_k, critic_value_mode, dropout,
        )
        nn.init.constant_(self.actor.policy_head.log_std, log_std_init)
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        if control_mode in ("guided", "flow_guided", "flow_spread"):
            nn.init.zeros_(self.actor.policy_head.mean.weight)
            nn.init.zeros_(self.actor.policy_head.mean.bias)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Recorded into every checkpoint so a viewer can rebuild the exact same
        # network without being told the architecture on the command line. A
        # checkpoint that only carries tensors is not reloadable in practice:
        # six architectures share the file extension and nothing else.
        self.meta = {
            "algo": "mappo_advanced",
            "critic_value_mode": critic_value_mode,
            "dropout": dropout,
            "architecture": architecture,
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "sparse_k": sparse_k,
            "control_mode": control_mode,
            "residual_scale": residual_scale,
            "guidance_speed": guidance_speed,
            "action_semantics": (
                "direct_local_frenet" if control_mode == "local"
                else "world_frame" if control_mode == "world"
                else "controller_residual"
            ),
            "max_agents": max_agents,
        }
        # A checkpoint built with max_agents=N can be reloaded for inference
        # at any n_agents <= N (with padding); one built with max_agents=0 is
        # fixed-N. See load().
        self.max_agents = max(max_agents, n_agents) if max_agents else 0
        self.buffer = ContextRolloutBuffer()
        self.world_model = None
        self.imagination_horizon = 0
        self.imagination_blend = 0.0
        self.imagination_uncertainty = 0.05

    def configure_world_model(
        self, model, horizon: int = 3, blend: float = 0.25,
        uncertainty_threshold: float = 0.05,
    ) -> None:
        if self.meta["control_mode"] != "world":
            raise ValueError("existing world models require world-frame actions")
        self.world_model = model.to(self.device).eval()
        self.imagination_horizon = max(0, int(horizon))
        self.imagination_blend = float(np.clip(blend, 0.0, 1.0))
        self.imagination_uncertainty = float(uncertainty_threshold)

    @torch.no_grad()
    def _model_value_expansion(self, data, real_returns):
        if self.world_model is None or self.imagination_horizon <= 0:
            return real_returns, {
                "mve_fraction": 0.0, "mve_uncertainty": 0.0,
                "mve_target_delta": 0.0,
            }
        required = {
            "next_obs", "next_states", "next_adjacency", "next_positions",
            "next_velocities", "next_geometry_features", "scenario_id",
        }
        if not required <= set(data):
            raise ValueError(
                f"world-model rollout buffer is missing {sorted(required - set(data))}"
            )

        original_shape = data["rewards"].shape
        n_agents = original_shape[-1]
        imagined_chunks = []
        accepted_chunks = []
        uncertainty_chunks = []
        total = int(np.prod(original_shape[:-1]))
        flattened = {
            "obs": data["next_obs"].reshape(total, n_agents, -1),
            "state": data["next_states"].reshape(total, -1),
            "adjacency": data["next_adjacency"].reshape(total, n_agents, n_agents),
            "positions": data["next_positions"].reshape(total, n_agents, 3),
            "velocities": data["next_velocities"].reshape(total, n_agents, 3),
            "geometry_features": data["next_geometry_features"].reshape(total, -1),
            "scenario_id": data["scenario_id"].reshape(total),
            "rewards": data["rewards"].reshape(total, n_agents),
            "terminals": data.get("terminals", data["dones"]).reshape(total, n_agents),
        }
        for start in range(0, total, 2048):
            stop = min(start + 2048, total)
            current = {
                key: torch.as_tensor(value[start:stop], device=self.device)
                for key, value in flattened.items()
                if key not in ("rewards", "terminals")
            }
            real_reward = torch.as_tensor(
                flattened["rewards"][start:stop],
                dtype=torch.float32, device=self.device,
            )
            terminal = torch.as_tensor(
                flattened["terminals"][start:stop, 0],
                dtype=torch.float32, device=self.device,
            )
            imagined = real_reward.clone()
            factor = self.gamma * (1.0 - terminal)
            accepted = torch.ones(stop - start, dtype=torch.bool, device=self.device)
            max_uncertainty = torch.zeros(stop - start, device=self.device)

            for _ in range(self.imagination_horizon):
                ctx = {
                    "adjacency": current["adjacency"],
                    "positions": current["positions"],
                    "velocities": current["velocities"],
                }
                action, _, _ = self.actor.get_actions(
                    current["obs"], ctx, deterministic=True
                )
                model_batch = {**current, "action": action}
                prediction = self.world_model.predict(model_batch)
                max_uncertainty = torch.maximum(
                    max_uncertainty, prediction["uncertainty"]
                )
                accepted &= prediction["uncertainty"] <= self.imagination_uncertainty
                imagined += factor[:, None] * prediction["rewards"]
                factor = (
                    factor * self.gamma * (1.0 - prediction["done_probability"])
                )
                current.update({
                    "obs": prediction["obs"],
                    "state": prediction["state"],
                    "positions": prediction["positions"],
                    "velocities": prediction["velocities"],
                    "adjacency": prediction["adjacency"],
                })

            ctx = {
                "adjacency": current["adjacency"],
                "positions": current["positions"],
                "velocities": current["velocities"],
            }
            bootstrap_action, _, _ = self.actor.get_actions(
                current["obs"], ctx, deterministic=True
            )
            bootstrap = self.critic(
                current["obs"], bootstrap_action, ctx, current["state"]
            ).squeeze(-1)
            imagined += factor[:, None] * bootstrap
            imagined_chunks.append(imagined.cpu().numpy())
            accepted_chunks.append(accepted.cpu().numpy())
            uncertainty_chunks.append(max_uncertainty.cpu().numpy())

        imagined = np.concatenate(imagined_chunks).reshape(original_shape)
        accepted = np.concatenate(accepted_chunks).reshape(original_shape[:-1])
        uncertainty = np.concatenate(uncertainty_chunks)
        mask = accepted[..., None]
        blended = np.where(
            mask,
            (1.0 - self.imagination_blend) * real_returns
            + self.imagination_blend * imagined,
            real_returns,
        )
        return blended.astype(np.float32), {
            "mve_fraction": float(accepted.mean()),
            "mve_uncertainty": float(uncertainty.mean()),
            "mve_target_delta": float(np.mean(np.abs(blended - real_returns))),
        }

    def save(self, path, training_state=None):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "meta": self.meta,
        }
        if training_state is not None:
            checkpoint["training_state"] = training_state
        torch.save(checkpoint, path)

    def load(self, path, load_optimizers: bool = True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        stored_meta = checkpoint.get("meta", {})
        for key, default in (("critic_value_mode", "q"), ("dropout", 0.1)):
            if stored_meta.get(key, default) != self.meta[key]:
                raise ValueError(f"checkpoint {key} mismatch; rebuild using checkpoint metadata")
        for key, default in (("control_mode", "world"), ("residual_scale", 0.2),
                             ("guidance_speed", 0.65), ("action_semantics", None)):
            if key == "action_semantics" and stored_meta.get(key) is None:
                continue
            if stored_meta.get(key, default) != self.meta[key]:
                raise ValueError(f"checkpoint {key} differs from requested control configuration")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return checkpoint.get("training_state", {})

    def env_action(self, actions, obs, env):
        from marl.geometric_control import policy_action

        if (self.meta["control_mode"] == "local"
                and self.meta.get("action_semantics") != "direct_local_frenet"):
            raise RuntimeError("local control must use direct_local_frenet action semantics")
        return policy_action(
            actions, obs, env, mode=self.meta["control_mode"],
            residual_scale=self.meta["residual_scale"],
            guidance_speed=self.meta["guidance_speed"],
        )

    # -- context plumbing ---------------------------------------------------

    def _ctx_to_torch(self, ctx: Dict[str, np.ndarray], batched: bool) -> Dict[str, torch.Tensor]:
        out = {}
        for k in CONTEXT_KEYS:
            v = ctx.get(k)
            if v is None:
                continue
            t = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=self.device)
            out[k] = t if batched else t.unsqueeze(0)
        mask = ctx.get(MASK_KEY)
        if mask is not None:
            m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device)
            if m.ndim == 1:
                if batched:
                    # A single [N] mask against a batched obs means "the same
                    # active set in every env" -- expand along the batch.
                    m = m.unsqueeze(0).expand(
                        int(np.asarray(ctx[CONTEXT_KEYS[0]]).shape[0])
                        if CONTEXT_KEYS[0] in ctx else m.shape[0]
                    )
                else:
                    m = m.unsqueeze(0)
            out[MASK_KEY] = m
        return out

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray,
        ctx: Dict[str, np.ndarray],
        state: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        """One environment step. Returns (actions, log_probs, values).

        When the observation carries padding rows (n_agents > the env's real
        robot count) the caller must pass ctx["agent_mask"] with True for the
        real agents; padding rows then produce inert actions that the caller
        discards, and real-agent outputs are identical to the unpadded call.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        ctx_t = self._ctx_to_torch(ctx, batched=False)
        state_t = (
            torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if state is not None else None
        )

        actions, log_probs, _ = self.actor.get_actions(obs_t, ctx_t, deterministic)
        values = self.critic(obs_t, actions, ctx_t, state_t)

        return (
            actions.squeeze(0).cpu().numpy(),
            log_probs.squeeze(0).cpu().numpy(),
            values.squeeze(0).squeeze(-1).cpu().numpy(),
        )

    @torch.no_grad()
    def act_batch(
        self,
        obs: np.ndarray,
        ctx: Dict[str, np.ndarray],
        state: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        """Select actions for ``[n_envs, n_agents, ...]`` observations."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        ctx_t = self._ctx_to_torch(ctx, batched=True)
        state_t = (
            torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state is not None else None
        )
        actions, log_probs, _ = self.actor.get_actions(obs_t, ctx_t, deterministic)
        values = self.critic(obs_t, actions, ctx_t, state_t)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
        )

    # -- update -------------------------------------------------------------

    def update(self, n_epochs: int = 10, batch_size: int = 256) -> Dict[str, float]:
        """PPO update. Mirrors MAPPO.update but passes the context dict through.

        Minibatches are drawn over timesteps so the agent dimension survives:
        flattening agents into the batch would hand every interaction model a
        one-node graph, silently turning each arm of an architecture comparison
        into the same per-agent MLP.
        """
        data = self.buffer.get()
        if not len(data["obs"]):
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        dev = self.device
        obs = torch.as_tensor(data["obs"], dtype=torch.float32, device=dev)
        actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=dev)
        old_log_probs = torch.as_tensor(data["log_probs"], dtype=torch.float32, device=dev)
        old_values = torch.as_tensor(data["values"], dtype=torch.float32, device=dev)
        rewards, dones, values_np = data["rewards"], data["dones"], data["values"]

        # agent_mask: True where the slot was a REAL agent during rollout.
        # Padding slots are zeroed in every per-agent quantity below so they
        # contribute nothing to GAE, the actor/critic losses, entropy or the
        # reported metrics. With no mask (legacy rollouts) everything is real.
        agent_masks = (
            torch.as_tensor(data["agent_mask"], dtype=torch.bool, device=dev)
            if "agent_mask" in data else None
        )
        ctx_all = {
            k: torch.as_tensor(data[k], dtype=torch.float32, device=dev)
            for k in CONTEXT_KEYS if k in data
        }
        # The mask must ride in ctx_all too: every encoder adapter reads it
        # from there, and the pre-update ratio diagnostic forwards through
        # the actor with ctx_all.
        if agent_masks is not None:
            ctx_all[MASK_KEY] = agent_masks
        states = (
            torch.as_tensor(data["states"], dtype=torch.float32, device=dev)
            if "states" in data else None
        )
        vectorized = obs.ndim == 4

        # Compute V(s_{t+1}) for every transition. This matters at vector-env
        # auto-reset boundaries: values[t+1] belongs to the next episode there.
        with torch.no_grad():
            if "next_obs" in data:
                next_obs = torch.as_tensor(
                    data["next_obs"], dtype=torch.float32, device=dev
                )
                next_ctx = {
                    k: torch.as_tensor(
                        data[f"next_{k}"], dtype=torch.float32, device=dev
                    )
                    for k in CONTEXT_KEYS if f"next_{k}" in data
                }
                if "next_agent_mask" in data:
                    next_ctx[MASK_KEY] = torch.as_tensor(
                        data["next_agent_mask"], dtype=torch.bool, device=dev
                    )
                next_states = (
                    torch.as_tensor(
                        data["next_states"], dtype=torch.float32, device=dev
                    )
                    if "next_states" in data else None
                )
                original_next_shape = next_obs.shape
                if vectorized:
                    flat_count = original_next_shape[0] * original_next_shape[1]
                    next_obs = next_obs.reshape(
                        flat_count, original_next_shape[2], original_next_shape[3]
                    )
                    next_ctx = {
                        key: value.reshape(flat_count, *value.shape[2:])
                        for key, value in next_ctx.items()
                    }
                    if next_states is not None:
                        next_states = next_states.reshape(flat_count, *next_states.shape[2:])
                next_actions, _, _ = self.actor.get_actions(
                    next_obs, next_ctx, deterministic=True
                )
                next_values = self.critic(
                    next_obs, next_actions, next_ctx, next_states
                ).squeeze(-1)
                next_values = next_values.cpu().numpy().reshape(values_np.shape)
            else:
                last_index = -1 if vectorized else slice(-1, None)
                last_obs = obs[last_index]
                last_ctx = {k: v[last_index] for k, v in ctx_all.items()}
                if agent_masks is not None:
                    last_ctx[MASK_KEY] = (
                        agent_masks.reshape(-1, *agent_masks.shape[2:])[
                            -1 if vectorized else slice(-1, None)
                        ]
                    )
                last_state = states[last_index] if states is not None else None
                last_actions = actions[last_index]
                last_value = self.critic(
                    last_obs, last_actions, last_ctx, last_state
                ).squeeze(-1).cpu().numpy()
                if not vectorized:
                    last_value = last_value.squeeze(0)
                next_values = np.concatenate([values_np[1:], last_value[None]], axis=0)

        # Padding rows carry zeros, not garbage: rewards/terminals were stored
        # masked by the caller, and the value/bootstrap terms are zeroed here so
        # GAE sees a dead branch that contributes exactly zero everywhere.
        if agent_masks is not None:
            m_np = agent_masks.cpu().numpy()
            rewards = rewards * m_np
            dones = dones * m_np
            values_np = values_np * m_np
            next_values = next_values * m_np
        terminals = data.get("terminals", dones)
        if agent_masks is not None:
            terminals = terminals * agent_masks.cpu().numpy()
        advantages = np.zeros_like(rewards)
        running = np.zeros_like(rewards[0])
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * next_values[step] * (1.0 - terminals[step])
                - values_np[step]
            )
            running = (
                delta
                + self.gamma * self.gae_lambda * (1.0 - dones[step]) * running
            )
            advantages[step] = running
        returns = advantages + values_np
        returns, mve_metrics = self._model_value_expansion(data, returns)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=dev)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=dev)
        # Advantage normalisation over REAL slots only. Padding rows are
        # structurally zero, so including them would shrink the variance and
        # shift the mean by a factor that depends on HOW MUCH padding the
        # batch happens to contain -- silently coupling the update's scale to
        # the padding ratio.
        if agent_masks is not None:
            adv_mask = agent_masks.reshape(-1).to(torch.bool)
            real = advantages.reshape(-1)[adv_mask]
            if real.numel() > 1:
                mean = real.mean()
                std = real.std()
                advantages = (advantages - mean) / (std + 1e-8)
                # Padding rows normalise to -mean/std; force back to exact 0.
                advantages = advantages * agent_masks.to(advantages.dtype)
            else:
                advantages = advantages * 0.0
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if vectorized:
            time_steps, n_envs, n_agents = obs.shape[:3]
            samples = time_steps * n_envs
            obs = obs.reshape(samples, n_agents, *obs.shape[3:])
            actions = actions.reshape(samples, n_agents, *actions.shape[3:])
            old_log_probs = old_log_probs.reshape(samples, n_agents)
            old_values = old_values.reshape(samples, n_agents)
            advantages = advantages.reshape(samples, n_agents)
            returns = returns.reshape(samples, n_agents)
            ctx_all = {
                k: value.reshape(samples, *value.shape[2:])
                for k, value in ctx_all.items()
            }
            if agent_masks is not None:
                agent_masks = agent_masks.reshape(samples, *agent_masks.shape[2:])
                # ctx_all holds the SAME tensor object; after the reshape
                # reassignment above it points at the old [T, E, N] view.
                ctx_all[MASK_KEY] = agent_masks
            if states is not None:
                states = states.reshape(samples, *states.shape[2:])
        else:
            samples, n_agents = obs.shape[:2]

        # Flat [samples, N] float mask: 1 where the slot is a real agent. When
        # agent_counts differ per sample this is genuinely mixed-N in one batch
        # -- the padding zeros just never reach a loss.
        mask_f = None
        if agent_masks is not None:
            mask_f = agent_masks.reshape(samples, n_agents).to(torch.float32)
            agent_counts = mask_f.sum(dim=1)                       # [samples]
        else:
            agent_counts = torch.full(
                (samples,), float(n_agents), device=dev
            )

        N = n_agents
        steps_per_batch = max(1, batch_size // N)

        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        metrics.update(approx_kl=0.0, clip_fraction=0.0)
        n_updates = 0

        def _padded_real_mean(per_agent: torch.Tensor) -> torch.Tensor:
            """Mean over real agent-slots only; 0 for an all-padding batch row.

            Dividing by the per-sample agent count (not by N) keeps the loss
            scale independent of how much padding a minibatch happens to
            contain, which is what makes mixed-N batches comparable.
            """
            if mask_f is None:
                return per_agent.mean()
            total = per_agent.sum()
            count = mask_f.sum().clamp(min=1.0)
            return total / count

        # These diagnostics are computed BEFORE the first gradient, once per rollout.
        with torch.no_grad():
            lp, _ = self.actor.evaluate_actions(obs, actions, ctx_all)
            initial_ratio = torch.exp(lp - old_log_probs)
            if mask_f is not None:
                # Padding rows have garbage old_log_probs/entropy; only real
                # slots enter the diagnostic.
                initial_ratio = initial_ratio * mask_f
            initial_ratio_error = (initial_ratio - mask_f).abs().max().item() if mask_f is not None \
                else (initial_ratio - 1).abs().max().item()
            var = torch.var(returns, unbiased=False)
            explained = (1 - torch.var(returns - old_values, unbiased=False) / var).item() if var > 1e-12 else 0.0

        for _ in range(n_epochs):
            order = np.random.permutation(samples)
            for start in range(0, samples, steps_per_batch):
                idx = order[start:start + steps_per_batch]

                ctx_b = {k: v[idx] for k, v in ctx_all.items()}
                state_b = states[idx] if states is not None else None
                mask_b = mask_f[idx] if mask_f is not None else None

                log_probs, entropy = self.actor.evaluate_actions(
                    obs[idx], actions[idx], ctx_b
                )
                values_pred = self.critic(obs[idx], actions[idx], ctx_b, state_b).squeeze(-1)

                ratio = torch.exp(log_probs - old_log_probs[idx])
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[idx]
                    if mask_b is not None:
                        ratio_diag = ratio * mask_b
                        log_ratio = log_ratio * mask_b
                        ones = mask_b
                    else:
                        ratio_diag, ones = ratio, None
                    if ones is None:
                        metrics["approx_kl"] += (ratio_diag - 1 - log_ratio).mean().item()
                        metrics["clip_fraction"] += ((ratio_diag - 1).abs() > self.clip_epsilon).float().mean().item()
                    else:
                        count = max(1.0, float(ones.sum()))
                        metrics["approx_kl"] += float(((ratio_diag - ones - log_ratio).sum() / count).item())
                        metrics["clip_fraction"] += float(
                            (((ratio_diag - ones).abs() > self.clip_epsilon) & ones.bool()).float().sum().item() / count
                        )
                adv_b = advantages[idx]
                surr = torch.min(
                    ratio * adv_b,
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b,
                )
                if mask_b is not None:
                    surr = surr * mask_b
                    entropy_term = entropy * mask_b
                else:
                    entropy_term = entropy
                actor_loss = -_padded_real_mean(surr) - self.entropy_coef * _padded_real_mean(entropy_term)

                # Trust region around the value that generated the rollout.
                ov = old_values[idx]
                clipped = ov + torch.clamp(values_pred - ov, -self.value_clip, self.value_clip)
                per_slot_critic = torch.max(
                    (values_pred - returns[idx]).pow(2),
                    (clipped - returns[idx]).pow(2),
                )
                if mask_b is not None:
                    per_slot_critic = per_slot_critic * mask_b
                critic_loss = self.value_loss_coef * _padded_real_mean(per_slot_critic)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                metrics["actor_loss"] += actor_loss.item()
                metrics["critic_loss"] += critic_loss.item()
                metrics["entropy"] += (
                    _padded_real_mean(entropy_term).item()
                    if mask_b is not None else entropy.mean().item()
                )
                n_updates += 1

        averaged = {k: v / max(1, n_updates) for k, v in metrics.items()}
        averaged.update(mve_metrics)
        averaged["pre_update_ratio_max_error"] = initial_ratio_error
        averaged["explained_variance"] = explained
        return averaged


class ContextRolloutBuffer:
    """Rollout storage that keeps the graph-context arrays alongside transitions."""

    def __init__(self):
        self.clear()

    def clear(self):
        self._d = {
            "obs": [], "actions": [], "rewards": [], "dones": [],
            "terminals": [],
            "log_probs": [], "values": [], "states": [],
            "adjacency": [], "positions": [], "velocities": [],
            "next_obs": [], "next_states": [],
            "next_adjacency": [], "next_positions": [], "next_velocities": [],
            "geometry_features": [], "next_geometry_features": [],
            "scenario_id": [],
            "agent_mask": [], "next_agent_mask": [],
        }

    def store(self, obs, actions, rewards, dones, log_probs, values,
              ctx: Dict[str, np.ndarray], state=None, next_obs=None,
              next_ctx: Optional[Dict[str, np.ndarray]] = None, next_state=None,
              geometry_features=None, next_geometry_features=None,
              terminals=None, scenario_id=None):
        self._d["obs"].append(obs)
        self._d["actions"].append(actions)
        self._d["rewards"].append(rewards)
        self._d["dones"].append(dones)
        self._d["terminals"].append(dones if terminals is None else terminals)
        self._d["log_probs"].append(log_probs)
        self._d["values"].append(values)
        self._d["states"].append(state)
        self._d["next_obs"].append(next_obs)
        self._d["next_states"].append(next_state)
        self._d["geometry_features"].append(geometry_features)
        self._d["next_geometry_features"].append(next_geometry_features)
        self._d["scenario_id"].append(scenario_id)
        # The mask rides along in the context dict but is stored as its own
        # boolean array -- `get()` casts everything it materialises to
        # float32, which would destroy the bool semantics the update reads.
        self._d["agent_mask"].append(ctx.get(MASK_KEY))
        self._d["next_agent_mask"].append(
            None if next_ctx is None else next_ctx.get(MASK_KEY)
        )
        for k in CONTEXT_KEYS:
            self._d[k].append(ctx.get(k))
            self._d[f"next_{k}"].append(
                None if next_ctx is None else next_ctx.get(k)
            )

    def get(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, v in self._d.items():
            if not v or v[0] is None:
                continue
            if k in ("agent_mask", "next_agent_mask"):
                # Booleans, not float32: the update logic reads them as masks.
                out[k] = np.asarray(v, dtype=bool)
            else:
                out[k] = np.asarray(v, dtype=np.float32)
        self.clear()
        return out

    def __len__(self):
        return len(self._d["obs"])
