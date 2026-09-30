"""One loader for every checkpoint format this repo produces.

Three trainers now write `.pt` files that share nothing but the extension:

  * `train_vascular_maddpg.py` / `train_vector.py` -> MADDPG actors
  * `train_gnn_mappo.py`                           -> GAT actor/critic
  * `train_advanced.py`                            -> one of six architectures

The viewers (`watch_gui.py`, `make_gif.py`) used to hardcode MADDPG, so they
raised a shape error on anything else. Rather than teach each viewer about each
trainer, this module rebuilds the right network from the checkpoint's own `meta`
block and hands back a uniform callable.

The other wrinkle is the calling convention. A MADDPG actor is a function of the
node observation alone; the graph architectures also need a per-step context
(positions, velocities, adjacency), because that is what their attention layers
consume. So the returned policy takes `(obs_dict, env)` rather than a bare
array -- the env is where the context comes from.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


def _context_from_env(env, obs_dict: dict) -> dict:
    """Same context the advanced trainer builds during rollout."""
    ctx = {}
    if hasattr(env, "robot_positions"):
        ctx["positions"] = env.robot_positions.copy()
    if hasattr(env, "robot_velocities"):
        ctx["velocities"] = env.robot_velocities.copy()
    if "adjacency" in obs_dict:
        ctx["adjacency"] = obs_dict["adjacency"]
    return ctx


def _infer_advanced_meta(ckpt: dict, env, fallback_arch: Optional[str]) -> dict:
    """Recover build parameters for checkpoints written before `meta` existed.

    Shapes are read off the tensors themselves, so an older file still loads;
    only the architecture name is genuinely unrecoverable and must be supplied.
    """
    actor = ckpt["actor"]
    if fallback_arch is None:
        raise SystemExit(
            "this checkpoint has no 'meta' block, so its architecture is unknown.\n"
            "Re-run the viewer with --architecture <name> (gat, edge_gat, "
            "transformer, sparse_transformer, hierarchical_gnn, mlp)."
        )

    obs_dim = env.observation_space["nodes"].shape[1]
    action_dim = 3
    hidden_dim = actor["policy_head.mean.weight"].shape[1]

    # state_dim is only visible on the critic's state encoder, when it has one.
    state_dim = 0
    for key in ("inner.state_encoder.weight",):
        if key in ckpt.get("critic", {}):
            state_dim = ckpt["critic"][key].shape[1]
            break

    return {
        "architecture": fallback_arch,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
        "state_dim": state_dim,
        "num_layers": 2,
        "num_heads": 4,
        "sparse_k": 8,
        "n_agents": getattr(env, "num_robots", 5),
    }


def load_policy(
    path: str,
    env,
    device: str = "cpu",
    architecture: Optional[str] = None,
    hidden_dim: int = 128,
) -> Callable[[dict, object], np.ndarray]:
    """Return `fn(obs_dict, env) -> action` for any checkpoint this repo writes.

    Actions are deterministic (distribution mean / no exploration noise), which
    is what you want for viewing and for reported evaluation numbers.
    """
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt.get("meta", {})
    env_obs_dim = env.observation_space["nodes"].shape[1]

    # An obs_mode mismatch is the single most common way this goes wrong; say so
    # in words instead of letting it surface as a matmul shape error.
    stored_dim = meta.get("obs_dim")
    if stored_dim is not None and stored_dim != env_obs_dim:
        expected = {20: "legacy", 36: "geometric", 42: "geometric_v2", 44: "geometric_dynamic", 52: "geometric_predictive"}.get(stored_dim, "unknown")
        raise SystemExit(
            f"checkpoint expects obs_dim={stored_dim} but the env provides "
            f"{env_obs_dim}.\nRe-run with --obs-mode {expected}"
        )

    is_advanced = "actor" in ckpt and any(
        k.startswith("policy_head.") for k in ckpt["actor"]
    )

    # ---- MADDPG (original trainer) --------------------------------------
    if not is_advanced:
        from marl.maddpg_policy import MADDPG

        maddpg = MADDPG(
            n_agents=getattr(env, "num_robots", 3),
            obs_dim=env_obs_dim,
            action_dim=3,
            hidden_dim=meta.get("hidden_dim", hidden_dim),
            device=device,
            state_dim=meta.get("state_dim", 0),
            share_parameters=meta.get("share_parameters", True),
        )
        maddpg.load(path)
        print(f"loaded MADDPG policy from {path}")

        def act_maddpg(obs_dict: dict, _env) -> np.ndarray:
            return maddpg.select_actions(obs_dict["nodes"], add_noise=False)

        return act_maddpg

    # ---- MAPPO / graph architectures ------------------------------------
    from marl.mappo_advanced import MAPPOAdvanced

    if not meta:
        meta = _infer_advanced_meta(ckpt, env, architecture)
        print("checkpoint has no meta block; rebuilt from tensor shapes")

    agent = MAPPOAdvanced(
        n_agents=meta["n_agents"],
        obs_dim=meta["obs_dim"],
        action_dim=meta["action_dim"],
        architecture=meta["architecture"],
        state_dim=meta.get("state_dim", 0),
        hidden_dim=meta.get("hidden_dim", 128),
        num_layers=meta.get("num_layers", 2),
        num_heads=meta.get("num_heads", 4),
        sparse_k=meta.get("sparse_k", 8),
        control_mode=meta.get("control_mode", "world"),
        residual_scale=meta.get("residual_scale", 0.2),
        guidance_speed=meta.get("guidance_speed", 0.65),
        critic_value_mode=meta.get("critic_value_mode", "q"),
        dropout=meta.get("dropout", 0.1),
        device=device,
    )
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval()
    agent.critic.eval()
    print(f"loaded {meta['architecture']} policy from {path}")

    def act_advanced(obs_dict: dict, env_ref) -> np.ndarray:
        ctx = _context_from_env(env_ref, obs_dict)
        state = obs_dict["clot_state"].flatten() if "clot_state" in obs_dict else None
        actions, _, _ = agent.act(
            obs_dict["nodes"], ctx, state, deterministic=True
        )
        return agent.env_action(actions, obs_dict, env_ref)

    return act_advanced
