"""Observation-only route guidance and explicit local-to-world actuation."""
from __future__ import annotations

import numpy as np


CONTROL_MODES = ("world", "local", "guided", "flow_guided", "flow_spread")


def route_guidance(nodes, max_speed=0.018, contact_radius=0.035, speed=0.65):
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.shape[-1] not in (36, 42):
        raise ValueError("geometric control requires 36-dimensional observations")
    waypoint = nodes[..., 6:9] * 0.25
    target_delta = nodes[..., 25:28] * 0.5
    near_target = nodes[..., 30] > 0.5
    displacement = np.where(near_target[..., None], target_delta, waypoint)
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)
    desired = displacement * np.minimum(
        0.8 / max_speed, speed / np.maximum(distance, 1e-8)
    )
    lubrication = np.maximum(nodes[..., 24:25], 0.35)
    command = (desired - nodes[..., 21:24]) / lubrication
    command /= np.maximum(np.linalg.norm(command, axis=-1, keepdims=True), 1.0)
    return command.astype(np.float32)


def local_to_world(action, env):
    action = np.asarray(action, dtype=np.float32)
    if hasattr(env, "envs"):
        converted = []
        offset = 0
        for child in env.envs:
            converted.append(local_to_world(action[offset:offset + child.n_envs], child))
            offset += child.n_envs
        if offset != action.shape[0]:
            raise ValueError("action batch does not match vector environment")
        return np.concatenate(converted, axis=0)
    stations = env.robot_stations
    if action.shape != stations.shape + (3,):
        raise ValueError("action shape does not match robot stations")
    return (
        action[..., 0:1] * env.tree.tangents[stations]
        + action[..., 1:2] * env.tree.normals[stations]
        + action[..., 2:3] * env.tree.binormals[stations]
    ).astype(np.float32)


def direct_local_action(action, env):
    """Execute a policy action directly in the Frenet local frame.

    This is deliberately a separate path from the hand-authored guidance
    controllers below.  The only processing is the existing generic bounded
    action normalization followed by the geometric Frenet-to-world transform;
    no flow, route, target, or residual command is read here.
    """
    action = np.asarray(action, dtype=np.float32)
    if not np.isfinite(action).all():
        raise FloatingPointError("direct local policy action is non-finite")
    bounded = np.clip(action, -1.0, 1.0)
    bounded /= np.maximum(np.linalg.norm(bounded, axis=-1, keepdims=True), 1.0)
    executed = local_to_world(bounded, env)
    if not np.isfinite(executed).all():
        raise FloatingPointError("direct local world action is non-finite")
    return executed


def flow_guidance(nodes, max_speed=0.018, contact_radius=0.035, speed=0.65):
    nodes = np.asarray(nodes, dtype=np.float32)
    if nodes.shape[-1] not in (36, 42):
        raise ValueError("geometric control requires 36-dimensional observations")
    lumen = nodes[..., 19] * 0.055
    occluded = lumen * nodes[..., 20]
    radial = lumen * (1.0 - nodes[..., 18])
    measured = np.maximum(nodes[..., 21], 0.0) * max_speed
    profile = np.maximum(1.0 - (radial / np.maximum(occluded, 1e-6)) ** 2, 0.05)
    center_speed = np.minimum(measured / profile, 0.064)
    center_speed = np.where(nodes[..., 21] >= 0.99, 0.064, center_speed)
    offset = occluded * np.sqrt(np.maximum(
        1.0 - 0.35 * max_speed / np.maximum(center_speed, 1e-6), 0.0
    ))
    offset = np.minimum(offset, np.maximum(lumen - 0.002, 0.0))
    outward = nodes[..., 15:18].copy()
    outward[..., 0] = 0.0
    length = np.linalg.norm(outward, axis=-1, keepdims=True)
    fallback = np.zeros_like(outward)
    fallback[..., 1] = 1.0
    outward = np.where(length > 1e-6, outward / np.maximum(length, 1e-6), fallback)
    waypoint = nodes[..., 6:9] * 0.25
    target_delta = nodes[..., 25:28] * 0.5
    near_target = nodes[..., 30] > 0.5
    displacement = np.where(near_target[..., None], target_delta, waypoint)
    displacement = displacement + outward * offset[..., None]
    distance = np.linalg.norm(displacement, axis=-1, keepdims=True)
    desired = displacement * np.minimum(
        0.8 / max_speed, speed / np.maximum(distance, 1e-8)
    )
    command = (desired - nodes[..., 21:24]) / np.maximum(nodes[..., 24:25], 0.35)
    command /= np.maximum(np.linalg.norm(command, axis=-1, keepdims=True), 1.0)
    return command.astype(np.float32)


def _spread_target_directions(nodes, env):
    if not hasattr(env, "clot_masses") or not np.any(env.clot_masses > 0):
        return nodes[..., 25:28], nodes[..., 29], nodes[..., 30]
    alive = np.flatnonzero(env.clot_masses[:env.active_clots] > 0)
    distances = np.full((env.num_robots, len(alive)), np.inf, np.float32)
    for index, clot in enumerate(alive):
        route_distance, _ = env._route(int(clot))
        distances[:, index] = route_distance[env.robot_stations]
    order = np.argsort(np.min(distances, axis=1))
    loads = np.zeros(len(alive), np.int32)
    assignment = np.zeros(env.num_robots, np.int32)
    penalty = max(float(env.tree.total_length), 1e-6) * 0.35
    for robot in order:
        score = distances[robot] + penalty * loads
        assignment[robot] = int(np.argmin(score))
        loads[assignment[robot]] += 1
    clot_positions = env.clot_positions[alive[assignment]]
    delta = clot_positions - env.robot_positions
    tangent = env.tree.tangents[env.robot_stations]
    normal = env.tree.normals[env.robot_stations]
    binormal = env.tree.binormals[env.robot_stations]
    local = np.stack([
        np.sum(delta * tangent, axis=1),
        np.sum(delta * normal, axis=1),
        np.sum(delta * binormal, axis=1),
    ], axis=1)
    distance = np.linalg.norm(delta, axis=1)
    contact_radius = getattr(env, "clot_contact_radius", 0.035)
    touching = distance <= contact_radius
    if getattr(env, "contact_mode", "euclidean") == "geodesic":
        touching &= env._contact_geodesic()[np.arange(env.num_robots), alive[assignment]] <= contact_radius
    return np.clip(local / 0.5, -1.0, 1.0), distance / 0.5, touching


def spread_flow_guidance(nodes, env, max_speed=0.018, contact_radius=0.035, speed=0.65):
    nodes = np.asarray(nodes, dtype=np.float32).copy()
    target_delta, target_distance, touching = _spread_target_directions(nodes, env)
    nodes[..., 25:28] = target_delta
    nodes[..., 29] = target_distance
    nodes[..., 30] = touching
    command = flow_guidance(nodes, max_speed=max_speed,
                            contact_radius=contact_radius, speed=speed)
    peer_delta = nodes[..., 32:35]
    peer_distance = nodes[..., 35:36]
    repulsion = -peer_delta * np.clip((0.75 - peer_distance) / 0.75, 0.0, 1.0)
    command = command + 0.35 * repulsion
    command /= np.maximum(np.linalg.norm(command, axis=-1, keepdims=True), 1.0)
    return command.astype(np.float32)


def policy_action(action, obs, env, mode="world", residual_scale=0.2,
                  guidance_speed=0.65):
    if mode not in CONTROL_MODES:
        raise ValueError(f"unknown control mode: {mode}")
    if mode != "local":
        if not np.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual scale must be finite and nonnegative")
    if mode in ("guided", "flow_guided", "flow_spread"):
        if not np.isfinite(guidance_speed) or not 0 < guidance_speed <= 1:
            raise ValueError("guidance speed must be in (0, 1]")
    action = np.asarray(action, dtype=np.float32)
    if mode == "world":
        return action
    if obs["nodes"].shape[-1] not in (36, 42):
        raise ValueError("local control requires geometric observations")
    if mode == "local":
        # Keep this branch free of all controller calls.  In particular,
        # residual_scale is intentionally irrelevant for direct RL semantics.
        return direct_local_action(action, env)
    if mode in ("guided", "flow_guided", "flow_spread"):
        if mode == "guided":
            guidance = route_guidance(obs["nodes"], speed=guidance_speed)
        elif mode == "flow_spread":
            guidance = spread_flow_guidance(obs["nodes"], env, speed=guidance_speed)
        else:
            guidance = flow_guidance(obs["nodes"], speed=guidance_speed)
        action = guidance + residual_scale * action
    action = np.clip(action, -1.0, 1.0)
    action /= np.maximum(np.linalg.norm(action, axis=-1, keepdims=True), 1.0)
    return local_to_world(action, env)
