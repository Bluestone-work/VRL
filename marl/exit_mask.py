"""Predict open-boundary crossings from camera estimates and the preoperative map."""
from __future__ import annotations

import numpy as np


MARGIN = 8.0  # prediction horizon in control intervals


def _crossings(ep, est, commands, drift, margin):
    n, k, _ = commands.shape
    transport = ep.env.transport
    positions = np.asarray(est.pos, float)
    edges = np.asarray(est.edge, int)
    frames = ep.ctl.frames(est)
    # Frames map world vectors into local coordinates. Apply the transpose
    # to each local candidate, as DeployablePursuit.to_world does.
    world = np.einsum('nij,nki->nkj', frames, commands)
    world /= np.maximum(np.linalg.norm(world, axis=-1, keepdims=True), 1.)
    speed = ep.env.config.robot_speed_mm_s
    dt = margin * ep.env.config.control_dt_s
    proposed = positions[:, None] + (drift[:, None] + speed * world) * dt
    # Only the surveyed map geometry and aperture enter this calculation.
    node, fraction, _ = transport._boundary_crossings(
        np.repeat(positions, k, axis=0), proposed.reshape(-1, 3),
        np.repeat(edges, k), np.full(n * k, ep.ctl.body), {'radius_mm': ep.sensor.healthy})
    return node.reshape(n, k), fraction.reshape(n, k)


def exit_mask(ep, est, commands, valid, margin=MARGIN, *, drift=None):
    drift = np.zeros_like(est.pos) if drift is None else np.asarray(drift, float)
    node, _ = _crossings(ep, est, commands, drift, margin)
    return (node >= 0) & est.active[:, None]


def apply_exit_mask(ep, est, commands, valid, margin=MARGIN, *, drift=None):
    """If all available commands cross, retain the one with the latest predicted exit."""
    drift = np.zeros_like(est.pos) if drift is None else np.asarray(drift, float)
    node, fraction = _crossings(ep, est, commands, drift, margin)
    masked = valid.copy()
    masked[(node >= 0) & est.active[:, None]] = 0.
    for row in np.flatnonzero(~masked.any(axis=1)):
        available = np.flatnonzero(valid[row])
        if available.size:
            masked[row, available[fraction[row, available].argmax()]] = 1.
    return masked, int(np.count_nonzero(valid > masked))


def compensate_exit_command(ep, est, commands, margin=1.0, *, drift=None):
    """Select among the issued command, drift cancellation and an inward command."""
    drift = np.zeros_like(est.pos) if drift is None else np.asarray(drift, float)
    commands = np.asarray(commands, dtype=float).reshape(-1, 3)
    node, _ = _crossings(ep, est, commands[:, None], drift, margin)
    unsafe = (node[:, 0] >= 0) & est.active
    if not unsafe.any():
        return commands
    frames = ep.ctl.frames(est)
    cancel = commands - np.einsum('nij,nj->ni', frames, drift) / ep.env.config.robot_speed_mm_s
    cancel /= np.maximum(np.linalg.norm(cancel, axis=1, keepdims=True), 1.)
    transport = ep.env.transport
    inlet = node[:, 0] == transport.ends[est.edge, 0]
    inward = transport.direction[est.edge] * np.where(inlet, 1., -1.)[:, None]
    inward = np.einsum('nij,nj->ni', frames, inward)
    candidates = np.stack([commands, cancel, inward], axis=1)
    nodes, fraction = _crossings(ep, est, candidates, drift, margin)
    best = np.where(nodes < 0, np.inf, fraction).argmax(axis=1)
    return candidates[np.arange(len(commands)), np.where(unsafe, best, 0)]
