"""EXP16--20 v2: causal local sensing, learned prediction and hybrid control.

All inference functions consume observed arrays. No simulator stepping, RNG,
future-state query or reward query is available to the controller. Units are
simulation steps (velocity is displacement per step), not seconds.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

SPEED = 0.018
RANGE = 0.12
HISTORY = 6
HORIZONS = (1, 3, 5)


class MotionGRU(nn.Module):
    """History of tracked position/velocity -> displacement mean and log variance."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(6, 48, batch_first=True)
        self.head = nn.Linear(48, 18)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        _, h = self.gru(x)
        raw = self.head(h[-1]).reshape(-1, 2, 3, 3)
        # Residual around constant velocity, in SPEED-normalised units.
        cv = x[:, -1, 3:6, None].transpose(1, 2)
        cv = cv * x.new_tensor(HORIZONS)[None, :, None]
        return cv + raw[:, 0], raw[:, 1].clamp(-7, 4)


class LocalWorldEnsemble(nn.Module):
    """Action-conditioned local transition ensemble; not a full vessel simulator.

    Targets: node delta (52), realised Frenet displacement (3), removed mass
    share (1), wall event (1). Collision risk is also estimated from predicted
    obstacle/peer clearance. Models have independently bootstrapped updates.
    """

    def __init__(self, members=5):
        super().__init__()
        self.members = nn.ModuleList([
            nn.Sequential(nn.Linear(55, 128), nn.SiLU(), nn.Linear(128, 128),
                          nn.SiLU(), nn.Linear(128, 57)) for _ in range(members)
        ])

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1)
        return torch.stack([m(x) for m in self.members])


def frame(env):
    st = env.robot_stations
    return np.stack((env.tree.tangents[st], env.tree.normals[st],
                     env.tree.binormals[st]), axis=-2)


def local(vec, basis):
    return np.einsum('...ij,...j->...i', basis, vec)


class SensorHistory:
    """Nearest visible obstacle track with explicit ID association in simulation.

    IDs are used only for association, never as learned inputs. Reacquisition,
    changing nearest obstacle and episode resets clear history. Real hardware
    would require a tracker; this is an explicit ideal-sensor assumption.
    """

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ids = np.full(shape, -1, np.int32)
        self.steps = np.full(shape, -1, np.int64)
        self.values = np.zeros((*shape, HISTORY, 6), np.float32)
        self.valid = np.zeros(shape, np.int32)

    def reset(self, rows=None):
        rows = slice(None) if rows is None else rows
        self.ids[rows] = -1
        self.steps[rows] = -1
        self.values[rows] = 0
        self.valid[rows] = 0

    def observe(self, positions, velocities, particles, particle_velocities, steps):
        distance = np.linalg.norm(positions[..., None, :] - particles[:, None], axis=-1)
        ids = distance.argmin(-1)
        near = np.take_along_axis(distance, ids[..., None], -1)[..., 0] <= RANGE
        point = np.take_along_axis(particles, ids[..., None], 1)
        vel = np.take_along_axis(particle_velocities, ids[..., None], 1)
        times = np.broadcast_to(np.asarray(steps)[:, None], ids.shape)
        changed = (ids != self.ids) | ~near | (times < self.steps)
        self.valid[changed] = 0
        self.values[changed] = 0
        update = (times != self.steps) | changed
        shifted = np.concatenate((self.values[..., 1:, :],
                                  np.concatenate((point, vel / SPEED), -1)[..., None, :]), -2)
        self.values[update] = shifted[update]
        self.valid[update] = np.minimum(self.valid[update] + 1, HISTORY)
        self.valid[~near] = 0
        self.ids[:] = np.where(near, ids, -1)
        self.steps[:] = times
        # Centre position history at its current observation for translation invariance.
        x = self.values.copy()
        x[..., :3] = (x[..., :3] - point[..., None, :]) / RANGE
        return x, point - positions, vel - velocities, near


class Features:
    def __init__(self, model=None, device='cpu'):
        self.model, self.device = model, device
        self.history = None

    def reset(self, rows=None):
        if self.history is not None:
            self.history.reset(rows)

    def augment(self, env, obs, vector=True):
        p = env.robot_positions if vector else env.robot_positions[None]
        v = env.robot_velocities if vector else env.robot_velocities[None]
        steps = env.steps if vector else np.array([env.steps])
        if self.history is None or self.history.shape != p.shape[:-1]:
            self.history = SensorHistory(p.shape[:-1])
        x, rel, relvel, seen = self.history.observe(
            p, v, env.particles.positions, env.particles.velocities, steps)
        basis = frame(env) if vector else frame(env)[None]
        horizon = 3.0
        predicted = rel + horizon * relvel
        sigma = np.full(p.shape[:-1], env.flow_speed * env.particles.lateral_drift
                        * np.sqrt(horizon), np.float32)
        ready = (self.history.valid >= HISTORY) & seen
        if self.model is not None and ready.any():
            with torch.no_grad():
                mu, lv = self.model(torch.as_tensor(x[ready], device=self.device))
            predicted[ready] = rel[ready] + mu[:, 1].cpu().numpy() * SPEED - horizon * v[ready]
            sigma[ready] = np.sqrt(np.exp(lv[:, 1].cpu().numpy()).sum(-1)) * SPEED
        nodes = obs['nodes'] if vector else obs['nodes'][None]
        c = env.particles.contact_distance
        # Expose only observations within the registered local sensing range.
        nodes[..., 36:44] = 0
        nodes[..., 36:39] = np.clip(local(rel, basis) / RANGE, -1, 1) * seen[..., None]
        nodes[..., 39] = np.where(seen, np.clip((np.linalg.norm(rel, axis=-1)-c)/.06, -1, 1), 1)
        nodes[..., 40] = np.clip(np.linalg.norm(relvel, axis=-1)/SPEED, 0, 1)*seen
        nodes[..., 41] = (np.linalg.norm(rel, axis=-1) < c)*seen
        nodes[..., 42] = np.where(seen, np.clip(np.linalg.norm(rel, axis=-1)/RANGE, 0, 1), 1)
        nodes[..., 43] = seen
        # Closest approach over the whole horizon catches pass-through collisions.
        drift = (predicted-rel)/horizon
        t = np.clip(-np.sum(rel*drift, -1)/np.maximum(np.sum(drift*drift, -1), 1e-10), 0, horizon)
        clearance = np.linalg.norm(rel+t[..., None]*drift, axis=-1)-c
        risk = 1/(1+np.exp(np.clip(clearance/np.maximum(sigma, c*.25), -30, 30)))
        nodes[..., 44:47] = np.clip(local(predicted, basis)/RANGE, -1, 1)*seen[..., None]
        nodes[..., 47] = np.where(seen, np.clip(clearance/.06, -1, 1), 1)
        nodes[..., 48] = np.clip(-np.sum(rel*relvel, -1)/np.maximum(np.linalg.norm(rel, axis=-1)*SPEED, 1e-9), 0, 1)*seen
        nodes[..., 49] = np.where(seen, t/horizon, 1)
        nodes[..., 50] = np.clip(sigma/RANGE, 0, 1)*seen
        nodes[..., 51] = risk*seen  # risk score, not claimed to be calibrated probability
        if 'agent_mask' in obs:
            nodes *= obs['agent_mask'][..., None]
        return obs


def bounded(action):
    action = np.clip(action, -1, 1)
    return action / np.maximum(np.linalg.norm(action, axis=-1, keepdims=True), 1)


def candidates(action):
    offsets = np.concatenate((np.zeros((1, 3)), .3*np.eye(3), -.3*np.eye(3)))
    return np.concatenate((bounded(action[..., None, :]+offsets),
                           np.zeros((*action.shape[:-1], 1, 3))), -2).astype(np.float32)


def analytic_cost(nodes, actions, horizon=3):
    """Local constant-frame surrogate, no ground-truth env rollout.

    A fixed frame is intentionally short horizon; curvature errors are included
    in validation. Length penalty small relative to collision risk.
    """
    n = nodes[..., None, :]
    delta = (actions*n[..., 24:25] + n[..., 21:24])*SPEED
    lumen = np.maximum(n[..., 19]*.055, .0011)
    wall_vector = n[..., 15:18] * (lumen*(1-n[..., 18]))[..., None]
    wall_risk = np.zeros(actions.shape[:-1], np.float32)
    obstacle_risk = np.zeros_like(wall_risk)
    # Forecast at 3 steps; linearly interpolate causal forecast only.
    obstacle_velocity = (n[..., 44:47]-n[..., 36:39])*RANGE/3 + n[..., 3:6]*SPEED
    peer_delta = n[..., 32:35]*.06
    for step in range(1, horizon+1):
        displacement = step*delta
        radial = wall_vector+displacement*np.array([0, 1, 1], np.float32)
        wall_risk += np.maximum(np.linalg.norm(radial, axis=-1)+.0011-lumen, 0)/.0011
        rel = n[..., 36:39]*RANGE+step*obstacle_velocity-displacement
        obstacle_risk += np.maximum(.00286+2*n[..., 50]*RANGE-np.linalg.norm(rel, axis=-1), 0)/.00286*n[..., 43]
        obstacle_risk += np.maximum(.0022-np.linalg.norm(peer_delta-displacement, axis=-1), 0)/.0022
    target = np.where(n[..., 30:31] > .5, n[..., 25:28]*.5, n[..., 6:9]*.25)
    progress = np.linalg.norm(target, axis=-1)-np.linalg.norm(target-horizon*delta, axis=-1)
    return 2*wall_risk+2*obstacle_risk-progress/SPEED+.03*np.linalg.norm(delta, axis=-1)/SPEED


class ActionController:
    def __init__(self, world=None, device='cpu'):
        self.world, self.device = world, device

    def choose(self, nodes, proposal):
        acts = candidates(proposal)
        cost = analytic_cost(nodes, acts)
        cost += .15*np.square(acts-bounded(proposal)[..., None, :]).sum(-1)
        accepted = np.zeros(cost.shape, bool)
        if self.world is not None:
            # Receding-horizon, constant candidate sequences; apply just first action.
            tiled = np.broadcast_to(nodes[..., None, :], (*acts.shape[:-1], 52)).copy()
            shape = cost.shape
            state = torch.as_tensor(tiled.reshape(-1, 52), device=self.device)
            action = torch.as_tensor(acts.reshape(-1, 3), device=self.device)
            predicted_cost = torch.zeros(len(state), device=self.device)
            trusted = torch.ones(len(state), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                for _ in range(3):
                    ensemble = self.world(state, action)
                    mean = ensemble.mean(0)
                    uncertainty = ensemble[..., 52:55].std(0).norm(dim=-1)*SPEED
                    trusted &= (uncertainty < .0045) & torch.isfinite(mean).all(-1)
                    predicted_cost += 2*mean[:, 56].sigmoid() - .5*mean[:, 55].clamp(0, 5)
                    predicted_cost += .03*mean[:, 52:55].norm(dim=-1)
                    state = (state+.1*mean[:, :52]).clamp(-1, 1)
                    predicted_cost += 2*state[:, 51].clamp(0, 1)
            # Reject the learned rollout for a robot unless every candidate is trusted;
            # otherwise comparisons mix objectives and favour rejected candidates.
            accepted = trusted.cpu().numpy().reshape(shape)
            all_trusted = accepted.all(-1, keepdims=True)
            cost += np.where(all_trusted, predicted_cost.cpu().numpy().reshape(shape), 0)
        chosen = cost.argmin(-1)
        result = np.take_along_axis(acts, chosen[..., None, None], -2)[..., 0, :]
        return result, {
            'interventions': int((np.linalg.norm(result-bounded(proposal), axis=-1)>1e-6).sum()),
            'model_trusted': int(accepted.all(-1).sum()) if self.world is not None else 0,
        }
