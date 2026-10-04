"""Shared local maneuver library and learned selector; no simulator access.

The joint projection is approximate and has no safety certificate. Learning
and conventional baselines use the same measurements and feasible-action code.
"""
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from marl.multicluster import MultiClusterController
from marl.fair_reactive import fair_reactive_action
from marl.partial_obs import PATH0, CLOT0

N_ACTIONS = 10
FEATURE_DIM = 111+14+3+N_ACTIONS


def project_joint(commands, packet, cfg, speed=1., iterations=16):
    """Approximate intersection of local wall/peer halfspaces and speed ball."""
    single = commands.ndim == 2
    result = commands[:, None, :].copy() if single else commands.copy()
    residual = np.zeros(result.shape[:2])
    for i in np.flatnonzero(packet.active):
        o = packet.navigation[i]
        constraints = []
        for j in np.flatnonzero(packet.peer_visible[i]):
            r = packet.peer_relative_mm[i, j]
            distance = float(np.linalg.norm(r))
            if distance < 1e-9:
                normal = np.array([1. if j > i else -1., 0., 0.])
            else:
                normal = r/distance
            relative_velocity = packet.peer_relative_velocity_mm_s[i, j]
            gap = distance-cfg.min_spacing_mm-cfg.spacing_buffer_mm
            bound = float(normal@o[:3]+(normal@relative_velocity+.5*gap/cfg.wait_horizon_s)/speed)
            constraints.append((normal, bound))
        radial = o[6:9].astype(float)
        radius_fraction = float(np.linalg.norm(radial))
        if radius_fraction > 1e-6:
            normal = radial/radius_fraction
            # The visible-path radius is an existing image feature, not a query
            # to exact environment geometry. Conservative minimum among paths.
            radii = [np.exp(np.clip(o[PATH0+10*k+9], -5, 5)) for k in range(4)
                     if o[PATH0+10*k] > 0]
            radius = min(radii) if radii else .08
            gap = (float(o[9])-.08)*radius
            bound = float(normal@o[:3]-normal@o[3:6]+.5*gap/(cfg.wait_horizon_s*speed))
            constraints.append((normal, bound))
        # Dykstra projections preserve the closest-command objective better
        # than arbitrary repeated clipping. Infeasibility is exposed below.
        corrections = [np.zeros_like(result[i]) for _ in range(len(constraints)+1)]
        x = result[i].copy()
        for _ in range(iterations):
            for k, (normal, bound) in enumerate(constraints):
                y = x+corrections[k]
                z = y-np.maximum(y@normal-bound, 0.)[:, None]*normal
                corrections[k] = y-z; x = z
            y = x+corrections[-1]
            z = y/np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1.)
            corrections[-1] = y-z; x = z
        result[i] = x
        residual[i] = np.maximum(np.stack([x@n-b for n, b in constraints]+[np.zeros(len(x))]), 0.).max(axis=0)
    result[~packet.active] = 0.
    return (result[:, 0], residual[:, 0]) if single else (result, residual)


class ManeuverLibrary:
    """Measured local history; every row can be executed without a global map."""
    def __init__(self, cfg, speed=1., dt=.1):
        self.cfg, self.speed, self.dt = cfg, speed, dt
        self.base = MultiClusterController(cfg, robot_speed_mm_s=speed, control_dt_s=dt)
        self.reset()

    def reset(self):
        self.base.reset(self.cfg.control_seed)
        self.history = [deque(maxlen=30) for _ in range(self.cfg.clusters)]
        self.last_choice = np.zeros(self.cfg.clusters, np.int64)
        self.escape = np.zeros(self.cfg.clusters, np.int64)
        self.escape_slot = np.zeros(self.cfg.clusters, np.int64)
        self.steps = 0
        self.infeasible_agent_steps = 0

    def prepare(self, packet):
        n = self.cfg.clusters
        legacy, _ = self.base.act(packet)
        reactive = fair_reactive_action(packet.navigation, mode='path')
        raw = np.zeros((n, N_ACTIONS, 3))
        raw[:, 0], raw[:, 1], raw[:, 2] = legacy, reactive, .5*reactive
        valid = np.ones((n, N_ACTIONS), bool)
        for k in range(4):
            raw[:, 4+k] = packet.navigation[:, PATH0+10*k+1:PATH0+10*k+4]
            valid[:, 4+k] = packet.navigation[:, PATH0+10*k] > 0
        raw[:, 8] = -reactive
        raw[:, 9] = packet.navigation[:, CLOT0+2:CLOT0+5]
        candidates, errors = project_joint(raw, packet, self.cfg, self.speed)
        feature = np.zeros((n, FEATURE_DIM), np.float32)
        feature[:, :111] = np.clip(packet.navigation, -10., 10.)
        stuck = np.zeros(n, bool)
        for i in range(n):
            peers = np.flatnonzero(packet.peer_visible[i])
            peers = sorted(peers, key=lambda j: np.linalg.norm(packet.peer_relative_mm[i, j]))[:2]
            for slot, j in enumerate(peers):
                offset = 111+7*slot
                feature[i, offset:offset+7] = np.r_[1.,
                    packet.peer_relative_mm[i, j]/self.cfg.peer_sensing_radius_mm,
                    packet.peer_relative_velocity_mm_s[i, j]/(2*self.speed)]
            ident = int(packet.clot_ids[i, 0])
            distance = float(packet.navigation[i, CLOT0+1])*10
            mass = float(packet.navigation[i, CLOT0+5])
            h = self.history[i]
            if h and h[-1][0] != ident:
                h.clear(); self.escape[i] = 0
            h.append((ident, distance, mass))
            if len(h) == h.maxlen:
                start = np.array(list(h)[:10], float).mean(axis=0)
                end = np.array(list(h)[-10:], float).mean(axis=0)
                progress, lysis = start[1]-end[1], start[2]-end[2]
                stuck[i] = progress < .03 and lysis < .01 and ident >= 0
            else:
                progress, lysis = 0., 0.
            feature[i, 125:128] = [np.clip(progress, -2, 2), np.clip(lysis, -1, 1), float(stuck[i])]
            feature[i, 128+self.last_choice[i]] = 1.
        self.steps += 1
        return feature, candidates, valid, dict(legacy=legacy, stuck=stuck, projection_residual=errors)

    def conventional_choice(self, packet, valid, details, memory=False):
        choices = np.zeros(self.cfg.clusters, np.int64)
        if not memory:
            return choices
        for i in range(self.cfg.clusters):
            if self.escape[i] == 0 and details['stuck'][i]:
                self.escape[i] = 30
                paths = np.flatnonzero(valid[i, 4:8])
                if len(paths):
                    self.escape_slot[i] = (self.escape_slot[i]+1) % len(paths)
                    self.escape_slot[i] = int(paths[self.escape_slot[i]])
            if self.escape[i] > 0:
                choices[i] = 8 if self.escape[i] > 20 else 4+self.escape_slot[i]
                self.escape[i] -= 1
                if not valid[i, choices[i]]:
                    choices[i] = 0
        return choices

    def commit(self, choices, details):
        self.last_choice = np.asarray(choices, np.int64).copy()
        chosen_residual = details['projection_residual'][np.arange(self.cfg.clusters), choices]
        self.infeasible_agent_steps += int((chosen_residual > 1e-5).sum())


class LocalManeuverActorCritic(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, N_ACTIONS))
        self.critic = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        # Deterministic untrained policy is the shared joint-filter baseline.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad(): self.actor[-1].bias[0] = 3.

    def distribution(self, features, valid):
        logits = self.actor(features).masked_fill(~valid, -1e9)
        return Categorical(logits=logits)

    def value(self, features):
        return self.critic(features).squeeze(-1)
