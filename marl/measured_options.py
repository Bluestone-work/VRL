"""Observation-only persistent options and their shared, uncertified supervisor.

No class in this module accepts an environment. Targets are detection IDs,
never native vascular nodes. Both schedulers and learned actors see the same
centralized collection of local measurement packets.
"""
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from marl.multicluster import ClusterObservation
from marl.partial_obs import CLOT0, PATH0

N_OPTIONS = 7
OPTION_FEATURES = 111 + 14 + N_OPTIONS + 16 + 5 + 1


def option_observation(packet, previous_options, previous_targets=None):
    n = len(packet.active)
    features = np.zeros((n, OPTION_FEATURES), np.float32)
    features[:, :111] = np.clip(packet.navigation, -10., 10.)
    valid = np.zeros((n, N_OPTIONS), bool)
    valid[:, 0] = True
    for i in np.flatnonzero(packet.active):
        peers = sorted(np.flatnonzero(packet.peer_visible[i]),
                       key=lambda j: np.linalg.norm(packet.peer_relative_mm[i, j]))[:2]
        for slot, j in enumerate(peers):
            features[i, 111+7*slot:118+7*slot] = np.r_[1.,
                packet.peer_relative_mm[i, j]/6.,
                packet.peer_relative_velocity_mm_s[i, j]/2.]
        for k in range(4):
            valid[i, k+1] = packet.navigation[i, CLOT0+6*k] > 0
            ident = int(packet.clot_ids[i, k])
            if valid[i, k+1] and 0 <= ident < 4:
                features[i, 132+4*k+ident] = 1.
        valid[i, 5:] = True
    features[np.arange(n), 125+np.asarray(previous_options, int)] = 1.
    targets = np.full(n, -1) if previous_targets is None else np.asarray(previous_targets)
    features[np.arange(n), 148+np.where((targets >= 0) & (targets < 4), targets, 4)] = 1.
    features[:, -1] = packet.active
    return features, valid


def latch_targets(packet, options):
    targets = np.full(len(options), -1, np.int32)
    for i, option in enumerate(options):
        if 1 <= option <= 4:
            targets[i] = packet.clot_ids[i, option-1]
    return targets


def target_packet(packet, targets):
    """Reorder measured target slots without changing the input packet."""
    nav, ids = packet.navigation.copy(), packet.clot_ids.copy()
    for i, target in enumerate(targets):
        if target < 0:
            continue
        matches = np.flatnonzero(ids[i] == target)
        if not len(matches):
            continue
        slot = int(matches[0])
        order = [slot] + [j for j in range(4) if j != slot]
        nav[i, CLOT0:CLOT0+24] = packet.navigation[i, CLOT0:CLOT0+24].reshape(4, 6)[order].ravel()
        ids[i] = packet.clot_ids[i, order]
    return ClusterObservation(nav, ids, packet.peer_relative_mm,
        packet.peer_relative_velocity_mm_s, packet.peer_visible, packet.active)


def measured_assignment(packet, previous_targets=None, hysteresis_mm=.4):
    """Small exact assignment using measured Euclidean distances only."""
    n = len(packet.active)
    choices = np.zeros(n, np.int64)
    previous = np.full(n, -1) if previous_targets is None else previous_targets
    ids = np.flatnonzero(packet.active)
    candidates = []
    for i in ids:
        slots = [k for k in range(4) if packet.navigation[i, CLOT0+6*k] > 0]
        candidates.append(slots or [-1])
    best, score = None, np.inf
    for slots in product(*candidates):
        cost, assigned = 0., []
        for i, k in zip(ids, slots):
            if k < 0:
                continue
            target = int(packet.clot_ids[i, k])
            cost += float(packet.navigation[i, CLOT0+6*k+1])*10.
            cost -= hysteresis_mm * (target == previous[i])
            assigned.append(target)
        # A common task-allocation penalty; no true mass or geodesic ETA.
        cost += 4. * (len(assigned)-len(set(assigned)))
        if cost < score:
            score, best = cost, slots
    if best is not None:
        for i, k in zip(ids, best):
            choices[i] = k+1 if k >= 0 else 0
    return choices


def priority_options(packet, previous_targets=None):
    choice = measured_assignment(packet, previous_targets)
    for i in range(len(choice)):
        for j in range(i+1, len(choice)):
            if not packet.peer_visible[i, j]:
                continue
            r, v = packet.peer_relative_mm[i, j], packet.peer_relative_velocity_mm_s[i, j]
            t = float(np.clip(-r@v/max(float(v@v), 1e-12), 0., .75))
            if np.linalg.norm(r+t*v) >= 2.5:
                continue
            di = float(packet.navigation[i, CLOT0+1])
            dj = float(packet.navigation[j, CLOT0+1])
            # The closer-to-treatment robot keeps priority; ID breaks ties.
            yielding = j if (di, i) <= (dj, j) else i
            choice[yielding] = 5
    return choice


def joint_measured_projection(commands, packet, settings, speed=1.):
    """Project a *joint* command, with both robots' proposed velocities.

    The margins are assumptions, not bounds proved for the noisy tracker.
    Missing tracks have no hidden fallback. Residuals expose infeasibility.
    """
    n = len(commands)
    preferred = np.asarray(commands, float).copy()
    preferred[~packet.active] = 0.
    drift = speed * (packet.navigation[:, 3:6]-packet.navigation[:, :3])
    inequalities = []
    for i in np.flatnonzero(packet.active):
        for j in range(i+1, n):
            if not packet.peer_visible[i, j] or not packet.active[j]:
                continue
            r = packet.peer_relative_mm[i, j]
            distance = float(np.linalg.norm(r))
            normal = r/max(distance, 1e-9) if distance > 1e-9 else np.array([1., 0., 0.])
            normal_full = np.zeros((n, 3))
            normal_full[i], normal_full[j] = normal, -normal
            measured_drift_rel = packet.peer_relative_velocity_mm_s[i, j] - speed*(
                packet.navigation[j, :3]-packet.navigation[i, :3])
            gap = distance-settings['spacing_mm']-settings['position_margin_mm']
            bound = (normal@measured_drift_rel + gap/settings['horizon_s']
                     - settings['relative_velocity_margin_mm_s'])/speed
            inequalities.append((normal_full.ravel(), float(bound)))
        o = packet.navigation[i]
        radial = np.asarray(o[6:9], float)
        norm = np.linalg.norm(radial)
        if norm > 1e-6:
            normal = radial/norm
            radii = [np.exp(np.clip(o[PATH0+10*k+9], -5., 5.))
                     for k in range(4) if o[PATH0+10*k] > 0]
            radius = min(radii) if radii else .08
            clearance = (float(o[9])-settings['wall_margin_fraction'])*radius
            normal_full = np.zeros((n, 3)); normal_full[i] = normal
            bound = (clearance/settings['horizon_s'] - normal@drift[i])/speed
            inequalities.append((normal_full.ravel(), float(bound)))
    x = preferred.ravel().copy()
    corrections = [np.zeros_like(x) for _ in range(len(inequalities)+1)]
    for _ in range(settings['iterations']):
        for k, (a, b) in enumerate(inequalities):
            y = x+corrections[k]
            z = y-max(float(a@y-b), 0.)/max(float(a@a), 1e-12)*a
            corrections[k], x = y-z, z
        y = x+corrections[-1]
        shaped = y.reshape(n, 3)
        z = (shaped/np.maximum(np.linalg.norm(shaped, axis=1, keepdims=True), 1.)).ravel()
        corrections[-1], x = y-z, z
    result = x.reshape(n, 3)
    result[~packet.active] = 0.
    residual = max([max(float(a@result.ravel()-b), 0.) for a, b in inequalities] or [0.])
    return result, dict(residual=residual, constraints=len(inequalities),
        missing_tracks=int((~packet.active).sum()),
        changed=bool(np.max(np.abs(result-preferred)) > 1e-6))


class OptionActorCritic(nn.Module):
    """Shared robot actor with measured team context; same inputs to critic."""
    def __init__(self, hidden=96, memory_bias=3.):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OPTION_FEATURES, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(2*hidden, hidden), nn.Tanh(), nn.Linear(hidden, N_OPTIONS))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.actor[-1].weight); nn.init.zeros_(self.actor[-1].bias)
        with torch.no_grad():
            self.actor[-1].bias[0] = memory_bias

    def forward(self, features, valid):
        encoded = self.encoder(features)
        active = features[..., -1:]
        context = (encoded*active).sum(dim=1)/active.sum(dim=1).clamp_min(1.)
        together = torch.cat((encoded, context[:, None].expand_as(encoded)), dim=-1)
        logits = self.actor(together).masked_fill(~valid, -1e9)
        return Categorical(logits=logits), self.critic(context).squeeze(-1)


def smdp_gae(rewards, values, dones, durations, last_value, gamma, lam):
    """rewards already contain within-option discounting."""
    result = np.zeros(len(rewards), np.float32)
    carry = 0.
    for t in range(len(rewards)-1, -1, -1):
        future = last_value if t == len(rewards)-1 else values[t+1]
        live = 1.-float(dones[t])
        discount = gamma**int(durations[t])
        delta = rewards[t]+discount*future*live-values[t]
        carry = delta+discount*(lam**int(durations[t]))*live*carry
        result[t] = carry
    return result, result+np.asarray(values, np.float32)
