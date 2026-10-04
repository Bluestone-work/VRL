"""Measured conservative residual MARL; no dynamics-prediction policy feature.

The graph and PPO components are adaptations, not a new algorithm claim.
Both anchoring arms and their controls use identical bounded action permissions.
"""
import numpy as np
import torch
from torch.distributions import Categorical
from marl.measured_marl import MeasuredMARL

VARIANTS = ('r_mappo', 'mappo_anchor', 'graph_ppo', 'graph_anchor')


class ResidualMARL(MeasuredMARL):
    def __init__(self, variant, hidden=64, clusters=3, prior=2.0794415416798357):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__('vctpg_no_ac' if variant.startswith('graph') else 'r_mappo',
                         hidden, clusters, prior)
        self.variant = variant


def bounded_candidates(commands, candidates, valid, radius):
    """Bound departure from nominal only when forward nominal is admissible.

    Blocked/ambiguous states retain the exact old hold/escape controls and masks.
    Convex interpolation of unit-ball commands preserves the actuator limit.
    Feature dots are interpolated with the same coefficient as the controls.
    """
    if not 0 < radius <= 1:
        raise ValueError('Residual radius must lie in (0, 1]')
    output, features = commands.copy(), candidates.copy()
    for i in np.flatnonzero(valid[:, 0]):
        delta = commands[i] - commands[i, 0]
        scale = np.minimum(1., radius / np.maximum(np.linalg.norm(delta, axis=-1), 1e-12))
        output[i] = commands[i, 0] + scale[:, None]*delta
        # Keep default command bit-identical for full-trajectory identity checks.
        output[i, 0] = commands[i, 0]
        features[i, :, :3] = output[i]
        for k in (3, 4):
            features[i, :, k] = candidates[i, 0, k] + scale*(candidates[i, :, k]-candidates[i, 0, k])
        features[i, :, 5] = np.linalg.norm(output[i], axis=-1)
    return output, features


def reference_distribution(state, prior):
    return Categorical(logits=(prior*state['candidates'][..., 8]).masked_fill(~state['valid'], -1e9))


def measured_anchor_weight(state):
    """More conservative at stale or ambiguous measured geometry; no truth."""
    age = (state['nodes'][..., 21]/.3).clamp(0., 1.)
    ambiguous = state['nodes'][..., 20].clamp(0., 1.)
    return 1. + 2.*torch.maximum(age, ambiguous)


def anchor_penalty(dist, state, prior):
    reference = reference_distribution(state, prior)
    # Invalid categories have zero probability under both distributions.
    kl = (dist.probs*(dist.logits-reference.logits)).sum(-1)
    return measured_anchor_weight(state)*kl
