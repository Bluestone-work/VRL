"""Explicit experiment-only exploration constraints for physical MAPPO.

Project log standard deviations BEFORE collecting a fresh rollout. They are
unchanged by this helper throughout collection and PPO updates, so stored and
recomputed action densities describe the same policy. Checkpoints store the
projected parameters; deterministic inference and the environment are unchanged.
"""
import math

import torch


def exploration_cap(schedule, transitions):
    if schedule is None:
        return None
    expected = {'kind', 'start_std', 'end_std', 'anneal_transitions'}
    if not isinstance(schedule, dict) or set(schedule) != expected:
        raise ValueError('Exploration schedule requires exactly kind/start_std/end_std/anneal_transitions')
    if schedule['kind'] != 'log_std_cap_v1':
        raise ValueError('Unknown exploration schedule')
    for key in ('start_std', 'end_std'):
        value = schedule[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Exploration standard deviations must be finite numbers')
        if not math.exp(-5) <= value <= math.exp(2):
            raise ValueError('Exploration cap outside GaussianActor supported standard deviations')
    duration = schedule['anneal_transitions']
    if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
        raise ValueError('Annealing duration must be a positive integer')
    if schedule['end_std'] > schedule['start_std']:
        raise ValueError('Exploration cap must not increase')
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
        raise ValueError('Transition count must be a nonnegative integer')
    fraction = min(transitions / duration, 1.)
    return math.exp((1-fraction)*math.log(schedule['start_std']) + fraction*math.log(schedule['end_std']))


def apply_exploration_schedule(agent, schedule, transitions):
    cap = exploration_cap(schedule, transitions)
    if cap is None:
        return None
    if len(agent.buffer):
        raise RuntimeError('Never change exploration with an outstanding PPO rollout')
    with torch.no_grad():
        agent.actor.policy_head.log_std.clamp_(max=math.log(cap))
    agent.meta['exploration_constraint'] = dict(schedule=schedule.copy(),
        applied_at_transitions=transitions, std_cap=cap,
        semantics='raw Gaussian std cap projected before rollout; deterministic mean unchanged')
    return cap
