"""EXP0056 process-local adapter: frozen earlier experiments remain untouched."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import scripts.run_measured_marl as base
from scripts.tpg_episode import TPGEpisode
from marl.measured_marl import temporal_gae
from marl.conservative_residual import ResidualMARL, VARIANTS, bounded_candidates, anchor_penalty
from scripts.run_safety_marl import independent_gradient_clip

ROOT = base.ROOT
PROTOCOL = ROOT/'configs/experiments/EXP_0056_CONSERVATIVE_RESIDUAL.json'
old_hashes = base.source_hashes
old_make_episode = base.make_episode


def protocol():
    p = json.loads((ROOT/'configs/experiments/EXP_0054_MEASURED_MARL.json').read_text())
    p.update(json.loads(PROTOCOL.read_text()))
    return p


def source_hashes():
    result = old_hashes()
    for name in ('marl/conservative_residual.py', 'scripts/run_residual_marl.py',
                 'scripts/run_residual_marl_study.py', 'scripts/analyze_residual_marl.py',
                 'scripts/run_safety_marl.py', 'scripts/audit_marl_outcomes.py',
                 'configs/experiments/EXP_0056_CONSERVATIVE_RESIDUAL.json'):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


class ResidualEpisode(TPGEpisode):
    def build_low(self):
        super().build_low()
        self.low_commands, self.low_candidates = bounded_candidates(
            self.low_commands, self.low_candidates, self.low_valid, self.residual_radius)

    def step_control(self, actions):
        choices = np.asarray(actions, int)
        preferred = self.conventional_low()
        delta = np.linalg.norm(self.low_commands[np.arange(self.cfg.clusters), choices] -
                               self.low_commands[np.arange(self.cfg.clusters), preferred], axis=-1)
        active = self.scene.active
        self.residual_sum += float(delta[active].sum())
        self.residual_choices += int(active.sum())
        self.residual_max = max(self.residual_max, float(delta[active].max()) if active.any() else 0.)
        self.moving_residual_max = max(self.moving_residual_max,
            float(delta[active & self.low_valid[:, 0]].max()) if (active & self.low_valid[:, 0]).any() else 0.)
        return super().step_control(choices)

    def result(self, policy):
        row = super().result(policy)
        row.update(residual_radius=self.residual_radius,
            mean_requested_residual=self.residual_sum/max(self.residual_choices, 1),
            max_requested_residual=self.residual_max, max_moving_residual=self.moving_residual_max,
            low_action_semantics='When nominal admissible all alternatives are bounded residuals; otherwise original hold/escape',
            low_auxiliary_label=None, velocity_prediction_used=False,
            nominal_low_controller='frozen measured-memory controller; bounded residual alternatives')
        return row

    def __init__(self, *args, **kwargs):
        p = protocol()
        self.residual_radius = p['residual_radius']
        self.residual_sum = self.residual_max = self.moving_residual_max = 0.
        self.residual_choices = 0
        super().__init__(*args, **kwargs)
        self.protocol = {**self.protocol, 'reward':{**self.protocol['reward'], **p['reward_override']}}


def make_episode(*args, **kwargs):
    # Retain the frozen reset guard, changing only the episode implementation.
    with patch('scripts.run_tpg_learning.TPGEpisode', ResidualEpisode):
        return old_make_episode(*args, **kwargs)


def update(net, optimizer, records, level, bootstrap, p):
    if not records:
        return dict(samples=0)
    states = {k:torch.from_numpy(np.asarray([r[k] for r in records])) for k in base.STATE_KEYS}
    actions = torch.from_numpy(np.asarray([r['action'] for r in records]))
    oldlog = torch.from_numpy(np.asarray([r['logp'] for r in records], np.float32))
    adv, ret = temporal_gae(records, bootstrap, p['gamma_per_control_step'], p['gae_lambda_per_control_step'])
    adv, ret = torch.from_numpy(adv), torch.from_numpy(ret)
    joint = level == 'high' and net.joint_high
    mask = states['active'].any(-1) if joint else states['active']
    if not mask.any():
        return dict(samples=len(records), active_samples=0, skipped=True)
    selected = adv[mask]
    adv = (adv-selected.mean())/(selected.std(unbiased=False)+1e-8)
    anchored = level == 'low' and net.variant.endswith('anchor')
    logs = []
    for epoch in range(p['epochs']):
        epoch_kls = []
        for indices in torch.randperm(len(records)).split(p['minibatch_size']):
            valid = mask[indices]
            if not valid.any():
                continue
            state = {k:v[indices] for k,v in states.items()}
            dist, value = net(level, state)
            delta = dist.log_prob(actions[indices])-oldlog[indices]
            ratio = delta.exp()
            gain = torch.minimum(ratio*adv[indices], ratio.clamp(1-p['clip_coef'],1+p['clip_coef'])*adv[indices])
            entropy = dist.entropy()[valid].mean()
            anchor = anchor_penalty(dist, state, p['prior_logit_bias'])[valid].mean() if anchored else torch.zeros(())
            loss = (-gain[valid].mean() + p['value_coef']*(value-ret[indices]).square()[valid].mean()
                    - p['entropy_coef']*entropy + p['anchor_coef']*anchor)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite residual loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            an, cn = independent_gradient_clip(net, level)
            optimizer.step()
            kl = float(((ratio-1)-delta)[valid].mean().detach())
            epoch_kls.append(kl)
            logs.append([float(loss.detach()), kl, float(entropy.detach()), float(anchor.detach()), an, cn])
        if epoch_kls and np.mean(epoch_kls) > p['target_kl']:
            break
    result = dict(zip(('loss','kl','entropy','anchor','actor_unclipped_grad_norm','critic_unclipped_grad_norm'),
                      np.mean(logs, axis=0).tolist()))
    return dict(samples=len(records), active_samples=int(mask.sum()), anchored=anchored, **result)


def configure():
    class Reader:
        def read_text(self):
            return json.dumps(protocol())
    base.PROTOCOL = Reader()
    base.MeasuredMARL = ResidualMARL
    base.make_episode = make_episode
    base.source_hashes = source_hashes
    base.update = update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('train','evaluate'):
        p = sub.add_parser(mode)
        p.add_argument('--variant', choices=VARIANTS, default='r_mappo')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--out', type=Path, required=True)
        if mode == 'train':
            p.add_argument('--steps', type=int, required=True)
            p.add_argument('--scene-base', type=int, required=True)
        else:
            p.add_argument('--scene', type=int, required=True)
            p.add_argument('--duration', type=float, default=180.)
            p.add_argument('--checkpoint', type=Path)
            p.add_argument('--rule', action='store_true')
    args = parser.parse_args()
    configure()
    base.train(args) if args.mode == 'train' else base.evaluate(args)


if __name__ == '__main__':
    main()
