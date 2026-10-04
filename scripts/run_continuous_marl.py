"""EXP0058 process-local adapter: frozen earlier experiments remain untouched."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import scripts.run_measured_marl as base
from scripts.tpg_episode import TPGEpisode
from marl.measured_marl import temporal_gae
from marl.continuous_residual import ContinuousMARL, residual_command, continuous_anchor
from marl.conservative_residual import VARIANTS
from scripts.run_residual_marl import ResidualEpisode
from scripts.run_clean_residual_eval import enforce_runtime, source_hashes as clean_hashes
from scripts.run_safety_marl import independent_gradient_clip

ROOT = base.ROOT
PROTOCOL = ROOT/'configs/experiments/EXP_0058_CONTINUOUS_RESIDUAL.json'
old_hashes = clean_hashes
old_make_episode = base.make_episode


def protocol():
    p = json.loads((ROOT/'configs/experiments/EXP_0054_MEASURED_MARL.json').read_text())
    p.update(json.loads((ROOT/'configs/experiments/EXP_0056_CONSERVATIVE_RESIDUAL.json').read_text()))
    p.update(json.loads(PROTOCOL.read_text()))
    return p


def source_hashes():
    result = old_hashes()
    for name in ('marl/continuous_residual.py', 'scripts/run_continuous_marl.py',
                 'scripts/run_continuous_marl_study.py', 'scripts/analyze_continuous_marl.py',
                 'configs/experiments/EXP_0058_CONTINUOUS_RESIDUAL.json'):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


class ContinuousEpisode(ResidualEpisode):
    def __init__(self,*args,**kwargs):
        p=protocol()
        self.residual_radius=p['residual_radius']
        self.residual_sum=self.residual_max=self.moving_residual_max=0.
        self.residual_choices=self.nonzero_residual_choices=0
        TPGEpisode.__init__(self,*args,**kwargs)
        self.protocol={**self.protocol,'reward':{**self.protocol['reward'],**p['reward_override']}}

    def build_low(self):
        TPGEpisode.build_low(self)

    def step_control(self,actions):
        raw=np.asarray(actions)
        if raw.shape==(self.cfg.clusters,):
            return ResidualEpisode.step_control(self,raw)
        if raw.shape!=(self.cfg.clusters,3):raise ValueError('Invalid continuous action shape')
        preferred=self.conventional_low()
        nominal=self.low_commands[np.arange(self.cfg.clusters),preferred].copy()
        enabled=self.scene.active & self.low_valid[:,0]
        command=residual_command(nominal,raw,enabled,self.residual_radius)
        delta=np.linalg.norm(command-nominal,axis=-1)
        active=self.scene.active
        self.residual_sum+=float(delta[active].sum());self.residual_choices+=int(active.sum())
        self.nonzero_residual_choices+=int(np.sum((delta>1e-6)&active))
        maximum=float(delta[active].max()) if active.any() else 0.
        self.residual_max=max(self.residual_max,maximum)
        self.moving_residual_max=max(self.moving_residual_max,maximum)
        self.low_commands[np.arange(self.cfg.clusters),preferred]=command
        return TPGEpisode.step_control(self,preferred)

    def result(self,policy):
        row=super().result(policy)
        row.update(low_action_semantics='Continuous tanh residual when nominal forward admissible; exact conventional hold/retreat otherwise',
                   continuous_nonzero_residual_choices=self.nonzero_residual_choices)
        return row


def make_episode(*args, **kwargs):
    # Retain the frozen reset guard, changing only the episode implementation.
    with patch('scripts.run_tpg_learning.TPGEpisode', ContinuousEpisode):
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
    value_mask = states['active'].any(-1) if joint else states['active']
    mask = value_mask if level=='high' else value_mask & states['valid'][...,0]
    if not value_mask.any():
        return dict(samples=len(records), active_samples=0, skipped=True)
    selected = adv[mask]
    adv = (adv-selected.mean())/(selected.std(unbiased=False)+1e-8) if mask.any() else torch.zeros_like(adv)
    anchored = level == 'low' and net.variant.endswith('anchor')
    logs = []
    for epoch in range(p['epochs']):
        epoch_kls = []
        for indices in torch.randperm(len(records)).split(p['minibatch_size']):
            valid = mask[indices]
            value_valid=value_mask[indices]
            if not value_valid.any():
                continue
            state = {k:v[indices] for k,v in states.items()}
            dist, value = net(level, state)
            delta = dist.log_prob(actions[indices])-oldlog[indices]
            ratio = delta.exp()
            gain = torch.minimum(ratio*adv[indices], ratio.clamp(1-p['clip_coef'],1+p['clip_coef'])*adv[indices])
            entropy = dist.entropy()[valid].mean() if valid.any() else torch.zeros(())
            anchor = continuous_anchor(dist, state)[valid].mean() if anchored and valid.any() else torch.zeros(())
            actor_loss=-gain[valid].mean() if valid.any() else torch.zeros(())
            entropy_coef=p['low_entropy_coef'] if level=='low' else p['entropy_coef']
            loss = (actor_loss + p['value_coef']*(value-ret[indices]).square()[value_valid].mean()
                    - entropy_coef*entropy + p['anchor_coef']*anchor)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite residual loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            an, cn = independent_gradient_clip(net, level)
            optimizer.step()
            kl = float(((ratio-1)-delta)[valid].mean().detach()) if valid.any() else 0.
            epoch_kls.append(kl)
            logs.append([float(loss.detach()), kl, float(entropy.detach()), float(anchor.detach()), an, cn])
        if epoch_kls and np.mean(epoch_kls) > p['target_kl']:
            break
    result = dict(zip(('loss','kl','entropy','anchor','actor_unclipped_grad_norm','critic_unclipped_grad_norm'),
                      np.mean(logs, axis=0).tolist()))
    return dict(samples=len(records), active_samples=int(mask.sum()), anchored=anchored, **result)


def act(net,level,state,greedy=False):
    with torch.no_grad():
        dist,value=net(level,base.tensors(state))
        action=(dist.mean if level=='low' else dist.logits.argmax(-1)) if greedy else dist.sample()
        return action[0].numpy(),dist.log_prob(action)[0].numpy(),value[0].numpy()


def evaluate(args):
    runtime=enforce_runtime();p=protocol();source=source_hashes()
    torch.set_num_threads(1);torch.manual_seed(42)
    net=ContinuousMARL(args.variant,p['hidden_dim'],p['clusters'],p['prior_logit_bias'])
    payload=path=digest=None
    if args.checkpoint:
        path=args.checkpoint
        payload=torch.load(path,map_location='cpu',weights_only=False)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if payload['source_hashes']!=source or payload['runtime']!=runtime or payload['variant']!=args.variant:
            raise ValueError('Continuous checkpoint provenance mismatch')
        net.load_state_dict(payload['model'],strict=True)
        if not all(torch.equal(v,payload['model'][k]) for k,v in net.state_dict().items()):
            raise ValueError('Weight values changed during transfer')
    net.eval();args.out.parent.mkdir(parents=True,exist_ok=True)
    ep=make_episode(args.scene,duration=p['duration_s'])
    high_calls=low_calls=high_nonrule=low_nonrule=active_choices=0;prior_probability=0.
    started=time.monotonic()
    with args.out.open('x') as stream:
        try:
            while not ep.done:
                if ep.needs_decision:
                    conventional=ep.conventional_priority()
                    if args.rule:priority=conventional
                    else:
                        state=base.observation(ep);action,_,_=act(net,'high',state,True)
                        priority=base.priority_action(net,action,state);high_calls+=1
                        high_nonrule+=priority!=conventional
                    ep.choose_priority(priority)
                conventional=ep.conventional_low()
                if args.rule:choices=conventional
                else:
                    state=base.observation(ep)
                    choices,_,_=act(net,'low',state,True)
                    active_choices+=int(state['active'].sum())
                    low_calls+=1
                ep.step_control(choices)
            row=ep.result('tpg' if args.rule else args.variant)
            low_nonrule=ep.nonzero_residual_choices
            row.update(experiment=p['experiment'],variant='tpg' if args.rule else args.variant,
                training_seed=payload['training_seed'] if payload else None,
                training_steps=payload['transitions'] if payload else 0,
                checkpoint_sha256=digest,parent_checkpoint=str(path) if path else None,
                training_runtime=payload['runtime'] if payload else None,
                continuous_control=True,
                source_hashes=source,runtime=runtime,high_network_calls=high_calls,low_network_calls=low_calls,
                high_nonrule_choices=high_nonrule,low_nonrule_choices=low_nonrule,active_low_choices=active_choices,
                mean_rule_probability=None,wall_clock_s=time.monotonic()-started,
                baseline_scope=p['scope'],confirmation_accessed=False,new_training_steps=payload['transitions'] if payload else 0)
            if source!=source_hashes():raise RuntimeError('Source changed during evaluation')
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps({k:row[k] for k in ('variant','scene_seed','removal','cluster_safe_success')}),flush=True)
        finally:ep.close()


def configure():
    class Reader:
        def read_text(self):
            return json.dumps(protocol())
    base.PROTOCOL = Reader()
    base.MeasuredMARL = ContinuousMARL
    base.enforce_runtime = enforce_runtime
    base.act = act
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
    base.train(args) if args.mode == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
