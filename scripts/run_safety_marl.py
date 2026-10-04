"""Explicit process-local EXP0055 adapter; frozen EXP0054 files are never edited.

Reuses the admitted trainer with a protocol reader, episode reward override,
and checked parent initialization. Optimizer state is intentionally restarted
in BOTH reward regimes, so this is matched fine-tuning, not exact resume.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import scripts.run_measured_marl as base
from marl.measured_marl import temporal_gae

ROOT=base.ROOT
PROTOCOL=ROOT/'configs/experiments/EXP_0055_SAFETY_OBJECTIVE.json'


def hashes():
    result=base_source_hashes()
    for name in ('configs/experiments/EXP_0055_SAFETY_OBJECTIVE.json',
                 'scripts/run_safety_marl.py','scripts/run_safety_marl_study.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


base_source_hashes=base.source_hashes
BaseNetwork=base.MeasuredMARL
base_episode=base.make_episode
base_atomic_json=base.atomic_json


def independent_gradient_clip(net,level):
    actor=torch.nn.utils.clip_grad_norm_(getattr(net,level+'_actor').parameters(),.5)
    critic=torch.nn.utils.clip_grad_norm_(getattr(net,level+'_critic').parameters(),.5)
    if not torch.isfinite(actor) or not torch.isfinite(critic):
        raise FloatingPointError('Nonfinite separately clipped gradient')
    return float(actor),float(critic)


def update(net,optimizer,records,level,bootstrap,p):
    if not records:return dict(samples=0)
    states={k:torch.from_numpy(np.asarray([r[k] for r in records])) for k in base.STATE_KEYS}
    actions=torch.from_numpy(np.asarray([r['action'] for r in records]))
    oldlog=torch.from_numpy(np.asarray([r['logp'] for r in records],np.float32))
    adv,ret=temporal_gae(records,bootstrap,p['gamma_per_control_step'],p['gae_lambda_per_control_step'])
    adv,ret=torch.from_numpy(adv),torch.from_numpy(ret)
    joint=level=='high' and net.joint_high
    mask=states['active'].any(-1) if joint else states['active']
    if not mask.any():return dict(samples=len(records),active_samples=0,skipped=True)
    selected=adv[mask];adv=(adv-selected.mean())/(selected.std(unbiased=False)+1e-8)
    aux_enabled=level=='low' and net.variant=='vctpg_ac'
    if aux_enabled:
        commands=torch.from_numpy(np.asarray([r['commands'] for r in records],np.float32))
        labels=torch.from_numpy(np.asarray([r['measured_velocity'] for r in records],np.float32))
        fresh=torch.from_numpy(np.asarray([r['measurement_valid'] for r in records],bool))
    losses=[];kls=[];auxs=[];entropies=[];actor_norms=[];critic_norms=[]
    for epoch in range(p['epochs']):
        epoch_kl=[]
        for indices in torch.randperm(len(records)).split(p['minibatch_size']):
            valid=mask[indices]
            if not valid.any():continue
            state={k:v[indices] for k,v in states.items()}
            dist,value=net(level,state);logp=dist.log_prob(actions[indices]);delta=logp-oldlog[indices]
            ratio=delta.exp()
            gain=torch.minimum(ratio*adv[indices],ratio.clamp(1-p['clip_coef'],1+p['clip_coef'])*adv[indices])
            entropy=dist.entropy()[valid].mean()
            loss=-gain[valid].mean()+p['value_coef']*(value-ret[indices]).square()[valid].mean()-p['entropy_coef']*entropy
            auxiliary=torch.zeros(())
            if aux_enabled and fresh[indices].any():
                prediction=net.low_actor.predict(state,commands[indices])
                auxiliary=torch.nn.functional.smooth_l1_loss(prediction[fresh[indices]],labels[indices][fresh[indices]])
                loss=loss+p['auxiliary_weight']*auxiliary
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite safety-MARL loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            actor_norm,critic_norm=independent_gradient_clip(net,level)
            optimizer.step()
            kl=float(((ratio-1)-delta)[valid].mean().detach())
            losses.append(float(loss.detach()));kls.append(kl);epoch_kl.append(kl)
            auxs.append(float(auxiliary.detach()));entropies.append(float(entropy.detach()))
            actor_norms.append(actor_norm);critic_norms.append(critic_norm)
        if epoch_kl and np.mean(epoch_kl)>p['target_kl']:break
    return dict(samples=len(records),active_samples=int(mask.sum()),loss=float(np.mean(losses)),
        kl=float(np.mean(kls)),auxiliary=float(np.mean(auxs)),entropy=float(np.mean(entropies)),
        actor_unclipped_grad_norm=float(np.mean(actor_norms)),critic_unclipped_grad_norm=float(np.mean(critic_norms)),
        gradient_clipping='actor_and_critic_separately_at_0.5')


def effective_protocol(regime):
    p=json.loads((ROOT/'configs/experiments/EXP_0054_MEASURED_MARL.json').read_text())
    follow=json.loads(PROTOCOL.read_text())
    p.update(follow)
    p['reward_regime']=regime
    p['checkpoints']=follow['checkpoints']
    return p


def parent_path(p,variant,seed):
    return ROOT/p['parent_study']/'training'/f'train_{variant}_{seed}'/f"policy_{p['parent_steps']}.pt"


def checked_parent(p,variant,seed):
    root=ROOT/p['parent_study']
    audit=json.loads((root/'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:
        raise ValueError('Parent matrix not admitted')
    path=parent_path(p,variant,seed)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['source_hashes']!=base_source_hashes() or payload['variant']!=variant or payload['training_seed']!=seed:
        raise ValueError('Parent source or identity mismatch')
    if payload['transitions']!=p['parent_steps'] or payload['runtime']!=base.enforce_runtime():
        raise ValueError('Parent budget/runtime mismatch')
    recorded=json.loads((path.parent/'status.json').read_text())
    entry=next(x for x in recorded['checkpoints'] if x['steps']==p['parent_steps'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=entry['sha256']:
        raise ValueError('Parent checkpoint digest mismatch')
    return payload,dict(parent_checkpoint=str(path),parent_checkpoint_sha256=digest,
        parent_control_steps=p['parent_steps'],fresh_optimizer=True,exact_resume=False)


def configure(args):
    p=effective_protocol(args.regime)
    parent=None;provenance={}
    if args.mode=='train' or args.parent_policy:
        parent,provenance=checked_parent(p,args.variant,args.seed)
    elif args.checkpoint:
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if payload['protocol']['reward_regime']!=args.regime:
            raise ValueError('Wrong reward-regime checkpoint')
        _,provenance=checked_parent(p,args.variant,payload['training_seed'])

    class ProtocolReader:
        def read_text(self):return json.dumps(p)

    def network(*positional,**keywords):
        model=BaseNetwork(*positional,**keywords)
        if parent is not None:
            model.load_state_dict(parent['model'],strict=True)
        return model

    def episode(*positional,**keywords):
        ep=base_episode(*positional,**keywords)
        if args.regime=='safety_reward':
            ep.protocol={**ep.protocol,'reward':{**ep.protocol['reward'],**p['safety_reward_override']}}
        return ep

    def atomic(path,value):
        if Path(path).name=='manifest.json':
            value={**value,**provenance,'reward_regime':args.regime,
                'cumulative_requested_controls':p['parent_steps']+args.steps}
        return base_atomic_json(path,value)

    # Dependency injection is confined to this dedicated child process.
    base.PROTOCOL=ProtocolReader()
    base.MeasuredMARL=network
    base.make_episode=episode
    base.source_hashes=hashes
    base.atomic_json=atomic
    base.update=update
    return p,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    for name in ('train','evaluate'):
        p=sub.add_parser(name)
        p.add_argument('--regime',choices=('original_reward','safety_reward'),required=True)
        p.add_argument('--variant',choices=('r_mappo','r_ippo','vctpg_ac','vctpg_no_ac'),required=True)
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--out',type=Path,required=True)
        if name=='train':
            p.add_argument('--steps',type=int,required=True)
            p.add_argument('--scene-base',type=int,required=True)
        else:
            p.add_argument('--scene',type=int,required=True)
            p.add_argument('--duration',type=float,default=180.)
            p.add_argument('--checkpoint',type=Path)
            p.add_argument('--parent-policy',action='store_true')
    args=parser.parse_args()
    if args.mode=='train':
        args.parent_policy=False;args.checkpoint=None
    else:
        args.rule=False
        if bool(args.checkpoint)==bool(args.parent_policy):
            raise ValueError('Choose exactly one checkpoint or parent policy')
    p,provenance=configure(args)
    if args.mode=='train':
        base.train(args)
    else:
        base.evaluate(args)
        row=json.loads(args.out.read_text())
        row.update(provenance,reward_regime=args.regime,additional_training_steps=row['training_steps'],
            cumulative_training_steps=p['parent_steps']+row['training_steps'],parent_policy_diagnostic=args.parent_policy)
        if args.parent_policy:
            row['training_seed']=args.seed
            row['checkpoint_sha256']=provenance['parent_checkpoint_sha256']
        # Add declared lineage fields atomically; no numerical outcome changes.
        temporary=args.out.with_suffix('.tmp')
        temporary.write_text(json.dumps(row,allow_nan=False)+'\n')
        temporary.replace(args.out)


if __name__=='__main__':main()
