"""Frozen-source pilot training and paired evaluation for EXP0053."""
import argparse
from datetime import datetime
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch

from marl.tpg_learning import TPGAgent
from marl.measured_options import smdp_gae
from scripts.tpg_episode import TPGEpisode,PROTOCOL,ROOT,tpg_hashes
from scripts.run_option_learning import atomic_json,check_development_seed


def make_episode(seed,**kwargs):
    from unittest.mock import patch
    from environments.mca_physical_env import MCAPhysicalEnv
    check_development_seed(seed)
    original=MCAPhysicalEnv.reset
    def reset(env,*,seed=None,options=None):
        if seed is not None:check_development_seed(seed)
        return original(env,seed=seed,options=options)
    with patch.object(MCAPhysicalEnv,'reset',reset):
        ep=TPGEpisode(seed,**kwargs)
    try:check_development_seed(ep.manifest['accepted_seed'])
    except BaseException:ep.close();raise
    return ep


def high_state(ep):
    return dict(nodes=ep.nodes.copy(),pairs=ep.pairs.copy())


def low_state(ep):
    return dict(history=ep.low_history.copy(),candidates=ep.low_candidates.copy(),
                valid=ep.low_valid.copy(),active=ep.scene.active.copy())


def tensor(value):
    return torch.from_numpy(np.asarray(value))


def high_forward(net,state):
    return net.high(tensor(state['nodes'])[None],tensor(state['pairs'])[None])


def low_forward(net,state):
    return net.low(*(tensor(state[k])[None] for k in ('history','candidates','valid','active')))


def ppo_update(net,optimizer,records,kind,bootstrap,protocol):
    if not records:return dict(samples=0)
    keys=('nodes','pairs') if kind=='high' else ('history','candidates','valid','active')
    inputs={k:tensor(np.asarray([r[k] for r in records])) for k in keys}
    actions=tensor(np.asarray([r['action'] for r in records]))
    oldlog=torch.tensor([r['logp'] for r in records],dtype=torch.float32)
    advantages,returns=smdp_gae([r['reward'] for r in records],[r['value'] for r in records],
        [r['done'] for r in records],[r['duration'] for r in records],bootstrap,
        protocol['gamma_per_control_step'],protocol['gae_lambda_per_control_step'])
    advantages=tensor(advantages);advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    returns=tensor(returns);losses=[];kls=[];auxs=[]
    if kind=='low':
        commands=tensor(np.asarray([r['commands'] for r in records]))
        labels=tensor(np.asarray([r['measured_velocity'] for r in records]))
        label_valid=tensor(np.asarray([r['measurement_valid'] for r in records],bool))
    module=getattr(net,kind)
    for epoch in range(protocol['epochs']):
        epoch_kls=[]
        for ii in torch.randperm(len(records)).split(protocol['minibatch_size']):
            dist,value=module(*(inputs[k][ii] for k in keys))
            logp=dist.log_prob(actions[ii]);entropy=dist.entropy()
            if kind=='low':logp=logp.sum(-1);entropy=entropy.sum(-1)
            ratio=torch.exp(logp-oldlog[ii])
            gain=torch.minimum(ratio*advantages[ii],ratio.clamp(1-protocol['clip_coef'],1+protocol['clip_coef'])*advantages[ii])
            loss=-gain.mean()+.25*(value-returns[ii]).square().mean()-protocol['entropy_coef']*entropy.mean()
            auxiliary=torch.zeros(())
            if kind=='low' and label_valid[ii].any():
                prediction=net.low.predict_measured_velocity(inputs['history'][ii],commands[ii])
                auxiliary=torch.nn.functional.smooth_l1_loss(prediction[label_valid[ii]],labels[ii][label_valid[ii]])
                loss=loss+protocol['auxiliary_weight']*auxiliary
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TPG learning loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(),.5);optimizer.step()
            kl=float(((ratio-1)-(logp-oldlog[ii])).mean().detach())
            losses.append(float(loss.detach()));kls.append(kl);epoch_kls.append(kl);auxs.append(float(auxiliary.detach()))
        if np.mean(epoch_kls)>protocol['target_kl']:break
    return dict(samples=len(records),loss=float(np.mean(losses)),kl=float(np.mean(kls)),aux=float(np.mean(auxs)))


def train(args):
    p=json.loads(PROTOCOL.read_text());source=tpg_hashes()
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=False)
    net=TPGAgent(p['hidden_dim'])
    initial={k:v.detach().clone() for k,v in net.state_dict().items()}
    optimizers={name:torch.optim.Adam(getattr(net,name).parameters(),lr=p['learning_rate']) for name in ('high','low')}
    learn_high=args.variant in ('high','both');learn_low=args.variant in ('low','both')
    atomic_json(args.out/'manifest.json',dict(experiment=p['experiment'],training_seed=args.seed,variant=args.variant,
        requested_control_steps=args.steps,scene_base=args.scene_base,source_hashes=source,protocol=p,
        confirmation_accessed=False,started_at=datetime.now().astimezone().isoformat()))
    steps=episodes=updates=0;ep=None;pending=None;started=time.monotonic()
    high_samples=low_samples=0
    with (args.out/'attempts.jsonl').open('x',buffering=1) as attempts, \
            (args.out/'episodes.jsonl').open('x',buffering=1) as episode_log, \
            (args.out/'updates.jsonl').open('x',buffering=1) as update_log:
        def reset():
            seed=args.scene_base+episodes
            attempts.write(json.dumps(dict(event='reset_requested',scene_seed=seed,steps=steps))+'\n')
            result=make_episode(seed,duration=p['duration_s'])
            attempts.write(json.dumps(dict(event='reset_completed',scene_seed=seed,
                accepted_seed=result.manifest['accepted_seed'],scenario_hash=result.manifest['scenario_hash']))+'\n')
            return result
        try:
            ep=reset()
            while steps<args.steps:
                high_records=[];low_records=[];collected=0
                while collected<p['rollout_control_steps'] and steps<args.steps:
                    if ep.needs_decision:
                        if pending is not None:
                            high_records.append(pending);pending=None
                        hs=high_state(ep)
                        if learn_high:
                            with torch.no_grad():
                                dist,value=high_forward(net,hs);action=dist.sample()
                                pending=dict(**hs,action=int(action[0]),logp=float(dist.log_prob(action)[0]),
                                    value=float(value[0]),reward=0.,duration=0,done=False)
                            priority=int(action[0])
                        else:priority=ep.conventional_priority()
                        ep.choose_priority(priority)
                    ls=low_state(ep)
                    if learn_low:
                        with torch.no_grad():
                            dist,value=low_forward(net,ls);action=dist.sample()
                            lr=dict(**ls,action=action[0].numpy(),logp=float(dist.log_prob(action).sum(-1)[0]),value=float(value[0]))
                        choices=action[0].numpy()
                    else:choices=ep.conventional_low()
                    reward,done,aux=ep.step_control(choices)
                    if pending is not None:
                        pending['reward']+=p['gamma_per_control_step']**pending['duration']*reward
                        pending['duration']+=1;pending['done']=done
                    if learn_low:low_records.append(dict(**lr,reward=reward,done=done,duration=1,**aux))
                    steps+=1;collected+=1
                    if done:
                        if pending is not None:high_records.append(pending);pending=None
                        row=ep.result('sampled_'+args.variant);row.update(training_seed=args.seed,training_steps=steps)
                        episode_log.write(json.dumps(row,allow_nan=False)+'\n')
                        attempts.write(json.dumps(dict(event='completed',scene_seed=ep.scene_seed,steps=steps))+'\n')
                        ep.close();episodes+=1
                        ep=reset() if steps<args.steps else None
                low_bootstrap=0.
                if learn_low and ep is not None:
                    with torch.no_grad():_,v=low_forward(net,low_state(ep));low_bootstrap=float(v[0])
                # An unfinished high option stays pending across rollouts. It is
                # never relabeled as a fresh decision merely to fill a batch.
                high_bootstrap=pending['value'] if pending is not None else 0.
                high_stats=ppo_update(net,optimizers['high'],high_records,'high',high_bootstrap,p) if learn_high else dict(samples=0)
                low_stats=ppo_update(net,optimizers['low'],low_records,'low',low_bootstrap,p) if learn_low else dict(samples=0)
                high_samples+=high_stats['samples'];low_samples+=low_stats['samples'];updates+=1
                status=dict(phase='training',steps=steps,episodes=episodes,updates=updates,
                    high=high_stats,low=low_stats,wall_s=time.monotonic()-started,
                    fps=steps/(time.monotonic()-started))
                atomic_json(args.out/'status.json',status);update_log.write(json.dumps(status)+'\n');print(json.dumps(status),flush=True)
            if source!=tpg_hashes():raise RuntimeError('Training source changed')
            deltas={name:float(sum((v.detach()-initial[k]).square().sum().item()
                for k,v in net.state_dict().items() if k.startswith(name+'.'))**.5) for name in ('high','low')}
            path=args.out/f'policy_{steps}.pt'
            torch.save(dict(model=net.state_dict(),source_hashes=source,protocol=p,variant=args.variant,
                training_seed=args.seed,transitions=steps,model_l2_change=deltas),path)
            if pending is not None:attempts.write(json.dumps(dict(event='censored_high_option',duration=pending['duration'],not_trained=True))+'\n')
            if ep is not None:attempts.write(json.dumps(dict(event='budget_cutoff',scene_seed=ep.scene_seed,not_completed_episode=True))+'\n')
            atomic_json(args.out/'status.json',dict(**{**status,'phase':'completed'},checkpoint=str(path),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_l2_change=deltas,
                high_samples=high_samples,low_samples=low_samples))
        except BaseException as exc:
            atomic_json(args.out/'python_failure.json',dict(error=repr(exc),steps=steps,episodes=episodes));raise
        finally:
            if ep is not None:ep.close()


def evaluate(args):
    p=json.loads(PROTOCOL.read_text());source=tpg_hashes()
    torch.set_num_threads(1);torch.manual_seed(42)
    net=TPGAgent(p['hidden_dim']);payload=None
    if args.policy=='learned':
        if args.checkpoint is None:raise ValueError('Checkpoint required')
        payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if payload['source_hashes']!=source or payload['variant']!=args.variant:raise ValueError('Checkpoint provenance mismatch')
        net.load_state_dict(payload['model'])
    net.eval();args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',buffering=1) as stream:
        ep=make_episode(args.scene,duration=args.duration,clusters=args.clusters,coupling=args.coupling)
        try:
            high_calls=low_calls=0
            while not ep.done:
                if ep.needs_decision:
                    if args.policy=='untrained' or (args.policy=='learned' and args.variant in ('high','both')):
                        with torch.no_grad():dist,_=high_forward(net,high_state(ep));priority=int(dist.logits.argmax(-1)[0])
                        high_calls+=1
                    else:priority=ep.conventional_priority()
                    ep.choose_priority(priority)
                if args.policy=='untrained' or (args.policy=='learned' and args.variant in ('low','both')):
                    with torch.no_grad():dist,_=low_forward(net,low_state(ep));choices=dist.logits.argmax(-1)[0].numpy()
                    low_calls+=1
                else:choices=ep.conventional_low()
                ep.step_control(choices)
            row=ep.result(args.policy)
            row.update(variant=args.variant,training_seed=payload['training_seed'] if payload else None,
                training_steps=payload['transitions'] if payload else 0,
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if payload else None,
                high_network_calls=high_calls,low_network_calls=low_calls)
            if source!=tpg_hashes():raise RuntimeError('Evaluation source changed')
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps({k:row[k] for k in ('scene_seed','removal','cluster_safe_success','spacing_violation_pair_s','scheduler_calls','wait_agent_s')}),flush=True)
        finally:ep.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    tr=sub.add_parser('train');tr.add_argument('--seed',type=int,required=True);tr.add_argument('--steps',type=int,required=True)
    tr.add_argument('--scene-base',type=int,required=True);tr.add_argument('--variant',choices=('high','low','both'),required=True)
    tr.add_argument('--out',type=Path,required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--scene',type=int,required=True);ev.add_argument('--duration',type=float,default=180.)
    ev.add_argument('--policy',choices=('tpg','untrained','learned'),required=True)
    ev.add_argument('--variant',choices=('high','low','both'),default='both');ev.add_argument('--checkpoint',type=Path)
    ev.add_argument('--clusters',type=int,default=3);ev.add_argument('--coupling',type=float,default=0.)
    ev.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();train(args) if args.mode=='train' else evaluate(args)


if __name__=='__main__':main()
