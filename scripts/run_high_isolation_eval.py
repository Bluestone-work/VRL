"""Deploy frozen EXP0058 actors in a high/low factorial on the clean runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.run_continuous_marl import ROOT, protocol as old_protocol, source_hashes as old_hashes
from scripts.run_continuous_marl import ContinuousMARL, make_episode, act
from scripts.run_measured_marl import observation, tensors, priority_action
from scripts.run_clean_residual_eval import enforce_runtime

PROTOCOL=ROOT/'configs/experiments/EXP_0059_HIGH_LAYER_ISOLATION.json'


def protocol():
    return {**old_protocol(), **json.loads(PROTOCOL.read_text())}


def source_hashes():
    result=old_hashes()
    for name in ('configs/experiments/EXP_0059_HIGH_LAYER_ISOLATION.json',
                 'scripts/run_high_isolation_eval.py','scripts/run_high_isolation_study.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def checked(path,variant,seed):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['variant']!=variant or payload['training_seed']!=seed or payload['transitions']!=16384:
        raise ValueError('Checkpoint identity mismatch')
    return payload


def evaluate(args):
    runtime=enforce_runtime();p=protocol();source=source_hashes()
    torch.set_num_threads(1);torch.manual_seed(42)
    high_payload=low_payload=None
    if args.high_variant!='rule':high_payload=checked(args.high_checkpoint,args.high_variant,args.seed)
    low_payload=checked(args.low_checkpoint,args.low_variant,args.seed)
    net=ContinuousMARL(args.high_variant if args.high_variant!='rule' else 'r_mappo',p['hidden_dim'],p['clusters'],p['prior_logit_bias'])
    if high_payload is not None:
        net.high_actor.load_state_dict({k[len('high_actor.'):]:v for k,v in high_payload['model'].items() if k.startswith('high_actor.')},strict=True)
    net.low_actor.load_state_dict({k[len('low_actor.'):]:v for k,v in low_payload['model'].items() if k.startswith('low_actor.')},strict=True)
    net.eval();args.out.parent.mkdir(parents=True,exist_ok=True)
    ep=make_episode(args.scene,duration=p['duration_s']);high_calls=high_nonrule=low_calls=0
    started=time.monotonic()
    with args.out.open('x') as stream:
        try:
            while not ep.done:
                if ep.needs_decision:
                    conventional=ep.conventional_priority()
                    if args.high_variant=='rule':priority=conventional
                    else:
                        state=observation(ep);action,_,_=act(net,'high',state,True)
                        priority=priority_action(net,action,state);high_calls+=1
                        high_nonrule+=priority!=conventional
                    ep.choose_priority(priority)
                if args.high_variant=='rule' and False:pass
                state=observation(ep);raw,_,_=act(net,'low',state,True)
                ep.step_control(raw);low_calls+=1
            label=f'{args.high_variant}_high__{args.low_variant}_low'
            row=ep.result(label)
            row.update(experiment=p['experiment'],variant=label,high_variant=args.high_variant,
                low_variant=args.low_variant,training_seed=args.seed,training_steps=16384,
                high_checkpoint_sha256=hashlib.sha256(args.high_checkpoint.read_bytes()).hexdigest() if high_payload else None,
                low_checkpoint_sha256=hashlib.sha256(args.low_checkpoint.read_bytes()).hexdigest(),
                high_checkpoint=str(args.high_checkpoint) if high_payload else None,low_checkpoint=str(args.low_checkpoint),
                parent_high_runtime=high_payload['runtime'] if high_payload else None,
                parent_low_runtime=low_payload['runtime'],source_hashes=source,runtime=runtime,
                high_network_calls=high_calls,high_nonrule_choices=high_nonrule,low_network_calls=low_calls,
                confirmation_accessed=False,post_training_intervention=True)
            stream.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps({k:row[k] for k in ('variant','scene_seed','removal','cluster_safe_success')}),flush=True)
        finally:ep.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--high-variant',choices=('rule','r_mappo','graph_ppo','graph_anchor'),required=True);ap.add_argument('--low-variant',choices=('r_mappo','graph_ppo'),required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--scene',type=int,required=True);ap.add_argument('--high-checkpoint',type=Path);ap.add_argument('--low-checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.high_variant!='rule' and not args.high_checkpoint:raise ValueError('High checkpoint required')
    evaluate(args)
