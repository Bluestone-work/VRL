"""EXP0057: frozen EXP0056 policies deployed in a clean CPU runtime.

No retraining, no replacement of an EXP0056 row, no cross-runtime pooling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from marl.conservative_residual import ResidualMARL, VARIANTS
from scripts.run_residual_marl import ROOT, make_episode, protocol as parent_protocol, source_hashes as parent_hashes
from scripts.run_measured_marl import observation, tensors, act, priority_action, runtime_context

PROTOCOL=ROOT/'configs/experiments/EXP_0057_CLEAN_RUNTIME_EVALUATION.json'


def protocol():
    return {**parent_protocol(), **json.loads(PROTOCOL.read_text())}


def source_hashes():
    result=parent_hashes()
    for name in ('scripts/run_clean_residual_eval.py','scripts/run_clean_residual_study.py',
                 'scripts/analyze_clean_residual.py','configs/experiments/EXP_0057_CLEAN_RUNTIME_EVALUATION.json'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def enforce_runtime():
    p=protocol();context=runtime_context()
    if Path(sys.prefix).resolve()!=Path(p['clean_prefix']).resolve() or sys.version_info[:2]!=(3,12):
        raise ValueError('Requires the declared isolated system-Python runtime')
    if context['numpy']!='1.26.4' or context['torch']!='2.3.0+cpu':
        raise ValueError('Wrong NumPy/PyTorch versions')
    import scipy
    if scipy.__version__!='1.11.4':raise ValueError('Wrong SciPy version')
    context.update(python=sys.version,scipy=scipy.__version__,prefix=sys.prefix,
        include_system_site_packages=False,numpy_file=np.__file__,torch_file=torch.__file__,scipy_file=scipy.__file__)
    context['python_binary_sha256']=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
    context['torch_binary_sha256']=hashlib.sha256(Path(torch._C.__file__).read_bytes()).hexdigest()
    if any(not Path(file).resolve().is_relative_to(Path(sys.prefix).resolve())
           for file in (np.__file__,torch.__file__,scipy.__file__)):
        raise ValueError('Mixed external packages')
    return context


def checked_checkpoint(variant,seed):
    p=protocol();root=ROOT/p['parent_study']
    manifest=json.loads((root/'manifest.json').read_text())
    audit=json.loads((root/'AUDIT.json').read_text())
    if not all(audit['checks'][k] for k in ('source_matches','snapshot_matches','checkpoint_lineage','regression')):
        raise ValueError('Parent TRAINING provenance not admitted')
    folder=root/'training'/f'train_{variant}_{seed}'
    status=json.loads((folder/'status.json').read_text())
    path=folder/f"policy_{p['training_steps']}.pt"
    item=next(x for x in status['checkpoints'] if x['steps']==p['training_steps'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if status['phase']!='completed' or digest!=item['sha256']:
        raise ValueError('Parent checkpoint file mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload['source_hashes']!=parent_hashes() or payload['source_hashes']!=manifest['source_hashes']
        or payload['runtime']!=manifest['runtime'] or payload['variant']!=variant
        or payload['training_seed']!=seed or payload['transitions']!=p['training_steps']):
        raise ValueError('Parent identity mismatch')
    return payload,path,digest


def evaluate(args):
    runtime=enforce_runtime();p=protocol();source=source_hashes()
    torch.set_num_threads(1);torch.manual_seed(42)
    net=ResidualMARL(args.variant,p['hidden_dim'],p['clusters'],p['prior_logit_bias'])
    payload=path=digest=None
    if args.trained:
        payload,path,digest=checked_checkpoint(args.variant,args.seed)
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
                        state=observation(ep);action,_,_=act(net,'high',state,True)
                        priority=priority_action(net,action,state);high_calls+=1
                        high_nonrule+=priority!=conventional
                    ep.choose_priority(priority)
                conventional=ep.conventional_low()
                if args.rule:choices=conventional
                else:
                    state=observation(ep)
                    with torch.no_grad():dist,_=net('low',tensors(state))
                    choices=dist.logits.argmax(-1)[0].numpy();probabilities=dist.probs[0].numpy()
                    active=state['active'];low_nonrule+=int(np.sum((choices!=conventional)&active))
                    active_choices+=int(active.sum())
                    prior_probability+=float(probabilities[np.arange(p['clusters']),conventional][active].sum())
                    low_calls+=1
                ep.step_control(choices)
            row=ep.result('tpg' if args.rule else args.variant)
            row.update(experiment=p['experiment'],variant='tpg' if args.rule else args.variant,
                training_seed=payload['training_seed'] if payload else None,
                training_steps=payload['transitions'] if payload else 0,
                checkpoint_sha256=digest,parent_checkpoint=str(path) if path else None,
                training_runtime=payload['runtime'] if payload else None,
                weight_transfer_exact=True if payload else None,parent_matrix_failed=True,
                source_hashes=source,runtime=runtime,high_network_calls=high_calls,low_network_calls=low_calls,
                high_nonrule_choices=high_nonrule,low_nonrule_choices=low_nonrule,active_low_choices=active_choices,
                mean_rule_probability=prior_probability/max(active_choices,1),wall_clock_s=time.monotonic()-started,
                baseline_scope=p['scope'],confirmation_accessed=False,new_training_steps=0)
            if source!=source_hashes():raise RuntimeError('Source changed during evaluation')
            stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps({k:row[k] for k in ('variant','scene_seed','removal','cluster_safe_success')}),flush=True)
        finally:ep.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,default='r_mappo')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--scene',type=int,required=True)
    parser.add_argument('--trained',action='store_true')
    parser.add_argument('--rule',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.trained and args.rule:raise ValueError('Rule cannot load a checkpoint')
    evaluate(args)
