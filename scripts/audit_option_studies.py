"""Final read-only audit of completed EXP0051/52 artifacts and confirmation use."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    summaries=[];source_issues=[];seed_issues=[];feature_issues=[];all_training_steps=0
    pools=[]
    for p in (ROOT/'configs/experiments').glob('EXP_00*_*.json'):
        q=json.loads(p.read_text())
        if q.get('confirmation_scene_base') is not None and q.get('confirmation_scenes') is not None:
            pools.append((q['confirmation_scene_base'],q['confirmation_scene_base']+q['confirmation_scenes'],p.name))
    def check_seed(seed,where):
        for lo,hi,pool in pools:
            if lo<=seed<hi:seed_issues.append(dict(seed=seed,source=where,pool=pool))
    for code in ('EXP0051','EXP0052'):
        folder=ROOT/f'research/validation/{code}_STUDY_20261004'
        status=json.loads((folder/'status.json').read_text())
        if not status['phase'].startswith('completed'):
            raise SystemExit(f'{code} still in {status["phase"]}; final audit not written')
        manifest=json.loads((folder/'study_manifest.json').read_text())
        for name,h in manifest['source_hashes'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:source_issues.append(dict(study=code,file=name))
        runs=[]
        for task in manifest['tasks']:
            base=folder/'training'/task['label']
            train_status=json.loads((base/'status.json').read_text())
            steps=train_status['control_steps'];all_training_steps+=steps
            ckpt=base/f'policy_{steps}.pt'
            payload=torch.load(ckpt,map_location='cpu',weights_only=False)
            if payload['source_hashes']!=manifest['source_hashes']:source_issues.append(dict(study=code,checkpoint=str(ckpt)))
            if hashlib.sha256(ckpt.read_bytes()).hexdigest()!=train_status['checkpoint_sha256']:
                source_issues.append(dict(study=code,checkpoint_digest=str(ckpt)))
            for line in (base/'episode_attempts.jsonl').read_text().splitlines():
                event=json.loads(line)
                for field in ('scene_seed','accepted_seed'):
                    if field in event:check_seed(event[field],str(base))
            runs.append(dict(label=task['label'],control_steps=steps,
                model_l2_change=payload['model_l2_change'],actor_head_weight_norm=float(payload['model']['actor.2.weight'].norm())))
        study=dict(experiment=code,training_runs=runs,phases={})
        for phase in ('validation','coupling_stress'):
            directory=folder/phase
            a=json.loads((directory/'analysis.json').read_text())
            rows=[json.loads(l) for l in (directory/'attempts.jsonl').read_text().splitlines()]
            groups=defaultdict(list)
            for row in rows:
                check_seed(row['scene_seed'],str(directory))
                if row['status']!='completed':continue
                check_seed(row['reset_info']['accepted_seed'],str(directory))
                for rejection in row['reset_info']['rejected_attempts']:
                    # Reserved seeds explicitly rejected before reset are not accesses.
                    if rejection.get('reason')!='reserved_seed_excluded_before_reset':
                        check_seed(rejection['seed'],str(directory))
                for flag in ('exact_mass_input','true_velocity_input','true_frenet_input','safety_certificate','sealed_test_used'):
                    if row.get(flag) is not False:feature_issues.append(dict(study=code,phase=phase,arm=row['arm'],flag=flag))
                if row.get('completion_uses_observed_confirmation') is not True:
                    feature_issues.append(dict(study=code,phase=phase,arm=row['arm'],flag='observed_completion'))
                if row['source_hashes']!=manifest['source_hashes']:
                    source_issues.append(dict(study=code,phase=phase,arm=row['arm'],row_source_mismatch=True))
                groups[row['arm']].append(row)
            arms={}
            for label,data in groups.items():
                horizon=180.
                arms[label]=dict(completed=len(data),safe_successes=sum(r['cluster_safe_success'] for r in data),
                    spacing_violating_episodes=sum(not r['spacing_compliant'] for r in data),
                    false_visual_completions=sum(r['false_visual_completion'] for r in data),
                    removal=float(np.mean([r['removal'] for r in data])),
                    auc=float(np.mean([r['removal_auc_180'] for r in data])),
                    wall_cluster_s=float(np.mean([r['wall_contact_s'] for r in data])),
                    spacing_pair_s=float(np.mean([r['spacing_violation_pair_s'] for r in data])),
                    worst_minimum_spacing_mm=min((r['minimum_spacing_mm'] for r in data if r['minimum_spacing_mm'] is not None),default=None),
                    T90_horizon_penalty_s=float(np.mean([r['time_to_removal_s']['90'] if r['time_to_removal_s']['90'] is not None else horizon for r in data])))
            equivalence={}
            for variant in ('hierarchical','flat'):
                nominal='memory_n3' if code=='EXP0051' else ('balanced_n3' if variant=='hierarchical' else 'balanced_flat_n3')
                reference={r['scene_seed']:r for r in groups[nominal]}
                for label,data in groups.items():
                    if label.startswith(variant+'_s'):
                        equivalence[label]=dict(nominal=nominal,completed=len(data),
                            same_final_state=sum(r['final_state_hash']==reference[r['scene_seed']]['final_state_hash']
                                for r in data if r['scene_seed'] in reference))
            study['phases'][phase]=dict(requested=a['requested'],completed=a['completed'],
                audit_passed=a['audit_passed'],issues=a['issues'],independent_scenes=a['scenes'],
                arms=arms,final_state_equivalence=equivalence)
        summaries.append(study)
    result=dict(date='2026-10-04',studies=summaries,training_control_steps=all_training_steps,
        training_runs=sum(len(s['training_runs']) for s in summaries),
        requested_main_and_stress=sum(p['requested'] for s in summaries for p in s['phases'].values()),
        completed_main_and_stress=sum(p['completed'] for s in summaries for p in s['phases'].values()),
        current_source_issues=source_issues,confirmation_access_issues=seed_issues,
        actor_contract_flag_issues=feature_issues,
        all_numerical_pairing_gates_pass=all(p['audit_passed'] for s in summaries for p in s['phases'].values()),
        no_hardware_safety_certificate=True,
        statistical_scope='Development only; repeated policies and training seeds do not increase the number of independent scenes; prior native failures remain unresolved.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='studies'},indent=2))
    if source_issues or seed_issues or feature_issues:raise SystemExit(2)


if __name__=='__main__':main()
