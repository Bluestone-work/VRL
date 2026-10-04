"""Recompute stored-state hashes and check full training/legacy provenance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from scripts.multicluster_protocol import digest
from scripts.run_option_learning import check_development_seed, atomic_json
from scripts.run_residual_marl import ROOT, source_hashes
from scripts.run_measured_marl import runtime_context


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    root=parser.parse_args().root
    manifest=json.loads((root/'manifest.json').read_text());p=manifest['protocol']
    migrated='parent_study' in p
    training_root=ROOT/p['parent_study'] if migrated else root
    training_manifest=json.loads((training_root/'manifest.json').read_text())
    previous={}
    for name in ('EXP0053_TPG_GENERIC_STUDY_20261004','EXP0054_MEASURED_MARL_STUDY_20261004',
                 'EXP0055_SAFETY_OBJECTIVE_STUDY_20261004'):
        old=json.loads((ROOT/'research/validation'/name/'manifest.json').read_text())['source_hashes']
        previous[name]=dict(files=len(old),unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in old.items()))
    snapshots=[]
    for folder in ('preflight','results'):
        for path in sorted((root/folder).glob('*.jsonl')):
            row=json.loads(path.read_text())
            snapshots.append(dict(file=str(path),matches=digest(row['actual_initial_snapshot'])==row['actual_initial_snapshot_hash']))
    training=[];seeds_checked=0
    for i,seed in enumerate(p['training_seeds']):
        expected=p['training_scene_base']+i*p['training_scene_stride']
        for variant in p['variants']:
            folder=training_root/'training'/f'train_{variant}_{seed}'
            m=json.loads((folder/'manifest.json').read_text())
            status=json.loads((folder/'status.json').read_text())
            resets=[json.loads(x) for x in (folder/'attempts.jsonl').read_text().splitlines()]
            requested=[r['scene_seed'] for r in resets if r['event']=='reset_requested']
            for row in resets:
                for field in ('scene_seed','accepted_seed'):
                    if field in row:
                        check_development_seed(row[field]);seeds_checked+=1
            updates=[json.loads(x) for x in (folder/'updates.jsonl').read_text().splitlines()]
            finite=all(math.isfinite(value) for row in updates for value in row.values() if isinstance(value,(int,float)))
            training.append(dict(variant=variant,seed=seed,
                passed=m['scene_base']==expected and m['training_seed']==seed and m['variant']==variant
                    and m['runtime']==training_manifest['runtime'] and m['source_hashes']==training_manifest['source_hashes']
                    and status['steps']==p['training_steps'] and status['phase']=='completed'
                    and requested==list(range(expected,expected+len(requested))) and finite,
                high_samples=status['high_samples'],high_updates=status['high_updates'],low_updates=status['low_updates']))
    if migrated:
        from scripts.run_clean_residual_eval import source_hashes as clean_hashes, enforce_runtime
        current=clean_hashes()==manifest['source_hashes'] and enforce_runtime()==manifest['runtime']
    elif 'CONTINUOUS' in p['experiment']:
        from scripts.run_continuous_marl import source_hashes as continuous_hashes, enforce_runtime
        current=continuous_hashes()==manifest['source_hashes'] and enforce_runtime()==manifest['runtime']
    else:
        current=source_hashes()==manifest['source_hashes'] and runtime_context()==manifest['runtime']
    main_audit=json.loads((root/'AUDIT.json').read_text())
    outcome=json.loads((root/'OUTCOME_AUDIT.json').read_text())
    passed=current and main_audit['valid_comparison_gate'] and outcome['passed'] and all(x['unchanged'] for x in previous.values())
    passed=passed and all(x['matches'] for x in snapshots) and all(x['passed'] for x in training)
    result=dict(passed=passed,current_source_and_runtime_match=current,older_sources=previous,
        initial_snapshot_hashes_recomputed=snapshots,training=training,training_seed_fields_checked=seeds_checked,
        confirmation_accessed=False,parent_training_only=migrated,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(root/'FINAL_INTEGRITY_AUDIT.json',result)
    print(json.dumps(dict(passed=passed,snapshots_recomputed=len(snapshots),training_runs=len(training),
                         seed_fields_checked=seeds_checked,older_sources=previous),indent=2))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
