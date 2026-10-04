"""Post-hoc paired observer diagnosis, all 12 fixed development scenes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts import run_option_study as study
from scripts.run_option_learning import make_episode
from scripts.option_learning_episode import ROOT


def main():
    folder=ROOT/'research/validation/EXP0052_PAIRED_OBSERVER_ABLATION_20261004'
    arms=[dict(label='old_tracker_memory_n3',policy='memory',clusters=3,variant='hierarchical')]
    summary=study.evaluate_matrix(folder,arms,range(1820000000,1820000012),workers=6,
        phase='posthoc_old_observer_ablation')
    old=[json.loads(l) for l in (folder/'attempts.jsonl').read_text().splitlines()]
    source=ROOT/'research/validation/EXP0052_STUDY_20261004/validation/attempts.jsonl'
    new=[r for r in map(json.loads,source.read_text().splitlines()) if r['arm']=='memory_n3']
    errors=[]
    if not summary['audit_passed']:errors+=summary['issues']
    if len(new)!=12 or any(r['status']!='completed' for r in new):errors.append('New-observer reference incomplete')
    if len(old)!=12 or any(r['status']!='completed' for r in old):errors.append('Old-observer matrix incomplete')
    old_good={r['scene_seed']:r for r in old if r['status']=='completed'}
    new_good={r['scene_seed']:r for r in new if r['status']=='completed'}
    for seed in old_good.keys()&new_good.keys():
        a,b=old_good[seed],new_good[seed]
        for key in ('actual_initial_snapshot','actual_initial_snapshot_hash','sensor_assumptions','supervisor_settings'):
            if a[key]!=b[key]:errors.append(f'Mismatch {key} at {seed}')
        if a['reset_info']['actual_config']!=b['reset_info']['actual_config']:errors.append('Physical configuration mismatch')
        for name,h in a['source_hashes'].items():
            if b['source_hashes'].get(name)!=h:errors.append('Common-source change '+name)
    # Repeat the FIRST predeclared scene with a five-second renewal. This is
    # a separate diagnostic, never a replacement for a failed primary attempt.
    interval_equal=None
    if 1820000000 in old_good:
        ep=make_episode(1820000000,option_steps=50)
        try:
            while not ep.done:ep.step_option(ep.conventional_options('memory'))
            repeat=ep.result('memory')
            interval_equal=repeat['final_state_hash']==old_good[1820000000]['final_state_hash']
            (folder/'interval_repeat.json').write_text(json.dumps(repeat,indent=2)+'\n')
            if not interval_equal:errors.append('Memory renewal interval changed the trajectory')
        finally:ep.close()
    result=dict(exploratory_posthoc=True,learning_contribution=False,confirmation_accessed=False,
        admission_passed=not errors,issues=errors,old_memory_renewal_equivalence=interval_equal,
        reference=str(source),reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),comparisons={})
    if not errors:
        keys=('cluster_safe_success','removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s')
        for key in keys:
            result['comparisons'][key]=dict(old_mean=float(np.mean([r[key] for r in old_good.values()])),
                aligned_mean=float(np.mean([r[key] for r in new_good.values()])),
                aligned_minus_old=study.bootstrap([new_good[s][key]-old_good[s][key] for s in sorted(old_good)]))
    study.dump(folder/'observer_comparison.json',result)
    report=['# Paired command-aligned observer diagnosis','',
        'Post-hoc development diagnosis; not a learning contribution or hardware/safety proof.','',
        f"Admission: {'PASS' if not errors else 'BLOCKED'}; 12 fixed paired layouts.",'',
        '| Metric | Old tracking | Command-aligned tracking |','|---|---:|---:|']
    for key,value in result['comparisons'].items():
        report.append(f"| {key} | {value['old_mean']:.6f} | {value['aligned_mean']:.6f} |")
    report+=['','All common physical, sensing-noise and supervisor settings match; observer source additions are explicit.',
        'Memory renewal interval diagnostic: '+str(interval_equal)]
    report+=['- '+error for error in errors]
    (folder/'OBSERVER_COMPARISON.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if not errors else 2


if __name__=='__main__':raise SystemExit(main())
