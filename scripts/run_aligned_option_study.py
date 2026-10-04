"""Registered EXP0052 factories over the frozen process-isolated study runner.

The narrow dispatch below changes the entrypoint and baseline decision interval
explicitly; all changed choices are stored in arm manifests and result rows.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from scripts import run_option_study as base
from scripts.aligned_option_episode import PROTOCOL,aligned_hashes


def baseline_arms():
    protocol=json.loads(PROTOCOL.read_text())
    policies={'memory_n3':'memory','balanced_n3':'balanced','balanced_fast_n3':'balanced',
        'balanced_flat_n3':'balanced','priority_n3':'priority','memory_n1':'memory',
        'untrained_balanced_n3':'untrained','untrained_flat_n3':'untrained'}
    return [dict(label=label,policy=policy,clusters=1 if label=='memory_n1' else 3,
        variant='flat' if label in ('balanced_flat_n3','untrained_flat_n3') else 'hierarchical',
        decision_steps=protocol['baseline_decision_steps'][label]) for label,policy in policies.items()]


ORIGINAL_RUN=base.run_process
ORIGINAL_ANALYZE=base.analyze
ORIGINAL_DUMP=base.dump


def dump(path,value):
    if isinstance(value,dict) and value.get('experiment')=='EXP0051':
        value=dict(value,experiment='EXP0052')
    return ORIGINAL_DUMP(path,value)


def run_process(command,log,timeout):
    command=list(command)
    if command[0]=='scripts/run_option_learning.py':command[0]='scripts/run_aligned_options.py'
    label=Path(log).name.split('_scene')[0]
    settings={a['label']:a for a in baseline_arms()}
    if label in settings and 'evaluate' in command:
        command+=['--decision-steps',str(settings[label]['decision_steps'])]
    return ORIGINAL_RUN(command,log,timeout)


def analyze(folder,manifest,rows):
    summary=ORIGINAL_ANALYZE(folder,manifest,rows)
    arms=defaultdict(list)
    for r in rows:
        if r['status']=='completed':arms[r['arm']].append(r)
    checks={}
    for untrained,matched in [('untrained_balanced_n3','balanced_n3'),('untrained_flat_n3','balanced_flat_n3')]:
        if untrained not in {a['label'] for a in manifest['arms']}:continue
        a={r['scene_seed']:r['final_state_hash'] for r in arms[untrained]}
        b={r['scene_seed']:r['final_state_hash'] for r in arms[matched]}
        checks[untrained]=a==b and len(a)==len(manifest['scenes'])
        if not checks[untrained]:summary['issues'].append(untrained+' does not exactly match its baseline')
    summary['untrained_allocation_equivalence']=checks
    summary['experiment']='EXP0052'
    summary['audit_passed']=not summary['issues']
    extra=[]
    if summary['audit_passed']:
        for variant in ('hierarchical','flat'):
            group=defaultdict(list)
            for label,data in arms.items():
                if label.startswith(variant+'_s'):
                    for r in data:group[r['scene_seed']].append(r)
            if not group:continue
            for label in ('balanced_fast_n3','balanced_flat_n3'):
                if label not in arms:continue
                reference={r['scene_seed']:r for r in arms[label]}
                if group.keys()!=reference.keys():raise ValueError('Unpaired scheduling-time comparator')
                metrics={key:base.bootstrap([np.mean([r[key] for r in group[s]])-reference[s][key]
                    for s in sorted(group)]) for key in ('cluster_safe_success','removal','removal_auc_180',
                        'wall_contact_s','spacing_violation_pair_s','minimum_spacing_mm','supervisor_infeasible_steps')}
                extra.append(dict(learning=variant,baseline=label,
                    inference_unit='scene; three training seeds averaged within scene',metrics=metrics))
        summary['comparisons']+=extra
    else:
        summary['comparisons']=[]
        summary.pop('hierarchy_vs_flat',None)
    base.dump(folder/'analysis.json',summary)
    report=(folder/'REPORT.md').read_text().replace('# EXP0051 ', '# EXP0052 ',1)
    report=report.replace('Untrained equals memory: None\n','')
    report=report.replace('The legacy-memory arm has a different safety layer, so any difference alone is not a learning gain.\n','')
    report+='\n## Matched observer and prior\n\n'
    report+='All arms use command-aligned measured tracking. Untrained policies must match measured allocation at the same decision interval. Observer/prior upgrades alone are not learning gains.\n'
    report+='Exact untrained allocation equivalence: '+json.dumps(checks)+'\n'
    report+='Classical allocation at 0.1 s, 1 s and 5 s remains in the strong comparator set.\n'
    for item in extra:
        report+=f"\n### {item['learning']} minus {item['baseline']}\n\n"
        for key in ('cluster_safe_success','removal','wall_contact_s','spacing_violation_pair_s'):
            r=item['metrics'][key]
            report+=f"- {key}: {r['mean']:.6f}, scene bootstrap 95% [{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}].\n"
    if not summary['audit_passed']:
        report=report.replace('Audit: PASS','Audit: BLOCKED',1)
        report+='\nFinal audit BLOCKED: '+json.dumps(summary['issues'])+'\n'
    (folder/'REPORT.md').write_text(report)
    return summary


if __name__=='__main__':
    base.PROTOCOL=PROTOCOL
    base.option_hashes=aligned_hashes
    base.dump=dump
    base.baseline_arms=baseline_arms
    base.run_process=run_process
    base.analyze=analyze
    raise SystemExit(base.main())
