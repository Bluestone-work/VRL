"""Summarize complete paired screening without substituting training metrics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    manifest=json.loads((a.directory/'manifest.json').read_text())
    rows=[json.loads(l) for l in (a.directory/'episodes.jsonl').read_text().splitlines()]
    key=lambda r:(r['method'],r['suite'],r['anatomy'],r['clusters'],r['seed'])
    if len(rows)!=len(manifest['jobs']) or {key(r) for r in rows}!={tuple(j) for j in manifest['jobs']}:
        raise ValueError('incomplete or duplicate evaluation')
    if any('error' in r for r in rows): raise ValueError('evaluation errors present')
    indexed={key(r):r for r in rows}
    groups={}
    for r in rows: groups.setdefault((r['suite'],r['method']),[]).append(r)
    summaries=[];paired=[]
    fields=['task_success','strict_success','removal','removal_auc','wall_contact_s','lost',
            'neighborhood_residence_ratio','departures_after_entry']
    for (suite,method),rs in sorted(groups.items()):
        s=dict(suite=suite,method=method,n=len(rs))
        s.update({f:float(np.mean([r[f] for r in rs])) for f in fields})
        for f in ['t90_s','first_neighborhood_entry_s','near_command_norm','observed_near_motion_mm_s']:
            vals=[r[f] for r in rs if r[f] is not None]
            s[f]=float(np.mean(vals)) if vals else None;s[f+'_observed_n']=len(vals)
        s['exit_episode_rate']=float(np.mean([r['lost']>0 for r in rs]))
        s['failure_stages']=dict(Counter(r['failure_stage'] for r in rs))
        summaries.append(s)
    for suite in sorted({r['suite'] for r in rows}):
        for candidate,baseline in [('B','A'),('C','B'),('D','C'),('no_settle','settle'),('A','A_legacy')]:
            if (suite,candidate) not in groups or (suite,baseline) not in groups:continue
            cr=sorted(groups[suite,candidate],key=lambda r:key(r)[2:])
            br=[indexed[(baseline,suite,r['anatomy'],r['clusters'],r['seed'])] for r in cr]
            for c,b in zip(cr,br):
                assert c['scenario_hash']==b['scenario_hash'] and c['initial_state_hash']==b['initial_state_hash']
                assert c['variation']==b['variation'] and c['actual_healthy_mean_mm_s']==b['actual_healthy_mean_mm_s']
            paired.append(dict(suite=suite,contrast=f'{candidate}-{baseline}',n=len(cr),
                               delta={f:float(np.mean([c[f]-b[f] for c,b in zip(cr,br)])) for f in fields}))
    result=dict(summary=summaries,paired=paired,paired_scene_hashes_verified=True,training_seeds=1)
    (a.directory/'summary.json').write_text(json.dumps(result,indent=2))
    fmt=lambda x:'NA' if x is None else f'{x:.3f}'
    lines=['# EXP0062 paired development screening','',
           '14 anatomies × N=1/2/3 × one paired scene per suite; training seed 6201 only. 300 s horizon and existing success definitions unchanged.',
           '', '| Suite | Arm | n | All-clear % | Strict % | Removal % | AUC | T90 s (observed n) | Wall s | Exit episodes % |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| {s["suite"]} | {s["method"]} | {s["n"]} | {100*s["task_success"]:.1f} | {100*s["strict_success"]:.1f} | {100*s["removal"]:.1f} | {s["removal_auc"]:.3f} | {fmt(s["t90_s"])} ({s["t90_s_observed_n"]}) | {s["wall_contact_s"]:.3f} | {100*s["exit_episode_rate"]:.1f} |')
    lines += ['', 'T90 is averaged only over episodes reaching 90%; missing values remain censored and counts are shown. See summary.json for paired deltas and trajectory diagnostics.', '',
              '## Scope and limitations','',
              '- A/B already have a 32-step Transformer; C/D add an 8-step (0.8 s) GRU context. C−B measures added GRU context, not history versus no history.',
              '- Flow values are nominal healthy inlet calibration parameters. The pre-existing per-scene multiplier is retained (actual mean = nominal × multiplier/2); nominal intermediate flow does not guarantee disjoint actual flow support.',
              '- Training used 120 updates × 4 workers × 256 environment steps = 122880 environment steps per arm. Retained active-agent samples differ (~259–262k). The earlier 90k estimate was incorrect.',
              '- The existing GAE treats time limits as terminal and omits held-agent steps. This inherited training limitation is shared by all arms and prevents claiming fully corrected truncation handling.',
              '- D predicts next measured normalized velocity, not velocity displacement; the actual auxiliary weight is 0.01. Earlier EXP0062 module/config descriptions (delta velocity, 0.05) do not describe these trained checkpoints.',
              '- Old residual scale .5 allows up to .5 speed at zero prior; it does not force zero commands. Scale 1 restores scalar speed range, but route aiming still restricts direction. At exactly zero aim displacement and zero lateral action, the command remains zero.',
              '- A−A_legacy is a same-checkpoint inference intervention, not an equal-budget old/new mapping retraining comparison.',
              '- Neighborhood diagnostics use estimated position and a 0.3 mm preoperative-target neighborhood; observed motion includes command response and sensing noise, so it is not an isolated estimate of blood flow.',
              '- Single training seed and one scene per anatomy/N/suite: exploratory screening, no robust performance claim or three-seed confirmation.',
              '- The abandoned EXP0062_*_eval.jsonl files used uncalibrated flow and mismatched history actions. They are invalid, retained for audit, and excluded here.', '',
              '## Reproduction','',
              '```bash',
              'env PYTHONPATH=. OMP_NUM_THREADS=1 /home/wj/miniconda3/envs/v/bin/python -m scripts.evaluate_lysis_abcd --out research/validation/EXP0062_paired_reproduction --workers 20',
              f'env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.report_lysis_abcd {a.directory}',
              '```']
    (a.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
