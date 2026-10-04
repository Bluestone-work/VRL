"""Complete the missing same-layout alpha=0 cell of EXP0052 stress diagnosis."""
from collections import defaultdict
import json
import numpy as np
from scripts import run_aligned_option_study as aligned
from scripts.aligned_option_episode import ROOT


def main():
    reference=ROOT/'research/validation/EXP0052_STUDY_20261004/coupling_stress'
    manifest=json.loads((reference/'manifest.json').read_text())
    base=aligned.base
    base.PROTOCOL=aligned.PROTOCOL;base.option_hashes=aligned.aligned_hashes
    base.dump=aligned.dump;base.run_process=aligned.run_process
    base.baseline_arms=aligned.baseline_arms;base.analyze=aligned.analyze
    out=ROOT/'research/validation/EXP0052_SAME_LAYOUT_UNCOUPLED_20261004'
    summary=base.evaluate_matrix(out,manifest['arms'],manifest['scenes'],workers=6,coupling=0.,
        phase='posthoc_same_layout_uncoupled')
    original=[json.loads(l) for l in (reference/'attempts.jsonl').read_text().splitlines()]
    uncoupled=[json.loads(l) for l in (out/'attempts.jsonl').read_text().splitlines()]
    errors=[]
    if not summary['audit_passed']:errors+=summary['issues']
    if any(r['status']!='completed' for r in original+uncoupled):errors.append('Incomplete factorial cell')
    a={(r['arm'],r['scene_seed']):r for r in original if r['status']=='completed'}
    b={(r['arm'],r['scene_seed']):r for r in uncoupled if r['status']=='completed'}
    if a.keys()!=b.keys():errors.append('Unpaired arm/layout keys')
    for key in a.keys()&b.keys():
        for field in ('actual_initial_snapshot','sensor_assumptions','supervisor_settings',
                      'source_hashes','option_steps','checkpoint_sha256'):
            if a[key][field]!=b[key][field]:errors.append(f'Mismatched {field}: {key}')
        if a[key]['reset_info']['actual_config']!=b[key]['reset_info']['actual_config']:
            errors.append(f'Physical config mismatch: {key}')
    result=dict(exploratory_posthoc=True,admission_passed=not errors,issues=errors,
        contrast='coupled alpha=0.1 minus alpha=0, same four layouts and exact checkpoints',
        independent_scenes=4,hardware_calibrated=False,comparisons={})
    if not errors:
        for arm in sorted({key[0] for key in a}):
            keys=sorted(k for k in a if k[0]==arm)
            result['comparisons'][arm]=dict(
                uncoupled_violating_scenes=sum(not b[k]['spacing_compliant'] for k in keys),
                coupled_violating_scenes=sum(not a[k]['spacing_compliant'] for k in keys),
                uncoupled_minimum_mm=min((b[k]['minimum_spacing_mm'] for k in keys if b[k]['minimum_spacing_mm'] is not None),default=None),
                coupled_minimum_mm=min((a[k]['minimum_spacing_mm'] for k in keys if a[k]['minimum_spacing_mm'] is not None),default=None),
                metrics={m:dict(uncoupled=float(np.mean([b[k][m] for k in keys])),
                    coupled=float(np.mean([a[k][m] for k in keys])),
                    paired_difference=base.bootstrap([a[k][m]-b[k][m] for k in keys]))
                    for m in ('removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s','cluster_safe_success')})
    base.dump(out/'coupling_comparison.json',result)
    text=['# Same-layout synthetic coupling diagnosis','',
        'Post-hoc development check. Same policies and all four layouts; no hardware calibration or safety proof.','',
        f"Admission: {'PASS' if not errors else 'BLOCKED'}.",'',
        '| Arm | Spacing-violating scenes, alpha=0 | Spacing-violating scenes, alpha=0.1 | Mean spacing pair-s, alpha=0 | Mean spacing pair-s, alpha=0.1 |',
        '|---|---:|---:|---:|---:|']
    for arm,r in result['comparisons'].items():
        m=r['metrics']['spacing_violation_pair_s']
        text.append(f"| {arm} | {r['uncoupled_violating_scenes']} | {r['coupled_violating_scenes']} | {m['uncoupled']:.6f} | {m['coupled']:.6f} |")
    text+=['','Intervals are exploratory over only four scenes; repeated training seeds are not independent layouts.']
    text+=['- '+issue for issue in errors]
    (out/'COUPLING_COMPARISON.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
    return 0 if not errors else 2


if __name__=='__main__':raise SystemExit(main())
