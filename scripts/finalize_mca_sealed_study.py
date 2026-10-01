"""Select checkpoints on the validation split, then sealed-test the selections once.

For every arm and seed: evaluate each milestone checkpoint and the parent on the
registered validation layouts (scripts/select_mca_checkpoint.py, frozen arm
snapshot), pick the best (ties -> earlier; parent is step 0), sealed-test it
(scripts/evaluate_mca_sealed_test.py) and write a paired report.
usage: finalize_mca_sealed_study.py --study-dir research/runs/EXP_0039_ASSIGNED_20261001a --study EXP_0039 \
       --arms assigned,control --baseline-study BASELINE_EXP35
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/wj/miniconda3/envs/v/bin/python'
SEEDS = (42, 43, 44)


def run(cmd, cwd, cpus):
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run(['taskset', '-c', cpus, *cmd], cwd=cwd, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir', required=True, type=Path)
    p.add_argument('--study', required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--baseline-study', default='BASELINE_EXP35')
    p.add_argument('--cpus', default='0-5,8-23')
    p.add_argument('--parallel', type=int, default=18)
    p.add_argument('--test-workers', type=int, default=18)
    p.add_argument('--reference', help='NAME=path/to/summary.json:ARM, an arm of an earlier sealed study to pair against')
    args = p.parse_args()
    study_dir = args.study_dir.resolve(); arms = args.arms.split(',')
    out = ROOT/f'research/validation/{args.study}_SEALED_RESULTS'
    sel_dir = out/'selection'; sel_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in arms:
        snap = study_dir/arm/'source_snapshot'
        protocol = json.loads((study_dir/arm/'protocol.json').read_text())
        proto_rel = next((snap/'configs/experiments').glob(f'{protocol["experiment"]}.json')).relative_to(snap)
        milestones = protocol['milestones']
        for seed in SEEDS:
            cands = {m: study_dir/arm/f'seed_{seed}/policy_{m}.pt' for m in milestones}
            cands[0] = snap/protocol['initialization_checkpoints'][str(seed)]['path']
            for m, ckpt in cands.items():
                target = sel_dir/f'{arm}_seed_{seed}_{m}.json'
                if not target.exists():
                    jobs.append(([PYTHON, str(ROOT/'scripts/select_mca_checkpoint.py'), '--protocol', str(proto_rel),
                                  '--checkpoint', str(ckpt), '--out', str(target)], snap))
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(lambda j: run(j[0], j[1], args.cpus), jobs))
    report = dict(study=args.study, arms={}, baseline={})
    ledger = json.loads((ROOT/'research/sealed_test/LEDGER.json').read_text())
    for e in ledger['entries']:
        if e['study'] == args.baseline_study and e['complete']:
            seed = int(e['label'].rsplit('_', 1)[1])
            report['baseline'][seed] = dict(checkpoint_sha256=e['checkpoint_sha256'], result=e['result'], success_rate=e['success_rate'])
    for arm in arms:
        snap = study_dir/arm/'source_snapshot'
        protocol = json.loads((study_dir/arm/'protocol.json').read_text())
        proto_rel = next((snap/'configs/experiments').glob(f'{protocol["experiment"]}.json')).relative_to(snap)
        report['arms'][arm] = {}
        for seed in SEEDS:
            scores = {m: json.loads((sel_dir/f'{arm}_seed_{seed}_{m}.json').read_text())['success_rate']
                      for m in [0]+protocol['milestones']}
            best = max(scores, key=lambda m: (scores[m], -m))
            ckpt = (snap/protocol['initialization_checkpoints'][str(seed)]['path'] if best == 0
                    else study_dir/arm/f'seed_{seed}/policy_{best}.pt')
            cmd = [PYTHON, str(ROOT/'scripts/evaluate_mca_sealed_test.py'), '--protocol', str(proto_rel), '--checkpoint', str(ckpt),
                   '--study', args.study, '--label', f'{arm}_seed_{seed}_{best}', '--declare-candidates', str(len(arms)*len(SEEDS)),
                   '--workers', str(args.test_workers)]
            run(cmd, snap, args.cpus)
            ledger = json.loads((ROOT/'research/sealed_test/LEDGER.json').read_text())
            import hashlib
            digest = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
            entry = next(e for e in ledger['entries'] if e['checkpoint_sha256'] == digest and e['complete'] and not e['partial'])
            report['arms'][arm][seed] = dict(selected_step=best, validation_scores=scores, test_result=entry['result'],
                                             test_success_rate=entry['success_rate'])
    def successes(path):
        return np.array([r['success'] for r in json.loads(Path(path).read_text())['episodes']], float)
    rows = []
    for arm in arms:
        rates = [report['arms'][arm][s]['test_success_rate'] for s in SEEDS]
        rows.append(f'| {arm} | ' + ' / '.join(f'{100*r:.1f}' for r in rates) + f' | {100*np.mean(rates):.1f}% |')
    if report['baseline']:
        rates = [report['baseline'][s]['success_rate'] for s in SEEDS]
        rows.append('| baseline EXP35 500K | ' + ' / '.join(f'{100*r:.1f}' for r in rates) + f' | {100*np.mean(rates):.1f}% |')
    paired = []
    pair_names = None
    if args.reference:
        name, spec = args.reference.split('=', 1); path, ref_arm = spec.rsplit(':', 1)
        ref = json.loads(Path(path).read_text())['arms'][ref_arm]
        report['reference'] = {name: {s: ref[str(s)] for s in SEEDS}}
        rates = [ref[str(s)]['test_success_rate'] for s in SEEDS]
        rows.append(f'| {name} (reference) | ' + ' / '.join(f'{100*r:.1f}' for r in rates) + f' | {100*np.mean(rates):.1f}% |')
        pair_names = (arms[0], name)
        for s in SEEDS:
            paired.append(float((successes(report['arms'][arms[0]][s]['test_result'])-successes(ref[str(s)]['test_result'])).mean()))
    elif len(arms) == 2:
        a, b = arms; pair_names = (a, b)
        for s in SEEDS:
            paired.append(float((successes(report['arms'][a][s]['test_result'])-successes(report['arms'][b][s]['test_result'])).mean()))
    if paired:
        report['paired_diff'] = dict(zip(map(str, SEEDS), paired))
    (out/'summary.json').write_text(json.dumps(report, indent=1)+'\n')
    lines = [f'# {args.study} sealed-test result', '',
             'Selection on the registered validation split (200 layouts); one sealed-test evaluation (500 layouts) per selected checkpoint.', '',
             '| arm | seed 42 / 43 / 44 (%) | mean |', '|---|---|---:|', *rows, '']
    for arm in arms:
        lines.append(f'- {arm} selected steps: ' + ', '.join(f'seed {s}: {report["arms"][arm][s]["selected_step"]}' for s in SEEDS))
    if paired:
        lines.append(f'- paired {pair_names[0]} - {pair_names[1]} on identical test layouts: ' +
                     ' / '.join(f'{100*d:+.1f}' for d in paired) + f' pp, mean {100*np.mean(paired):+.1f} pp')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
