"""Process-isolated EXP0047 comparisons; retain every requested attempt."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from scripts.local_learning_episode import PROTOCOL, ROOT, learning_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('validation', 'confirmation'), default='validation')
    parser.add_argument('--contract', choices=('ideal', 'tracked', 'temporal', 'memory_prior'), default='ideal')
    parser.add_argument('--validation-scene-base', type=int,
                        help='Explicit fresh development transfer pool; forbidden in confirmation')
    parser.add_argument('--selection-manifest', type=Path,
                        help='Frozen method/checkpoint selection required before confirmation')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--single-memory', action='store_true',
                        help='Strong N=1 navigation-memory baseline, separately registered')
    parser.add_argument('--checkpoint', type=Path, action='append', default=[])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    protocol_path, hash_function = PROTOCOL, learning_hashes
    entrypoint = 'scripts/run_local_learning.py'
    if args.contract == 'tracked':
        from scripts.tracked_learning_episode import PROTOCOL as tracked_protocol, tracking_hashes
        protocol_path, hash_function = tracked_protocol, tracking_hashes
        entrypoint = 'scripts/run_tracked_learning.py'
    elif args.contract == 'temporal':
        from scripts.temporal_learning_episode import PROTOCOL as temporal_protocol, temporal_hashes
        protocol_path, hash_function = temporal_protocol, temporal_hashes
        entrypoint = 'scripts/run_temporal_learning.py'
    elif args.contract == 'memory_prior':
        from scripts.memory_prior_episode import PROTOCOL as memory_protocol, memory_prior_hashes
        protocol_path, hash_function = memory_protocol, memory_prior_hashes
        entrypoint = 'scripts/run_memory_prior_learning.py'
    if args.validation_scene_base is not None and args.stage != 'validation':
        parser.error('Scene override is development-only')
    protocol = json.loads(protocol_path.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    arms = []
    if args.baselines:
        arms += [dict(label=policy+'_n3', policy=policy, clusters=3, checkpoint=None)
                 for policy in ('v2_spacing', 'joint', 'memory')]
        arms.append(dict(label='joint_n1', policy='joint', clusters=1, checkpoint=None))
    if args.single_memory:
        arms.append(dict(label='memory_n1', policy='memory', clusters=1, checkpoint=None))
    for path in args.checkpoint:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        arms.append(dict(label=f"learned_s{payload['training_seed']}_t{payload['transitions']}_n3",
            policy='learned', clusters=3, checkpoint=str(path.resolve()),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if not arms:
        parser.error('Specify baselines and/or checkpoint(s)')
    assert len({a['label'] for a in arms}) == len(arms)
    base = protocol[args.stage+'_scene_base'] if args.validation_scene_base is None else args.validation_scene_base
    seeds = list(range(base, base+protocol[args.stage+'_scenes']))
    source = hash_function()
    if args.stage == 'validation':
        for path in (ROOT/'configs/experiments').glob('EXP_00*_*.json'):
            reserved = json.loads(path.read_text())
            start, count = reserved.get('confirmation_scene_base'), reserved.get('confirmation_scenes')
            if start is not None and count is not None and any(start <= seed < start+count for seed in seeds):
                parser.error('Development pool intersects a reserved confirmation pool')
    selection_sha = None
    if args.stage == 'confirmation':
        if args.selection_manifest is None:
            parser.error('Freeze method/checkpoint selection before opening confirmation scenes')
        selection = json.loads(args.selection_manifest.read_text())
        if selection['source_hashes'] != source:
            parser.error('Frozen selection source mismatch')
        if not set(a['label'] for a in arms) <= set(selection['arms']):
            parser.error('Unregistered confirmation arm')
        for arm in arms:
            if arm.get('checkpoint_sha256') and arm['checkpoint_sha256'] != selection['checkpoints'][arm['label']]:
                parser.error('Frozen checkpoint mismatch')
        selection_sha = hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest()
    manifest = dict(stage=args.stage, arms=arms, scenes=seeds, control_seed=42,
        requested=len(arms)*len(seeds), source_hashes=source, sealed_test_used=False,
        registered_at=datetime.now().astimezone().isoformat(), workers=args.workers,
        observation_contract=args.contract,
        development_scene_override=args.validation_scene_base,
        selection_manifest_sha256=selection_sha,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    parts = args.out/'parts'; parts.mkdir()
    def run(arm, seed):
        prefix = parts/f"{arm['label']}_scene{seed}"
        path = prefix.with_suffix('.jsonl')
        cmd = [sys.executable, '-X', 'faulthandler', entrypoint, 'evaluate',
            '--policy', arm['policy'], '--clusters', str(arm['clusters']), '--control-seed', '42',
            '--scene-base', str(seed), '--episodes', '1', '--out', str(path.resolve())]
        if arm['checkpoint']: cmd += ['--checkpoint', arm['checkpoint']]
        common = dict(arm=arm['label'], scene_seed=seed, clusters=arm['clusters'], method=arm['policy'])
        started = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180.)
            prefix.with_suffix('.stderr.txt').write_text(p.stderr)
            if p.returncode:
                row = dict(common, status='process_exit', returncode=p.returncode,
                           stderr_path=str(prefix.with_suffix('.stderr.txt')))
            else:
                row = json.loads(path.read_text().splitlines()[0])
                assert row['scene_seed'] == seed and row['status'] == 'completed'
                assert row['source_hashes'] == source
                row['arm'] = arm['label']
        except subprocess.TimeoutExpired:
            row = dict(common, status='timeout')
        except (OSError, ValueError, IndexError, AssertionError) as exc:
            row = dict(common, status='invalid_output', error=repr(exc))
        row['process_wall_s'] = time.monotonic()-started
        return row
    rows = []
    with (args.out/'attempts.jsonl').open('x', buffering=1) as stream, \
            ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, arm, seed) for arm in arms for seed in seeds]
        for future in as_completed(futures):
            row = future.result(); rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            progress = dict(recorded=len(rows), requested=manifest['requested'],
                failed=sum(r['status'] != 'completed' for r in rows), last_arm=row['arm'])
            temp = args.out/'progress.tmp'
            temp.write_text(json.dumps(progress, indent=2)+'\n');temp.replace(args.out/'progress.json')
            print(json.dumps({**progress, 'last_status': row['status']}), flush=True)
    assert hash_function() == source
    summary = dict(requested=manifest['requested'], recorded=len(rows),
        completed=sum(r['status'] == 'completed' for r in rows),
        failed_attempts=[r for r in rows if r['status'] != 'completed'], arms={})
    for arm in arms:
        a = [r for r in rows if r['arm'] == arm['label']]
        complete = [r for r in a if r['status'] == 'completed']
        success = sum(r['cluster_safe_success'] for r in complete)
        summary['arms'][arm['label']] = dict(completed=len(complete), requested=len(a),
            success_missing_bounds=[success/len(a), (success+len(a)-len(complete))/len(a)],
            conditional_means={key: sum(float(r[key]) for r in complete)/len(complete)
                for key in ('task_success', 'safe_success', 'cluster_safe_success', 'removal',
                    'removal_auc_180', 'wall_contact_s', 'spacing_violation_pair_s')} if complete else {})
    summary['numerical_gate_passed'] = len(rows) == manifest['requested'] and not summary['failed_attempts']
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if not summary['numerical_gate_passed']: raise SystemExit(2)


if __name__ == '__main__':
    main()
