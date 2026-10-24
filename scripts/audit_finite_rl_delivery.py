"""Audit finite-SAC coverage, paired physics, RL training and stored GIF sources.

This read-only audit never selects a checkpoint or runs the simulator. Missing
records are reported as pending; a complete delivery requires every audit below.
"""
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, os.sched_getaffinity(0) - {6, 7})
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scripts.eval_image_flow_sac import scenes
from scripts.report_image_flow_sac import metrics

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / 'research/validation/EXP0099_FINITE_IMAGE_SAC_20261011'
RUN = ROOT / 'research/runs/EXP0099_FINITE_IMAGE_SAC_20261011/seed0'
MEDIA = Path('/home/wj/桌面/VRL_有限时域图像SAC_训练测试_3D_GIF_20261011')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def records(path):
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def scene_key(method, cell, job):
    return (method, cell, *job)


def row_key(row):
    return scene_key(row['method'], row['cell'], (
        row['anatomy'], row['clusters'], row['seed'], row['flow_inlet_mm_s'],
        row['latency_steps'], row['variation_s']))


def joint_change(main, reference):
    delta = {k: main[k] - reference[k] if main[k] is not None and reference[k] is not None else None
             for k in ('strict_success', 'removal', 'wall_contact_s', 'route_cross_track_rmse_mm')}
    qualifies = all(delta[k] is not None and delta[k] >= -1e-12 for k in ('strict_success', 'removal'))
    qualifies &= all(delta[k] is not None and delta[k] <= 1e-12 for k in ('wall_contact_s', 'route_cross_track_rmse_mm'))
    return dict(jointly_nondegrading=qualifies, deltas=delta)


def audit(verify_media=False):
    errors = []
    pending = []
    config = read(RUN / 'config.json', {})
    training = read(RUN / 'TRAINING_COMPLETE.json', {})
    hashes = {source: (ROOT / source).exists() and hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
              for source, digest in config.get('source_sha256', {}).items()}
    if not hashes or not all(hashes.values()):
        errors.append('Training source hash mismatch or missing hashes')
    if training.get('actor_gradient_updates', 0) <= 0:
        errors.append('No recorded RL actor updates')
    training_traces = list((RUN / 'training_traces').glob('*.npz'))
    verified_training = read(VALIDATION / 'TRAINING_VERIFIED.json', {})
    if len(training_traces) != verified_training.get('completed_training_episodes'):
        errors.append('Completed training episode / trace coverage mismatch')
    multi_execution = read(VALIDATION/'execution_verification_N3/VERIFIED.json', {})
    if (VALIDATION/'execution_verification_N3/FAILED.json').exists():
        errors.append('N=3 attribution verification failed; inspect physical/control array differences')
    elif not multi_execution.get('attribution_does_not_change_physics'):
        pending.append('Frozen N=3 spacing-filter attribution verification')
    selection = read(VALIDATION / 'SELECTED.json') or read(VALIDATION / 'DIAGNOSTIC_SELECTED.json')
    tune = ['parent'] + [k + '=' + str((RUN / f'policy_it{i}.pt').relative_to(ROOT))
                        for k in ('sac_finite', 'sac_finite_stochastic') for i in (40, 80, 160, 240)]
    stages = [('tune', 'tune', 49000000, tune, False)]
    main = None
    frozen = {}
    if selection:
        import torch
        checkpoint = ROOT / selection['checkpoint']
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        frozen = dict(checkpoint=selection['checkpoint'], method_kind=selection['method_kind'],
                      actor_gradient_updates=payload.get('actor_gradient_updates'),
                      jointly_admissible=selection['jointly_admissible'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['sha256']:
            errors.append('Frozen checkpoint hash mismatch')
        if not payload.get('trained_with_rl') or payload.get('actor_gradient_updates', 0) <= 0:
            errors.append('Frozen actor was not trained with RL')
        if selection.get('sealed_test_accessed') is not False:
            errors.append('Selection does not confirm sealed-test exclusion')
        kind = selection['method_kind']
        main = kind + '=' + selection['checkpoint']
        initial = kind + '=' + str((RUN / 'initial.pt').relative_to(ROOT))
        standard = ['parent', 'old_ppo', initial, main]
        stages = [('tune', 'tune', 49000000, tune, False),
                  ('full_hard', 'hard', 0, standard, True),
                  ('stress', 'stress', 50000000, standard, True),
                  ('ablation_no_images', 'stress', 50000000, [main], True),
                  ('ablation_no_pulse', 'stress', 50000000, [main], True),
                  ('ablation_contact_range', 'stress', 50000000, [main], True),
                  ('v5', 'v5', 0, ['parent', main], True)]
    else:
        pending.append('Frozen checkpoint selection')

    coverage = {}
    performance = {}
    for folder, pool, offset, methods, traces_required in stages:
        expected = {scene_key(method, cell, (an, n, seed + offset, flow, delay, strength))
                    for cell, (an, n, seed, flow, delay, strength) in scenes(pool) for method in methods}
        rows = records(VALIDATION / folder / 'episodes.jsonl')
        observed = Counter(row_key(row) for row in rows)
        duplicate = sum(n - 1 for n in observed.values())
        extra = set(observed) - expected
        missing = expected - set(observed)
        failures = records(VALIDATION / folder / 'failures.jsonl')
        if duplicate or extra or failures:
            errors.append(f'{folder}: duplicate={duplicate}, unexpected={len(extra)}, execution_errors={len(failures)}')
        if missing:
            pending.append(f'{folder}: {len(missing)} missing episodes')
        paired = defaultdict(set)
        noise_mismatch = 0
        for row in rows:
            k = row_key(row)
            paired[k[1:]].add(row['scenario_hash'])
            if row['method'].startswith('sac_finite_stochastic='):
                wanted = int(hashlib.sha256(json.dumps(['SAC_policy_noise_v1', k[2:]], sort_keys=True).encode()).hexdigest()[:8], 16)
                noise_mismatch += int(row.get('policy_rng_seed') != wanted
                                      or row.get('deployment_action_sampling') != 'seeded stochastic')
        mismatch = sum(len(hashes) > 1 for hashes in paired.values())
        if mismatch:
            errors.append(f'{folder}: {mismatch} mismatched paired scenario hashes')
        if noise_mismatch:
            errors.append(f'{folder}: {noise_mismatch} policy-noise protocol mismatches')
        valid_traces = 0
        if traces_required:
            trace_keys = Counter()
            for trace in (VALIDATION / folder / 'traces').glob('*.npz'):
                try:
                    with np.load(trace) as z:
                        row = json.loads(str(z['metrics']))
                        k = row_key(row)
                        trace_keys[k] += 1
                        count = len(z['t'])
                        required = ('pos', 'mass', 'wall', 'removal', 'sent', 'hold',
                                    'targets', 'reference_local', 'policy_local', 'flow_factor')
                        if count < 2 or not np.all(np.diff(z['t']) > 0):
                            raise ValueError('Invalid time axis')
                        if any(len(z[name]) != count or not np.all(np.isfinite(z[name])) for name in required):
                            raise ValueError('Invalid recorded physical/control arrays')
                        if abs(float(z['removal'][-1]) - row['removal']) > 1e-10:
                            raise ValueError('Final trace / metric clearance mismatch')
                        valid_traces += 1
                except Exception as exc:
                    errors.append(f'{trace}: {exc}')
            if set(trace_keys) - expected or any(n > 1 for n in trace_keys.values()):
                errors.append(f'{folder}: unexpected or duplicate trace keys')
            missing_traces = expected - set(trace_keys)
            if missing_traces:
                pending.append(f'{folder}: {len(missing_traces)} missing traces')
        coverage[folder] = dict(expected=len(expected), completed=len(observed), missing=len(missing),
                                duplicate=duplicate, execution_errors=len(failures),
                                scenario_hash_mismatches=mismatch, policy_noise_mismatches=noise_mismatch,
                                valid_traces=valid_traces)
        if not missing and main and 'parent' in methods and main in methods:
            groups = defaultdict(lambda: defaultdict(list))
            for row in rows:
                groups['all'][row['method']].append(row)
                groups[row['cell'].split('|')[0]][row['method']].append(row)
            performance[folder] = {}
            for condition, methods_rows in groups.items():
                primary = metrics(methods_rows[main])
                comparisons = {ref: joint_change(primary, metrics(methods_rows[ref]))
                               for ref in ('parent', initial) if ref in methods_rows}
                performance[folder][condition] = dict(n=primary['n'], main_metrics=primary,
                                                     comparisons=comparisons)

    progress = read(MEDIA / 'progress.json', {})
    if progress.get('pending', 1):
        pending.append(f"GIFs: {progress.get('pending', 'unknown')} remaining")
    media_checked = 0
    if verify_media:
        from PIL import Image
        tasks = read(MEDIA / 'tasks.json', [])
        expected_media = len(training_traces) + 1 + 4704
        if len(tasks) != expected_media:
            pending.append(f'Media registry: {len(tasks)} / {expected_media}')
        for task in tasks:
            gif = MEDIA / task['gif']
            meta = gif.with_suffix('.json')
            trace = gif.with_suffix('.npz')
            if not all(p.exists() for p in (gif, meta, trace)):
                pending.append(str(gif))
                continue
            try:
                details = read(meta)
                if details['state'] != 'complete' or details['id'] != task['id']:
                    raise ValueError('GIF provenance mismatch')
                source = Path(task['source'])
                if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(trace.read_bytes()).digest():
                    raise ValueError('Copied source trace hash mismatch')
                if details.get('reference_mismatch'):
                    raise ValueError('Replay/reference metrics mismatch')
                with Image.open(gif) as im:
                    if im.n_frames < 2:
                        raise ValueError('Static GIF')
                    im.seek(im.n_frames - 1)
                    im.load()
                media_checked += 1
            except Exception as exc:
                errors.append(f'{gif}: {exc}')
    result = dict(updated=time.strftime('%F %T'), complete=not errors and not pending and verify_media,
                  errors=errors, pending=pending, training=training, source_hash_checks=hashes,
                  training_trace_files=len(training_traces), frozen=frozen, coverage=coverage,
                  multirobot_execution_verification=multi_execution,
                  performance=performance, media=progress, media_verified=media_checked,
                  sealed_test_accessed=False, mechanical_fragmentation_modeled=False,
                  observation_assumptions=dict(
                      robot_motion='Delayed biplane pixel detection; initial operator acquisition simulated',
                      clot_presence='Simulator mass > 0 flag, not independently image-segmented',
                      robot_active='Simulator active flag',
                      actor_presence_context='Context index 47 contains the mean clot-presence flag',
                      fully_pixel_derived_task_state=False),
                  interpretation='Coverage and provenance audit; paired performance is descriptive, not multi-seed confirmation.')
    target = VALIDATION / ('DELIVERY_VERIFIED.json' if result['complete'] else 'DELIVERY_AUDIT_CURRENT.json')
    temp = target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temp.replace(target)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-media', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    result = audit(args.verify_media)
    print(json.dumps({k: result[k] for k in ('complete', 'errors', 'pending', 'coverage', 'media_verified')},
                     ensure_ascii=False, indent=2))
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
