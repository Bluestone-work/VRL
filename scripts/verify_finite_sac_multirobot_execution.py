"""Verify that action-attribution measurement leaves a frozen N=3 rollout unchanged.

Use one fixed original-paper MCA case, including the spacing filter. The
attribution-enabled rollout is the existing full-hard matrix trace; this script
only replays it with attribution disabled. It never selects on this case.
"""
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, os.sched_getaffinity(0) - {6, 7})
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import hashlib
import json
from pathlib import Path
import time
import numpy as np

from scripts.eval_image_flow_sac import ROOT, scenes, work

V = ROOT/'research/validation/EXP0099_FINITE_IMAGE_SAC_20261011'
OUT = V/'execution_verification_N3'


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    import fcntl
    lock = (OUT/'.verify.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT/'VERIFIED.json').exists():
        return
    cell, job = next((cell, job) for cell, job in scenes('hard')
                     if cell.startswith('low_delay') and job[0] == 'mca_m1_lvo' and job[1] == 3)
    print('Fixed N=3 verification case:', cell, job, flush=True)
    while True:
        chosen_path = V/'SELECTED.json'
        if not chosen_path.exists():
            chosen_path = V/'DIAGNOSTIC_SELECTED.json'
        if chosen_path.exists():
            chosen = json.loads(chosen_path.read_text())
            method = chosen['method_kind']+'='+chosen['checkpoint']
            uid = hashlib.sha256(json.dumps([method, cell, job], sort_keys=True).encode()).hexdigest()[:16]
            name = f'{job[0]}_N3_seed{job[2]}_{uid}.npz'
            actual = V/'full_hard/traces'/name
            log = V/'full_hard/episodes.jsonl'
            recorded = False
            if log.exists():
                try:
                    for line in log.open():
                        row = json.loads(line)
                        if row['method'] == method and row['cell'] == cell and row['seed'] == job[2] and row['clusters'] == 3:
                            recorded = True
                            break
                except json.JSONDecodeError:
                    pass
            if recorded and actual.exists():
                break
        time.sleep(30)
    checkpoint = ROOT/chosen['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != chosen['sha256']:
        raise RuntimeError('Frozen checkpoint changed')
    reference = work((method, cell, job, {}, str(OUT/'traces_False'), False))
    replay = OUT/'traces_False'/name
    with np.load(actual) as enabled, np.load(replay) as disabled:
        keys = ('t', 'pos', 'mass', 'wall', 'removal', 'sent', 'hold', 'flow_factor',
                'targets', 'reference_local', 'policy_local')
        equality = {key: np.array_equal(enabled[key], disabled[key]) for key in keys}
        primary = json.loads(str(enabled['metrics']))
    result = dict(case=dict(cell=cell, job=job), method=method, equality=equality,
                  attribution_does_not_change_physics=all(equality.values()),
                  includes_multirobot_spacing_filter=True,
                  sources=dict(attribution_enabled=str(actual), attribution_disabled=str(replay)),
                  scenario_hash=reference['scenario_hash'], policy_rng_seed=reference.get('policy_rng_seed'),
                  strict_success=reference['strict_success'], removal=reference['removal'],
                  wall_contact_s=reference['wall_contact_s'],
                  selected_before_this_verification=True,
                  updated=time.strftime('%F %T'))
    if reference['scenario_hash'] != primary['scenario_hash'] or not all(equality.values()):
        (OUT/'FAILED.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        raise RuntimeError('Attribution changed the N=3 physical/control trace')
    (OUT/'VERIFIED.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
