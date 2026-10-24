"""Keep saved delivery status factual while the existing long batches finish.

Never launches duplicate simulations or changes policy selection. The legacy
V5 restoration remains owned by resume_previous_v5_after_rl_media.py.
"""
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, os.sched_getaffinity(0) - {6, 7})
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

from collections import Counter
import fcntl
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT/'research/validation/EXP0099_FINITE_IMAGE_SAC_20261011'
DESKTOP = Path('/home/wj/桌面')
MEDIA = ['VRL_有限时域图像SAC_训练测试_3D_GIF_20261011',
         'VRL_V3_全部场景_3D_GIF_20261011',
         'VRL_RL_局部导航_训练测试_GIF_20261011']


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def legacy_audit(name):
    folder = DESKTOP/name
    tasks = read(folder/'tasks.json', [])
    errors = []
    seen = Counter(task['id'] for task in tasks)
    if any(count != 1 for count in seen.values()):
        errors.append('Duplicate media task identifiers')
    for task in tasks:
        gif = folder/task['gif']
        meta = gif.with_suffix('.json')
        trace = gif.with_suffix('.npz')
        if not all(path.exists() and path.stat().st_size > 0 for path in (gif, meta, trace)):
            errors.append(str(gif))
            continue
        record = read(meta)
        if record.get('id') != task['id'] or record.get('state') != 'complete' or record.get('frames', 0) < 2:
            errors.append(f'{gif}: incomplete or mismatched provenance')
        if record.get('reference_mismatch'):
            errors.append(f'{gif}: reference/replay mismatch')
    result = dict(updated=time.strftime('%F %T'), complete=bool(tasks) and not errors,
                  expected=len(tasks), media_verified=len(tasks)-len(errors), errors=errors,
                  validation='All task IDs, GIF/NPZ/JSON files and recorded render checks; original reference metrics where recorded.')
    save(folder/'DELIVERY_VERIFIED.json', result)
    return result


def main():
    os.chdir(ROOT)
    lock = (VALIDATION/'.delivery_monitor.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    from scripts.update_rl_overview import main as update_overview
    legacy_checked = {}
    while True:
        try:
            progress = {name: read(DESKTOP/name/'progress.json', {}) for name in MEDIA}
            state = read(VALIDATION/'CURRENT_STATUS.json', {})
            finite_verified = read(VALIDATION/'DELIVERY_VERIFIED.json', {})
            if (VALIDATION/'COMPLETE.json').exists() and not finite_verified.get('complete'):
                import importlib
                from scripts import audit_finite_rl_delivery
                finite_verified = importlib.reload(audit_finite_rl_delivery).audit(verify_media=True)
                # Regenerate with current reporting clarifications; controller
                # sources and the frozen selection remain untouched.
                from scripts.finish_finite_image_flow_sac import document, selection_record
                document(selection_record()['checkpoint'])
            for name in MEDIA[1:]:
                if progress[name].get('pending') == 0 and name not in legacy_checked:
                    legacy_checked[name] = legacy_audit(name)
            all_complete = bool(finite_verified.get('complete')) and all(
                legacy_checked.get(name, {}).get('complete', False) for name in MEDIA[1:])
            counts = {}
            for stage in ('tune', 'full_hard', 'stress', 'ablation_no_images', 'ablation_no_pulse', 'ablation_contact_range', 'v5'):
                source = VALIDATION/stage/'episodes.jsonl'
                counts[stage] = sum(bool(line.strip()) for line in source.open()) if source.exists() else 0
            result = dict(updated=time.strftime('%F %T'), all_requested_batches_complete=all_complete,
                          finite_stage=state.get('stage'), finite_episodes=counts,
                          media=progress, finite_delivery_verified=bool(finite_verified.get('complete')),
                          legacy_delivery_verified=legacy_checked)
            save(VALIDATION/'ALL_BATCHES_STATUS.json', result)
            update_overview()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if all_complete:
                save(VALIDATION/'ALL_REQUESTED_MEDIA_COMPLETE.json', result)
                return
        except Exception:
            print(traceback.format_exc(), flush=True)
        time.sleep(60)


if __name__ == '__main__':
    main()
