"""Report the complete frozen speed-projection check; never promote a partial matrix."""
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, os.sched_getaffinity(0) - {6, 7})
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
import time

from scripts.finish_image_flow_sac import ROOT, rows, run

V = ROOT/'research/validation/EXP0096_SAFE_IMAGE_SAC_20261011'
R = ROOT/'research/runs/EXP0096_SAFE_IMAGE_SAC_20261011/seed0'


def main():
    import fcntl
    lock = (V/'.projection_finish.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while len(rows(V/'tune_speed_projection/episodes.jsonl')) < 864:
        if rows(V/'tune_speed_projection/failures.jsonl'):
            raise RuntimeError('Projection evaluation has failures; retain them and inspect')
        time.sleep(30)
    actual = rows(V/'tune_speed_projection/episodes.jsonl')
    if len(actual) != 864:
        raise RuntimeError('Unexpected projection coverage')
    parent = [r for r in rows(V/'tune/episodes.jsonl') if r['method'] == 'parent']
    target = V/'speed_projection_parent.jsonl'
    target.write_text(''.join(json.dumps(r) + '\n' for r in parent))
    from scripts.report_image_flow_sac import report
    summary = report([target, V/'tune_speed_projection/episodes.jsonl'], V/'tune_speed_projection_report')
    if any(r['n'] != 108 for r in summary['results'].values()):
        raise RuntimeError('Unequal projection coverage')
    baseline = summary['results']['parent']
    admissible = {name: r for name, r in summary['results'].items() if name != 'parent'
        and r['strict_success'] >= baseline['strict_success'] - 1e-12
        and r['removal'] >= baseline['removal'] - 1e-12
        and r['wall_contact_s'] <= baseline['wall_contact_s'] + 1e-12
        and r['route_cross_track_rmse_mm'] <= baseline['route_cross_track_rmse_mm'] + 1e-12}
    record = dict(evaluated=864, paired_training_anatomy_cases=108, admissible=admissible,
                  one_dimensional_training=False, description='Frozen three-dimensional actor with speed-only execution projection',
                  sealed_test_accessed=False, updated=time.strftime('%F %T'))
    (V/'SPEED_PROJECTION_COMPLETE.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    if not admissible:
        (V/'NO_JOINT_SPEED_PROJECTION.json').write_text(json.dumps(summary['results'], ensure_ascii=False, indent=2))
        run('scripts.export_image_flow_sac_media', ['--out', '/home/wj/桌面/VRL_安全图像SAC_训练测试_3D_GIF_20261011',
            '--run', str(R), '--validation', str(V), '--workers', '4', '--recorded-only'], V/'media_archive.log')
        (V/'CURRENT_STATUS.json').write_text(json.dumps(dict(stage='interim_unqualified_and_projection_archived',
            updated=time.strftime('%F %T'),full_matrix_executed=False,projection=record), ensure_ascii=False, indent=2))
    run('scripts.update_rl_overview', [], V/'overview.log')
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
