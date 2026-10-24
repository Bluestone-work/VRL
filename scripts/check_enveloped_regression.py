"""Recorded replay of the observed upstream regression; not a selection matrix."""
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, os.sched_getaffinity(0) - {6, 7})
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import json
import multiprocessing as mp
from pathlib import Path

from scripts.eval_image_flow_sac import ROOT, work


def main():
    os.chdir(ROOT)
    output = ROOT/'research/validation/EXP0098_ENVELOPED_IMAGE_SAC_20261011/diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = 'research/runs/EXP0098_ENVELOPED_IMAGE_SAC_20261011/seed0/policy_it20.pt'
    specs = [(method, 'upstream_regression|train', ('mca_m1_lvo', 1, 2179000001, .05, 1, 0.), {}, str(output/'traces'), True)
             for method in ('parent', 'sac_enveloped='+checkpoint)]
    with mp.get_context('spawn').Pool(2) as pool:
        results = pool.map(work, specs)
    (output/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    for row in results:
        print(json.dumps({key: row[key] for key in ('method', 'strict_success', 'removal', 'wall_contact_s', 'elapsed_s')}, ensure_ascii=False))
    from scripts.report_image_flow_sac import report
    report([output/'episodes.jsonl'], output/'report')
    (output/'LIMITATION.md').write_text('这是修复已知逆流减速故障的单场景回归诊断，不是独立验证或模型选择依据；不能用它声称全场景共同改善。完整评估按 EXP0098 预登记协议进行。\n')


if __name__ == '__main__':
    main()
