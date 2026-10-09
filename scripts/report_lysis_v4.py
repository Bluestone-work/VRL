"""Build the v4 baseline/ablation report tables from research/validation/V4_LYSIS_20261008/*.jsonl.

Groups (only files that exist are included):
  main       ours, turbo, pac_nmpc, stpg, classical_settle (strongest classical)
  ablation   ours vs ours minus a component (allocator, residual, TPG, settling, wall guard, topology fix)
usage: report_lysis_v4.py [--out research/validation/V4_LYSIS_20261008/TABLES.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.summarize_lysis import load, table

D = Path('research/validation/V4_LYSIS_20261008')
MAIN = ['ours', 'turbo', 'pac_nmpc', 'stpg', 'classical_settle', 'ours_classical']
ABL = ['ours', 'ours_alloc_only', 'ours_low_only', 'ours_low_guardprior', 'ours_low_guardprior_post', 'ours_low_gru', 'ours_low_mlp', 'ours_no_tpg',
       'classical_settle', 'classical_settle_notpg_park', 'classical_settle_help',
       'alloc_auction_settle', 'classical_settle_notopo', 'ours_classical', 'pursuit_tpg', 'ours_classical_help',
       'ours_classical_notpg']


def section(title, names, ref):
    files = [D/f'{m}.jsonl' for m in names if (D/f'{m}.jsonl').exists()]
    if not files:
        return f'\n## {title}\n\n(no results yet)\n'
    rows = [r for r in load(files) if r['method'] in names]
    have = sorted({r['method'] for r in rows})
    ref = ref if ref in have else have[0]
    return f'\n## {title}\n\nmethods: {", ".join(have)}; paired reference: {ref}\n' + table(rows, ref)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=D/'TABLES.md'); a = p.parse_args()
    s = '# Benchmark v4 tables (development split, 14 anatomies x 10 scenes, image sensing)\n'
    s += section('Main comparison', MAIN, 'ours')
    s += section('Ablations', ABL, 'ours')
    s += section('Actuation-mismatch stress (gain U[0.7,1.3], noise 0.05)', ['classical_settle', 'stress_classical_settle',
                                                                            'stress_ours_low_only'], 'classical_settle')
    a.out.write_text(s+'\n'); print(s)


if __name__ == '__main__':
    main()
