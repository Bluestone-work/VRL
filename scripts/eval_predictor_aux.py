"""Accuracy of the predictor's auxiliary flow / gain head, by flow regime.

The displacement head was validated (EXP0090); the auxiliary head never was. It
matters now because a WAND-style feed-forward term subtracts an estimated
disturbance from the commanded action, so the estimate's error enters the command
directly. The head was trained with weight 0.1 alongside the displacement NLL and
is documented as diagnostic, so its accuracy is an open question, not a given.

Reports RMSE of the 3-D local-frame flow velocity (robot-speed units) and of the
effective response gain, split by the inlet flow the episode was generated with,
against two references: predicting the dataset mean, and predicting zero flow.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
if hasattr(os, 'sched_getaffinity'):
    _a = os.sched_getaffinity(0) - {6, 7}
    if _a: os.sched_setaffinity(0, _a)
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARCHIVE = Path('/home/wj/VRL_archive_20261016/research/EXP0090_20261011')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ARCHIVE / 'data')
    ap.add_argument('--split', default='val_heldout')
    ap.add_argument('--predictors', default=','.join(str(ARCHIVE / 'predictor' / f'predictor_s{i}.pt')
                                                     for i in range(3)))
    ap.add_argument('--max-files', type=int, default=60)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    from scripts.train_dyn_predictor import build, load
    from marl.dyn_predictor import Ensemble
    seqs, metas = load(a.data, a.split)
    if not seqs:
        raise SystemExit(f'no {a.split} files under {a.data}')
    if a.max_files and len(seqs) > a.max_files:
        seqs, metas = seqs[:a.max_files], metas[:a.max_files]
    D = build(seqs, metas)
    ens = Ensemble([p for p in a.predictors.split(',')])
    hist = torch.as_tensor(D['hist']); hmask = torch.as_tensor(D['hmask'])
    ctx = torch.as_tensor(D['ctx']); plan = torch.as_tensor(D['plan'])
    aux_true = np.asarray(D['aux'], float)            # [N, 4] = flow(3) + gain(1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(hist), 4096):
            sl = slice(i, i + 4096)
            ens.predict(hist[sl], hmask[sl], ctx[sl], plan[sl])
            preds.append(np.asarray(ens.last_aux, float))
    aux_hat = np.concatenate(preds)[:len(aux_true)]
    flow_t, flow_p = aux_true[:, :3], aux_hat[:, :3]
    gain_t, gain_p = aux_true[:, 3], aux_hat[:, 3]
    cond = np.asarray(D['cond'], float)               # [:,0] = inlet flow
    def rmse(e):
        return float(np.sqrt((e ** 2).sum(-1).mean())) if e.ndim > 1 else float(np.sqrt((e ** 2).mean()))
    lines = [f'# Auxiliary head accuracy ({a.split}, n={len(flow_t)})', '',
             '向量流速 RMSE 单位 = robot-speed (1.0 = 执行器单位速度)', '',
             '| 组 | n | 流速 RMSE(预测) | 预测均值 | 预测零 | 增益 RMSE(预测) | 增益 预测均值 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    groups = [('all', np.ones(len(flow_t), bool))]
    for f in sorted(set(np.round(cond[:, 0], 4))):
        groups.append((f'inlet {f:g} mm/s', np.isclose(cond[:, 0], f)))
    for name, m in groups:
        if m.sum() < 20:
            continue
        ft, fp = flow_t[m], flow_p[m]
        gt, gp = gain_t[m], gain_p[m]
        lines.append(f'| {name} | {int(m.sum())} | {rmse(ft - fp):.4f} | '
                     f'{rmse(ft - ft.mean(0)):.4f} | {rmse(ft):.4f} | '
                     f'{rmse(gt - gp):.4f} | {rmse(gt - gt.mean()):.4f} |')
    mag_t = np.linalg.norm(flow_t, axis=-1)
    mag_p = np.linalg.norm(flow_p, axis=-1)
    corr = float(np.corrcoef(mag_t, mag_p)[0, 1])
    cos = float(np.mean(np.sum(flow_t * flow_p, -1) /
                        np.maximum(np.linalg.norm(flow_t, axis=-1) * np.linalg.norm(flow_p, axis=-1), 1e-9)))
    lines += ['', f'- 流速**大小**相关 r = {corr:+.3f}',
              f'- 流速**方向**平均余弦相似度 = {cos:+.3f}',
              f'- 真实流速大小: 均值 {mag_t.mean():.4f}, 中位 {np.median(mag_t):.4f}, 90 分位 {np.percentile(mag_t,90):.4f}',
              f'- 预测流速大小: 均值 {mag_p.mean():.4f}, 中位 {np.median(mag_p):.4f}']
    text = '\n'.join(lines)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n')


if __name__ == '__main__':
    main()
