"""EXP0090: train / evaluate the action-conditioned motion predictor ensemble (marl.dyn_predictor).

train:    train_dyn_predictor.py train --data DIR --out DIR --seeds 0,1,2 [--epochs 6]
evaluate: train_dyn_predictor.py eval --data DIR --models m0.pt,m1.pt,m2.pt --out DIR
Evaluation compares, on val_train (training anatomies, new seeds) and val_heldout (held-out anatomies):
  constant velocity  displacement = v_img * (frame age + h dt)
  nominal dynamics   displacement = speed * (sent command * age + sum of planned commands * dt)
  online affine      v = gain * command * speed + drift identified online (marl.observed_motion; deployable)
  learned ensemble   (this model)
per horizon h in {0, 0.1, 0.3, 0.6, 1.2} s, by inlet speed, latency, wall clearance and dynamics strength;
calibration = fraction of |error| / sigma below 1 and 2 (Gaussian: 0.68 / 0.95), mean NLL.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from marl.dyn_predictor import CTX_DIM, H_FUT, H_HIST, HIST_DIM, HORIZONS, Predictor
from scripts.collect_dyn_data import COL

DT, SPEED = .1, 1.


def load(data, split):
    seqs, metas = [], []
    for f in sorted(glob.glob(str(Path(data)/f'{split}_*.npz'))):
        d = np.load(f); R = d['rows']; m = json.loads(str(d['meta'])); n = m['n']
        for i in range(n):
            seqs.append(R[i::n]); metas.append(m)
    return seqs, metas


def build(seqs, metas):
    """Concatenate sequences and build sample arrays (local frame)."""
    lens = np.array([len(s) for s in seqs]); off = np.r_[0, np.cumsum(lens)]
    A = np.concatenate(seqs); sid = np.repeat(np.arange(len(seqs)), lens); tt = np.concatenate([np.arange(L) for L in lens])
    ok = (A[:, COL['active']] > 0) & (A[:, COL['live']] > 0)
    idx = np.flatnonzero(ok)
    F = A[:, COL['F']].reshape(-1, 3, 3)
    hist = np.zeros((len(idx), H_HIST, HIST_DIM), np.float32); hmask = np.ones((len(idx), H_HIST), bool)
    for k in range(H_HIST):
        j = idx-(H_HIST-1-k); valid = (j >= 0) & (sid[np.clip(j, 0, None)] == sid[idx])
        jj = np.clip(j, 0, None); Fi = F[idx]
        hist[:, k, 0:3] = np.einsum('nab,nb->na', Fi, A[jj, COL['vel']])
        hist[:, k, 3:6] = np.einsum('nab,nb->na', Fi, A[jj, COL['sent']])
        hist[:, k, 6:9] = np.einsum('nab,nb->na', Fi, A[jj, COL['pos']]-A[idx, COL['pos']])
        hist[:, k, 9] = A[jj, COL['age']]/.3; hist[:, k, 10] = A[jj, COL['hold']]
        hist[~valid, k] = 0.; hmask[:, k] = ~valid
    plan = np.zeros((len(idx), H_FUT, 3), np.float32)
    for k in range(H_FUT):
        j = idx+k; valid = (j < len(A)) & (sid[np.clip(j, None, len(A)-1)] == sid[idx]); jj = np.clip(j, None, len(A)-1)
        plan[:, k] = np.einsum('nab,nb->na', F[idx], A[jj, COL['intended']])*valid[:, None]
    lab = np.zeros((len(idx), len(HORIZONS), 3), np.float32); lmask = np.zeros((len(idx), len(HORIZONS)), bool)
    for q, h in enumerate(HORIZONS):
        j = idx+h; jj = np.clip(j, None, len(A)-1)
        valid = (j < len(A)) & (sid[jj] == sid[idx]) & (A[jj, COL['active']] > 0)
        lab[:, q] = np.einsum('nab,nb->na', F[idx], A[jj, COL['truth']]-A[idx, COL['pos']]); lmask[:, q] = valid
    aux = np.concatenate([np.einsum('nab,nb->na', F[idx], A[idx, COL['flow']]), A[idx, COL['gain']][:, None]], 1)
    ctx = A[idx, COL['ctx']]
    cond = np.array([[metas[s]['flow'], metas[s]['latency'], metas[s]['s']] for s in sid[idx]], np.float32)
    clear = ctx[:, 3]*.5
    return dict(hist=hist, hmask=hmask, ctx=ctx, plan=plan, lab=lab, lmask=lmask, aux=aux.astype(np.float32),
                cond=cond, clear=clear, A=A, idx=idx, sid=sid, tt=tt, F=F)


def nll(mu, lv, y, m):
    e = ((mu-y)**2*torch.exp(-lv)+lv).sum(-1)*.5
    return (e*m).sum()/m.sum().clamp(min=1)


def train(a):
    seqs, metas = load(a.data, 'train'); D = build(seqs, metas); dev = a.device
    norm = float(np.sqrt((D['lab'][D['lmask']]**2).mean()))
    T = {k: torch.as_tensor(D[k]).to(dev) for k in ('hist', 'hmask', 'ctx', 'plan', 'lab', 'lmask', 'aux')}
    print('samples', len(D['idx']), 'label rms (mm)', norm, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); paths = []
    for seed in [int(x) for x in a.seeds.split(',')]:
        torch.manual_seed(seed); np.random.seed(seed)
        m = Predictor(a.d).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
        N = len(D['idx']); steps = a.epochs*(N//a.batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        log = []
        for ep in range(a.epochs):
            perm = torch.randperm(N, device=dev); tot = 0.
            for b in perm.split(a.batch)[:N//a.batch]:
                mu, lv, ax = m(T['hist'][b], T['hmask'][b], T['ctx'][b], T['plan'][b])
                l = nll(mu, lv, T['lab'][b]/norm, T['lmask'][b].float())+.1*((ax-T['aux'][b])**2).mean()
                opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); sched.step()
                tot += float(l)
            log.append(tot/(N//a.batch)); print(f'seed {seed} epoch {ep} loss {log[-1]:.4f}', flush=True)
        p = out/f'predictor_s{seed}.pt'
        torch.save(dict(state={k: v.cpu() for k, v in m.state_dict().items()}, d=a.d, norm=norm, seed=seed, loss=log,
                        samples=int(N), horizons=list(HORIZONS)), p)
        paths.append(str(p))
    (out/'train_manifest.json').write_text(json.dumps(dict(paths=paths, epochs=a.epochs, batch=a.batch, d=a.d, norm=norm,
                                                          data=str(a.data), samples=int(len(D['idx']))), indent=2))


def online_affine(D):
    """Replay marl.observed_motion.ObservedMotion on each sequence (deployable baseline); return displacement [S, H, 3]."""
    from marl.observed_motion import ObservedMotion
    A, idx, sid, tt, F = D['A'], D['idx'], D['sid'], D['tt'], D['F']
    gain = np.ones(len(A)); drift = np.zeros((len(A), 3)); rm = np.ones(len(A))
    for s in np.unique(sid):
        rows = np.flatnonzero(sid == s); om = ObservedMotion(1, dt=DT)
        for r in rows:
            now = tt[r]*DT
            try:
                om.update(now, A[r, COL['sent']][None], now-A[r, COL['age']], A[r, COL['pos']][None],
                          np.array([A[r, COL['active']] > 0]), np.zeros(1, int))
            except ValueError:
                pass
            gain[r] = om.gain[0]; drift[r] = om.drift[0]; rm[r] = om.rmse[0]
    g, dr = gain[idx], drift[idx]
    sent_l = np.einsum('nab,nb->na', F[idx], A[idx, COL['sent']]); dr_l = np.einsum('nab,nb->na', F[idx], dr)
    age = A[idx, COL['age']]
    lag = (g[:, None]*sent_l*SPEED+dr_l)*age[:, None]
    cum = np.cumsum((g[:, None, None]*D['plan']*SPEED+dr_l[:, None])*DT, axis=1)
    out = np.stack([lag+(cum[:, h-1] if h > 0 else 0.) for h in HORIZONS], 1)
    sig2 = (rm[idx]*DT)**2
    var = sig2[:, None, None]*np.array([max(h, 1) for h in HORIZONS])[None, :, None]+1e-4
    return out, np.broadcast_to(var, out.shape)


def evaluate(a):
    from marl.dyn_predictor import Ensemble
    ens = Ensemble(a.models.split(',')); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rep = {}
    for split in ('val_train', 'val_heldout'):
        seqs, metas = load(a.data, split); D = build(seqs, metas)
        A, idx, F = D['A'], D['idx'], D['F']; age = A[idx, COL['age']]
        v_l = np.einsum('nab,nb->na', F[idx], A[idx, COL['vel']]); sent_l = np.einsum('nab,nb->na', F[idx], A[idx, COL['sent']])
        cv = np.stack([v_l*(age+h*DT)[:, None] for h in HORIZONS], 1)
        cumplan = np.cumsum(D['plan']*SPEED*DT, axis=1)
        nom = np.stack([sent_l*SPEED*age[:, None]+(cumplan[:, h-1] if h > 0 else 0.) for h in HORIZONS], 1)
        oa, oa_var = online_affine(D)
        mus, vars_, epis = [], [], []
        for b in np.array_split(np.arange(len(idx)), max(1, len(idx)//4096)):
            mu, var, epi = ens.predict(D['hist'][b], D['hmask'][b], D['ctx'][b], D['plan'][b]); mus.append(mu); vars_.append(var); epis.append(epi)
        lm, lvar, lepi = np.concatenate(mus), np.concatenate(vars_), np.concatenate(epis)
        y, M = D['lab'], D['lmask']
        preds = dict(constant_velocity=(cv, None), nominal=(nom, None), online_affine=(oa, oa_var), learned=(lm, lvar+lepi))
        groups = dict(all=np.ones(len(idx), bool))
        for f in (.025, .05, .1):
            groups[f'flow_{f:g}'] = np.isclose(D['cond'][:, 0], f)
        for L in (1, 2, 3):
            groups[f'latency_{L}'] = D['cond'][:, 1] == L
        groups['s0'] = D['cond'][:, 2] == 0; groups['s>0'] = D['cond'][:, 2] > 0
        groups['clearance<0.1mm'] = D['clear'] < .1; groups['clearance>=0.1mm'] = D['clear'] >= .1
        R = {}
        for gname, g in groups.items():
            R[gname] = {}
            for name, (p, var) in preds.items():
                cell = {}
                for q, h in enumerate(HORIZONS):
                    sel = g & M[:, q]
                    if not sel.any():
                        continue
                    e = p[sel, q]-y[sel, q]
                    c = dict(rmse_mm=float(np.sqrt((e**2).sum(-1).mean())), n=int(sel.sum()))
                    if var is not None:
                        z = np.abs(e)/np.sqrt(var[sel, q])
                        c.update(within1=float((z < 1).mean()), within2=float((z < 2).mean()),
                                 nll=float((.5*(e**2/var[sel, q]+np.log(2*np.pi*var[sel, q]))).sum(-1).mean()))
                    cell[f'{h*DT:.1f}s'] = c
                R[gname][name] = cell
        rep[split] = R
        epi_std = np.sqrt(lepi[:, -1].sum(-1))
        rep[split+'_epistemic_std_1.2s'] = dict(p50=float(np.median(epi_std)), p90=float(np.percentile(epi_std, 90)),
                                                p95=float(np.percentile(epi_std, 95)), p99=float(np.percentile(epi_std, 99)))
    (out/'prediction_eval.json').write_text(json.dumps(rep, indent=2))
    lines = ['# EXP0090 prediction evaluation (RMSE mm of the true displacement from the newest delayed estimate)', '']
    for split in ('val_train', 'val_heldout'):
        lines += [f'## {split}', '', '| group | method | ' + ' | '.join(f'{h*DT:.1f}s' for h in HORIZONS) + ' | calib 1sigma/2sigma @1.2s |',
                  '|---|---|' + '---|'*(len(HORIZONS)+1)]
        for gname, R in rep[split].items():
            for name, cell in R.items():
                cal = cell.get('1.2s', {})
                lines.append(f'| {gname} | {name} | ' + ' | '.join(f"{cell[k]['rmse_mm']:.3f}" if k in cell else '-' for k in
                             [f'{h*DT:.1f}s' for h in HORIZONS]) + (f" | {cal['within1']:.2f}/{cal['within2']:.2f} |" if 'within1' in cal else ' | - |'))
        lines += ['', f"epistemic std @1.2 s: {rep[split+'_epistemic_std_1.2s']}", '']
    (out/'PREDICTION_REPORT.md').write_text('\n'.join(lines)); print('\n'.join(lines[:40]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('train', 'eval')); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--d', type=int, default=96); ap.add_argument('--device', default='cuda:0'); ap.add_argument('--models')
    a = ap.parse_args()
    train(a) if a.cmd == 'train' else evaluate(a)


if __name__ == '__main__':
    main()
