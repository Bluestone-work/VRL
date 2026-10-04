"""Behaviour cloning of the fair local student (marl.local_student) on collect_cluster_bc data.
Loss: 1-cos on teacher-moving active clusters + BCE(stop, pos_weight). usage: train_local_bc.py --data DIR --val DIR --out DIR
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from marl.local_student import LocalStudent, save_local


def load(d):
    nav, slot, loc, stop = [], [], [], []
    for f in sorted(glob.glob(f'{d}/*.npz')):
        z = np.load(f)
        act = z['d_robot_active'].reshape(-1).astype(bool)
        nav.append(z['nav'].reshape(-1, z['nav'].shape[-1])[act]); slot.append(z['target_slot'].reshape(-1)[act])
        loc.append(z['teacher_local'].reshape(-1, 3)[act]); stop.append(z['teacher_stop'].reshape(-1)[act])
    return [np.concatenate(x) for x in (nav, slot, loc, stop)]


def step_loss(model, nav, slot, loc, stop, pw):
    d, s = model(nav, slot)
    norm = loc.norm(dim=-1); mv = (~stop) & (norm > 1e-9)
    cos = nn.functional.cosine_similarity(d, loc, dim=-1)
    l_dir = (1-cos)[mv].mean()
    l_stop = nn.functional.binary_cross_entropy_with_logits(s, stop.float(), pos_weight=torch.tensor(pw, device=s.device))
    return l_dir+l_stop, cos[mv].mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--val', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--stop-weight', type=float, default=5.); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); torch.manual_seed(a.seed); a.out.mkdir(parents=True, exist_ok=False)
    tr = [torch.as_tensor(x, device=a.device) for x in load(a.data)]
    va = [torch.as_tensor(x, device=a.device) for x in load(a.val)]
    tr[0], va[0] = tr[0].float(), va[0].float(); tr[2], va[2] = tr[2].float(), va[2].float()
    tr[3], va[3] = tr[3].bool(), va[3].bool(); tr[1], va[1] = tr[1].long(), va[1].long()
    model = LocalStudent().to(a.device); opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    n = len(tr[0]); steps = a.epochs*((n+a.batch-1)//a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    g = torch.Generator(device='cpu').manual_seed(a.seed); log = []
    for ep in range(a.epochs):
        perm = torch.randperm(n, generator=g).to(a.device)
        for i in range(0, n, a.batch):
            idx = perm[i:i+a.batch]
            loss, _ = step_loss(model, *(x[idx] for x in tr), a.stop_weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            vl, vc = step_loss(model, *va, a.stop_weight)
        log.append(dict(epoch=ep, val_loss=vl.item(), val_cos=vc)); print(json.dumps(log[-1]), flush=True)
    save_local(model, a.out/'student.pt', train_samples=n, val_samples=len(va[0]), log=log, args={k: str(v) for k, v in vars(a).items()})


if __name__ == '__main__':
    main()
