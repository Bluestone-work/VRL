"""Train the frontier selector (cross-entropy over candidates). Validation: held-out training-anatomy episodes
are not separated by anatomy here; a random 15 % of decisions is used for early checking only.
usage: train_frontier_selector.py --data DIR --out PT"""
import argparse, glob, json
import numpy as np, torch
from torch import nn
from marl.frontier_selector import FrontierSelector
p = argparse.ArgumentParser(); p.add_argument('--data'); p.add_argument('--out'); p.add_argument('--epochs', type=int, default=60); p.add_argument('--seed', type=int, default=0)
a = p.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
Xs, Ms, Ys = [], [], []
for f in sorted(glob.glob(f'{a.data}/*.npz')):
    z = np.load(f); n = z['X'].shape[1]; pad = 6-n
    Xs.append(np.pad(z['X'], ((0, 0), (0, pad), (0, 0)))); Ms.append(np.pad(z['mask'], ((0, 0), (0, pad)))); Ys.append(z['y'])
X, M, Y = np.concatenate(Xs), np.concatenate(Ms), np.concatenate(Ys)
idx = rng.permutation(len(X)); nv = len(X)//7; va, tr = idx[:nv], idx[nv:]
X, M, Y = torch.tensor(X), torch.tensor(M), torch.tensor(Y)
m = FrontierSelector(X.shape[-1]); flat = X[tr][M[tr]]; m.mu.copy_(flat.mean(0)); m.sd.copy_(flat.std(0).clamp_min(1e-3))
opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-4)
def run(ids, train):
    logits = m(X[ids]).masked_fill(~M[ids], -1e9); loss = nn.functional.cross_entropy(logits, Y[ids])
    if train:
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), (logits.argmax(-1) == Y[ids]).float().mean().item()
for ep in range(a.epochs):
    for b in np.array_split(rng.permutation(tr), max(1, len(tr)//512)):
        run(torch.tensor(b), True)
with torch.no_grad():
    vl, vacc = run(torch.tensor(va), False)
    # heuristic baseline accuracy on the same decisions: argmax cos_goal - 2 leaf - 3 dead - 3 tabu
    Xv, Mv, Yv = X[va], M[va], Y[va]
    h = (Xv[..., 0]-2*Xv[..., 3]-3*Xv[..., 4]-3*Xv[..., 5]).masked_fill(~Mv, -1e9)
    hacc = (h.argmax(-1) == Yv).float().mean().item()
print(json.dumps(dict(decisions=len(X), val_loss=vl, val_acc=vacc, heuristic_val_acc=hacc)))
torch.save(dict(state=m.state_dict(), n_feat=X.shape[-1], val_acc=vacc, heuristic_val_acc=hacc, decisions=len(X)), a.out)
