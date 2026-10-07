"""Distil the privileged lookahead teacher into a deployable temporal option policy.

Loss: (1-alpha) * CE(hard teacher label) + alpha * CE(softmax(-(cost-min)/tau)) — the soft target tells the
student which options are nearly as good as the argmax and which are catastrophic (cost >= 100 = a predicted
obstacle collision). Episodes, not samples, are split into train / validation so validation measures
generalisation to unseen layouts. The checkpoint is selected on validation teacher-cost regret
(cost of the student's argmax minus the teacher's minimum), the quantity closed-loop safety depends on.
Inputs are deployable tokens only; costs are used for the loss and never enter the network.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from marl.obstacle_control import DiscreteTemporalPolicy, save_discrete_checkpoint, token_dim


def load(dirs):
    eps = []
    for d in dirs:
        for f in sorted(Path(d).glob('*.npz')):
            z = np.load(f, allow_pickle=False)
            if 'costs' not in z.files:
                continue
            eps.append((f.stem, z['seq'], z['mask'], z['label'], z['costs'], int(z['period']) if 'period' in z.files else 1))
    return eps


def cat(eps):
    return (torch.as_tensor(np.concatenate([e[1] for e in eps])), torch.as_tensor(np.concatenate([e[2] for e in eps]), dtype=torch.bool),
            torch.as_tensor(np.concatenate([e[3] for e in eps]), dtype=torch.long), torch.as_tensor(np.concatenate([e[4] for e in eps]), dtype=torch.float32))


def evaluate(net, X, M, Y, C, bs=2048):
    net.eval(); P = []; dev = next(net.parameters()).device
    with torch.no_grad():
        for s in range(0, len(Y), bs):
            P.append(net.logits(X[s:s+bs].to(dev), M[s:s+bs].to(dev)).argmax(-1).cpu())
    P = torch.cat(P); regret = C.gather(1, P[:, None])[:, 0]-C.min(1).values
    acc = float((P == Y).float().mean()); rec = [float((P[Y == k] == k).float().mean()) for k in range(8) if (Y == k).any()]
    return dict(acc=acc, macro_recall=float(np.mean(rec)), regret=float(regret.mean()),
                catastrophic=float((regret >= 50).float().mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arch', choices=('transformer', 'gru', 'mlp'), default='transformer')
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--bs', type=int, default=512); p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha', type=float, default=.7); p.add_argument('--tau', type=float, default=5.)
    p.add_argument('--val-frac', type=float, default=.15); p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cpu')
    p.add_argument('--init', help='warm-start checkpoint (DAgger rounds)')
    p.add_argument('--cost-w', type=float, default=0., help='weight of the expected teacher-regret term')
    p.add_argument('--tie-pref', type=int, help='option index preferred among exactly tied best options')
    a = p.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    eps = load(a.data); idx = rng.permutation(len(eps)); nv = max(1, int(round(a.val_frac*len(eps))))
    val = [eps[i] for i in idx[:nv]]; tr = [eps[i] for i in idx[nv:]]
    Xt, Mt, Yt, Ct = cat(tr); Xv, Mv, Yv, Cv = cat(val)
    if a.tie_pref is not None:
        # the teacher's argmin breaks exact ties by the lowest index ('pursuit'); 72 % of v2 decisions have >= 2
        # options tied at the best cost. Relabel ties to the preferred option when it is among the tied best.
        for Y_, C_ in ((Yt, Ct), (Yv, Cv)):
            C2 = torch.nan_to_num(C_, posinf=1e4); tied = (C2-C2.min(1, keepdim=True).values).abs() < 1e-6
            Y_[tied[:, a.tie_pref]] = a.tie_pref
    soft = torch.softmax(-(Ct-Ct.min(1, keepdim=True).values)/a.tau, 1)
    acts = Ct.shape[1]; period = {e[5] for e in eps}; assert len(period) == 1, 'mixed decision periods'; period = period.pop()
    Ct = torch.nan_to_num(Ct, posinf=1e4); Cv = torch.nan_to_num(Cv, posinf=1e4)
    soft = torch.softmax(-(Ct-Ct.min(1, keepdim=True).values)/a.tau, 1)
    dim = token_dim(False); net = DiscreteTemporalPolicy(a.arch, dim=dim, actions=acts).to(a.device)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=True), strict=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    cfg = dict(arch=a.arch, dim=dim, actions=acts, period=period, obs_vel=False, window=16, teacher='lookahead_teacher', alpha=a.alpha, tau=a.tau, cost_w=a.cost_w, tie_pref=a.tie_pref,
               train_episodes=[e[0] for e in tr], val_episodes=[e[0] for e in val], init=a.init)
    best = None; a.out.parent.mkdir(parents=True, exist_ok=True); log = []
    print(json.dumps(dict(train_samples=len(Yt), val_samples=len(Yv), train_eps=len(tr), val_eps=len(val))), flush=True)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Yt)); tot = 0.
        for s in range(0, len(Yt), a.bs):
            b = perm[s:s+a.bs]; lg = net.logits(Xt[b].to(a.device), Mt[b].to(a.device))
            lp = torch.log_softmax(lg, 1)
            loss = (1-a.alpha)*torch.nn.functional.nll_loss(lp, Yt[b].to(a.device))-a.alpha*(soft[b].to(a.device)*lp).sum(1).mean()
            if a.cost_w > 0:   # expected teacher regret under the policy: mass on collision options costs ~100x
                reg = (Ct[b]-Ct[b].min(1, keepdim=True).values).clamp(max=200.).to(a.device)/100.
                loss = loss+a.cost_w*(lp.exp()*reg).sum(1).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step(); tot += float(loss)*len(b)
        sched.step(); v = evaluate(net, Xv, Mv, Yv, Cv); v.update(epoch=ep+1, loss=tot/len(Yt)); log.append(v)
        print(json.dumps(v), flush=True)
        if best is None or v['regret'] < best['regret']:
            best = v; save_discrete_checkpoint(a.out, net, dict(cfg, selected_epoch=ep+1, val=v))
    Path(str(a.out)+'.log.json').write_text(json.dumps(log, indent=1))
    print(json.dumps(dict(best=best)), flush=True)


if __name__ == '__main__':
    main()
