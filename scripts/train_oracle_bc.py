"""Behaviour cloning of the primitive oracle into the deployable VP policy (Oracle -> BC -> DAgger).

Input: VP observation only (biplane frames + token history). Loss: 0.3 x CE(oracle label) + 0.7 x CE(soft
target softmax(-(cost - min)/tau)) + 4 x expected oracle regret (probability mass on options the oracle scores
as clearly worse, e.g. predicted collisions). Episode-level train / validation split; the checkpoint with the
lowest validation loss (the training objective on held-out episodes) is kept (no dev-scene selection). Output is a VP-format checkpoint (scripts.train_vision_ppo).
usage: train_oracle_bc.py --data DIR [DIR ...] --out student.pt [--init prev.pt]
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', nargs='+', required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--init')
    p.add_argument('--epochs', type=int, default=15); p.add_argument('--bs', type=int, default=512); p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--tau', type=float, default=.1); p.add_argument('--balance', type=float, default=0., help='class-weight exponent (0 = none, 1 = inverse frequency)'); p.add_argument('--regret-w', type=float, default=4.)
    p.add_argument('--val-frac', type=float, default=.15); p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cuda:0')
    a = p.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    from marl.obstacle_control import token_dim
    from scripts.train_vision_ppo import FRAMES, FRAME_GAP, N_ACT, PERIOD, VisionStatePolicy, save
    files = sorted(f for d in a.data for f in glob.glob(str(Path(d)/'*.npz'))); idx = rng.permutation(len(files))
    nv = max(1, int(round(a.val_frac*len(files)))); val = [files[i] for i in idx[:nv]]; tr = [files[i] for i in idx[nv:]]

    def cat(fs):
        Z = [np.load(f) for f in fs]
        C = torch.nan_to_num(torch.as_tensor(np.concatenate([z['costs'] for z in Z])), posinf=1e4)
        return (torch.as_tensor(np.concatenate([z['img'] for z in Z])).float(), torch.as_tensor(np.concatenate([z['seq'] for z in Z])),
                torch.as_tensor(np.concatenate([z['mask'] for z in Z])), torch.as_tensor(np.concatenate([z['label'] for z in Z])).long(), C)
    Xi, Xs, Xm, Y, C = cat(tr); Vi, Vs, Vm, VY, VC = cat(val); dev = torch.device(a.device)
    soft = torch.softmax(-(C-C.min(1, keepdim=True).values)/a.tau, 1); reg = ((C-C.min(1, keepdim=True).values).clamp(max=200.)/100.)
    vsoft = torch.softmax(-(VC-VC.min(1, keepdim=True).values)/a.tau, 1); vreg = ((VC-VC.min(1, keepdim=True).values).clamp(max=200.)/100.)
    cnt = torch.bincount(Y, minlength=N_ACT).float().clamp(min=1.)
    cw = (cnt.sum()/(N_ACT*cnt))**a.balance; cw = cw/(cw[Y].mean())
    net = VisionStatePolicy(token_dim(False)).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=True), strict=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    print(json.dumps(dict(train_eps=len(tr), val_eps=len(val), train=len(Y), val=len(VY), label_share=np.round(np.bincount(Y.numpy(), minlength=N_ACT)/len(Y), 3).tolist())), flush=True)
    best = None; cfg = dict(dim=token_dim(False), actions=N_ACT, period=PERIOD, frames=FRAMES, frame_gap=FRAME_GAP, method='Oracle-BC-DAgger',
                            sensing='image', topo_pursuit=True, data=[str(d) for d in a.data], init=a.init, tau=a.tau)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(Y)); tot = 0.
        for s in range(0, len(Y), a.bs):
            b = perm[s:s+a.bs]; lp = torch.log_softmax(net(Xi[b].to(dev), Xs[b].to(dev), Xm[b].to(dev)), 1)
            wb = cw[Y[b]].to(dev)                      # class weights (--balance): rare avoidance labels count more
            loss = (-.3*(wb*lp.gather(1, Y[b].to(dev)[:, None])[:, 0]).mean()-.7*(wb*(soft[b].to(dev)*lp).sum(1)).mean()
                    + a.regret_w*(lp.exp()*reg[b].to(dev)).sum(1).mean())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step(); tot += float(loss)*len(b)
        sched.step(); net.eval(); P = []; vloss = 0.
        with torch.no_grad():
            for s in range(0, len(VY), 2048):
                lpv = torch.log_softmax(net(Vi[s:s+2048].to(dev), Vs[s:s+2048].to(dev), Vm[s:s+2048].to(dev)), 1).cpu()
                vloss += float((-.3*lpv.gather(1, VY[s:s+2048, None])[:, 0]-.7*(vsoft[s:s+2048]*lpv).sum(1)
                                + a.regret_w*(lpv.exp()*vreg[s:s+2048]).sum(1)).sum()); P.append(lpv.argmax(1))
        P = torch.cat(P); r = VC.gather(1, P[:, None])[:, 0]-VC.min(1).values
        v = dict(epoch=ep+1, loss=tot/len(Y), val_loss=vloss/len(VY), acc=float((P == VY).float().mean()), regret=float(r.mean()), catastrophic=float((r >= 50).float().mean()),
                 pred_share=np.round(np.bincount(P.numpy(), minlength=N_ACT)/len(P), 3).tolist())
        print(json.dumps(v), flush=True)
        # selection on the validation training objective. (v1 selected on argmax regret: the oracle's 2 s cost
        # hardly penalises lost progress, so an always-'stop' policy had the lowest regret and was selected.)
        if best is None or v['val_loss'] < best['val_loss']:
            best = v; a.out.parent.mkdir(parents=True, exist_ok=True); save(a.out, net.cpu(), dict(cfg, selected_epoch=ep+1, val=v)); net.to(dev)
    print(json.dumps(dict(best=best)), flush=True)


if __name__ == '__main__':
    main()
