"""VP-PPO: visual-state world-model-guided PPO over local motion primitives (obstacle benchmark v3).

Deployable actor input (history, so motion of obstacles and of the cluster itself is observable):
  images   biplane microscope crops around the tracked cluster (top x-y, side x-z; 1.6 mm field, 32x32 px),
           obstacles rendered as absorbers; 4 frames at t, t-0.2, t-0.4, t-0.6 s -> 8 channels
  state    16-step history (1.6 s) of the structured token: route direction from the pre-operative map,
           own velocity estimate, previous command, map clearance / wall direction, the 6 nearest detected
           obstacles (partial ground truth: true local positions and sizes + noise, misses, latency; no
           identities, no velocities) and peers. The heuristic's command is NOT in the token (route direction
           is passed in its slot).
Actions: 7 primitives of marl.lookahead_teacher.PRIMITIVE_NAMES, one per 0.5 s (no embedded avoidance
  algorithm; heuristics are baselines only). Output bias favours 'advance' (route following) at start.
Optional initialisation (--bc): behaviour cloning on privileged primitive-teacher labels (soft targets + expected
  teacher regret), then PPO with a decaying KL(pi || pi_BC) term (--kl-bc) so fine-tuning starts from the teacher.
Training-only signals: asymmetric critic with privileged truth; world-model head predicting, 1 s ahead, the
  relative positions of the 3 currently nearest obstacles, own displacement, min true gap and collision.
Reward per decision (simulator outcomes only), scaled by a running std of the discounted return:
  +1.0 x route progress (mm, clip 0.2 / step) +50 x removed mass fraction -20 x new obstacle event
  -10 x obstacle contact s -2 x near-miss s -10 x wall contact s (-20 once when wall total crosses 1 s)
  -50 x lost -0.004 / step ; terminal +50 success, +50 more if Safe, -50 if not all clots cleared.
DR per episode: actuation gain U[.75,1.25]. Train anatomies only, N=1, seeds 2620000000+.
usage: train_vision_ppo.py --out DIR --steps 3000000 --workers 20
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PERIOD, N_ACT, FRAMES, FRAME_GAP = 5, 7, 4, 2
K_PRED, H_PRED = 3, 2
PRIV_DIM = 6*8+4
AUX_DIM = 3*K_PRED+3+2
REW = dict(wall=10., wall1=20.)     # wall-contact weight per s and one-off penalty at 1 s (set per run)


class VisionStatePolicy(nn.Module):
    """CNN over stacked biplane frames + GRU over token history -> fused latent -> primitive logits."""
    def __init__(self, tok_dim, d=128, actions=N_ACT):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(2*FRAMES, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(), nn.Flatten(), nn.Linear(64*16, d), nn.LayerNorm(d))
        self.tok = nn.Sequential(nn.Linear(tok_dim, d), nn.LayerNorm(d), nn.GELU()); self.gru = nn.GRU(d, d, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(2*d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.pi = nn.Linear(256, actions); nn.init.zeros_(self.pi.weight); nn.init.zeros_(self.pi.bias)

    def latent(self, img, seq, mask):
        x = self.tok(seq).masked_fill(mask[..., None], 0.); h = self.gru(x)[0][:, -1]
        return self.fuse(torch.cat([self.cnn(img), h], -1))

    def forward(self, img, seq, mask):
        return self.pi(self.latent(img, seq, mask))


def save(path, net, cfg):
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, path)
    Path(str(path)+'.json').write_text(json.dumps(cfg, sort_keys=True))


def load(path):
    cfg = json.loads(Path(str(path)+'.json').read_text())
    net = VisionStatePolicy(int(cfg['dim'])); net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    net.eval(); return net, cfg


class Frames:
    def __init__(self):
        self.buf = []

    def push(self, crops):
        self.buf.append(crops[0].copy()); self.buf = self.buf[-(FRAME_GAP*(FRAMES-1)+1):]
        idx = [max(len(self.buf)-1-FRAME_GAP*j, 0) for j in range(FRAMES)]
        return np.concatenate([self.buf[i] for i in idx], 0)[None]          # [1, 2*FRAMES, 32, 32]


def priv_features(ep, pos, prevP, dtP):
    P, R, _ = ep.field.positions(); f = np.zeros(PRIV_DIM, np.float32); body = ep.ctl.body
    if len(R):
        g = np.linalg.norm(P-pos, axis=1)-body-R; idx = np.argsort(g)[:6]
        V = (P-prevP)/dtP if prevP is not None and prevP.shape == P.shape else np.zeros_like(P)
        for j, k in enumerate(idx):
            f[8*j:8*j+3] = (P[k]-pos)/3.; f[8*j+3] = R[k]/.5; f[8*j+4] = g[k]; f[8*j+5:8*j+8] = np.clip(V[k], -3, 3)
    env = ep.env; t = env.transport
    f[48] = float(np.sum(env.masses))/ep.initial; f[49] = 1-env.elapsed_s/env.config.episode_duration_s
    f[50] = ep.wall_total/10.; f[51] = float(ep.field.events.sum())/5.
    return f


def rollout(net, anatomy, seed, sample=True, gain=1., record=True, horizon=300.):
    """One episode with image sensing. Returns per-decision arrays (record) and the benchmark row."""
    from marl.lookahead_teacher import primitive_local
    from marl.obstacle_control import History, token, token_dim
    from scripts.benchmark_obstacles import Episode
    from scripts.train_obstacle_drl import route_remaining
    ep = Episode(1, anatomy, seed, horizon=horizon, sensing='image')
    hist = History(1, dim=token_dim(False)); fr = Frames(); rng = np.random.default_rng(seed+5)
    D = {k: [] for k in ('img', 'seq', 'mask', 'act', 'logp', 'priv', 'rew', 'pos', 'obs', 'ev')}
    k = 0; act = 0; r_acc = 0.; wall1 = False; prevP = None; done = False
    try:
        while not done:
            est, tgt, rule, hold = ep.observe()
            nominal = np.array(ep.ctl.nominal, float)                         # route direction only, no heuristic
            T, _ = token(ep.env, ep.sensor, est, ep.ctl, nominal, hold, tgt, ep.prev_local, False)
            seq, mask = hist.push(T); img = fr.push(ep.sensor.crops)
            if k % PERIOD == 0:
                if record and D['act']:
                    D['rew'].append(r_acc)
                r_acc = 0.
                live = bool(est.active[0] and tgt[0] >= 0)
                with torch.no_grad():
                    lg = net(torch.as_tensor(img), torch.as_tensor(seq), torch.as_tensor(mask))[0]
                if sample:
                    d = torch.distributions.Categorical(logits=lg); act = int(d.sample()) if live else 0; lp = float(d.log_prob(torch.tensor(act)))
                else:
                    act = int(lg.argmax()) if live else 0; lp = 0.
                if record:
                    pos = ep.env.positions_mm[0].astype(float); P = ep.field.positions()[0].copy()
                    D['img'].append(img[0].astype(np.float16)); D['seq'].append(seq[0]); D['mask'].append(mask[0]); D['act'].append(act)
                    D['logp'].append(lp); D['priv'].append(priv_features(ep, pos, prevP, PERIOD*ep.env.config.control_dt_s))
                    D['pos'].append(pos); D['obs'].append(P); D['ev'].append(int(ep.field.events.sum())); prevP = P
            local = primitive_local(ep, est, nominal, act)*gain
            g0 = route_remaining(ep.ctl, 0, ep.env.positions_mm[0].astype(float)) if tgt[0] >= 0 else 0.
            done, out = ep.step(est, local, hold)
            g1 = route_remaining(ep.ctl, 0, ep.env.positions_mm[0].astype(float)) if tgt[0] >= 0 else 0.
            r = (1.*float(np.clip(g0-g1, -.2, .2))*(tgt[0] >= 0)+50.*float(out['removed'])-20.*float(np.sum(out['obs_events']))
                 -10.*float(np.sum(out['obs_contact']))-2.*float(np.sum(out['obs_near']))-REW['wall']*float(np.sum(out['wall']))
                 -50.*float(np.sum(out['lost']))-.004)
            if ep.wall_total >= 1. and not wall1:
                r -= REW['wall1']; wall1 = True
            r_acc += r; k += 1
        succ = bool(ep.info['success']); safe = ep.safe()
        r_acc += (50. if succ else -50.)+(50. if safe else 0.)
        if record:
            D['rew'].append(r_acc)
        row = ep.row('vp_ppo'); row['sensing_model'] = 'image'
    finally:
        ep.close()
    return D, row


def labels(D, body, R):
    n = len(D['act']); Y = np.zeros((n, AUX_DIM), np.float32); M = np.zeros(n, np.float32)
    for t in range(n-H_PRED):
        if not len(R):
            break
        P0, p0 = D['obs'][t], D['pos'][t]; g = np.linalg.norm(P0-p0, axis=1)-body-R; idx = np.argsort(g)[:K_PRED]
        P1, p1 = D['obs'][t+H_PRED], D['pos'][t+H_PRED]
        Y[t, :3*len(idx)] = ((P1[idx]-p1)/3.).ravel(); Y[t, 3*K_PRED:3*K_PRED+3] = (p1-p0)
        Y[t, -2] = np.clip(min(float(np.min(np.linalg.norm(D['obs'][t+j]-D['pos'][t+j], axis=1)-body-R)) for j in (1, 2)), -.5, 2.)
        Y[t, -1] = float(D['ev'][t+H_PRED] > D['ev'][t]); M[t] = 1.
    return Y, M


def run_worker(job):
    """Wrapper that also returns obstacle radii and body size for world-model labels."""
    state, dim, seed, anatomy, rew = job
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1); REW.update(rew)
    net = VisionStatePolicy(dim); net.load_state_dict(state); net.eval()
    gain = float(np.random.default_rng(seed).uniform(.75, 1.25))
    import scripts.benchmark_obstacles as bo
    keep = {}
    orig_close = bo.Episode.close

    def close(self):
        keep['R'] = self.field.positions()[1].copy(); keep['body'] = float(self.ctl.body); orig_close(self)
    bo.Episode.close = close
    try:
        D, row = rollout(net, anatomy, seed, sample=True, gain=gain)
    except (ValueError, RuntimeError) as e:
        return dict(error=repr(e))
    finally:
        bo.Episode.close = orig_close
    Y, M = labels(D, keep['body'], keep['R'])
    out = {k: np.asarray(v) for k, v in D.items() if k in ('img', 'seq', 'mask', 'act', 'logp', 'priv', 'rew')}
    out.update(aux=Y, aux_m=M, row={k: row[k] for k in ('cluster_safe_success', 'task_success', 'obstacle_events', 'wall_contact_s', 'removal', 'elapsed_s')})
    return out


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim+PRIV_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, tok, priv):
        return self.f(torch.cat([tok, priv], -1)).squeeze(-1)


class RunningStd:
    def __init__(self):
        self.n, self.m, self.s = 1e-4, 0., 1.

    def update(self, x):
        x = np.asarray(x, float); b = len(x); bm, bv = x.mean(), x.var(); d = bm-self.m; tot = self.n+b
        self.m += d*b/tot; self.s = (self.s*self.n+bv*b+d*d*self.n*b/tot)/tot; self.n = tot

    @property
    def std(self):
        return float(np.sqrt(self.s)+1e-8)


def gae(rew, val, gamma, lam):
    adv = np.zeros_like(rew); g = 0.
    for t in reversed(range(len(rew))):
        nv = val[t+1] if t+1 < len(rew) else 0.
        delta = rew[t]+gamma*nv-val[t]; g = delta+gamma*lam*g; adv[t] = g
    return adv, adv+val


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--steps', type=int, default=3_000_000)
    p.add_argument('--workers', type=int, default=20); p.add_argument('--eps-per-iter', type=int, default=20)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--ent', type=float, default=.01)
    p.add_argument('--gamma', type=float, default=.99); p.add_argument('--lam', type=float, default=.95)
    p.add_argument('--aux-w', type=float, default=.5); p.add_argument('--prior', type=float, default=1.5)
    p.add_argument('--bc', type=Path, help='primitive-teacher data dir: behaviour-cloning initialisation of the actor')
    p.add_argument('--bc-epochs', type=int, default=8); p.add_argument('--bc-tau', type=float, default=.1, help='soft-target temperature (teacher cost units; costs differ by ~0.1-1)'); p.add_argument('--kl-bc', type=float, default=0., help='KL(pi || pi_BC) weight, decays linearly')
    p.add_argument('--wall-w', type=float, default=10.); p.add_argument('--wall1-w', type=float, default=20.)
    p.add_argument('--ppo-epochs', type=int, default=4); p.add_argument('--mb', type=int, default=1024)
    p.add_argument('--resume-actor', type=Path, help='continue from an actor checkpoint (critic, optimiser and reward scale restart)')
    p.add_argument('--start-steps', type=int, default=0, help='environment steps already spent by --resume-actor')
    p.add_argument('--no-aux', action='store_true'); p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cuda:1')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed)
    from marl.obstacle_control import token_dim
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    dim = token_dim(False); dev = torch.device(a.device)
    actor = VisionStatePolicy(dim)
    with torch.no_grad():
        actor.pi.bias[0] = 0. if a.bc else a.prior                            # 'advance' = route following
    actor.to(dev); critic = Critic(dim).to(dev); aux = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, AUX_DIM)).to(dev)
    if a.resume_actor:
        actor.load_state_dict(torch.load(a.resume_actor, map_location='cpu', weights_only=True), strict=True); actor.to(dev)
    bc_net = None
    if a.bc:
        import copy as _copy, glob
        Z = [np.load(f) for f in sorted(glob.glob(str(a.bc/'*.npz')))]
        BI = torch.as_tensor(np.concatenate([z['img'] for z in Z])).float(); BS = torch.as_tensor(np.concatenate([z['seq'] for z in Z]))
        BM = torch.as_tensor(np.concatenate([z['mask'] for z in Z])); BY = torch.as_tensor(np.concatenate([z['label'] for z in Z])).long()
        BC_ = torch.as_tensor(np.concatenate([z['costs'] for z in Z])).float(); BC_ = torch.nan_to_num(BC_, posinf=1e4)
        soft = torch.softmax(-(BC_-BC_.min(1, keepdim=True).values)/a.bc_tau, 1)
        bopt = torch.optim.AdamW(actor.parameters(), lr=3e-4, weight_decay=.01)
        for ep_ in range(a.bc_epochs):
            perm = torch.randperm(len(BY)); tot = 0.; acc = 0.
            for s0 in range(0, len(BY), 512):
                b = perm[s0:s0+512]; lg = actor(BI[b].to(dev), BS[b].to(dev), BM[b].to(dev)); lp = torch.log_softmax(lg, 1)
                reg = (BC_[b]-BC_[b].min(1, keepdim=True).values).clamp(max=200.).to(dev)/100.
                l = -.3*lp.gather(1, BY[b].to(dev)[:, None]).mean()-.7*(soft[b].to(dev)*lp).sum(1).mean()+4.*(lp.exp()*reg).sum(1).mean()
                bopt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.); bopt.step()
                tot += float(l)*len(b); acc += float((lg.argmax(1).cpu() == BY[b]).float().sum())
            print(json.dumps(dict(bc_epoch=ep_+1, loss=tot/len(BY), acc=acc/len(BY), samples=len(BY))), flush=True)
        bc_net = _copy.deepcopy(actor).eval()
        for q in bc_net.parameters():
            q.requires_grad_(False)
    params = list(actor.parameters())+list(critic.parameters())+list(aux.parameters())
    opt = torch.optim.Adam(params, lr=a.lr); rs = RunningStd()
    cfg = dict(bc=str(a.bc) if a.bc else None, kl_bc=a.kl_bc, dim=dim, actions=N_ACT, period=PERIOD, frames=FRAMES, frame_gap=FRAME_GAP, method='VP-PPO', sensing='image', prior=a.prior)
    (a.out/'config.json').write_text(json.dumps(dict(vars(a), out=str(a.out)), indent=1, default=str))
    save(a.out/'iter000.pt', actor.cpu(), dict(cfg, iter=0, steps=0)); actor.to(dev)
    log = (a.out/'train_log.jsonl').open('a'); seed = 2620000000+1000000*a.seed; rng = np.random.default_rng(a.seed)
    total, it, next_ck = a.start_steps, 0, (a.start_steps//500_000+1)*500_000
    seed += a.start_steps//100
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=6) as pool:
        while total < a.steps:
            it += 1; t0 = time.time(); frac = total/a.steps
            for g in opt.param_groups:
                g['lr'] = a.lr*max(.1, 1-frac)
            ent_w = a.ent*max(.1, 1-frac)
            state = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
            jobs = [(state, dim, seed+j, train[rng.integers(len(train))], dict(wall=a.wall_w, wall1=a.wall1_w)) for j in range(a.eps_per_iter)]; seed += a.eps_per_iter
            eps = [e for e in pool.map(run_worker, jobs) if 'error' not in e]
            steps = int(sum(round(e['row']['elapsed_s']/.1) for e in eps)); total += steps
            cat = lambda k: torch.as_tensor(np.concatenate([e[k] for e in eps])).to(dev)
            IM = cat('img').float(); X = cat('seq'); MK = cat('mask'); AC = cat('act').long(); LP0 = cat('logp').float(); PRV = cat('priv').float()
            AY = cat('aux'); AM = cat('aux_m')
            for e in eps:   # return-scale normalisation (standard reward scaling)
                g = 0.; disc = []
                for r in e['rew']:
                    g = a.gamma*g+r; disc.append(g)
                rs.update(disc)
            with torch.no_grad():
                V = torch.cat([critic(X[s:s+4096, -1], PRV[s:s+4096]) for s in range(0, len(X), 4096)]).cpu().numpy()
            advs, rets = [], []; o = 0
            for e in eps:
                n = len(e['act']); ad, rt = gae(e['rew']/rs.std, V[o:o+n], a.gamma, a.lam); advs.append(ad); rets.append(rt); o += n
            ADV = torch.as_tensor(np.concatenate(advs)).float().to(dev); RET = torch.as_tensor(np.concatenate(rets)).float().to(dev)
            V0 = torch.as_tensor(V).float().to(dev); ADV = (ADV-ADV.mean())/(ADV.std()+1e-8)
            st = dict(pl=0., vl=0., al=0., ent=0., kl=0.); nb = 0
            for _ in range(a.ppo_epochs):
                perm = torch.randperm(len(X), device=dev)
                for s in range(0, len(X), a.mb):
                    b = perm[s:s+a.mb]; z = actor.latent(IM[b], X[b], MK[b]); d = torch.distributions.Categorical(logits=actor.pi(z))
                    lp = d.log_prob(AC[b]); ratio = (lp-LP0[b]).exp()
                    pl = -torch.min(ratio*ADV[b], ratio.clamp(.8, 1.2)*ADV[b]).mean(); ent = d.entropy().mean()
                    v = critic(X[b, -1], PRV[b]); vc = V0[b]+(v-V0[b]).clamp(-.2, .2)
                    vl = torch.max((v-RET[b])**2, (vc-RET[b])**2).mean()
                    yp = aux(z); m = AM[b]
                    al = ((((yp[:, :-1]-AY[b, :-1])**2).mean(1)+nn.functional.binary_cross_entropy_with_logits(yp[:, -1], AY[b, -1], reduction='none'))*m).sum()/m.sum().clamp(min=1)
                    loss = pl+.5*vl-ent_w*ent+(0. if a.no_aux else a.aux_w*al)
                    if bc_net is not None and a.kl_bc > 0:
                        with torch.no_grad():
                            lq = torch.log_softmax(bc_net(IM[b], X[b], MK[b]), 1)
                        lpa = torch.log_softmax(actor.pi(z), 1)
                        loss = loss+a.kl_bc*max(0., 1-frac)*(lpa.exp()*(lpa-lq)).sum(1).mean()
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, .5); opt.step()
                    st['pl'] += float(pl); st['vl'] += float(vl); st['al'] += float(al); st['ent'] += float(ent); st['kl'] += float((LP0[b]-lp).mean()); nb += 1
            rows = [e['row'] for e in eps]
            rec = dict(iter=it, steps=total, episodes=len(eps), safe=np.mean([r['cluster_safe_success'] for r in rows]),
                       task=np.mean([r['task_success'] for r in rows]), obstacle_events=np.mean([r['obstacle_events'] for r in rows]),
                       wall=np.mean([r['wall_contact_s'] for r in rows]), removal=np.mean([r['removal'] for r in rows]),
                       opt=np.bincount(AC.cpu().numpy(), minlength=N_ACT).tolist(), ret=float(np.mean([e['rew'].sum() for e in eps])),
                       ret_std=rs.std, sec=time.time()-t0, **{k: v/max(nb, 1) for k, v in st.items()})
            log.write(json.dumps(rec, default=float)+'\n'); log.flush(); print(json.dumps(rec, default=float), flush=True)
            if total >= next_ck or total >= a.steps:
                save(a.out/f'step{total//1000:05d}k.pt', actor.cpu(), dict(cfg, iter=it, steps=total)); actor.to(dev); next_ck += 500_000


if __name__ == '__main__':
    main()
