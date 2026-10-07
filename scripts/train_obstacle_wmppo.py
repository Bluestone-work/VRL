"""WM-OPPO: world-model-guided option PPO with an asymmetric privileged critic, started from the reactive prior.

Actor (deployable): DiscreteTemporalPolicy over the 16-step token history (detector boxes, map, own state; no
simulator truth), choosing one of the 9 options of marl.lookahead_teacher every 0.5 s. Its output bias is
initialised so that argmax = option 8 'switch' (pursuit unless a detected obstacle surface is within 0.3 mm,
then wall-aware APF): the untrained policy IS the strongest deployable heuristic, and PPO only learns when to
deviate from it.
Critic (training only): token + privileged state (true nearest-obstacle positions, velocities and gaps, true
wall gap, remaining route and mass). Turbo/Medany also use simulator truth for reward and success only.
World-model head (training only): from the actor's latent state, predict the relative positions of the
currently nearest 3 obstacles 1 s ahead, the minimum true surface gap over the next 1 s and whether a
collision occurs in it. The loss shapes the actor's representation; nothing of it is used at deployment.
Reward per decision (sum over its 5 control steps; simulator outcomes, never an input):
  +0.5 x route progress (mm, clipped 0.2 per step) +50 x removed mass fraction -30 x new obstacle event
  -10 x obstacle contact s -10 x wall contact s -30 x lost -0.05 ; terminal +50 if Safe Success.
--reward v2 (metric-aligned): additionally -30 at the first obstacle event and -30 when cumulative wall contact
crosses 1 s (the two events that make Safe Success fail), Safe bonus +100.
--reward v3: v2 plus route progress x 2.0 (was 0.5) and -100 at the end of an episode that did not clear every
clot (v1 converged to always-avoid wall_apf: 58 % time-outs at iteration 100).
Collisions do not end the episode (an early termination shortens the penalty stream; v3.2 exploit).
Domain randomisation per episode: position noise U[.02,.08] mm, latency U{1,2}, dropout U[0,.05],
actuation gain U[.75,1.25]. Training anatomies only, N=1, seeds 2615000000+ (disjoint from dev / sealed).
usage: train_obstacle_wmppo.py --out DIR --iters 120 --workers 22 [--init distilled.pt]
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

PERIOD, K_PRED, H_PRED = 5, 3, 2          # decision every 5 control steps; predict 3 obstacles, 2 decisions (1 s) ahead
PRIV_DIM = 6*8+4
AUX_DIM = 3*K_PRED+2


def priv_features(ep, pos):
    """Privileged critic input: 6 nearest true obstacles (rel pos, radius, gap, velocity) + wall/route/mass/time."""
    P, R, _ = ep.field.positions(); f = np.zeros(PRIV_DIM, np.float32); body = ep.ctl.body
    if len(R):
        g = np.linalg.norm(P-pos, axis=1)-body-R; idx = np.argsort(g)[:6]
        V = (P-ep._prevP)/max(ep._dtP, 1e-9) if getattr(ep, '_prevP', None) is not None and ep._prevP.shape == P.shape else np.zeros_like(P)
        for j, k in enumerate(idx):
            f[8*j:8*j+3] = (P[k]-pos)/3.; f[8*j+3] = R[k]/.5; f[8*j+4] = g[k]; f[8*j+5:8*j+8] = np.clip(V[k], -3, 3)
    ax, r, rad = ep.sensor.map_coordinates(ep.est_last, 0) if ep.est_last is not None else (pos, 1., 0.)
    f[48] = (r-rad-body)/max(r, 1e-9); f[49] = ep._route_rem/100.; f[50] = float(ep.env.masses.sum())/ep.initial
    f[51] = 1-ep.env.elapsed_s/ep.env.config.episode_duration_s
    return f


def run_episode(job):
    state, cfg, seed, anatomy = job
    os.environ['OMP_NUM_THREADS'] = '1'; torch.set_num_threads(1)
    from marl.deployable_sensing import DeployableConfig
    from marl.lookahead_teacher import option_local
    from marl.obstacle_control import DiscreteTemporalPolicy, History, token, token_dim
    from scripts.benchmark_obstacles import Episode
    from scripts.train_obstacle_drl import route_remaining
    rng = np.random.default_rng(seed+77)
    sc = DeployableConfig(position_sigma_mm=float(rng.uniform(.02, .08)), latency_steps=int(rng.integers(1, 3)),
                          dropout_prob=float(rng.uniform(0, .05))) if cfg['dr'] else DeployableConfig()
    gain = float(rng.uniform(.75, 1.25)) if cfg['dr'] else 1.
    try:
        ep = Episode(1, anatomy, seed, sense_cfg=sc)
    except (ValueError, RuntimeError) as e:
        return dict(error=repr(e))
    net = DiscreteTemporalPolicy(cfg['arch'], dim=token_dim(False), actions=9); net.load_state_dict(state); net.eval()
    hist = History(1, dim=token_dim(False)); ep._hit = False; ep._wall1 = False; ep._prevP = None; ep._dtP = 1.; ep.est_last = None; ep._route_rem = 0.
    S, M, A, LP, PR, RW, DN, POS, OBS, EV = [], [], [], [], [], [], [], [], [], []
    k = 0; act = 8; r_acc = 0.; done = False
    try:
        while not done:
            est, tgt, rule, hold = ep.observe(); ep.est_last = est
            T, _ = token(ep.env, ep.sensor, est, ep.ctl, rule, hold, tgt, ep.prev_local, False)
            seq, mask = hist.push(T)
            live = bool(est.active[0] and tgt[0] >= 0 and not hold[0])
            if k % PERIOD == 0:
                if S:
                    RW.append(r_acc); DN.append(False)
                r_acc = 0.
                pos = ep.env.positions_mm[0].astype(float); ep._route_rem = route_remaining(ep.ctl, 0, pos) if tgt[0] >= 0 else 0.
                with torch.no_grad():
                    lg = net.logits(torch.as_tensor(seq), torch.as_tensor(mask))[0]
                d = torch.distributions.Categorical(logits=lg)
                act = int(d.sample()) if live else 8
                S.append(seq[0]); M.append(mask[0]); A.append(act); LP.append(float(d.log_prob(torch.tensor(act))))
                PR.append(priv_features(ep, pos)); POS.append(pos.copy()); OBS.append(ep.field.positions()[0].copy())
                EV.append(int(ep.field.events.sum())); ep._prevP = OBS[-1]; ep._dtP = PERIOD*ep.env.config.control_dt_s
            local = option_local(ep, est, rule, act)*gain
            pos0 = ep.env.positions_mm[0].astype(float); g0 = route_remaining(ep.ctl, 0, pos0) if tgt[0] >= 0 else 0.
            done, out = ep.step(est, local, hold)
            g1 = route_remaining(ep.ctl, 0, ep.env.positions_mm[0].astype(float)) if tgt[0] >= 0 else 0.
            prog = float(np.clip(g0-g1, -.2, .2))*(tgt[0] >= 0)
            r_acc += ((2. if cfg.get('reward') == 'v3' else .5)*prog+50.*float(out['removed'])-30.*float(out['obs_events'])-10.*float(np.sum(out['obs_contact']))
                      -10.*float(np.sum(out['wall']))-30.*float(np.sum(out['lost']))-.01)
            if cfg.get('reward', 'v1') in ('v2', 'v3'):
                # v2, aligned with the binary Safe Success: the step that makes the episode unsafe costs extra
                # (first obstacle event; cumulative wall contact crossing 1 s)
                ev_now = int(ep.field.events.sum())
                if ev_now > 0 and not ep._hit:
                    r_acc -= 30.; ep._hit = True
                if ep.wall_total >= 1. and not ep._wall1:
                    r_acc -= 30.; ep._wall1 = True
            k += 1
        safe = ep.safe()
        rw = cfg.get('reward', 'v1')
        bonus = (100. if rw in ('v2', 'v3') else 50.) if safe else 0.
        if rw == 'v3' and not ep.info['success']:
            bonus -= 100.   # v3: time-out without clearing every clot (v1/v2 policies settled on always-avoid and stalled)
        RW.append(r_acc+bonus); DN.append(True)
        row = ep.row('wmppo')
    finally:
        ep.close()
    # world-model labels: true relative positions of the currently nearest K obstacles H_PRED decisions later,
    # minimum true gap and collision over the next H_PRED decisions
    n = len(A); Y = np.zeros((n, AUX_DIM), np.float32); YM = np.zeros(n, np.float32); body = float(ep.ctl.body)
    R = ep.field.positions()[1]
    for t in range(n):
        if t+H_PRED >= n or not len(R):
            continue
        g = np.linalg.norm(OBS[t]-POS[t], axis=1)-body-R; idx = np.argsort(g)[:K_PRED]
        fut = (OBS[t+H_PRED][idx]-POS[t+H_PRED])/3.; Y[t, :3*len(idx)] = fut.ravel()
        gm = min(float(np.min(np.linalg.norm(OBS[t+j]-POS[t+j], axis=1)-body-R)) for j in range(1, H_PRED+1))
        Y[t, 3*K_PRED] = np.clip(gm, -.5, 2.); Y[t, 3*K_PRED+1] = float(EV[t+H_PRED] > EV[t]); YM[t] = 1.
    return dict(seq=np.stack(S), mask=np.stack(M), act=np.array(A), logp=np.array(LP, np.float32), priv=np.stack(PR),
                rew=np.array(RW, np.float32), done=np.array(DN), aux=Y, aux_m=YM,
                summary=dict(anatomy=anatomy, seed=seed, safe=bool(safe), task=bool(row['task_success']),
                             obstacle_events=int(row['obstacle_events']), wall=float(row['wall_contact_s']),
                             removal=float(row['removal']), opt=np.bincount(np.array(A), minlength=9).tolist()))


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim+PRIV_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, tok, priv):
        return self.f(torch.cat([tok, priv], -1)).squeeze(-1)


def gae(rew, val, done, gamma, lam):
    adv = np.zeros_like(rew); g = 0.
    for t in reversed(range(len(rew))):
        nv = 0. if done[t] else val[t+1]
        delta = rew[t]+gamma*nv-val[t]; g = delta+gamma*lam*(0. if done[t] else g); adv[t] = g
    return adv, adv+val


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--iters', type=int, default=120)
    p.add_argument('--workers', type=int, default=22); p.add_argument('--eps-per-iter', type=int, default=22)
    p.add_argument('--arch', default='gru', choices=('gru', 'transformer', 'mlp')); p.add_argument('--init')
    p.add_argument('--prior', type=float, default=4.); p.add_argument('--aux-w', type=float, default=.5)
    p.add_argument('--ent', type=float, default=.003); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=.98); p.add_argument('--lam', type=float, default=.95)
    p.add_argument('--reward', default='v1', choices=('v1', 'v2', 'v3')); p.add_argument('--no-dr', action='store_true'); p.add_argument('--no-aux', action='store_true')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--device', default='cuda:1')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.manual_seed(a.seed)
    from marl.obstacle_control import DiscreteTemporalPolicy, save_discrete_checkpoint, token_dim
    train = json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    dim = token_dim(False); dev = torch.device(a.device)
    actor = DiscreteTemporalPolicy(a.arch, dim=dim, actions=9)
    if a.init:   # representation from a distilled student; the option head is re-initialised to the prior
        st = torch.load(a.init, map_location='cpu', weights_only=True)
        actor.load_state_dict({k: v for k, v in st.items() if not k.startswith(('pi.', 'value.'))}, strict=False)
    nn.init.zeros_(actor.pi[-1].weight); nn.init.zeros_(actor.pi[-1].bias)
    with torch.no_grad():
        actor.pi[-1].bias[8] = a.prior
    actor.to(dev); critic = Critic(dim).to(dev)
    aux = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, AUX_DIM)).to(dev)
    opt = torch.optim.Adam([dict(params=actor.parameters(), lr=a.lr), dict(params=aux.parameters(), lr=3e-4),
                            dict(params=critic.parameters(), lr=3e-4)])
    cfg = dict(arch=a.arch, dr=not a.no_dr, reward=a.reward); log = (a.out/'train_log.jsonl').open('a')
    meta = dict(vars(a), out=str(a.out), init=a.init, method='WM-OPPO'); (a.out/'config.json').write_text(json.dumps(meta, indent=1, default=str))
    ckcfg = dict(arch=a.arch, dim=dim, actions=9, obs_vel=False, window=16, period=PERIOD, method='WM-OPPO', prior=a.prior, init=a.init)
    save_discrete_checkpoint(a.out/'iter000.pt', actor.cpu(), dict(ckcfg, iter=0)); actor.to(dev)
    seed = 2615000000+1000000*a.seed; rng = np.random.default_rng(a.seed)
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=8) as pool:
        for it in range(1, a.iters+1):
            t0 = time.time(); state = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
            jobs = [(state, cfg, seed+j, train[rng.integers(len(train))]) for j in range(a.eps_per_iter)]; seed += a.eps_per_iter
            eps = [e for e in pool.map(run_episode, jobs) if 'error' not in e]
            X = torch.as_tensor(np.concatenate([e['seq'] for e in eps])).to(dev); Mk = torch.as_tensor(np.concatenate([e['mask'] for e in eps])).to(dev)
            Ac = torch.as_tensor(np.concatenate([e['act'] for e in eps])).to(dev); LP0 = torch.as_tensor(np.concatenate([e['logp'] for e in eps])).to(dev)
            PRV = torch.as_tensor(np.concatenate([e['priv'] for e in eps])).to(dev)
            AY = torch.as_tensor(np.concatenate([e['aux'] for e in eps])).to(dev); AM = torch.as_tensor(np.concatenate([e['aux_m'] for e in eps])).to(dev)
            with torch.no_grad():
                V = torch.cat([critic(X[s:s+4096, -1], PRV[s:s+4096]) for s in range(0, len(X), 4096)]).cpu().numpy()
            advs, rets = [], []; o = 0
            for e in eps:
                n = len(e['act']); ad, rt = gae(e['rew'], V[o:o+n], e['done'], a.gamma, a.lam); advs.append(ad); rets.append(rt); o += n
            ADV = torch.as_tensor(np.concatenate(advs)).float().to(dev); RET = torch.as_tensor(np.concatenate(rets)).float().to(dev)
            ADV = (ADV-ADV.mean())/(ADV.std()+1e-8)
            stats = dict(pl=0., vl=0., al=0., ent=0., kl=0.); nb = 0
            for _ in range(4):
                perm = torch.randperm(len(X), device=dev)
                for s in range(0, len(X), 1024):
                    b = perm[s:s+1024]; h = actor.backbone(X[b], Mk[b]); lg = actor.pi(h); d = torch.distributions.Categorical(logits=lg)
                    lp = d.log_prob(Ac[b]); ratio = (lp-LP0[b]).exp()
                    pl = -torch.min(ratio*ADV[b], ratio.clamp(.8, 1.2)*ADV[b]).mean(); ent = d.entropy().mean()
                    vl = ((critic(X[b, -1], PRV[b])-RET[b])**2).mean()
                    yp = aux(h); m = AM[b]
                    al = ((((yp[:, :3*K_PRED+1]-AY[b, :3*K_PRED+1])**2).mean(1)
                           + nn.functional.binary_cross_entropy_with_logits(yp[:, -1], AY[b, -1], reduction='none'))*m).sum()/m.sum().clamp(min=1)
                    loss = pl+.5*vl-a.ent*ent+(0. if a.no_aux else a.aux_w*al)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters())+list(aux.parameters()), 1.); opt.step()
                    stats['pl'] += float(pl); stats['vl'] += float(vl); stats['al'] += float(al); stats['ent'] += float(ent)
                    stats['kl'] += float((LP0[b]-lp).mean()); nb += 1
            sm = [e['summary'] for e in eps]
            rec = dict(iter=it, episodes=len(eps), samples=len(X), safe=np.mean([x['safe'] for x in sm]), task=np.mean([x['task'] for x in sm]),
                       obstacle_events=np.mean([x['obstacle_events'] for x in sm]), wall=np.mean([x['wall'] for x in sm]),
                       removal=np.mean([x['removal'] for x in sm]), opt=np.sum([x['opt'] for x in sm], 0).tolist(),
                       ret=float(np.mean([e['rew'].sum() for e in eps])), sec=time.time()-t0, **{k: v/max(nb, 1) for k, v in stats.items()})
            log.write(json.dumps(rec, default=float)+'\n'); log.flush(); print(json.dumps(rec, default=float), flush=True)
            if it % 10 == 0 or it == a.iters:
                save_discrete_checkpoint(a.out/f'iter{it:03d}.pt', actor.cpu(), dict(ckcfg, iter=it)); actor.to(dev)


if __name__ == '__main__':
    main()
