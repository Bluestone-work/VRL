"""EXP0093: global-layer x local-layer factorial comparison (external methods are local adaptations).

Global (task allocation + multi-cluster coordination):
  A_TPG   ours: makespan-optimal geodesic allocation A (exhaustive) + temporal plan graph (fixed precedence)
  STPG    A + switchable TPG re-optimised every 2 s (Jiang, Lin & Li, AAAI 2025; exact search for N <= 3)
  OT      optimal-transport-style anonymous goal assignment minimising total geodesic transport cost, each
          cluster then visits its goals nearest-first (MPC-OT, Khan et al., CDC 2025: OT allocation; its
          continuous tracking layer is replaced by the local layer under test); no TPG, spacing shield only
  NONE    A, no TPG (spacing shield only)
Local (navigation, given the current target and route):
  ours      prediction-assisted command selection with the learned predictor ensemble + fallback (EXP0091 config)
  online    same selector with the non-learned online affine estimator
  pursuit   route pursuit + fixed arrival settling + WallGuard (classical)
  pac       PAC-NMPC sampling NMPC (Gonzales et al., CASE 2025; local adaptation)
  mppi      MPPI (Williams et al., ICRA 2016 / T-RO 2018; model: measured drift + unit command) + same settling
  cbf       CBF-QP wall-safety filter (Ames et al., IEEE TAC 2017) on pursuit + same settling
  turbo     Transformer PPO direct velocity policy (An et al., Nat. Mach. Intell. 2026; retrained in this domain)
The shared execution layer (TPG hold if the global layer has one, spacing shield, actuator limit) is identical.
"""
from __future__ import annotations

import itertools

import numpy as np

PRED = {}       # filled by the evaluator: ensemble paths, fallback std, weights, turbo checkpoint


def ot_plan(ep):
    """Total-transport-cost assignment of every clot to one cluster (min sum of start->clot geodesics; ties broken by
    makespan of the resulting nearest-first tours), then nearest-first visiting order."""
    env = ep.env; G = ep.bm.station_geodesic(env); st = np.asarray(env.robot_stations); cs = np.asarray(env.clot_stations)
    n, m = len(st), len(cs); best = None
    for lab in itertools.product(range(n), repeat=m):
        transport = sum(G[st[lab[j]], cs[j]] for j in range(m))
        tours = []
        for i in range(n):
            todo = [j for j in range(m) if lab[j] == i]; cur = st[i]; order = []; L = 0.
            while todo:
                j = min(todo, key=lambda q: G[cur, cs[q]]); L += G[cur, cs[j]]; cur = cs[j]; order.append(j); todo.remove(j)
            tours.append((L, order))
        key = (transport, max(t[0] for t in tours))
        if best is None or key < best[0]:
            best = (key, [t[1] for t in tours])
    return best[1]


def make(spec_global, n, anatomy, seed, **kw):
    from scripts.benchmark_lysis import LysisEpisode
    g = spec_global
    if g in ('OT', 'NONE'):
        kw.update(tpg=False, fallback='park')
    ep = LysisEpisode(n, anatomy, seed, **kw)
    ep.stpg = None
    if g == 'OT':
        ep.plan = ot_plan(ep); ep.plan_targets = ep.bm.PlanTargets(ep.plan)
    if g == 'STPG' and ep.coord is not None:
        from marl.lysis_baselines import SwitchableTPG
        ep.stpg = SwitchableTPG(ep.coord, period_s=2.)
    return ep


class _Settle:
    """Fixed arrival settling applied after a tracking controller (MPPI / CBF have no dwell logic)."""
    def __init__(self, radius=.3):
        self.radius = radius

    def __call__(self, ep, est, tgt, local):
        out = local.copy()
        for i, c in enumerate(tgt):
            if c >= 0:
                d = float(np.linalg.norm(ep.env.clot_positions_mm[c]-est.pos[i]))
                if d < self.radius:
                    out[i] *= d/self.radius
        return out


class MPPILocal:
    def __init__(self, ep):
        from scripts.benchmark_lysis import WallGuard
        from marl.paper_navigation import MPPIController
        self.guard, self.mppi, self.settle = WallGuard(ep), MPPIController(ep.sensor, ep.n, ep.ctl.body), _Settle()

    def __call__(self, ep, est, tgt, rule, hold):
        est.obstacles = [[] for _ in range(ep.n)]
        g = self.guard(ep, est, tgt, rule, hold)
        out = self.mppi.act(ep.env.elapsed_s, est, ep.ctl, g, hold)
        out = self.guard.exec.act(ep.env.elapsed_s, est, ep.ctl, out, hold)
        return self.settle(ep, est, tgt, out)


class CBFLocal:
    def __init__(self, ep):
        from scripts.benchmark_lysis import WallGuard
        from marl.paper_navigation import CBFQPFilter
        self.guard, self.cbf, self.settle = WallGuard(ep), CBFQPFilter(ep.sensor, ep.n, ep.ctl.body), _Settle()

    def __call__(self, ep, est, tgt, rule, hold):
        est.obstacles = [[] for _ in range(ep.n)]
        g = self.guard(ep, est, tgt, rule, hold)
        return self.settle(ep, est, tgt, self.cbf.act(ep.env.elapsed_s, est, ep.ctl, g, hold))


def local(name, ep):
    from scripts.benchmark_lysis import SettleGuard
    if name == 'pursuit':
        return SettleGuard(ep)
    if name == 'pac':
        from marl.lysis_baselines import PacNMPC
        return PacNMPC(ep)
    if name == 'mppi':
        return MPPILocal(ep)
    if name == 'cbf':
        return CBFLocal(ep)
    if name == 'turbo':
        from scripts.train_lysis_local import LocalController
        return LocalController(ep, PRED['turbo'])
    from marl.dyn_select import PredictiveSelector
    if name == 'online':
        return PredictiveSelector(ep, 'online', weights=PRED['weights'])
    if name == 'ours':
        from marl.dyn_predictor import Ensemble
        return PredictiveSelector(ep, 'learned', Ensemble(PRED['ensemble']), PRED['fallback_std'], PRED['weights'])
    raise ValueError(name)
