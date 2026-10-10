"""Scheduled-Settle RL (2026-10-10): learned bounded parameter residuals around the Adaptive Settle controller.

Motivation (EXP0073 hard-condition screening, development data): no classical controller wins every condition.
Adaptive Settle is best with short image latency (Strict 88.1 % at 0.05 mm/s, 1 step) but its stall-release rule
misfires with 3-step latency (0 %), where Fixed Settle / STPG reach 69 %; with composite dynamics PAC-NMPC leads.
Which rule / parameters work depends on the current latency and dynamics. Latency is observable (camera frame age);
the cluster's response and the flow are inferable from the history of sent commands versus observed motion. The
learned policy therefore schedules the parameters of the classical controller from deployable history.

Action a in [-1, 1]^5 (a = 0 reproduces scripts.benchmark_lysis.AdaptiveSettleGuard exactly; regression-tested):
  a0  speed residual          speed <- clip(speed + 0.3 a0, 0, 1)            (after settling / release)
  a1  release gate            a1 < 0 disables the stall-release (a1 = -1 everywhere = Fixed Settle behaviour)
  a2  settle-radius scale     r <- r * 2 ** (0.5 a2)                          (x0.71 ... x1.41)
  a3, a4  lateral aim         the pursuit direction is re-aimed at carrot + 0.5 (r_map - r_body)(a3 n + a4 b)
The shared map wall guard and stall re-planning (WallGuard) and the spacing shield act exactly as for every method.
Controller-internal state exposed to the policy (deployable): releasing flag, response EMA, current radius, d / r.
"""
from __future__ import annotations

import numpy as np

ACT_DIM = 5
SPEED_RES, RADIUS_EXP, LAT_FRAC = .3, .5, .5


class ScheduledSettle:
    """AdaptiveSettleGuard with per-step bounded parameter residuals. Call with a=None (or zeros) for the prior."""

    def __init__(self, ep, radius=.3):
        from scripts.benchmark_lysis import WallGuard
        self.guard = WallGuard(ep); self.radius = radius; n = ep.n
        self.hist = [[] for _ in range(n)]; self.release_until = np.zeros(n); self.resp = np.ones(n)
        self.prev_cmd = np.zeros((n, 3)); self.r_now = np.full(n, radius); self.d_now = np.full(n, 9.)
        self.releasing = np.zeros(n, bool)

    def _reaim(self, ep, est, rule, a):
        ctl = ep.ctl; F = ctl.frames(est); body = float(ep.env.config.robot_radius_mm); rule = rule.copy()
        for i in range(ep.n):
            if a[i, 3] == 0. and a[i, 4] == 0.:
                continue
            m = float(np.linalg.norm(rule[i]))
            if m <= 0.:
                continue
            st = int(np.argmin(np.linalg.norm(ctl.pts-ctl.carrot[i], axis=1)))
            nrm, bin_ = ep.env.tree.normals[st].astype(float), ep.env.tree.binormals[st].astype(float)
            lat = LAT_FRAC*max(float(ep.sensor.healthy[st])-body, 0.)
            d = ctl.carrot[i]+lat*(a[i, 3]*nrm+a[i, 4]*bin_)-est.pos[i]
            rule[i] = F[i]@(m*d/max(np.linalg.norm(d), 1e-9))
        return rule

    def __call__(self, ep, est, tgt, rule, hold, a=None):
        n = ep.n
        a = np.zeros((n, ACT_DIM)) if a is None else np.clip(np.asarray(a, float), -1, 1)
        if np.any(a[:, 3:5]):
            rule = self._reaim(ep, est, rule, a)
        out = self.guard(ep, est, tgt, rule, hold)
        t = ep.env.elapsed_s; F = ep.ctl.frames(est)
        for i, c in enumerate(tgt):
            w = F[i].T@self.prev_cmd[i]; sp = float(np.linalg.norm(w))
            if sp > .5:
                self.resp[i] = .95*self.resp[i]+.05*float(np.clip(est.vel[i]@w/sp**2, 0., 2.))
            if c < 0:
                self.hist[i].clear(); self.releasing[i] = False; continue
            d = float(np.linalg.norm(ep.env.clot_positions_mm[c]-est.pos[i]))
            r = self.radius*float(np.clip(self.resp[i], .5, 1.5))
            if a[i, 2] != 0.:
                r *= 2.**(RADIUS_EXP*a[i, 2])
            self.r_now[i], self.d_now[i] = r, d
            self.hist[i].append(d); self.hist[i] = self.hist[i][-10:]
            gate = a[i, 1] >= 0.
            if not gate:
                self.release_until[i] = min(self.release_until[i], t)     # cancel an ongoing release
            if d < r and t >= self.release_until[i]:
                if gate and len(self.hist[i]) == 10 and self.hist[i][0]-d < .03 and d > .12:
                    self.release_until[i] = t+2.
                else:
                    out[i] *= d/r
            self.releasing[i] = t < self.release_until[i]
            if a[i, 0] != 0.:
                m = float(np.linalg.norm(out[i]))
                if m > 0.:
                    out[i] *= float(np.clip(m+SPEED_RES*a[i, 0], 0., 1.))/m
                elif a[i, 0] > 0. and np.any(rule[i]):
                    out[i] = rule[i]/float(np.linalg.norm(rule[i]))*SPEED_RES*a[i, 0]
        self.prev_cmd = out.copy()
        return out

    def state(self, n):
        """[n, 4] controller-internal features for the policy (all deployable)."""
        return np.stack([self.releasing.astype(float), np.clip(self.resp, 0., 2.)/2.,
                         self.r_now/(2*self.radius), np.clip(self.d_now/np.maximum(self.r_now, 1e-6), 0., 5.)/5.], 1)[:n]


# ------------------------------------------------------------------------------------------------------- v2 (10-10)
# EXP SCHED v1 diagnosis: the release gate was learned inverted / not at all (mean |a1| ~ 0.03 around a hard
# threshold), and a correct frame-age switch between Fixed and Adaptive Settle (SwitchSettle, tau 0.15 s) already
# equals the best classical controller in every delay-dominated panel (Strict 88.1 / 50.0 / 69.0 %). Mode selection
# is therefore left to that deployable rule; RL learns only what the rules cannot: continuous speed, settle-radius
# and lateral corrections that compensate flow and response (privileged bound vs best classical: strong flow
# 45 vs 12 %, composite dynamics 76 vs 36 %, OOD 69 vs 36 %).
ACT2_DIM = 4      # speed residual, settle-radius scale, lateral n, lateral b


class SwitchResidual(ScheduledSettle):
    """Frame-age switch rule (Fixed Settle if frame age > tau, else Adaptive Settle) + bounded continuous residuals.
    a = 0 reproduces AdaptiveSettleGuard at frame age <= tau and SettleGuard above it (regression-tested)."""

    def __init__(self, ep, radius=.3, tau=.15):
        super().__init__(ep, radius); self.tau = float(tau); self.fixed_mode = np.zeros(ep.n, bool)

    def __call__(self, ep, est, tgt, rule, hold, a=None):
        n = ep.n
        a = np.zeros((n, ACT2_DIM)) if a is None else np.clip(np.asarray(a, float), -1, 1)
        full = np.zeros((n, ACT_DIM)); full[:, 0] = a[:, 0]; full[:, 3:5] = a[:, 2:4]
        if np.any(full[:, 3:5]):
            rule = self._reaim(ep, est, rule, full)
        fixed = ep.env.elapsed_s-float(getattr(est, 'frame_time_s', ep.env.elapsed_s)) > self.tau
        self.fixed_mode[:] = fixed
        out = self.guard(ep, est, tgt, rule, hold)
        t = ep.env.elapsed_s; F = ep.ctl.frames(est)
        for i, c in enumerate(tgt):
            w = F[i].T@self.prev_cmd[i]; sp = float(np.linalg.norm(w))
            if sp > .5:
                self.resp[i] = .95*self.resp[i]+.05*float(np.clip(est.vel[i]@w/sp**2, 0., 2.))
            if c < 0:
                self.hist[i].clear(); self.releasing[i] = False; continue
            d = float(np.linalg.norm(ep.env.clot_positions_mm[c]-est.pos[i]))
            r = self.radius if fixed else self.radius*float(np.clip(self.resp[i], .5, 1.5))
            if a[i, 1] != 0.:
                r *= 2.**(RADIUS_EXP*a[i, 1])
            self.r_now[i], self.d_now[i] = r, d
            self.hist[i].append(d); self.hist[i] = self.hist[i][-10:]
            if fixed:
                self.release_until[i] = min(self.release_until[i], t)
                if d < r:
                    out[i] *= d/r
            elif d < r and t >= self.release_until[i]:
                if len(self.hist[i]) == 10 and self.hist[i][0]-d < .03 and d > .12:
                    self.release_until[i] = t+2.
                else:
                    out[i] *= d/r
            self.releasing[i] = t < self.release_until[i]
            if a[i, 0] != 0.:
                m = float(np.linalg.norm(out[i]))
                if m > 0.:
                    out[i] *= float(np.clip(m+SPEED_RES*a[i, 0], 0., 1.))/m
                elif a[i, 0] > 0. and np.any(rule[i]):
                    out[i] = rule[i]/float(np.linalg.norm(rule[i]))*SPEED_RES*a[i, 0]
        self.prev_cmd = out.copy()
        return out

    def state(self, n):
        return np.concatenate([super().state(n), self.fixed_mode[:n, None].astype(float)], 1)


# ------------------------------------------------------------------------------------------------------- v3
# The useful variable left after the frame-age switch is the uncommanded motion caused by flow and response
# mismatch.  This controller exposes a causal, deployable estimate of that drift and gives the learner only two
# residuals.  With a zero action it is exactly the SwitchResidual prior, so an uninformative policy cannot destroy
# the rule baseline.
ACT_FLOW_DIM = 2              # speed residual, signed drift-compensation coefficient
# Conservative by design: early experiments showed that unconstrained flow
# compensation can help OOD flow while hurting variable-response scenes.  The
# residual remains useful, but is capped so the tested SwitchSettle prior keeps
# control when the motion estimate is uncertain.
FLOW_COMP_MAX = .4


class FlowResidual(SwitchResidual):
    """SwitchSettle prior plus a learned, bounded correction for measured flow drift.

    The drift estimate uses only the previous sent world command and the current image velocity.  A slow gain EMA
    separates response scaling from the additive component.  The actor controls the speed residual and a signed
    coefficient on ``-drift``; radius, release and frame-age mode remain owned by the tested rule.
    """

    def __init__(self, ep, radius=.3, tau=.15):
        super().__init__(ep, radius, tau)
        self.gain_est = np.ones(ep.n)
        self.drift = np.zeros((ep.n, 3))
        self.drift_conf = np.zeros(ep.n)
        self.cmd_history = []
        self.motion_history = [[] for _ in range(ep.n)]
        self.last_frame_time = -np.inf

    def __call__(self, ep, est, tgt, rule, hold, a=None):
        n = ep.n
        a = np.zeros((n, ACT_FLOW_DIM)) if a is None else np.clip(np.asarray(a, float), -1, 1)
        # Estimate the plant mismatch before updating the base controller's previous command.  The velocity is
        # delayed and may be held; match it to the command nearest the midpoint of the camera-frame interval.
        now = float(ep.env.elapsed_s); dt = float(getattr(ep.env.config, 'control_dt_s', .1))
        self.cmd_history.append((now-dt*.5, np.asarray(getattr(ep, 'sent_world', np.zeros((n, 3))), float).copy()))
        self.cmd_history = self.cmd_history[-64:]
        frame_time = float(getattr(est, 'frame_time_s', now))
        new_frame = frame_time > self.last_frame_time + 1e-7
        if new_frame:
            mid = frame_time - dt*.5
            cmd = min(self.cmd_history, key=lambda x: abs(x[0]-mid))[1]
        else:
            cmd = None
        vel = np.asarray(est.vel, float)
        if new_frame:
            for i in range(n):
                if not est.active[i] or hold[i]:
                    continue
                den = float(cmd[i] @ cmd[i])
                if den > .12**2 and np.isfinite(vel[i]).all():
                    h = self.motion_history[i]
                    h.append((cmd[i].copy(), vel[i].copy()))
                    del h[:-24]
                    if len(h) >= 6:
                        C = np.asarray([x[0] for x in h]); V = np.asarray([x[1] for x in h])
                        C0, V0 = C-C.mean(0), V-V.mean(0)
                        spread = float((C0*C0).sum())
                        if spread > .02:
                            # Fit v = gain * command + additive drift.  Centering separates the two terms even
                            # when the flow is aligned with the route; a projection-only EMA would absorb flow
                            # into the gain and make compensation impossible.
                            g = float(np.clip((C0*V0).sum()/spread, 0., 2.))
                            b = V.mean(0)-g*C.mean(0)
                            self.gain_est[i] = .8*self.gain_est[i] + .2*g
                            self.drift[i] = .8*self.drift[i] + .2*np.clip(b, -1., 1.)
                            self.drift_conf[i] = min(1., max(0., (len(h)-5)/12.) * min(1., spread/.08))
                else:
                    self.drift_conf[i] *= .99
            self.last_frame_time = frame_time

        # Only the speed residual is passed to the tested switch controller.
        base_a = np.zeros((n, ACT2_DIM)); base_a[:, 0] = a[:, 0]
        out = super().__call__(ep, est, tgt, rule, hold, base_a)
        if not np.any(np.abs(a[:, 1]) > 1e-12):
            return out
        F = ep.ctl.frames(est)
        world = np.einsum('nij,nj->ni', np.transpose(F, (0, 2, 1)), out)
        coeff = np.clip(a[:, 1], -1., 1.)*FLOW_COMP_MAX*self.drift_conf
        for i in range(n):
            if not est.active[i] or hold[i] or abs(coeff[i]) < 1e-6:
                continue
            w = world[i] - coeff[i]*self.drift[i]
            world[i] = w/max(float(np.linalg.norm(w)), 1.)
        out = np.einsum('nij,nj->ni', F, world)
        self.prev_cmd = out.copy()
        return out

    def state(self, n):
        base = super().state(n)
        drift = np.clip(self.drift[:n], -1., 1.)
        return np.concatenate([base, self.gain_est[:n, None]/2., drift,
                                np.linalg.norm(drift, axis=1, keepdims=True),
                                self.drift_conf[:n, None]], 1)
