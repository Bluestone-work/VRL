"""Benchmark v5 (2026-10-08): patient-specific and time-varying dynamics on a known pre-operative geometry.

Clinical rationale: CTA gives the vessel tree and clot locations before treatment, but the dynamics a controller
faces differ between patients and over time: pulsatile blood flow (heart rate, waveform, patient-level mean flow),
how the magnetic cluster actually responds to the commanded field (gain, direction bias, lag, slow drift, shape
change in narrow lumens), near-wall hydrodynamics and adhesion, and how fast a clot lyses (composition). None of
these is observable directly; a controller only sees the image estimates (and their history). Simulator truth is
used only to generate the dynamics.

One scalar `strength` s scales every factor (s = 0 reproduces benchmark v4 exactly; s = 1 is the registered
physiological range; s = 1.5 is out-of-distribution for policies trained on s <= 1.25):
  flow mean        x logU[1/(1+s), 1+s]               on top of the scene's own flow multiplier
  pulsatility      q(t) = 1 + A w(2 pi f t + phase),  A ~ U[0, 0.8 s] (venous anatomies x 0.3), f = HR/60,
                   HR ~ U[50, 120] bpm, w = normalised two-harmonic arterial waveform (systolic peak), q >= 0.05
  actuation gain   per cluster U[1-0.3 s, 1+0.3 s], slow drift (OU on log-gain, sd 0.1 s, time constant 20 s)
  direction bias   per cluster fixed rotation by U[0, 15 s] deg about a random axis
  response lag     first-order, tau ~ U[0, 0.3 s] s
  shape change     speed x clip(gap / 0.3 mm, 0.4, 1)^beta in narrow lumens, beta ~ U[0, 0.6 s]
                   (gap = local lumen radius - body radius; elongated swarms are slower and less steerable)
  near-wall drag   lubrication floor U[1-0.6 s, 1] (speed reduction within 4 body radii of the wall)
  wall adhesion    after a wall contact the next command is scaled by 1 - U[0, 0.5 s]
  lysis rate       x logU[1/(1+0.5 s), 1+0.5 s] (clot composition), unknown to the controller
  execution noise  N(0, 0.03 s) on the world command
Parameter ranges are engineering choices for an in-vitro-scaled model, registered here before any v5 result;
they are not patient measurements.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

VENOUS = {'cerebral_venous_sinus', 'popliteal_calf_dvt', 'iliac_may_thurner'}


def waveform(phase):
    """Normalised arterial flow waveform (mean 0, max 1): systolic peak + dicrotic component."""
    w = np.sin(phase)+.45*np.sin(2*phase-.6)
    return w/1.35


class Variation:
    def __init__(self, ep, strength, rng, pulsatile=True):
        self.ep, self.s, self.rng = ep, float(strength), rng
        env, n, s = ep.env, ep.n, float(strength)
        u = lambda lo, hi, size=None: rng.uniform(lo, hi, size)
        self.flow_mean = float(np.exp(u(-np.log1p(s), np.log1p(s)))) if s > 0 else 1.
        venous = env.config.anatomy in VENOUS
        self.amp = u(0, .8*s)*(.3 if venous else 1.)
        if not pulsatile:                       # steady-flow control condition (same random stream otherwise)
            self.amp = 0.
        self.hr = u(50, 120); self.phase0 = u(0, 2*np.pi)
        self.gain0 = u(1-.3*s, 1+.3*s, n); self.log_drift = np.zeros(n)
        self.rot = []
        for _ in range(n):
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax); ang = np.deg2rad(u(0, 15*s))
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            self.rot.append(np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*K@K)
        self.tau = u(0, .3*s); self.beta = u(0, .6*s)
        self.adhesion = u(0, .5*s); self.noise = .03*s
        env.transport.lubrication_floor = float(u(1-.6*s, 1.))
        self.lysis = float(np.exp(u(-np.log1p(.5*s), np.log1p(.5*s)))) if s > 0 else 1.
        env.config = replace(env.config, lysis_mass_per_s=env.config.lysis_mass_per_s*self.lysis)
        self.u_eff = np.zeros((n, 3)); self.touch = np.zeros(n, bool)
        self.dt = float(env.config.control_dt_s)
        self._install()

    def q(self, t):
        return max(self.flow_mean*(1+self.amp*waveform(2*np.pi*self.hr/60.*t+self.phase0)), .05)

    def _install(self):
        tp = self.ep.env.transport; inner = tp.advance; var = self

        def scaled(sol, q):
            if sol is None:
                return None
            out = dict(sol); out['station_inflow_mm3_s'] = sol['station_inflow_mm3_s']*q
            if 'mean_speed_mm_s' in sol:
                out['mean_speed_mm_s'] = sol['mean_speed_mm_s']*q
            return out

        def advance(positions_mm, edge, body_radius_mm, commands_mm_s, active, solution, duration_s, *, after_substep=None):
            q = var.q(var.ep.env.elapsed_s)
            cb = None
            if after_substep is not None:
                def cb(*a):
                    return scaled(after_substep(*a), q)
            return inner(positions_mm, edge, body_radius_mm, commands_mm_s, active, scaled(solution, q), duration_s,
                         after_substep=cb)
        tp.advance = advance

    def actuate(self, world):
        """World-frame command the controller sent -> command the cluster actually follows."""
        env, n = self.ep.env, self.ep.n
        self.log_drift += -self.log_drift*self.dt/20.+.1*self.s*np.sqrt(2*self.dt/20.)*self.rng.normal(size=n)
        g = self.gain0*np.exp(self.log_drift)
        u = np.stack([self.rot[i]@world[i] for i in range(n)])*g[:, None]
        if self.tau > 1e-6:
            self.u_eff = self.u_eff+min(self.dt/self.tau, 1.)*(u-self.u_eff)
        else:
            self.u_eff = u
        u = self.u_eff.copy()
        if self.beta > 0:
            _, lumen, radial, _ = env.transport.coordinates(env.positions_mm[:n], env.edges[:n], env.solution)
            gap = np.maximum(lumen-env.body_radius[:n], 0.)
            u *= (np.clip(gap/.3, .4, 1.)**self.beta)[:, None]
        u[self.touch] *= 1-self.adhesion
        if self.noise > 0:
            u += self.rng.normal(0., self.noise, u.shape)*(np.linalg.norm(world, axis=1, keepdims=True) > 0)
        return u

    def observe_outcome(self, info):
        self.touch = np.asarray(info['wall_contact_s'], float)[:self.ep.n] > 0

    def summary(self):
        return dict(strength=self.s, flow_mean=self.flow_mean, pulsatility=self.amp, heart_rate_bpm=self.hr,
                    gain=self.gain0.tolist(), lag_s=self.tau, shape_beta=self.beta, adhesion=self.adhesion,
                    lubrication_floor=float(self.ep.env.transport.lubrication_floor), lysis_rate_x=self.lysis)
