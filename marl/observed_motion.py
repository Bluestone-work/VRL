"""Timestamp-aligned affine motion identification from images and sent commands.

No environment object is accepted: only measured positions, acquisition clocks,
visibility flags, and the command log enter this estimator.  ``drift`` is an
effective model intercept, not a claim to recover the simulator's true flow.
"""
from __future__ import annotations

from collections import deque
import numpy as np


def fit_affine(commands, velocities, ridge=.025):
    c, v = np.asarray(commands, float), np.asarray(velocities, float)
    c0, v0 = c-c.mean(0), v-v.mean(0)
    excitation = float(np.sum(c0*c0))
    gain = float(np.clip((np.sum(c0*v0)+ridge)/(excitation+ridge), .25, 2.))
    drift = v.mean(0)-gain*c.mean(0)
    residual = v-gain*c-drift
    rmse = float(np.sqrt(np.mean(residual**2)))
    confidence = min(1., max(0., (len(c)-5)/15)) * excitation/(excitation+.08) / (1+(rmse/.15)**2)
    return gain, np.clip(drift, -2., 2.), rmse, confidence


class ObservedMotion:
    def __init__(self, n, dt=.1, window=32):
        self.n, self.dt = n, float(dt)
        self.commands = deque(maxlen=160)
        self.samples = [deque(maxlen=window) for _ in range(n)]
        self.gain = np.ones(n); self.drift = np.zeros((n, 3))
        self.rmse = np.ones(n); self.confidence = np.zeros(n)
        self.aligned_command = np.zeros((n, 3)); self.velocity = np.zeros((n, 3))
        self.previous_time = None; self.previous_position = None
        self.previous_edge = None; self.last_control_time = None
        self.fresh = False; self.interval = 0.; self.valid = np.zeros(n, bool)

    def interval_command(self, lo, hi):
        total = np.zeros((self.n, 3)); covered = 0.
        for start, end, command in self.commands:
            width = max(0., min(end, hi)-max(start, lo))
            total += width*command; covered += width
        if covered < hi-lo-1e-6:
            return None
        return total/max(hi-lo, 1e-8)

    def update(self, now, sent_command, frame_time, position, active, edge):
        now, frame_time = float(now), float(frame_time)
        if self.last_control_time is not None and now <= self.last_control_time+1e-8:
            return False
        if self.last_control_time is not None:
            self.commands.append((self.last_control_time, now, np.asarray(sent_command, float).copy()))
        self.last_control_time = now
        self.fresh = False; self.valid[:] = False
        if frame_time > now+1e-6 or not np.isfinite(frame_time):
            raise ValueError('Camera acquisition timestamp is in the future or invalid')
        if self.previous_time is not None and frame_time <= self.previous_time+1e-8:
            return False
        pos = np.asarray(position, float); edge = np.asarray(edge)
        if self.previous_time is not None:
            self.interval = frame_time-self.previous_time
            c = self.interval_command(self.previous_time, frame_time)
            if c is not None:
                v = (pos-self.previous_position)/self.interval
                valid = np.asarray(active, bool) & (edge == self.previous_edge)
                valid &= np.isfinite(v).all(1) & (np.linalg.norm(v, axis=1) < 4.)
                self.aligned_command = c; self.velocity = v; self.valid = valid
                for i in range(self.n):
                    if not valid[i]:
                        self.confidence[i] *= .8
                        continue
                    self.samples[i].append((c[i].copy(), v[i].copy()))
                    if len(self.samples[i]) >= 6:
                        C, V = zip(*self.samples[i])
                        gain, drift, rmse, confidence = fit_affine(C, V)
                        self.gain[i] = gain
                        self.drift[i] = drift
                        self.rmse[i] = rmse; self.confidence[i] = confidence
        self.previous_time = frame_time; self.previous_position = pos.copy(); self.previous_edge = edge.copy()
        self.fresh = True
        return True

