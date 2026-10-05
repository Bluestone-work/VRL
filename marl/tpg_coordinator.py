"""Spacing-safe coordination of several clusters by a temporal plan graph on the pre-operative roadmap.

Motivation (measured, benchmark v1): with N>=2 the spacing filter resolves conflicts reactively, by
deflecting commands sideways; inside a vessel this pushes clusters into the wall (N=3: 14.3 % of
episodes with >=1 s wall contact vs 2.9 % without the filter), and stopping-only variants deadlock.
Magnetic interference depends on the 3-D Euclidean distance between clusters, not on vessel topology,
and every cluster's route is known before treatment (allocation A on the roadmap). Conflicts can
therefore be resolved in advance, in time instead of space.

1. Paths. Each cluster's whole tour (start -> its clots, in plan order) as a station polyline,
   resampled at `step_mm`, with the arclength of every clot along it.
2. Conflict zones. For every pair (i, j), the cells (s, u) of the two arclength grids whose points are
   closer than D = d_min + buffer form connected components; each component is one zone with an
   s-interval on i's path and a u-interval on j's path.
3. Offline schedule (prioritised planning). Clusters in priority order (longest planned tour first)
   are simulated at the actuator speed with a dwell at each clot (expected lysis time) and parked at
   their tour end. A lower-priority cluster may enter a zone only when the higher-priority cluster is
   not inside it at that time; otherwise it waits just before the zone. The schedule fixes, per zone,
   which cluster goes first. Because it is one consistent timed plan, the resulting order graph is
   acyclic (no deadlock by construction, under the schedule's assumptions).
4. Online execution. A cluster whose next zone has an unfinished predecessor stops just before the
   zone entry until the predecessor's measured progress has left the zone. Progress is the monotone
   projection of the cluster's position on its own path. Nothing is deflected sideways.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class TPGCoordinator:
    def __init__(self, env, plan, station_path, d_min_mm=2., buffer_mm=.6, step_mm=.2, speed_mm_s=1.,
                 dwell_s=None, stop_margin_mm=.3):
        self.env, self.n = env, len(plan)
        self.D, self.step, self.stop_margin = d_min_mm+buffer_mm, step_mm, stop_margin_mm
        pts = env.transport.points.astype(np.float64)
        starts, clots = np.asarray(env.robot_stations), np.asarray(env.clot_stations)
        self.paths, self.clot_s = [], []
        for i, seq in enumerate(plan):
            st, cur = [int(starts[i])], int(starts[i])
            for c in seq:
                p = station_path(cur, int(clots[c]))[::-1]
                st += p[1:]; cur = int(clots[c])
            poly = pts[st]
            seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
            s = np.r_[0., np.cumsum(seg)]
            grid = np.arange(0., s[-1]+1e-9, step_mm) if s[-1] > 0 else np.array([0.])
            res = np.stack([np.interp(grid, s, poly[:, k]) for k in range(3)], 1)
            self.paths.append(res)
            cs, acc = [], 0.
            for c in seq:
                cs.append(float(np.argmin(np.linalg.norm(res-env.clot_positions_mm[c], axis=1)))*step_mm)
            self.clot_s.append(cs)
        rate = env.config.lysis_mass_per_s
        self.dwell = dwell_s if dwell_s is not None else 1.5*float(np.mean(env.initial_mass))/max(rate, 1e-9)
        self.speed = speed_mm_s
        self._zones()
        self._schedule()
        self.progress = np.zeros(self.n, int)

    def _zones(self):
        """zones: list of dict(i, j, s0, s1, u0, u1) (grid indices)."""
        self.zones = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                A, B = self.paths[i], self.paths[j]
                d = np.linalg.norm(A[:, None, :]-B[None, :, :], axis=-1) < self.D
                lab, k = ndimage.label(d)
                for z in range(1, k+1):
                    ss, uu = np.nonzero(lab == z)
                    self.zones.append(dict(i=i, j=j, s0=int(ss.min()), s1=int(ss.max()), u0=int(uu.min()), u1=int(uu.max())))

    def _timeline(self, i, waits):
        """Arrival time at every grid index of path i given extra waits {index: seconds} (dwell at clots)."""
        n = len(self.paths[i]); t = np.zeros(n); dwell_at = {int(round(c/self.step)): self.dwell for c in self.clot_s[i]}
        for k in range(1, n):
            t[k] = t[k-1]+self.step/self.speed+dwell_at.get(k-1, 0.)+waits.get(k-1, 0.)
        return t

    def _occupancy(self, i, t, z_lo, z_hi):
        """Time interval cluster i spends inside index range [z_lo, z_hi] (parked forever at its end)."""
        enter = t[z_lo]
        last = len(t)-1
        if z_hi >= last:
            return enter, np.inf
        dwell = sum(self.dwell for c in self.clot_s[i] if z_lo <= int(round(c/self.step)) <= z_hi)
        return enter, t[z_hi]+self.step/self.speed+dwell

    def _schedule(self):
        lengths = [len(p) for p in self.paths]
        self.priority = sorted(range(self.n), key=lambda i: -(lengths[i]*self.step+self.dwell*len(self.clot_s[i])))
        self.waits = {i: {} for i in range(self.n)}
        self.order = {}                      # zone index -> cluster that goes first
        timelines = {}
        done = []
        for i in self.priority:
            for _ in range(50):
                t = self._timeline(i, self.waits[i]); changed = False
                for zi, z in enumerate(self.zones):
                    if i not in (z['i'], z['j']):
                        continue
                    other = z['j'] if z['i'] == i else z['i']
                    if other not in done:
                        continue
                    a0, a1 = (z['s0'], z['s1']) if z['i'] == i else (z['u0'], z['u1'])
                    b0, b1 = (z['u0'], z['u1']) if z['i'] == i else (z['s0'], z['s1'])
                    oe, ox = self._occupancy(other, timelines[other], b0, b1)
                    me, mx = self._occupancy(i, t, a0, a1)
                    if zi in self.order:
                        continue
                    if mx <= oe:                    # i passes completely before the other arrives
                        self.order[zi] = i
                    elif me >= ox:                  # the other has left before i arrives
                        self.order[zi] = other
                    elif np.isfinite(ox):           # wait before the zone until the other has left
                        k = max(a0-1-int(round(self.stop_margin/self.step)), 0)
                        self.waits[i][k] = self.waits[i].get(k, 0.)+(ox-me)
                        self.order[zi] = other; changed = True; break
                    else:                           # the other parks inside: i must have passed before it arrives
                        self.order[zi] = i          # (infeasible if i is later; execution then lets i wait forever)
                if not changed:
                    break
            timelines[i] = self._timeline(i, self.waits[i]); done.append(i)
        self.timelines = timelines

    def update_progress(self):
        P = self.env.positions_mm
        for i in range(self.n):
            path = self.paths[i]; k = self.progress[i]
            w = path[k:k+int(3./self.step)]
            if len(w):
                self.progress[i] = k+int(np.argmin(np.linalg.norm(w-P[i], axis=1)))

    def gate(self):
        """Boolean [n]: True = this cluster must hold position now."""
        self.update_progress()
        hold = np.zeros(self.n, bool)
        margin = int(round(self.stop_margin/self.step))+1
        for zi, z in enumerate(self.zones):
            first = self.order.get(zi)
            if first is None:
                continue
            second = z['j'] if first == z['i'] else z['i']
            f0, f1 = (z['s0'], z['s1']) if first == z['i'] else (z['u0'], z['u1'])
            s0, s1 = (z['s0'], z['s1']) if second == z['i'] else (z['u0'], z['u1'])
            first_done = self.progress[first] > f1 or not self.env.active[first]
            ps = self.progress[second]
            if not first_done and s0-margin-int(1./self.step) <= ps < s0:   # approaching the zone entry
                hold[second] = True
        return hold
