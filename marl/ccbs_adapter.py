"""A fixed-route continuous-time conflict-search adapter.

This implements the CCBS comparison under the same route, speed, clearance,
and dwell assumptions as ``TPGCoordinator``. Search changes precedence in
conflict zones only, so it isolates temporal coordination from route planning.
For the three-cluster benchmark exhaustive branching is small and deterministic.
"""
from __future__ import annotations

import itertools
import math
import numpy as np


class CCBSPlanner:
    def __init__(self, coordinator, max_nodes=4096):
        self.c = coordinator
        self.max_nodes = int(max_nodes)
        self.nodes_expanded = 0
        self.planning_time_s = 0.0
        self.infeasible = False
        self.order = {}
        self.cost = math.inf

    def _timeline(self, i, waits):
        c = self.c
        n = len(c.paths[i])
        dwell_at = {int(round(x / c.step)): c.dwell for x in c.clot_s[i]}
        t = np.zeros(n)
        for k in range(1, n):
            t[k] = t[k - 1] + c.step / c.speed + dwell_at.get(k - 1, 0.) + waits.get(k - 1, 0.)
        return t

    def _simulate(self, order):
        c = self.c
        waits = {i: {} for i in range(c.n)}
        timelines = {}
        total_wait = 0.
        for i in sorted(range(c.n), key=lambda x: -len(c.paths[x])):
            for _ in range(32):
                t = self._timeline(i, waits[i]); changed = False
                for zi, z in enumerate(c.zones):
                    if i not in (z['i'], z['j']) or order.get(zi) == i:
                        continue
                    other = z['j'] if z['i'] == i else z['i']
                    if other not in timelines:
                        continue
                    a0, a1 = (z['s0'], z['s1']) if z['i'] == i else (z['u0'], z['u1'])
                    b0, b1 = (z['u0'], z['u1']) if z['i'] == i else (z['s0'], z['s1'])
                    me, mx = t[a0], t[min(a1, len(t) - 1)]
                    ot = timelines[other]
                    oe, ox = ot[b0], ot[min(b1, len(ot) - 1)]
                    if me < ox and mx > oe:
                        k = max(a0 - int(round(c.stop_margin / c.step)) - 1, 0)
                        delay = max(0., ox - me)
                        waits[i][k] = waits[i].get(k, 0.) + delay
                        total_wait += delay; changed = True; break
                if not changed:
                    break
            timelines[i] = self._timeline(i, waits[i])
        finish = sum(float(t[-1]) for t in timelines.values())
        return finish + total_wait, waits

    def plan(self):
        import time
        started = time.monotonic()
        if not self.c.zones:
            self.order = {}; self.cost = float(sum(len(p) * self.c.step / self.c.speed for p in self.c.paths))
        else:
            zones = list(range(len(self.c.zones)))
            for bits in itertools.product((0, 1), repeat=len(zones)):
                if self.nodes_expanded >= self.max_nodes:
                    self.infeasible = True; break
                order = {}
                for zi, bit in zip(zones, bits):
                    z = self.c.zones[zi]; order[zi] = z['i'] if bit == 0 else z['j']
                cost, _ = self._simulate(order); self.nodes_expanded += 1
                if cost < self.cost:
                    self.cost, self.order = cost, order
            if not self.order:
                self.infeasible = True
        self.planning_time_s = time.monotonic() - started
        self.c.order = dict(self.order)
        return self

    def metadata(self):
        return {"algorithm": "CCBS-fixed-route-adapter", "route_policy": "fixed",
                "speed_mm_s": float(self.c.speed), "clearance_mm": float(getattr(self.c, "D", 0.)),
                "nodes_expanded": int(self.nodes_expanded),
                "planning_time_s": float(self.planning_time_s),
                "infeasible": bool(self.infeasible), "conflict_zones": len(self.c.zones),
                "order": {str(k): int(v) for k, v in self.order.items()}}
