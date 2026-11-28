"""Bounded search over continuous-time precedence on fixed sampled vessel routes."""
import itertools
import math
import time
from collections import deque

import numpy as np


class CCBSPlanner:
    def __init__(self, coordinator, max_nodes=4096, deadline_s=math.inf):
        self.c = coordinator
        self.max_nodes = int(max_nodes)
        self.deadline_s = deadline_s
        self.nodes_expanded = 0
        self.planning_time_s = 0.
        self.infeasible = False
        self.search_exhausted = False
        self.order = {}
        self.cost = math.inf

    def _schedule_cost(self, order, nominal):
        # A release event is the first grid point strictly outside a zone.
        # A cluster parked at its final point never releases its terminal zone.
        c = self.c
        events = [{0, len(p) - 1} for p in c.paths]
        precedences = []
        for z, first in zip(c.zones, order):
            i, j = z['i'], z['j']
            exit_index, entry_index = z['s1'] + 1, z['u0']
            if first == j:
                i, j = j, i
                exit_index, entry_index = z['u1'] + 1, z['s0']
            if exit_index >= len(c.paths[i]) or entry_index == 0:
                return math.inf
            # The follower can wait before the zone. A wait must not move its
            # initial state, including when the first zone starts at index 1.
            events[i].add(exit_index)
            events[j].add(entry_index)
            precedences.append(((i, exit_index), (j, entry_index), 0.))
        edges = list(precedences)
        for i, indices in enumerate(events):
            seq = sorted(indices)
            for a, b in zip(seq, seq[1:]):
                edges.append(((i, a), (i, b), nominal[i][b] - nominal[i][a]))

        # Earliest feasible event times are longest paths. Even a zero-time
        # precedence cycle cannot be dispatched by the strict release gate.
        incoming = {(i, k): 0 for i, points in enumerate(events) for k in points}
        outgoing = {k: [] for k in incoming}
        times = dict.fromkeys(incoming, 0.)
        for a, b, dt in edges:
            outgoing[a].append((b, dt))
            incoming[b] += 1
        queue = deque(k for k, count in incoming.items() if count == 0)
        visited = 0
        while queue:
            a = queue.popleft()
            visited += 1
            for b, dt in outgoing[a]:
                times[b] = max(times[b], times[a] + dt)
                incoming[b] -= 1
                if incoming[b] == 0:
                    queue.append(b)
        if visited != len(times):
            return math.inf
        ends = [times[i, len(p) - 1] for i, p in enumerate(c.paths)]
        return sum(ends) if max(ends) <= self.deadline_s else math.inf

    def plan(self):
        started = time.monotonic()
        c = self.c
        nominal = []
        for i, path in enumerate(c.paths):
            durations = np.full(len(path) - 1, c.step / c.speed, dtype=float)
            for position in c.clot_s[i]:
                index = int(round(position / c.step))
                if index < len(durations):
                    durations[index] += c.dwell
            nominal.append(np.r_[0., np.cumsum(durations)])
        for order in itertools.product(*[(z['i'], z['j']) for z in c.zones]):
            if self.nodes_expanded >= self.max_nodes:
                self.search_exhausted = True
                break
            self.nodes_expanded += 1
            cost = self._schedule_cost(order, nominal)
            if cost < self.cost:
                self.cost, self.order = cost, dict(enumerate(order))
        self.infeasible = not self.search_exhausted and not math.isfinite(self.cost)
        self.planning_time_s = time.monotonic() - started
        c.order = dict(self.order)
        return self

    @property
    def has_plan(self):
        return math.isfinite(self.cost)

    def metadata(self):
        return {"algorithm": "CCBS-fixed-route-adapter", "route_policy": "fixed",
                "speed_mm_s": float(self.c.speed), "clearance_mm": float(getattr(self.c, "D", 0.)),
                "nodes_expanded": self.nodes_expanded, "planning_time_s": self.planning_time_s,
                "infeasible": self.infeasible, "node_limit_reached": self.search_exhausted,
                "has_plan": self.has_plan, "conflict_zones": len(self.c.zones),
                "order": {str(k): int(v) for k, v in self.order.items()}}
