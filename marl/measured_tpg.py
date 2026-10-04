"""Measured 3-D local routes and rolling dependency reservations.

Only measured crops/tracks enter this module. No simulator handle or native
topology is accepted. Unknown global route distance is represented by None.
This is a local TPG prototype, not a complete MAPF solver or safety certificate.
"""
from dataclasses import dataclass
from itertools import product
import heapq
import numpy as np


@dataclass
class MeasuredScene:
    positions: np.ndarray
    velocities: np.ndarray
    active: np.ndarray
    ages: np.ndarray
    crops: tuple
    targets: np.ndarray
    target_alive: np.ndarray
    time_s: float


def scene_from_processor(processor, packet, now):
    """Copy only delayed measurements and recorded commands, never env state."""
    n = processor.n
    positions, velocities = np.zeros((n, 3)), np.zeros((n, 3))
    ages = np.full(n, processor.spec.track_lifetime_s + 1.)
    crops = []
    from marl.tracked_sensors import empty_crop
    for i in range(n):
        crop = processor.crop_cache.get(i)
        crops.append(crop[1] if crop is not None and packet.active[i] else empty_crop())
        track = processor.tracks.get(('robot', i))
        if track is None or not packet.active[i]:
            continue
        stamp, p, _ = track
        age = max(0., now-stamp)
        drift = processor.drift.get(i, np.zeros(3))
        positions[i] = p+age*drift+processor.speed*processor.commands.integral(stamp, now)[i]
        velocities[i] = drift+processor.speed*processor.commands.at(now)[i]
        ages[i] = max(age, now-crop[0])
    return MeasuredScene(positions, velocities, packet.active.copy(), ages,
                         tuple(crops), processor.goals.copy(), processor.alive.copy(), float(now))


@dataclass
class LocalRoute:
    points: np.ndarray
    remaining_mm: object
    visible_length_mm: float
    lower_bound_mm: float
    radius_mm: float
    ambiguous: bool
    curvature: float

    @property
    def tangent(self):
        if len(self.points) < 2:
            return np.zeros(3)
        for point in self.points[1:]:
            d = point-self.points[0]
            if np.linalg.norm(d) > 1e-8:
                return d/np.linalg.norm(d)
        return np.zeros(3)


def polyline_prefix(points, length):
    out = [np.asarray(points[0], float)]
    for p in points[1:]:
        delta = p-out[-1]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            continue
        if distance >= length:
            out.append(out[-1]+delta*length/distance)
            break
        out.append(np.asarray(p, float)); length -= distance
    return np.asarray(out)


def measured_route(crop, position, target, settings):
    """Dijkstra on observed connectivity; never connect nearby separate tubes.

    Clipped frontier choice uses a labeled optimistic lower bound. It is not a
    claimed full route length or proof that the target is reachable.
    """
    position, target = np.asarray(position, float), np.asarray(target, float)
    lower = float(np.linalg.norm(target-position))
    if not len(crop.segments):
        return LocalRoute(position[None], None, 0., lower, 0., True, 0.)
    a, b = crop.segments[:, 0], crop.segments[:, 1]
    ab = b-a; lengths2 = np.sum(ab*ab, axis=1)
    def project(p):
        t = np.clip(np.sum((p-a)*ab, axis=1)/np.maximum(lengths2, 1e-12), 0., 1.)
        points = a+t[:, None]*ab
        distances = np.linalg.norm(points-p, axis=1)
        idx = int(np.argmin(distances))
        return idx, t, points, distances
    start_seg, sf, sp, sd = project(position)
    goal_seg, gf, gp, gd = project(target)
    # Adjacent segments sharing an observed endpoint are not distinct tubes.
    close = np.flatnonzero(sd <= sd[start_seg]+settings['attachment_ambiguity_mm'])
    ambiguous = any(j != start_seg and not any(np.linalg.norm(x-y) < 1e-6
        for x in crop.segments[j] for y in crop.segments[start_seg]) for j in close)
    points, adjacency, lookup = [], [], {}
    def vertex(p):
        key = tuple(np.round(p, 6))
        if key not in lookup:
            lookup[key] = len(points); points.append(np.asarray(p)); adjacency.append([])
        return lookup[key]
    start = vertex(sp[start_seg])
    goal = vertex(gp[goal_seg]) if gd[goal_seg] <= settings['target_attach_mm'] else None
    for j, (p, q) in enumerate(crop.segments):
        splits = [(0., vertex(p)), (1., vertex(q))]
        if j == start_seg: splits.append((sf[j], start))
        if goal is not None and j == goal_seg: splits.append((gf[j], goal))
        splits.sort()
        for (_, u), (_, v) in zip(splits, splits[1:]):
            length = float(np.linalg.norm(points[u]-points[v]))
            if u != v and length > 1e-10:
                adjacency[u].append((v, length)); adjacency[v].append((u, length))
    distances, parent, queue = {start: 0.}, {start: -1}, [(0., start)]
    while queue:
        distance, u = heapq.heappop(queue)
        if distance != distances[u]: continue
        for v, length in adjacency[u]:
            candidate = distance+length
            if candidate < distances.get(v, np.inf):
                distances[v], parent[v] = candidate, u
                heapq.heappush(queue, (candidate, v))
    known = goal is not None and goal in distances and not ambiguous
    if known:
        end = goal
    else:
        leaves = [u for u in distances if u != start and len(adjacency[u]) == 1]
        choices = leaves or [u for u in distances if u != start]
        end = min(choices, key=lambda u: distances[u]+np.linalg.norm(points[u]-target)) if choices else start
    chain = [end]
    while parent[chain[-1]] != -1: chain.append(parent[chain[-1]])
    chain.reverse()
    route = [position]+[points[u] for u in chain]
    if known: route.append(target)
    route = np.asarray(route)
    length = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    directions = np.diff(route, axis=0)
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 1e-7]/norms[norms > 1e-7, None]
    curvature = float(np.max(1-np.sum(directions[1:]*directions[:-1], axis=1))) if len(directions)>1 else 0.
    radius = float(np.min(crop.radii[start_seg]))
    return LocalRoute(route, length if known else None, length, lower, radius, ambiguous, curvature)


def segment_distance(a, b, c, d):
    """Exact minimum Euclidean separation of two closed 3-D line segments."""
    a, b, c, d = map(lambda x: np.asarray(x, float), (a, b, c, d))
    u, v, w = b-a, d-c, a-c
    aa, bb, cc, dd, ee = u@u, u@v, v@v, u@w, v@w
    candidates = []
    for s in (0., 1.):
        t = np.clip((ee+bb*s)/max(cc, 1e-15), 0., 1.)
        candidates.append(np.linalg.norm(w+s*u-t*v))
    for t in (0., 1.):
        s = np.clip((bb*t-dd)/max(aa, 1e-15), 0., 1.)
        candidates.append(np.linalg.norm(w+s*u-t*v))
    determinant = aa*cc-bb*bb
    if determinant > 1e-14:
        s, t = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(np.linalg.norm(w+s*u-t*v))
    return float(min(candidates))


def polyline_distance(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if len(first) == 1: first = np.repeat(first, 2, axis=0)
    if len(second) == 1: second = np.repeat(second, 2, axis=0)
    return min(segment_distance(a,b,c,d) for a,b in zip(first, first[1:])
               for c,d in zip(second, second[1:]))


def measured_assignment(scene, previous, route_settings):
    """Known local route costs or explicitly optimistic, penalized unknown costs."""
    ids = np.flatnonzero(scene.active); targets = np.flatnonzero(scene.target_alive)
    result = np.asarray(previous, int).copy()
    if not len(targets): return np.full(len(result), -1, int)
    costs = {}
    for i in ids:
        for j in targets:
            route = measured_route(scene.crops[i], scene.positions[i], scene.targets[j], route_settings)
            estimate = route.remaining_mm if route.remaining_mm is not None else route.lower_bound_mm+2.
            costs[i, j] = estimate-.4*(previous[i] == j)
    best, score = None, np.inf
    for assignment in product(targets, repeat=len(ids)):
        cost = sum(costs[i,j] for i,j in zip(ids,assignment))+4.*(len(assignment)-len(set(assignment)))
        if cost < score: score, best = cost, assignment
    if best is not None: result[ids] = best
    return result


class RollingTPG:
    """Persistent acyclic precedence for measured local route reservations.

    Edges release only after repeated fresh observations of separation. Missing
    observations never release a reservation. Acyclicity alone does not ensure
    physical feasibility, safe waiting, or progress in a blocked one-way tube.
    """
    def __init__(self, n, settings):
        self.n, self.settings = n, settings
        self.edges = np.zeros((n,n), bool)
        self.clear_counts = np.zeros((n,n), int)
        self.conflicts = np.zeros((n,n), bool)
        self.distances = np.full((n,n), np.inf)
        self.calls = self.priority_switches = self.releases = 0
        self.previous_order = tuple(range(n))

    def inspect(self, routes, active, fresh):
        conflicts = self.conflicts.copy()
        fresh = np.broadcast_to(np.asarray(fresh, bool), (self.n,))
        threshold = self.settings['spacing_mm']+self.settings['margin_mm']
        for i in range(self.n):
            for j in range(i+1,self.n):
                if not (active[i] and active[j]):
                    self.clear_counts[i,j] = 0
                    continue
                pair_fresh = bool(fresh[i] and fresh[j])
                distance = polyline_distance(routes[i],routes[j])
                self.distances[i,j] = self.distances[j,i] = distance
                if distance < threshold:
                    conflicts[i,j] = conflicts[j,i] = True
                    self.clear_counts[i,j] = 0
                elif pair_fresh and distance >= threshold+.2:
                    self.clear_counts[i,j] += 1
                    if self.clear_counts[i,j] >= self.settings['release_confirmations']:
                        conflicts[i,j] = conflicts[j,i] = False
                        if self.edges[i,j] or self.edges[j,i]: self.releases += 1
                        self.edges[i,j] = self.edges[j,i] = False
                else:
                    self.clear_counts[i,j] = 0
        new = bool(np.any(conflicts & ~(self.edges|self.edges.T)))
        self.conflicts = conflicts
        return new

    def schedule(self, requested_order):
        order = tuple(map(int,requested_order))
        if sorted(order) != list(range(self.n)): raise ValueError('A permutation is required')
        # Respect existing reservations while selecting among currently ready nodes.
        rank = {node:i for i,node in enumerate(order)}
        pending = set(order); admitted=[]
        while pending:
            ready = [i for i in pending if not any(self.edges[j,i] for j in pending)]
            if not ready: raise RuntimeError('Cyclic reservations are invalid')
            node = min(ready,key=rank.get); admitted.append(node); pending.remove(node)
        rank = {node:i for i,node in enumerate(admitted)}
        for i in range(self.n):
            for j in range(i+1,self.n):
                if self.conflicts[i,j] and not (self.edges[i,j] or self.edges[j,i]):
                    a,b = (i,j) if rank[i]<rank[j] else (j,i)
                    self.edges[a,b] = True
        self.calls += 1
        self.priority_switches += int(tuple(admitted) != self.previous_order)
        self.previous_order = tuple(admitted)
        return tuple(admitted)

    def ready(self, active):
        return np.asarray(active,bool) & ~self.edges.any(axis=0)


def tangent_frame(tangent):
    t = np.asarray(tangent,float)
    t = t/max(float(np.linalg.norm(t)),1e-9)
    if np.linalg.norm(t)<.5: return np.eye(3)
    reference = np.eye(3)[int(np.argmin(np.abs(t)))]
    n = np.cross(t,reference); n /= np.linalg.norm(n)
    return np.stack((t,n,np.cross(t,n)))
