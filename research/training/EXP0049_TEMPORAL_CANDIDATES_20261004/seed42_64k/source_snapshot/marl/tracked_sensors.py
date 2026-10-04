"""EXP0048 image-like measurements and observation-only tracking adapter.

The renderer may read simulator geometry/body state to synthesize measurements.
The measurement processor has no environment and cannot read routes, edge IDs,
true velocity/flow, exact clot mass, or a true Frenet coordinate frame.
Synthetic segmentation and camera registration remain unvalidated assumptions.
"""
from collections import deque
from dataclasses import dataclass
import heapq
import numpy as np

from marl.multicluster import ClusterObservation
from marl.partial_obs import CLOT0, PATH0, PART0, MATE0


@dataclass(frozen=True)
class TrackingSpec:
    position_error_mm: float = .02
    centerline_error_mm: float = .02
    radius_relative_error: float = .05
    preop_error_mm: float = .05
    dropout: float = .05
    clot_classification_error: float = .01
    latency_s: float = .1
    track_lifetime_s: float = .3
    velocity_alpha: float = .25
    absence_confirmations: int = 3
    geometry_view_mm: float = 4.
    peer_view_mm: float = 6.
    particle_view_mm: float = 1.5
    voxel_mm: float = .02


@dataclass(frozen=True)
class GeometryCrop:
    segments: np.ndarray             # image-space segment endpoints, [M,2,3]
    radii: np.ndarray                # measured endpoint radii, [M,2]


@dataclass(frozen=True)
class ImageFrame:
    time_s: float
    robot_centroids: np.ndarray       # missing detections are NaN
    particle_ids: np.ndarray          # tracker association labels, not edge IDs
    particle_centroids: np.ndarray
    geometry: tuple
    clot_visible: np.ndarray
    clot_present: np.ndarray          # binary classifier, never mass fractions
    clot_centroids: np.ndarray


def empty_crop():
    return GeometryCrop(np.empty((0, 2, 3)), np.empty((0, 2)))


class SimulatedLocalImager:
    """Renderer only. No route, flow, body velocity, station, or edge queries."""
    def __init__(self, env, spec, seed):
        self.env, self.spec = env, spec
        self.rng = np.random.default_rng(seed)
        points = np.asarray(env.transport.points, float)
        unique, inverse = np.unique(np.round(points, 8), axis=0, return_inverse=True)
        error = self.rng.normal(0., spec.centerline_error_mm, unique.shape)
        self.measured_skeleton = points+error[inverse]
        self.radius_gain = np.maximum(.2, self.rng.normal(1., spec.radius_relative_error, len(points)))
        self.preoperative_targets = np.asarray(env.clot_positions_mm, float)+self.rng.normal(
            0., spec.preop_error_mm, np.shape(env.clot_positions_mm))

    def _crop(self, center):
        if not np.isfinite(center).all(): return empty_crop()
        e, s = self.env, self.spec
        points = self.measured_skeleton+self.rng.normal(0., .005, (1, 3))
        points = np.round(points/s.voxel_mm)*s.voxel_mm
        ends = np.asarray(e.transport.ends)
        a, b = points[ends[:, 0]], points[ends[:, 1]]
        ab, delta = b-a, a-center
        aa = np.sum(ab*ab, axis=1)
        bb = 2*np.sum(delta*ab, axis=1)
        cc = np.sum(delta*delta, axis=1)-s.geometry_view_mm**2
        disc = bb*bb-4*aa*cc
        valid = (disc >= 0) & (aa > 1e-12)
        lo = np.maximum(0., (-bb-np.sqrt(np.maximum(disc, 0.)))/(2*np.maximum(aa, 1e-12)))
        hi = np.minimum(1., (-bb+np.sqrt(np.maximum(disc, 0.)))/(2*np.maximum(aa, 1e-12)))
        valid &= hi-lo > 1e-6
        if not np.any(valid): return empty_crop()
        # Radius is an image-like measurement of currently rendered lumen.
        # Global radius arrays never leave the renderer.
        radius = np.asarray(e.solution['radius_mm'])*self.radius_gain
        ra, rb = radius[ends[:, 0]], radius[ends[:, 1]]
        segment = np.stack((a+lo[:, None]*ab, a+hi[:, None]*ab), axis=1)[valid]
        measured_radius = np.stack((ra+(rb-ra)*lo, ra+(rb-ra)*hi), axis=1)[valid]
        return GeometryCrop(segment, np.maximum(measured_radius, .01))

    def render(self, time_s):
        e, s = self.env, self.spec
        n = e.num_robots
        robots = np.asarray(e.positions_mm[:n], float)+self.rng.normal(0., s.position_error_mm, (n, 3))
        visible = np.asarray(e.active[:n], bool) & (self.rng.random(n) >= s.dropout)
        robots[~visible] = np.nan
        crops = tuple(self._crop(p) for p in robots)
        ids = np.flatnonzero(e.active[n:])+n
        p = np.asarray(e.positions_mm[ids], float)
        if len(p) and np.any(visible):
            detected = np.min(np.linalg.norm(p[:, None]-robots[None, visible], axis=-1), axis=1) <= s.particle_view_mm
            detected &= self.rng.random(len(p)) >= s.dropout
            ids, p = ids[detected], p[detected]
        else:
            ids, p = ids[:0], p[:0]
        p = p+self.rng.normal(0., s.position_error_mm, p.shape)
        clots = np.asarray(e.clot_positions_mm, float)
        cv = (np.min(np.linalg.norm(clots[:, None]-robots[None, visible], axis=-1), axis=1) <= s.peer_view_mm
              if np.any(visible) else np.zeros(len(clots), bool))
        cv &= self.rng.random(len(clots)) >= s.dropout
        present = np.asarray(e.masses > 0, bool).copy()
        present ^= self.rng.random(len(clots)) < s.clot_classification_error
        present[~cv] = False
        centers = clots+self.rng.normal(0., s.position_error_mm, clots.shape)
        centers[~cv | ~present] = np.nan
        return ImageFrame(float(time_s), robots, ids-n, p, crops, cv, present, centers)


def geometry_features(crop, position, body_radius, view_radius):
    """Reconstruct only a cropped image graph, rooted by measured proximity."""
    nav = np.zeros(40, np.float32)
    if not len(crop.segments): return np.zeros(3), 0., nav
    a, b = crop.segments[:, 0], crop.segments[:, 1]
    ab = b-a
    fraction = np.clip(np.sum((position-a)*ab, axis=1)/np.maximum(np.sum(ab*ab, axis=1), 1e-12), 0, 1)
    projected = a+fraction[:, None]*ab
    index = int(np.argmin(np.linalg.norm(projected-position, axis=1)))
    axis = projected[index]
    radius = max(float(crop.radii[index, 0]+fraction[index]*(crop.radii[index, 1]-crop.radii[index, 0])), .01)
    radial = position-axis
    clearance = (radius-np.linalg.norm(radial)-body_radius)/radius
    nodes, lookup, adjacency, node_radius = [], {}, [], []
    def vertex(point, r):
        key = tuple(np.round(point, 6))
        if key not in lookup:
            lookup[key] = len(nodes); nodes.append(point); adjacency.append([]); node_radius.append(float(r))
        return lookup[key]
    def connect(u, v):
        length = float(np.linalg.norm(nodes[u]-nodes[v]))
        if u != v and length > 1e-9:
            adjacency[u].append((v, length)); adjacency[v].append((u, length))
    pairs = []
    for seg, radii in zip(crop.segments, crop.radii):
        pairs.append((vertex(seg[0], radii[0]), vertex(seg[1], radii[1])))
    start = vertex(axis, radius)
    for j, (u, v) in enumerate(pairs):
        if j == index:
            connect(u, start); connect(start, v)
        else:
            connect(u, v)
    distance, parent = {start: 0.}, {start: -1}
    queue = [(0., start)]
    while queue:
        d, u = heapq.heappop(queue)
        if d != distance[u] or d >= view_radius: continue
        for v, length in adjacency[u]:
            candidate = d+length
            if candidate < distance.get(v, np.inf):
                distance[v], parent[v] = candidate, u
                heapq.heappush(queue, (candidate, v))
    children = set(parent.values())
    paths = []
    for end in parent:
        if end == start or end in children: continue
        chain = [end]
        while parent[chain[-1]] != start: chain.append(parent[chain[-1]])
        chain.reverse()
        steer = next((v for v in chain if np.linalg.norm(nodes[v]-position) > .06), chain[-1])
        direction = nodes[steer]-position
        far = nodes[end]-position
        direction = direction/max(float(np.linalg.norm(direction)), 1e-9)
        far = far/max(float(np.linalg.norm(far)), 1e-9)
        # Only a locally visible interior endpoint can be a measured dead end.
        dead = len(adjacency[end]) == 1 and np.linalg.norm(nodes[end]-position) < view_radius-.2
        paths.append((direction, far, min(distance[end], view_radius)/view_radius, float(dead), np.log(max(node_radius[end], .01))))
    paths.sort(key=lambda p: tuple(-p[0]))
    for k, (direction, far, length, dead, log_radius) in enumerate(paths[:4]):
        nav[10*k:10*k+10] = np.r_[1., direction, far, length, dead, log_radius]
    return radial/radius, float(clearance), nav


class MeasurementProcessor:
    """Pure measured-data policy interface, deliberately without an env field."""
    def __init__(self, n, preoperative_targets, spec, *, speed=1., body_radius=.08, duration=180.):
        self.n, self.spec, self.speed, self.body_radius, self.duration = n, spec, speed, body_radius, duration
        self.goals = np.array(preoperative_targets, float, copy=True)
        self.alive = np.ones(len(self.goals), bool)
        self.absent = np.zeros(len(self.goals), int)
        self.tracks = {}
        self.crop_cache = {}
        self.last_commands = np.zeros((n, 3))
        self.last_frame_time = -np.inf

    def ingest(self, frame):
        if frame.time_s <= self.last_frame_time: return
        self.last_frame_time = frame.time_s
        objects = [(('robot', i), p) for i, p in enumerate(frame.robot_centroids) if np.isfinite(p).all()]
        objects += [(('particle', int(i)), p) for i, p in zip(frame.particle_ids, frame.particle_centroids)]
        for key, position in objects:
            velocity = np.zeros(3)
            if key in self.tracks:
                old_time, old_position, old_velocity = self.tracks[key]
                measured = (position-old_position)/max(frame.time_s-old_time, 1e-9)
                velocity = self.spec.velocity_alpha*measured+(1-self.spec.velocity_alpha)*old_velocity
            self.tracks[key] = (frame.time_s, position.copy(), velocity)
        for i, crop in enumerate(frame.geometry):
            if len(crop.segments): self.crop_cache[i] = (frame.time_s, crop)
        for j in np.flatnonzero(frame.clot_visible):
            if frame.clot_present[j]:
                self.absent[j] = 0
                if np.isfinite(frame.clot_centroids[j]).all():
                    self.goals[j] = .5*self.goals[j]+.5*frame.clot_centroids[j]
            else:
                self.absent[j] += 1
                if self.absent[j] >= self.spec.absence_confirmations: self.alive[j] = False

    def observe(self, now):
        n, s = self.n, self.spec
        positions, velocities = np.zeros((n, 3)), np.zeros((n, 3))
        active = np.zeros(n, bool)
        for i in range(n):
            track = self.tracks.get(('robot', i))
            crop = self.crop_cache.get(i)
            if track is None or crop is None: continue
            stamp, p, v = track
            if now-stamp > s.track_lifetime_s+1e-9 or now-crop[0] > s.track_lifetime_s+1e-9: continue
            active[i] = True; positions[i] = p+(now-stamp)*v; velocities[i] = v
        rel = positions[None]-positions[:, None]
        relv = velocities[None]-velocities[:, None]
        visible = (np.linalg.norm(rel, axis=-1) <= s.peer_view_mm) & active[:, None] & active[None]
        np.fill_diagonal(visible, False)
        rel[~visible] = 0.; relv[~visible] = 0.
        nav = np.zeros((n, 111), np.float32)
        clot_ids = np.full((n, 4), -1, np.int32)
        for i in np.flatnonzero(active):
            o = nav[i]
            o[:3], o[3:6] = self.last_commands[i], velocities[i]/self.speed
            offset, clearance, paths = geometry_features(self.crop_cache[i][1], positions[i], self.body_radius, s.geometry_view_mm)
            o[6:9], o[9], o[10], o[PATH0:PATH0+40] = offset, clearance, max(0., 1-now/self.duration), paths
            targets = self.goals-positions[i]
            distances = np.linalg.norm(targets, axis=1)
            order = np.argsort(np.where(self.alive, distances, np.inf))[:4]
            for k, j in enumerate(order):
                if not self.alive[j]: continue
                clot_ids[i, k] = j
                o[CLOT0+6*k:CLOT0+6*k+6] = np.r_[1., distances[j]/10,
                    targets[j]/max(distances[j], 1e-9), 1.]
            particles = []
            for (kind, ident), (stamp, p, v) in self.tracks.items():
                if kind != 'particle' or now-stamp > s.track_lifetime_s+1e-9: continue
                delta = p+(now-stamp)*v-positions[i]
                distance = np.linalg.norm(delta)
                if distance <= s.particle_view_mm: particles.append((distance, delta, v-velocities[i]))
            for k, (_, delta, velocity) in enumerate(sorted(particles, key=lambda p: p[0])[:4]):
                o[PART0+7*k:PART0+7*k+7] = np.r_[1., delta/s.particle_view_mm, velocity/self.speed]
            peers = sorted(np.flatnonzero(visible[i]), key=lambda j: np.linalg.norm(rel[i, j]))
            for k, j in enumerate([j for j in peers if np.linalg.norm(rel[i, j]) < s.particle_view_mm][:2]):
                o[MATE0+4*k:MATE0+4*k+4] = np.r_[1., rel[i, j]/s.particle_view_mm]
        return ClusterObservation(nav, clot_ids, rel, relv, visible, active)


class TrackedSensorAdapter:
    def __init__(self, env, config, spec=TrackingSpec()):
        self.env, self.config, self.spec = env, config, spec
        self.reset(0)

    def reset(self, seed):
        self.imager = SimulatedLocalImager(self.env, self.spec, seed)
        self.processor = MeasurementProcessor(self.env.num_robots, self.imager.preoperative_targets,
            self.spec, speed=self.env.config.robot_speed_mm_s,
            body_radius=self.env.config.robot_radius_mm, duration=self.env.config.episode_duration_s)
        self.queue = deque()

    def observe(self):
        now = float(self.env.elapsed_s)
        self.queue.append(self.imager.render(now))
        while self.queue and self.queue[0].time_s <= now-self.spec.latency_s+1e-9:
            self.processor.ingest(self.queue.popleft())
        return self.processor.observe(now)

    def execute(self, commands):
        world = np.asarray(commands, float)
        world = world/np.maximum(np.linalg.norm(world, axis=1, keepdims=True), 1.)
        self.processor.last_commands = world.copy()
        return world
