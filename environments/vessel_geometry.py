"""Branch-aware vessel centerline geometry.

The original environment stored the centerline as a flat point cloud and used a
global nearest-neighbour query for wall projection and flow lookup. That has a
concrete failure mode at a bifurcation: the closest centerline point to a robot
travelling up the superior branch can sit on the *inferior* branch, so the robot
is handed the wrong tangent, the wrong wall and the wrong flow direction.

`VesselTree` keeps the branch topology explicit instead:

  * every station carries a branch id and an arclength measured from the inlet,
  * a station graph records which stations are actually connected, including the
    second junction of an anastomosis (which makes the vessel a graph, not a
    tree, so tree-only shortest-path logic would be wrong),
  * `route_to` runs Dijkstra on that graph, so "how far along the vessel is this
    clot" and "which way do I turn to reach it" are exact geodesics rather than
    Euclidean approximations that point straight through a vessel wall,
  * nearest-station queries can be restricted to a branch neighbourhood, which
    supplies the temporal continuity that stops robots teleporting between
    branches,
  * per-branch flow fractions follow Murray's law, so distal branches carry less
    flow than the parent instead of every branch advecting at one global speed.

Nothing here knows about robots, clots or rewards; it is pure geometry and is
unit-tested on its own in `tests/test_vessel_geometry.py`.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Legacy 3-segment layouts. Kept as the default randomisation pool so results
# stay comparable with earlier runs.
SCENARIOS = ("straight", "bifurcation", "anastomosis", "stenotic")

# Multi-generation / anatomical topologies built by `vessel_tree_generator`.
# These are what the supervisor's "too simple" critique asks for: 4 generations
# of Murray-law branching, and an MCA thrombectomy template.
GENERATED_SCENARIOS = ("multilevel", "mca_stroke")


def _anatomical_names() -> tuple[str, ...]:
    """Names of the anatomical territory scenarios.

    Imported lazily: `vessel_anatomy` imports this module for `Branch` and
    `VesselTree`, so a module-level import here would be circular.
    """
    from environments.vessel_anatomy import TERRITORIES

    return tuple(TERRITORIES)


# Anatomical territories, modelled on the sites where thrombus actually lodges.
ANATOMICAL_SCENARIOS = _anatomical_names()

# Named pools a caller can hand to `randomize_scenario`. "legacy" is the
# default so existing runs are unchanged; "anatomical" and "all" are what a new
# training run would use.
SCENARIO_POOLS: dict[str, tuple[str, ...]] = {
    "legacy": SCENARIOS,
    "generated": GENERATED_SCENARIOS,
    "anatomical": ANATOMICAL_SCENARIOS,
    "arterial": (),      # filled below from the territory metadata
    "venous": (),
    "all": SCENARIOS + GENERATED_SCENARIOS + ANATOMICAL_SCENARIOS,
}


def _fill_pools() -> None:
    from environments.vessel_anatomy import TERRITORIES

    for key in ("arterial", "venous"):
        SCENARIO_POOLS[key] = tuple(
            name for name, spec in TERRITORIES.items() if spec.circulation == key
        )


_fill_pools()


def resolve_pool(pool: str | Sequence[str]) -> tuple[str, ...]:
    """Turn a pool name or an explicit scenario list into a tuple of names."""
    if isinstance(pool, str):
        try:
            names = SCENARIO_POOLS[pool]
        except KeyError:
            raise ValueError(
                f"unknown scenario pool {pool!r}; "
                f"choose from {sorted(SCENARIO_POOLS)}"
            ) from None
        if not names:
            raise ValueError(f"scenario pool {pool!r} is empty")
        return names
    names = tuple(pool)
    bad = [n for n in names if n not in ALL_SCENARIOS]
    if bad:
        raise ValueError(f"unknown scenarios {bad}")
    if not names:
        raise ValueError("scenario pool is empty")
    return names


# Everything a caller may legally ask for.
ALL_SCENARIOS = SCENARIOS + GENERATED_SCENARIOS + ANATOMICAL_SCENARIOS


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Row-wise normalisation that leaves (near-)zero rows at zero."""
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norm, 1e-8)).astype(np.float32)


def _catmull_rom(control: np.ndarray, samples: int) -> np.ndarray:
    """Sample a Catmull-Rom spline through `control` (>=2 points).

    Real vessels are curved and the original environment interpolated straight
    line segments, which made "follow the tangent" a trivially constant policy
    inside each branch. A C1 spline gives curvature that the agent has to track.
    The endpoints are duplicated so the curve is interpolating at both ends.
    """
    control = np.asarray(control, dtype=np.float32)
    if control.shape[0] < 2:
        raise ValueError("need at least 2 control points")
    if control.shape[0] == 2:
        t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
        return ((1.0 - t) * control[0] + t * control[1]).astype(np.float32)

    padded = np.concatenate([control[:1], control, control[-1:]], axis=0)
    n_spans = padded.shape[0] - 3
    # Distribute samples over spans, keeping the final endpoint exactly.
    ts = np.linspace(0.0, float(n_spans), samples, dtype=np.float32)
    span = np.clip(np.floor(ts).astype(np.int32), 0, n_spans - 1)
    u = (ts - span)[:, None]

    p0, p1 = padded[span], padded[span + 1]
    p2, p3 = padded[span + 2], padded[span + 3]
    u2, u3 = u * u, u * u * u
    # Uniform Catmull-Rom basis (tension 0.5).
    out = (
        0.5
        * (
            (2.0 * p1)
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
        )
    )
    return out.astype(np.float32)


@dataclass
class Branch:
    """One vessel segment: a contiguous run of stations in the global arrays."""

    branch_id: int
    start: int          # first global station index (inclusive)
    stop: int           # last global station index (inclusive)
    parent: int         # parent branch id, -1 for the inlet branch
    flow_fraction: float  # share of inlet flow carried by this branch

    @property
    def size(self) -> int:
        return self.stop - self.start + 1


class VesselTree:
    """Branch-aware centerline with geodesic routing and Frenet frames.

    Station arrays are global and concatenated in branch order; `branch_ids`
    maps each station back to its branch. `station_graph` holds the true
    connectivity as an adjacency list of (neighbour, edge_length) pairs.
    """

    def __init__(
        self,
        points: np.ndarray,
        radii: np.ndarray,
        branches: list[Branch],
        extra_links: list[tuple[int, int]] | None = None,
        scenario: str = "custom",
    ) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.radii = np.ascontiguousarray(radii, dtype=np.float32)
        self.branches = branches
        self.scenario = scenario
        self.n_stations = int(self.points.shape[0])

        self.branch_ids = np.zeros((self.n_stations,), dtype=np.int32)
        for br in branches:
            self.branch_ids[br.start : br.stop + 1] = br.branch_id

        self._build_frames()
        self._build_graph(extra_links or [])
        self._build_arclength()

    # ------------------------------------------------------------------ frames

    def _build_frames(self) -> None:
        """Per-station tangent, plus a rotation-minimising normal/binormal.

        A naive per-station Frenet frame flips sign wherever curvature passes
        through zero, which would inject discontinuities straight into the
        observation. Parallel transport of the normal along the curve keeps the
        frame continuous.
        """
        pts = self.points
        tangents = np.zeros_like(pts)
        for br in self.branches:
            seg = pts[br.start : br.stop + 1]
            if seg.shape[0] < 2:
                tangents[br.start] = np.array([1.0, 0.0, 0.0], np.float32)
                continue
            # Central differences inside, one-sided at the two ends.
            d = np.zeros_like(seg)
            d[1:-1] = seg[2:] - seg[:-2]
            d[0] = seg[1] - seg[0]
            d[-1] = seg[-1] - seg[-2]
            tangents[br.start : br.stop + 1] = _unit(d)
        self.tangents = tangents

        normals = np.zeros_like(pts)
        binormals = np.zeros_like(pts)
        for br in self.branches:
            t = tangents[br.start : br.stop + 1]
            # Seed with any axis not parallel to the first tangent.
            seed = np.array([0.0, 0.0, 1.0], np.float32)
            if abs(float(np.dot(t[0], seed))) > 0.9:
                seed = np.array([0.0, 1.0, 0.0], np.float32)
            n_prev = _unit((seed - np.dot(seed, t[0]) * t[0])[None, :])[0]
            for k in range(t.shape[0]):
                # Project the previous normal onto the current normal plane
                # (double-reflection-free rotation minimising frame).
                n_k = n_prev - float(np.dot(n_prev, t[k])) * t[k]
                nrm = float(np.linalg.norm(n_k))
                if nrm < 1e-6:
                    alt = np.array([1.0, 0.0, 0.0], np.float32)
                    if abs(float(np.dot(alt, t[k]))) > 0.9:
                        alt = np.array([0.0, 1.0, 0.0], np.float32)
                    n_k = alt - float(np.dot(alt, t[k])) * t[k]
                    nrm = float(np.linalg.norm(n_k))
                n_k = (n_k / max(nrm, 1e-8)).astype(np.float32)
                normals[br.start + k] = n_k
                binormals[br.start + k] = np.cross(t[k], n_k)
                n_prev = n_k
        self.normals = normals
        self.binormals = binormals

    # ------------------------------------------------------------------- graph

    def _build_graph(self, extra_links: list[tuple[int, int]]) -> None:
        """Adjacency list over stations, weighted by Euclidean edge length."""
        graph: list[list[tuple[int, float]]] = [[] for _ in range(self.n_stations)]

        def link(a: int, b: int) -> None:
            w = float(np.linalg.norm(self.points[a] - self.points[b]))
            graph[a].append((b, w))
            graph[b].append((a, w))

        # Intra-branch chain.
        for br in self.branches:
            for i in range(br.start, br.stop):
                link(i, i + 1)

        # Parent -> child: attach the child's first station to whichever parent
        # station is geometrically closest, which is the junction point.
        for br in self.branches:
            if br.parent < 0:
                continue
            parent = self.branches[br.parent]
            pslice = self.points[parent.start : parent.stop + 1]
            j = int(np.argmin(np.linalg.norm(pslice - self.points[br.start], axis=1)))
            link(parent.start + j, br.start)

        # Anastomosis-style merges supplied by the caller.
        for a, b in extra_links:
            link(int(a), int(b))

        self.station_graph = graph

    def _build_arclength(self) -> None:
        """Geodesic distance from the inlet station to every other station."""
        self.inlet_station = int(self.branches[0].start)
        self.arclength, _ = self._dijkstra(self.inlet_station)
        finite = self.arclength[np.isfinite(self.arclength)]
        self.total_length = float(finite.max()) if finite.size else 1.0

    def _dijkstra(self, source: int) -> tuple[np.ndarray, np.ndarray]:
        """Shortest-path distance and predecessor tree rooted at `source`."""
        dist = np.full((self.n_stations,), np.inf, dtype=np.float32)
        dist[source] = 0.0
        # parent[v] is the station v was finalised from; the root points at
        # itself. Following parents always terminates at the root because the
        # predecessor relation is a tree, which is what makes this safe in the
        # presence of zero-length edges (coincident stations at a junction,
        # where a local "strictly smaller distance" test would stall).
        parent = np.arange(self.n_stations, dtype=np.int32)
        visited = np.zeros((self.n_stations,), dtype=bool)
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            d, u = heapq.heappop(queue)
            if visited[u]:
                continue
            visited[u] = True
            for v, w in self.station_graph[u]:
                if visited[v]:
                    continue
                nd = d + w
                # `<=` rather than `<` so a zero-length edge still establishes a
                # parent link, otherwise a coincident junction station keeps its
                # self-parent and the route dead-ends there.
                if nd < dist[v] or (nd <= dist[v] and parent[v] == v and v != source):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(queue, (nd, v))
        return dist, parent

    def route_to(self, target_station: int) -> tuple[np.ndarray, np.ndarray]:
        """Geodesic field toward one target station.

        Returns:
            distance: [n_stations] along-vessel distance to the target.
            next_hop: [n_stations] the neighbouring station to step to in order
                to approach the target. This is what turns "which branch do I
                take at the fork" into a lookup instead of a guess.

        Dijkstra rooted at the target gives both: edges are undirected so
        distance-from-target equals distance-to-target, and the predecessor tree
        is exactly the next-hop field.
        """
        distance, parent = self._dijkstra(int(target_station))
        return distance, parent

    # -------------------------------------------------------------- projection

    def nearest_station(
        self, positions: np.ndarray, hint: np.ndarray | None = None, window: int = 12
    ) -> np.ndarray:
        """Closest station per position, optionally restricted near `hint`.

        Without a hint this is the old global argmin. With one, the search is
        limited to stations within `window` graph steps of the hint, which is
        what prevents a robot in the superior branch from being snapped onto the
        inferior branch just because that station happens to be marginally
        closer in Euclidean terms.

        The hint is the previous step's station, so the restriction encodes
        continuity of motion: a robot cannot change branch without passing
        through the junction.
        """
        positions = np.asarray(positions, dtype=np.float32)
        if hint is None:
            d = np.linalg.norm(
                positions[:, None, :] - self.points[None, :, :], axis=2
            )
            return np.argmin(d, axis=1).astype(np.int32)

        out = np.zeros((positions.shape[0],), dtype=np.int32)
        for i, pos in enumerate(positions):
            cand = self._neighbourhood(int(hint[i]), window)
            local = np.linalg.norm(self.points[cand] - pos, axis=1)
            out[i] = cand[int(np.argmin(local))]
        return out

    def _neighbourhood(self, station: int, window: int) -> np.ndarray:
        """Stations within `window` graph hops of `station` (BFS)."""
        seen = {station}
        frontier = [station]
        for _ in range(window):
            nxt = []
            for u in frontier:
                for v, _w in self.station_graph[u]:
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            if not nxt:
                break
            frontier = nxt
        return np.fromiter(sorted(seen), dtype=np.int32, count=len(seen))

    def project(
        self, positions: np.ndarray, robot_radius: float, hint: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Clamp positions into the lumen using segment (not point) projection.

        Point-based projection quantises the wall to the station spacing, which
        makes the lumen a chain of spheres rather than a tube. Projecting onto
        the *segment* through the nearest station gives a smooth wall.

        Returns:
            clamped: positions moved inside the lumen.
            outside: bool mask, True where the position had left the lumen.
            station: nearest station index per position.
            axis_point: the closest point on the centerline polyline.
        """
        positions = np.asarray(positions, dtype=np.float32)
        station = self.nearest_station(positions, hint=hint)
        axis_point, radius = self._axis_point(positions, station)

        offset = positions - axis_point
        dist = np.linalg.norm(offset, axis=1)
        limit = np.maximum(radius - robot_radius, 1e-4)
        outside = dist > limit
        safe = np.maximum(dist, 1e-8)[:, None]
        clamped = axis_point + offset / safe * limit[:, None]
        result = np.where(outside[:, None], clamped, positions).astype(np.float32)
        return result, outside, station, axis_point

    def _axis_point(
        self, positions: np.ndarray, station: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closest point on the polyline near `station`, with lumen radius there.

        Considers the two segments incident to the station along its own branch
        and keeps whichever projection is closer.
        """
        best_point = self.points[station].copy()
        best_radius = self.radii[station].copy()
        best_d = np.linalg.norm(positions - best_point, axis=1)

        for step in (-1, 1):
            other = station + step
            # Stay inside the same branch: crossing a branch boundary in index
            # space is not a real geometric neighbour.
            same = (
                (other >= 0)
                & (other < self.n_stations)
                & (self.branch_ids[np.clip(other, 0, self.n_stations - 1)]
                   == self.branch_ids[station])
            )
            other = np.clip(other, 0, self.n_stations - 1)
            a = self.points[station]
            b = self.points[other]
            ab = b - a
            denom = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
            t = np.clip(np.sum((positions - a) * ab, axis=1) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(positions - proj, axis=1)
            radius = (1.0 - t) * self.radii[station] + t * self.radii[other]

            take = same & (d < best_d)
            best_point = np.where(take[:, None], proj, best_point)
            best_radius = np.where(take, radius, best_radius)
            best_d = np.where(take, d, best_d)

        return best_point.astype(np.float32), best_radius.astype(np.float32)

    # -------------------------------------------------------------------- flow

    def flow(
        self,
        positions: np.ndarray,
        station: np.ndarray,
        inlet_speed: float,
        reference_radius: float,
        radius_override: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advection velocity, obeying continuity and Murray's law.

        Two corrections over a single global `flow_speed`:

        1. Continuity. Volumetric flow through a branch is fixed, so the mean
           axial speed scales as (r_ref / r_local)^2. A stenosis that halves the
           radius therefore quadruples the local speed. The original constant
           made a stenosis "narrower but equally slow", which removed the only
           interesting thing about that scenario.
        2. Murray's law. Each branch carries `flow_fraction` of the inlet flow,
           so daughter vessels are slower than the parent rather than every
           branch advecting identically.

        `radius_override` lets the caller pass the *occluded* radius (clot
        present) so that a clot both narrows the lumen and accelerates flow past
        it, which is the physical coupling the point-clot model lacked.
        """
        positions = np.asarray(positions, dtype=np.float32)
        tangent = self.tangents[station]
        axis_point, geom_radius = self._axis_point(positions, station)
        radius = geom_radius if radius_override is None else radius_override
        radius = np.maximum(radius, 1e-4)

        fractions = np.array(
            [br.flow_fraction for br in self.branches], dtype=np.float32
        )
        frac = fractions[self.branch_ids[station]]

        # Mean axial speed from continuity, capped so a near-total occlusion
        # cannot produce an unbounded velocity (and a jump larger than the
        # integrator step could tunnel through the wall).
        speed = inlet_speed * frac * np.square(reference_radius / radius)
        speed = np.minimum(speed, inlet_speed * 8.0)

        # Poiseuille profile, normalised to the *mean* so that the continuity
        # scaling above refers to mean flow: u(r) = 2*u_mean*(1 - (r/R)^2).
        radial = np.linalg.norm(positions - axis_point, axis=1)
        profile = 2.0 * np.clip(1.0 - np.square(radial / radius), 0.0, 1.0)
        return (tangent * (speed * profile)[:, None]).astype(np.float32)

    # --------------------------------------------------------------- lookahead

    def lookahead(
        self, station: np.ndarray, offsets: tuple[int, ...], next_hop: np.ndarray | None
    ) -> np.ndarray:
        """Centerline points ahead of each station.

        Returns [n_positions, len(offsets), 3] world-space points.

        When `next_hop` is supplied the walk follows the geodesic toward the
        agent's target, so at a fork the lookahead bends into the branch the
        agent actually needs. That is the single most important piece of missing
        information in the original 20-D observation: without it the agent
        cannot see a bifurcation at all, let alone choose a side.

        Without `next_hop` the walk advances by station index within the branch,
        which is the downstream direction.
        """
        station = np.asarray(station, dtype=np.int32)
        max_offset = max(offsets)
        # Walk one hop at a time and record the stations we want.
        path = np.zeros((station.shape[0], max_offset + 1), dtype=np.int32)
        path[:, 0] = station
        cur = station.copy()
        for k in range(1, max_offset + 1):
            if next_hop is not None:
                cur = next_hop[cur]
            else:
                nxt = cur + 1
                same_branch = (nxt < self.n_stations) & (
                    self.branch_ids[np.clip(nxt, 0, self.n_stations - 1)]
                    == self.branch_ids[cur]
                )
                cur = np.where(same_branch, np.clip(nxt, 0, self.n_stations - 1), cur)
            path[:, k] = cur
        picks = path[:, list(offsets)]
        return self.points[picks]

    # ------------------------------------------------------------------ helpers

    def stations_of_branch(self, branch_id: int) -> np.ndarray:
        br = self.branches[branch_id]
        return np.arange(br.start, br.stop + 1, dtype=np.int32)

    @property
    def n_branches(self) -> int:
        return len(self.branches)


# ---------------------------------------------------------------- construction


def build_vessel_tree(
    scenario: str,
    rng: np.random.Generator,
    base_radius: float = 0.055,
    min_radius: float = 0.0135,
) -> VesselTree:
    """Sample a randomized curved vessel tree for one of the SCENARIOS.

    Radii follow Murray's law: for a parent splitting into daughters,
    r_parent^3 = sum(r_daughter^3). Combined with the flow fractions this makes
    the branching physically consistent instead of every vessel having the same
    radius and speed.

    New scenarios (multi-generation trees and anatomical presets) are handled by
    `vessel_tree_generator` before falling back to the legacy 3-segment layouts.
    """
    # Route new multi-generation and anatomical scenarios to the new generator.
    if scenario in ANATOMICAL_SCENARIOS:
        from environments.vessel_anatomy import build_territory

        return build_territory(scenario, rng=rng, min_radius=min_radius)

    if scenario in ("multilevel", "mca_stroke"):
        from environments.vessel_tree_generator import (
            generate_tree,
            preset_mca_stroke,
            segments_to_vessel_tree,
        )

        if scenario == "mca_stroke":
            return preset_mca_stroke(base_radius=base_radius, rng=rng)

        # multilevel: generic multi-generation branching tree
        root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        root_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        segments = generate_tree(
            root_pos=root_pos,
            root_dir=root_dir,
            root_radius=base_radius,
            generations=3,
            branch_factor=2,
            asymmetry=float(rng.uniform(0.0, 0.3)),
            tortuosity=float(rng.uniform(0.05, 0.15)),
            segment_length=float(rng.uniform(0.12, 0.18)),
            bifurcation_angle=float(rng.uniform(25.0, 40.0)),
            rng=rng,
        )
        tree = segments_to_vessel_tree(segments, n_per_segment=20)
        tree.scenario = scenario
        return tree

    # Legacy scenarios (bifurcation, stenotic, anastomosis, aneurysm)
    if scenario not in ALL_SCENARIOS:
        raise ValueError(f"scenario must be one of {ALL_SCENARIOS}, got {scenario!r}")

    jitter = lambda s: rng.uniform(-s, s, size=3).astype(np.float32)
    inlet = np.array([0.08, 0.5, 0.5], np.float32) + jitter(0.03)
    fork = np.array([0.46, 0.5, 0.5], np.float32) + jitter(0.04)

    # Curvature: a mid control point pushed off the inlet-fork chord.
    def curved(a: np.ndarray, b: np.ndarray, bow: float, n: int) -> np.ndarray:
        mid = 0.5 * (a + b)
        chord = b - a
        # Any direction perpendicular to the chord, randomly rotated about it.
        ref = np.array([0.0, 0.0, 1.0], np.float32)
        if abs(float(np.dot(_unit(chord[None, :])[0], ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], np.float32)
        u = _unit(np.cross(chord, ref)[None, :])[0]
        v = _unit(np.cross(chord, u)[None, :])[0]
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        push = (np.cos(phi) * u + np.sin(phi) * v) * bow
        return _catmull_rom(np.stack([a, mid + push, b]), n)

    trunk_r = base_radius * float(rng.uniform(0.9, 1.1))
    specs: list[tuple[np.ndarray, int, float, float]] = []  # pts, parent, radius, flow
    extra_links: list[tuple[int, int]] = []

    if scenario == "straight":
        outlet = np.array([0.92, 0.5, 0.5], np.float32) + jitter(0.05)
        pts = curved(inlet, outlet, float(rng.uniform(0.03, 0.10)), 120)
        specs.append((pts, -1, trunk_r, 1.0))

    elif scenario == "anastomosis":
        up = np.array([0.88, 0.74, 0.66], np.float32) + jitter(0.04)
        dn = np.array([0.88, 0.26, 0.34], np.float32) + jitter(0.04)
        merge = np.array([0.95, 0.5, 0.5], np.float32)
        # Murray split, then the two limbs re-merge into a common outlet.
        split = float(rng.uniform(0.42, 0.58))
        r_up = trunk_r * split ** (1.0 / 3.0)
        r_dn = trunk_r * (1.0 - split) ** (1.0 / 3.0)
        specs.append((curved(inlet, fork, float(rng.uniform(0.02, 0.07)), 55), -1, trunk_r, 1.0))
        specs.append((curved(fork, up, float(rng.uniform(0.03, 0.09)), 50), 0, r_up, split))
        specs.append((curved(fork, dn, float(rng.uniform(0.03, 0.09)), 50), 0, r_dn, 1.0 - split))
        specs.append((curved(up, merge, float(rng.uniform(0.01, 0.05)), 28), 1, r_up, split))
        specs.append((curved(dn, merge, float(rng.uniform(0.01, 0.05)), 28), 2, r_dn, 1.0 - split))
        # Close the loop: the two distal limbs meet, which is what makes an
        # anastomosis a graph with a cycle rather than a tree.
        extra_links.append((-1, -2))  # resolved to real indices below

    else:  # bifurcation, stenotic
        up = np.array([0.90, 0.76, 0.70], np.float32) + jitter(0.05)
        dn = np.array([0.90, 0.24, 0.30], np.float32) + jitter(0.05)
        split = float(rng.uniform(0.40, 0.60))
        r_up = trunk_r * split ** (1.0 / 3.0)
        r_dn = trunk_r * (1.0 - split) ** (1.0 / 3.0)
        specs.append((curved(inlet, fork, float(rng.uniform(0.02, 0.08)), 55), -1, trunk_r, 1.0))
        specs.append((curved(fork, up, float(rng.uniform(0.04, 0.11)), 65), 0, r_up, split))
        specs.append((curved(fork, dn, float(rng.uniform(0.04, 0.11)), 65), 0, r_dn, 1.0 - split))

    # Assemble global arrays.
    points_list, radii_list, branches = [], [], []
    cursor = 0
    for bid, (pts, parent, radius, flow) in enumerate(specs):
        n = pts.shape[0]
        points_list.append(pts)
        # Mild distal taper within each branch, on top of the Murray step.
        taper = np.linspace(1.0, float(rng.uniform(0.88, 1.0)), n, dtype=np.float32)
        radii_list.append(np.maximum(radius * taper, min_radius))
        branches.append(
            Branch(branch_id=bid, start=cursor, stop=cursor + n - 1,
                   parent=parent, flow_fraction=float(flow))
        )
        cursor += n

    points = np.concatenate(points_list, axis=0)
    radii = np.concatenate(radii_list, axis=0)

    # Resolve the anastomosis merge link to the two distal end stations.
    links: list[tuple[int, int]] = []
    for a, b in extra_links:
        links.append((branches[a].stop, branches[b].stop))

    if scenario == "stenotic":
        # Focal stenosis on a randomly chosen branch, as a smooth Gaussian
        # constriction rather than a boxcar so the wall stays differentiable.
        bid = int(rng.integers(0, len(branches)))
        br = branches[bid]
        idx = np.arange(br.start, br.stop + 1)
        centre = float(rng.uniform(0.35, 0.7)) * (br.size - 1) + br.start
        width = max(float(rng.uniform(0.06, 0.14)) * br.size, 2.0)
        severity = float(rng.uniform(0.45, 0.70))  # residual radius fraction
        bump = np.exp(-0.5 * np.square((idx - centre) / width))
        radii[idx] *= 1.0 - (1.0 - severity) * bump
        radii = np.maximum(radii, min_radius)

    return VesselTree(points, radii, branches, extra_links=links, scenario=scenario)
