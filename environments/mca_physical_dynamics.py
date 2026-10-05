"""Explicit-mm tube transport for EXP22B; independent of legacy step physics.

Piecewise circular tubes and quasi-steady edge flow are reduced-order models,
not a bifurcation CFD solution. Branch transitions use local tube geometry,
never targets or a navigation controller. Actions are held in WORLD coordinates
for a control interval. Exits absorb bodies instead of projecting them back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environments.mca_physiology import _finite_scalar


@dataclass
class TransportResult:
    positions_mm: np.ndarray
    edge: np.ndarray
    active: np.ndarray
    path_mm: np.ndarray
    wall_contact_s: np.ndarray
    blocked_s: np.ndarray
    exit_time_s: np.ndarray
    exit_node: np.ndarray
    substeps: int
    elapsed_s: float


class PhysicalTubeTransport:
    """Flow, collision-wall projection and open-boundary bookkeeping in mm.

    The CFL-like limit bounds displacement by local radius AND edge length.
    It controls time resolution, never changes blood or commanded speed.
    Exceeding the compute budget raises; no partial rollout is returned.
    Body-body effects are reported by the environment, not silently modelled
    as impulses with uncalibrated units.
    """

    def __init__(self, flow_model, *, spatial_fraction=.1, max_substeps=100000,
                 lubrication_floor=1.0, junction_model='graph'):
        if junction_model not in ('graph', 'union'):
            raise ValueError('junction_model must be graph or union')
        self.junction_model = junction_model
        self.model = flow_model
        self.tree = flow_model.tree
        self.scale = flow_model.mm_per_unit
        self.fraction = _finite_scalar(spatial_fraction, 'spatial_fraction')
        self.lubrication_floor = _finite_scalar(lubrication_floor, 'lubrication_floor')
        if self.fraction > .25 or self.lubrication_floor > 1:
            raise ValueError('spatial_fraction <= .25 and lubrication_floor <= 1 required')
        if not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError('max_substeps must be a positive integer')
        self.max_substeps = max_substeps
        self.points = np.asarray(self.tree.points, np.float64) * self.scale
        # Collapse only coincident graph junctions, retaining each nonzero
        # edge's OWN endpoint radius (a small tributary cannot narrow a trunk).
        group = np.arange(self.tree.n_stations)

        def root(node):
            while group[node] != node:
                node = group[node]
            return node

        edges = []
        for node in flow_model.order[1:]:
            parent = int(flow_model.parent[node])
            length = np.linalg.norm(self.points[node] - self.points[parent])
            if length <= 1e-8:
                group[root(node)] = root(parent)
            else:
                edges.append((parent, node))
        if not edges:
            raise ValueError('Geometry has no positive-length edges')
        self.groups = np.array([root(n) for n in range(len(group))])
        self.ends = np.asarray(edges, np.int32)
        self.a = self.points[self.ends[:, 0]]
        self.ab = self.points[self.ends[:, 1]] - self.a
        self.length = np.linalg.norm(self.ab, axis=1)
        self.direction = self.ab / self.length[:, None]
        edge_groups = self.groups[self.ends]
        self.edge_groups = edge_groups
        candidates, gates = [], []
        for e, groups in enumerate(edge_groups):
            nearby = np.flatnonzero(np.isin(edge_groups, groups).any(axis=1))
            candidates.append(nearby)
            gate = []
            for other in nearby:
                shared = np.intersect1d(groups, edge_groups[other])
                own_nodes = self.ends[e][np.isin(groups, shared)]
                other_nodes = self.ends[other][np.isin(edge_groups[other], shared)]
                gate.append((int(own_nodes[0]), int(other_nodes[0])))
            gates.append(gate)
        width = max(map(len, candidates))
        self.candidates = np.zeros((len(edges), width), np.int32)
        self.candidate_valid = np.zeros((len(edges), width), bool)
        self.gates = np.zeros((len(edges), width, 2), np.int32)
        for e, choices in enumerate(candidates):
            self.candidates[e, :len(choices)] = choices
            self.candidate_valid[e, :len(choices)] = True
            self.gates[e, :len(choices)] = gates[e]
        if self.junction_model == 'union':
            # Take-off segments can be shorter than a body diameter, so the tubes a body can be inside
            # are those within two junction hops of its edge, not only the edges sharing its ends.
            hop = [set(c[v].tolist()) for c, v in zip(self.candidates, self.candidate_valid)]
            two = [sorted(set().union(*(hop[o] for o in hop[e]))) for e in range(len(hop))]
            width = max(map(len, two))
            self.candidates = np.zeros((len(two), width), np.int32)
            self.candidate_valid = np.zeros((len(two), width), bool)
            for e, c in enumerate(two):
                self.candidates[e, :len(c)] = c; self.candidate_valid[e, :len(c)] = True
        self.root_group = int(self.groups[flow_model.root])
        self.terminal_groups = set(self.groups[n] for n in flow_model.order
                                   if not flow_model.children[n])

    def coordinates(self, positions, edge, solution):
        """Piecewise-linear centerline and lumen radius; constant edge flux."""
        a, ab = self.a[edge], self.ab[edge]
        t = np.clip(np.sum((positions - a) * ab, axis=-1) / self.length[edge] ** 2, 0, 1)
        axis = a + t[..., None] * ab
        radii = solution['radius_mm'][self.ends[edge]]
        radius = radii[..., 0] * (1 - t) + radii[..., 1] * t
        radial = np.linalg.norm(positions - axis, axis=-1)
        return axis, radius, radial, t

    def nearest_edges(self, positions):
        positions = np.asarray(positions, np.float64)
        t = np.clip(np.sum((positions[:, None] - self.a) * self.ab, axis=-1)
                    / self.length ** 2, 0, 1)
        axis = self.a + t[..., None] * self.ab
        return np.argmin(np.sum((positions[:, None] - axis) ** 2, axis=-1), axis=1)

    def velocity_mm_s(self, positions, edge, solution):
        axis, radius, radial, _ = self.coordinates(positions, edge, solution)
        del axis
        flux = solution['station_inflow_mm3_s'][self.ends[edge, 1]]
        mean = np.divide(flux, np.pi * radius ** 2, out=np.zeros_like(radius), where=radius > 0)
        eta = np.divide(radial, radius, out=np.ones_like(radius), where=radius > 0)
        return (2 * mean * np.maximum(1 - eta ** 2, 0))[:, None] * self.direction[edge]

    def _velocity(self, positions, edge, radius, commands, solution):
        _, lumen, radial, _ = self.coordinates(positions, edge, solution)
        surface_gap = np.maximum(lumen - radial - radius, 0)
        factor = self.lubrication_floor + (1 - self.lubrication_floor) * np.clip(
            surface_gap / np.maximum(4 * radius, 1e-12), 0, 1)
        return self.velocity_mm_s(positions, edge, solution) + commands * factor[:, None]

    def _project_union(self, proposed, previous, edge, body_radius, solution):
        """Union-of-tubes lumen (junction_model='union').

        The accessible region near a body is the union of the capsules (segment + hemispherical caps,
        radius = local lumen radius minus body radius) of its current edge and every edge sharing one of
        its junctions. Feasibility depends only on the 3-D position: a side branch opens where its tube
        pierces the parent wall, and a body can enter any daughter by moving into it. The membership edge
        (used for the local flow field and the next neighbourhood) is the containing capsule with the
        smallest radial/radius ratio. Outside every capsule the body is projected onto the nearest one
        (wall contact); if no capsule admits the body at all, it keeps its previous position (blocked)."""
        n = len(proposed)
        fixed = proposed.copy(); new_edge = edge.copy()
        wall = np.zeros(n, bool); blocked = np.zeros(n, bool)
        for i in range(n):
            choices = self.candidates[edge[i]][self.candidate_valid[edge[i]]]
            axis, radius, radial, _ = self.coordinates(np.repeat(proposed[i:i+1], len(choices), 0), choices, solution)
            limit = radius-body_radius[i]
            ok = limit >= 0
            inside = ok & (radial <= limit)
            if inside.any():
                k = np.flatnonzero(inside)[np.argmin((radial/np.maximum(radius, 1e-12))[inside])]
                new_edge[i] = choices[k]
                continue
            if ok.any():
                viol = np.where(ok, radial-limit, np.inf); k = int(np.argmin(viol))
                fixed[i] = axis[k]+(proposed[i]-axis[k])*(limit[k]/max(radial[k], 1e-30))
                new_edge[i] = choices[k]; wall[i] = True
                continue
            fixed[i] = previous[i]; blocked[i] = True
        return fixed, new_edge, wall, blocked

    def _project(self, proposed, previous, edge, body_radius, solution):
        if self.junction_model == 'union':
            return self._project_union(proposed, previous, edge, body_radius, solution)
        # Stay on the current edge until the proposed point crosses one of its
        # endpoints. Nearest-edge projection around a junction allows a large
        # Euler step to teleport into a daughter branch; that produced 20--30 mm
        # refinement jumps even when the local motion was smooth.
        # coordinates() clips t for interpolation. Crossing detection must use
        # the unbounded axial coordinate or every internal endpoint is a wall.
        current_t = np.sum((proposed - self.a[edge]) * self.direction[edge], axis=-1) / self.length[edge]
        new_edge = edge.copy()
        axial_motion = np.sum((proposed - previous) * self.direction[edge], axis=-1)
        # A newly entered curved tube may contain the point in its proximal
        # cap. While it moves inward, t<0 is NOT a crossing back upstream.
        # Without this direction test small step sizes change outlet identity.
        endpoint_tol = 1e-10 / self.length[edge]
        crossed = ((current_t <= endpoint_tol) & (axial_motion < 0)) | (
                    (current_t >= 1 - endpoint_tol) & (axial_motion > 0))
        for i in np.flatnonzero(crossed):
            endpoint = 0 if axial_motion[i] < 0 else 1
            group = self.edge_groups[edge[i], endpoint]
            choices = self.candidates[edge[i]]
            valid = self.candidate_valid[edge[i]]
            choices = choices[valid]
            choices = choices[(choices != edge[i]) &
                              np.any(self.edge_groups[choices] == group, axis=1)]
            if not len(choices):
                continue
            delta = proposed[i] - previous[i]
            # Orient the candidate with the requested crossing direction. At a
            # junction, the best aligned edge wins; a narrow candidate is kept
            # in the list so the body is blocked rather than reprojected.
            away = np.where(self.edge_groups[choices, 0] == group, 1., -1.)
            scores = (self.direction[choices] * away[:, None]) @ delta
            pick = int(np.argmax(scores))
            new_edge[i] = choices[pick]
        axis, local_radius, _, _ = self.coordinates(proposed, new_edge, solution)
        # A transition is allowed only through an endpoint whose opening fits.
        transitioned = crossed & (new_edge != edge)
        blocked_gate = np.zeros(len(edge), dtype=bool)
        if np.any(transitioned):
            endpoint = np.where(axial_motion[transitioned] < 0, 0, 1)
            old = edge[transitioned]
            group = self.edge_groups[old, endpoint]
            new = new_edge[transitioned]
            new_endpoint = (self.edge_groups[new, 1] == group).astype(int)
            opening = np.minimum(solution['radius_mm'][self.ends[old, endpoint]],
                                 solution['radius_mm'][self.ends[new, new_endpoint]]) - body_radius[transitioned]
            blocked_transition = opening < 0
            ids = np.flatnonzero(transitioned)[blocked_transition]
            blocked_gate[ids] = True
            new_edge[ids] = edge[ids]
            axis[ids], local_radius[ids], _, _ = self.coordinates(
                proposed[ids], edge[ids], solution)
            transitioned[ids] = False
        delta = proposed - axis
        radial = np.linalg.norm(delta, axis=-1)
        limit = local_radius - body_radius
        blocked = (limit < 0) | blocked_gate
        wall = radial > np.maximum(limit, 0)
        fixed = axis + delta * np.minimum(1., np.maximum(limit, 0) / np.maximum(radial, 1e-30))[:, None]
        # A narrowing may block advance. Keeping the previous valid position
        # makes this a contact event; there is no minimum navigable radius.
        fixed[blocked] = previous[blocked]
        new_edge[blocked] = edge[blocked]
        return fixed, new_edge, wall & ~blocked, blocked

    def _boundary_crossings(self, previous, proposed, edge, body_radius, solution):
        node = np.full(len(edge), -1, np.int32)
        fraction = np.ones(len(edge))
        point = proposed.copy()
        for sign, endpoint, groups in ((-1, 0, {self.root_group}),
                                       (1, 1, self.terminal_groups)):
            possible = np.isin(self.edge_groups[edge, endpoint], list(groups))
            boundary = self.points[self.ends[edge, endpoint]]
            direction = self.direction[edge] * sign
            before = np.sum((previous - boundary) * direction, axis=-1)
            after = np.sum((proposed - boundary) * direction, axis=-1)
            crossing = possible & (before <= 1e-9) & (after > 0)
            alpha = np.divide(-before, after - before, out=np.ones_like(before),
                              where=after > before)
            alpha = np.clip(alpha, 0, 1)
            intersection = previous + alpha[:, None] * (proposed - previous)
            radial = np.linalg.norm(intersection - boundary, axis=-1)
            aperture = solution['radius_mm'][self.ends[edge, endpoint]] - body_radius
            crossing &= (aperture >= 0) & (radial <= aperture + 1e-9)
            node[crossing] = self.ends[edge[crossing], endpoint]
            fraction[crossing] = alpha[crossing]
            point[crossing] = intersection[crossing]
        return node, fraction, point

    def advance(self, positions_mm, edge, body_radius_mm, commands_mm_s, active,
                solution, duration_s, *, after_substep=None):
        duration_s = _finite_scalar(duration_s, 'duration_s')
        pos = np.array(positions_mm, dtype=np.float64, copy=True)
        edge = np.array(edge, dtype=np.int32, copy=True)
        active = np.array(active, dtype=bool, copy=True)
        radius = np.asarray(body_radius_mm, np.float64)
        commands = np.asarray(commands_mm_s, np.float64)
        n = len(pos)
        if (pos.shape != (n, 3) or commands.shape != pos.shape or edge.shape != (n,) or
                active.shape != (n,) or radius.shape != (n,) or
                not np.isfinite(pos).all() or not np.isfinite(commands).all() or
                not np.isfinite(radius).all() or np.any(radius <= 0) or
                np.any((edge < 0) | (edge >= len(self.ends)))):
            raise ValueError('Invalid body positions, commands, radii, edges or mask')
        _, local_radius, distance, _ = self.coordinates(pos, edge, solution)
        if np.any(active & (distance + radius > local_radius + 1e-7)):
            raise ValueError('Active body starts outside the accessible lumen')
        path, walls, blocks = np.zeros(n), np.zeros(n), np.zeros(n)
        exit_time = np.full(n, np.nan)
        exit_node = np.full(n, -1, np.int32)
        elapsed, count = 0., 0
        while elapsed < duration_s - 1e-14 and np.any(active):
            if count >= self.max_substeps:
                raise RuntimeError('Physical integration budget exceeded; flow was NOT capped')
            ids = np.flatnonzero(active)
            current = pos[ids]
            e, a, command = edge[ids], radius[ids], commands[ids]
            # Use reachable incident edges to anticipate acceleration at a
            # junction, and both edge radii to anticipate a local stenosis.
            candidates = self.candidates[e]
            end_r = solution['radius_mm'][self.ends[candidates]]
            minimum_r = np.maximum(end_r.min(axis=-1), a[:, None])
            flux = solution['station_inflow_mm3_s'][self.ends[candidates, 1]]
            max_speed = np.where(self.candidate_valid[e],
                                 2 * flux / (np.pi * minimum_r ** 2), 0).max(axis=-1)
            max_speed += np.linalg.norm(command, axis=-1)
            local_min = np.where(self.candidate_valid[e], minimum_r, np.inf).min(axis=-1)
            edge_min = np.where(self.candidate_valid[e], self.length[candidates], np.inf).min(axis=-1)
            distance_limit = np.minimum(self.fraction * local_min, .25 * edge_min)
            dt_limit = np.divide(distance_limit, max_speed, out=np.full(len(ids), np.inf),
                                 where=max_speed > 0)
            v0 = self._velocity(current, e, a, command, solution)
            # Resolve the next centreline endpoint as an integration event.
            # A radius-based CFL limit alone still straddles discontinuous
            # edge directions/fluxes at an arbitrary phase of each step size.
            axial_position = np.sum((current - self.a[e]) * self.direction[e], axis=-1)
            axial_speed = np.sum(v0 * self.direction[e], axis=-1)
            to_end = np.where(axial_speed >= 0, self.length[e] - axial_position, axial_position)
            event_dt = np.divide(to_end, np.abs(axial_speed), out=np.full(len(ids), np.inf),
                                 where=(to_end > 1e-10) & (np.abs(axial_speed) > 0))
            dt = min(duration_s - elapsed, float(dt_limit.min()), float(event_dt.min()))
            if dt <= 0 or not np.isfinite(dt):
                raise FloatingPointError('Invalid physical integration step')
            # Projected explicit midpoint: evaluate shear/taper velocity at a
            # half step in the local lumen. Using start velocity for the full
            # step accumulates first-order errors for fast passive tracers.
            # Projection and edge selection are purely geometric; the held
            # world command is not steered or augmented by a controller.
            midpoint, mid_edge, _, _ = self._project(current + .5 * dt * v0, current, e, a, solution)
            vmid = self._velocity(midpoint, mid_edge, a, command, solution)
            # The vector field is discontinuous across graph junctions. A
            # midpoint on a daughter must not apply that daughter's velocity
            # retroactively on the parent (which can select another branch).
            # Use the incoming Euler direction for crossing steps and midpoint
            # only inside a single edge; geometric events stay one-sided.
            euler = current + dt * v0
            axial = np.sum((euler - self.a[e]) * self.direction[e], axis=-1) / self.length[e]
            smooth = (axial >= 0) & (axial <= 1) & (mid_edge == e)
            proposal = current + dt * np.where(smooth[:, None], vmid, v0)
            fixed, next_edge, wall, blocked = self._project(proposal, current, e, a, solution)
            # Absorb on the accepted, wall-constrained path. The raw proposal
            # can cross an outlet outside its aperture, then project back into
            # the opening; checking only the raw path misses that exit forever
            # and leaves a body stuck in a fictitious cap beyond the outlet.
            node, alpha, boundary_point = self._boundary_crossings(current, fixed, e, a, solution)
            exited = node >= 0
            fixed[exited] = boundary_point[exited]
            wall[exited], blocked[exited] = False, False
            old = pos.copy()
            previous_active = active.copy()
            pos[ids], edge[ids] = fixed, next_edge
            path[ids] += np.linalg.norm(fixed - current, axis=-1)
            walls[ids] += wall * dt
            blocks[ids] += blocked * dt
            exit_time[ids[exited]] = elapsed + dt * alpha[exited]
            exit_node[ids[exited]] = node[exited]
            active[ids[exited]] = False
            effective_dt = np.zeros(n)
            effective_dt[ids] = dt * np.where(exited, alpha, 1)
            if after_substep is not None:
                updated = after_substep(old, pos, edge, previous_active, effective_dt, dt)
                if updated is not None:
                    solution = updated
            elapsed += dt
            count += 1
        return TransportResult(pos, edge, active, path, walls, blocks,
                               exit_time, exit_node, count, duration_s)
