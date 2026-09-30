"""Connectivity-aware clot-task allocation for the vascular swarm.

Adapts the connectivity/contiguity idea from C²-Explorer (multi-robot
exploration under communication constraints) to vascular thrombolysis:

  * C²-Explorer keeps a communicating team connected by preferring frontier
    cells reachable WITHOUT leaving the connectivity region, and spreads
    robots over CONTIGUOUS areas so paths do not overlap more than needed.
  * Here the analogue is: each clot is a task, and robots should be assigned
    so that the geodesic corridors they follow toward their clots share as
    little vessel length as necessary. Strictly disjoint paths are NOT the
    goal -- in a branching tree the trunk is unavoidable -- the goal is to
    avoid *unnecessary* shared corridors and edge congestion.

Cost components per (robot, clot) pair (all normalised, weights recorded):

  geodesic distance  -- along-vessel travel cost to reach the clot
  path overlap       -- fraction of the robot's corridor already used by the
                        routes of robots assigned before it
  edge congestion    -- penalty when many assigned routes pass the same edge
                        (the trunk inevitably carries some; the term is about
                        excess)
  flow cost          -- upstream (against-flow) approach is more expensive
                        than downstream
  switching penalty  -- cost of abandoning the clot the robot is currently
                        shaped toward (target thrashing)
  clot capacity      -- lysis saturates at `lysis_saturation` robots per
                        clot; assigning beyond capacity wastes a robot

The allocator emits per-robot target indices AND an optional corridor hint
(the next-hop field of the assigned clot's route), which the caller can use
for a corridor hint. It deliberately does NOT emit actions: the final 3-D
local action remains the output of the Direct Local GAT-MAPPO policy
(``control_mode='local'``). The assignment is applied to the environment via
``env.set_task_assignments`` / the vector env's task-assignment override,
which only changes WHICH clot the observation's shaping/lookahead points at.

Baselines in the same module so the comparison is apples-to-apples:
  * ``nearest_assignment``       -- geodesic-nearest live clot (the env default)
  * ``flow_spread_assignment``   -- the flow-guided era's spread rule
  * ``connectivity_assignment``  -- this module's allocator

Reference: C²-Explorer connectivity/contiguity concept, adapted. No claim of
implementing C²-Explorer itself; the communication model there has no direct
vascular counterpart and is replaced by corridor overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_WEIGHTS = {
    "geodesic": 1.0,
    "overlap": 1.0,
    "congestion": 0.5,
    "flow": 0.25,
    "switching": 0.5,
    "capacity": 0.75,
    "particle": 0.8,
    "path_length": 0.25,
    "prediction": 0.9,
}


@dataclass
class ConnectivityResult:
    """Allocation outcome plus the diagnostics that justify it."""

    assignments: np.ndarray                  # [n_robots] clot index or -1
    target_load: np.ndarray                  # [n_alive] robots per clot
    distinct_targets: int
    cost_matrix: np.ndarray                  # [n_robots, n_alive] full cost
    geodesic_matrix: np.ndarray              # [n_robots, n_alive] raw distance
    overlap_matrix: np.ndarray               # [n_robots, n_alive] overlap term
    edge_counts: dict[int, int]              # station -> assigned routes through it
    overlap_cost: float                      # total shared-arclength fraction
    weights: dict[str, float] = field(default_factory=dict)


def _alive_clots(env) -> np.ndarray:
    masses = env.clot_masses[: env.active_clots]
    return np.flatnonzero(masses > 0).astype(np.int32)


def _route_path(tree, robot_station: int, clot_station: int,
                next_hop: np.ndarray, max_len: int = 4096) -> np.ndarray:
    """Stations along the geodesic corridor from a robot to a clot.

    Follows the clot route's next-hop field from the robot's station. The
    predecessor field points AT the target, so walking hop-by-hop from the
    robot terminates at the clot station. A cycle guard (max_len) protects
    against malformed routes; it cannot trigger for a Dijkstra tree.
    """
    path = [int(robot_station)]
    cur = int(robot_station)
    for _ in range(max_len):
        nxt = int(next_hop[cur])
        if nxt == cur:
            break
        path.append(nxt)
        cur = nxt
        if cur == int(clot_station):
            break
    return np.asarray(path, dtype=np.int32)


def connectivity_assignment(
    env,
    weights: dict[str, float] | None = None,
    previous_assignments: np.ndarray | None = None,
    risk_aware: bool = False,
    predictive: bool = False,
) -> ConnectivityResult:
    """Assign robots to clots with corridor-overlap awareness.

    Greedy sequential allocation: robots are assigned in increasing
    geodesic-distance order; each takes the clot minimising
    ``w_geo * dist + w_ov * overlap + w_cong * congestion + w_flow * flow
    + w_switch * switch + w_cap * capacity_overflow``. Greedy keeps the
    allocator O(R*C) route lookups and deterministic, which a Hungarian
    assignment on the same cost matrix would not be once overlap couples the
    rows (the cost of row i depends on the assignment of rows < i).

    The clot capacity term saturates at the env's ``lysis_saturation``:
    the (k+1)-th robot on a clot pays the overflow, mimicking the actual
    saturating lysis physics rather than a hard cap, so excess robots still
    go somewhere useful (the least-bad clot) instead of being parked idle.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    alive = _alive_clots(env)
    n_robots = int(env.num_robots)
    result_shape = (n_robots, max(alive.size, 1))

    geodesic = np.full(result_shape, np.inf, np.float64)
    overlap = np.zeros(result_shape, np.float64)
    flow_cost = np.zeros(result_shape, np.float64)
    particle_risk = np.zeros(result_shape, np.float64)

    if alive.size == 0 or n_robots == 0:
        return ConnectivityResult(
            assignments=np.full((n_robots,), -1, np.int32),
            target_load=np.zeros((alive.size,), np.int32),
            distinct_targets=0,
            cost_matrix=np.zeros((n_robots, 0)),
            geodesic_matrix=geodesic[:, : alive.size] if alive.size else geodesic,
            overlap_matrix=overlap,
            edge_counts={},
            overlap_cost=0.0,
            weights=w,
        )

    tree = env.tree
    stations = np.asarray(env.robot_stations, dtype=np.int32)
    scale = max(float(tree.total_length), 1e-6)

    routes = {}
    hop_fields = {}
    for clot in alive:
        distance, hop = env._route(int(clot))
        routes[int(clot)] = distance
        hop_fields[int(clot)] = hop
        geodesic[:, int(np.flatnonzero(alive == clot)[0])] = distance[stations]
    geodesic_norm = geodesic / scale

    # Flow cost: dot(flow direction, route direction toward the clot). Going
    # against the local flow is expensive; with it is free. The flow vector
    # is the env's own local flow at the robot's station.
    flow_vec = env.tree.flow(
        env.robot_positions, stations, env.flow_speed, env.tube_radius
    )
    for column, clot in enumerate(alive):
        hop = hop_fields[int(clot)]
        nxt = hop[stations]
        step = tree.points[nxt] - tree.points[stations]
        norm = np.linalg.norm(step, axis=1)
        unit = step / np.maximum(norm, 1e-8)[:, None]
        # fraction of flow opposing the route direction
        opposing = np.clip(-np.sum(flow_vec * unit, axis=1)
                           / max(float(env.max_speed), 1e-8), 0.0, 1.0)
        flow_cost[:, column] = opposing

    # Dynamic-obstacle risk uses only particles at the current time.  The
    # allocator never queries a future particle position or a rollout outcome.
    # Sampling route stations keeps this closed-loop planner bounded while
    # still penalising a corridor whose present clearance is poor.
    if risk_aware and getattr(env, "particles", None) is not None:
        particles = env.particles
        row = getattr(env, "_row", None)
        particle_positions = particles.positions if row is None else particles.positions[int(row):int(row) + 1]
        particle_velocities = particles.velocities if row is None else particles.velocities[int(row):int(row) + 1]
        if particle_positions.shape[0] == 1:
            particle_positions = particle_positions[0]
            particle_velocities = particle_velocities[0]
        else:
            particle_positions = particle_positions[0]
            particle_velocities = particle_velocities[0]
        robot_positions = np.asarray(env.robot_positions)
        robot_velocities = np.asarray(getattr(env, "robot_velocities", np.zeros_like(robot_positions)))
        current_dist = np.linalg.norm(robot_positions[:, None, :] - particle_positions[None, :, :], axis=2)
        nearest = np.argmin(current_dist, axis=1)
        current_clearance = current_dist[np.arange(n_robots), nearest] - float(particles.contact_distance)
        current_rel_speed = np.linalg.norm(
            robot_velocities - particle_velocities[nearest], axis=1
        ) / max(float(env.max_speed), 1e-8)
        current_risk = np.exp(-np.clip(current_clearance, -0.02, 0.2) / 0.02) + 0.25 * current_rel_speed
        for column, clot in enumerate(alive):
            # Route geometry is shared across robots; evaluate the remaining
            # corridor from each robot's station so branch-specific risk is kept.
            for robot in range(n_robots):
                route = _route_path(tree, int(stations[robot]), int(clot), hop_fields[int(clot)])
                points = tree.points[route[::max(1, route.size // 24)]]
                clearance = np.linalg.norm(points[:, None, :] - particle_positions[None, :, :], axis=2).min()
                corridor_risk = np.exp(-np.clip(clearance - float(particles.contact_distance), -0.02, 0.2) / 0.02)
                particle_risk[robot, column] = 0.5 * current_risk[robot] + 0.5 * corridor_risk
                if predictive:
                    future_particles = particle_positions + 0.45 * particle_velocities
                    future_clearance = np.linalg.norm(
                        points[:, None, :] - future_particles[None, :, :], axis=2
                    ).min() - float(particles.contact_distance)
                    particle_risk[robot, column] += 0.75 / (
                        1.0 + np.exp(np.clip(future_clearance / 0.02, -30.0, 30.0))
                    )
        particle_risk = np.clip(particle_risk, 0.0, 4.0)

    capacity = float(getattr(env, "lysis_saturation", 4.0))
    loads = np.zeros((alive.size,), np.int32)
    assignments = np.full((n_robots,), -1, np.int32)
    edge_counts: dict[int, int] = {}

    # Robots closer to SOME clot pick first: their corridors are the least
    # negotiable, and later (far) robots can still trade overlap for distance.
    order = np.argsort(geodesic.min(axis=1))
    prev = (
        np.full((n_robots,), -1, np.int32)
        if previous_assignments is None
        else np.asarray(previous_assignments, dtype=np.int32)
    )

    overlap_cost_total = 0.0
    for robot in order:
        column_costs = (
            w["geodesic"] * geodesic_norm[robot]
            + w["overlap"] * overlap[robot]
            + w["flow"] * flow_cost[robot]
        )
        if risk_aware:
            # Geodesic distance is also the physically meaningful path-length
            # lower bound; the explicit term documents that tradeoff separately
            # from obstacle risk and makes it available for auditing.
            column_costs = column_costs + w["particle"] * particle_risk[robot]
            column_costs = column_costs + w["path_length"] * geodesic_norm[robot]
        # Congestion: penalise edges already used by MORE than one assigned
        # route. The first user of the trunk is free (unavoidable); the second
        # and later pay.
        if edge_counts:
            congestion = np.zeros((alive.size,), np.float64)
            for column, clot in enumerate(alive):
                path = _route_path(
                    tree, int(stations[robot]), int(clot),
                    hop_fields[int(clot)],
                )
                used = sum(edge_counts.get(int(s), 0) for s in path)
                # fraction of this corridor that is congested
                congestion[column] = used / max(path.size, 1)
            column_costs = column_costs + w["congestion"] * congestion
        # Switching: leaving a clot the robot was previously shaped toward.
        if prev[robot] >= 0:
            switch = np.ones((alive.size,), np.float64)
            for column, clot in enumerate(alive):
                if int(clot) == int(prev[robot]):
                    switch[column] = 0.0
            column_costs = column_costs + w["switching"] * switch
        # Capacity overflow: robots beyond the saturation count pay. With a
        # single live clot the overflow is uniform and the robot still lands
        # there (there is nowhere else to go) -- the term shapes behaviour
        # when a CHOICE exists, and the soft form keeps the assignment total.
        overflow = np.maximum(loads - capacity + 1.0, 0.0)
        column_costs = column_costs + w["capacity"] * overflow
        if alive.size > 1:
            # Prefer not to exceed capacity while any unsaturated clot exists.
            saturated = loads >= capacity
            if np.any(saturated) and not np.all(saturated):
                column_costs = column_costs + np.where(saturated, 1e6, 0.0)

        best = int(np.argmin(column_costs))
        assignments[int(robot)] = int(alive[best])
        loads[best] += 1

        # Record this route's edges and add overlap term for later robots.
        path = _route_path(
            tree, int(stations[robot]), int(alive[best]),
            hop_fields[int(alive[best])],
        )
        for s in path:
            s = int(s)
            edge_counts[s] = edge_counts.get(s, 0) + 1
        path_len = max(path.size, 1)
        shared = sum(edge_counts[int(s)] - 1 for s in path)
        overlap_cost_total += shared / path_len
        for column in range(alive.size):
            other = _route_path(
                tree, int(stations[robot]), int(alive[column]),
                hop_fields[int(alive[column])],
            )
            overlap[robot, column] = np.mean(
                [min(edge_counts.get(int(s), 0), 1) for s in other]
            )

    # Normalise the reported overlap cost by the number of routes.
    overlap_cost_total = overlap_cost_total / max(int((assignments >= 0).sum()), 1)

    return ConnectivityResult(
        assignments=assignments,
        target_load=loads,
        distinct_targets=int(np.count_nonzero(loads)),
        cost_matrix=(geodesic_norm + overlap).astype(np.float32),
        geodesic_matrix=geodesic.astype(np.float32),
        overlap_matrix=overlap.astype(np.float32),
        edge_counts=edge_counts,
        overlap_cost=float(overlap_cost_total),
        weights=w,
    )


def nearest_assignment(env) -> ConnectivityResult:
    """Baseline: geodesic-nearest live clot (the env's own default rule)."""
    alive = _alive_clots(env)
    n_robots = int(env.num_robots)
    if alive.size == 0:
        return ConnectivityResult(
            assignments=np.full((n_robots,), -1, np.int32),
            target_load=np.zeros((0,), np.int32),
            distinct_targets=0,
            cost_matrix=np.zeros((n_robots, 0)),
            geodesic_matrix=np.zeros((n_robots, 0)),
            overlap_matrix=np.zeros((n_robots, 0)),
            edge_counts={},
            overlap_cost=0.0,
        )
    geodesic = np.full((n_robots, alive.size), np.inf, np.float64)
    for column, clot in enumerate(alive):
        distance, _ = env._route(int(clot))
        geodesic[:, column] = distance[np.asarray(env.robot_stations, np.int32)]
    assignments = np.argmin(geodesic, axis=1).astype(np.int32)
    assignments = alive[assignments]
    loads = np.zeros((alive.size,), np.int32)
    for target in assignments:
        loads[int(np.flatnonzero(alive == target)[0])] += 1
    return ConnectivityResult(
        assignments=assignments.astype(np.int32),
        target_load=loads,
        distinct_targets=int(np.count_nonzero(loads)),
        cost_matrix=geodesic.astype(np.float32),
        geodesic_matrix=geodesic.astype(np.float32),
        overlap_matrix=np.zeros((n_robots, alive.size), np.float32),
        edge_counts={},
        overlap_cost=0.0,
    )


def flow_spread_assignment(env, congestion_penalty: float = 0.35) -> ConnectivityResult:
    """Baseline: the flow-guided era's spread rule (distance + load penalty)."""
    alive = _alive_clots(env)
    n_robots = int(env.num_robots)
    if alive.size == 0:
        return ConnectivityResult(
            assignments=np.full((n_robots,), -1, np.int32),
            target_load=np.zeros((0,), np.int32),
            distinct_targets=0,
            cost_matrix=np.zeros((n_robots, 0)),
            geodesic_matrix=np.zeros((n_robots, 0)),
            overlap_matrix=np.zeros((n_robots, 0)),
            edge_counts={},
            overlap_cost=0.0,
        )
    geodesic = np.full((n_robots, alive.size), np.inf, np.float64)
    for column, clot in enumerate(alive):
        distance, _ = env._route(int(clot))
        geodesic[:, column] = distance[np.asarray(env.robot_stations, np.int32)]
    scale = max(float(env.tree.total_length), 1e-6)
    norm = geodesic / scale
    loads = np.zeros((alive.size,), np.int32)
    assignments = np.full((n_robots,), -1, np.int32)
    order = np.argsort(norm.min(axis=1))
    for robot in order:
        score = norm[int(robot)] + congestion_penalty * loads
        best = int(np.argmin(score))
        assignments[int(robot)] = int(alive[best])
        loads[best] += 1
    return ConnectivityResult(
        assignments=assignments.astype(np.int32),
        target_load=loads,
        distinct_targets=int(np.count_nonzero(loads)),
        cost_matrix=norm.astype(np.float32),
        geodesic_matrix=geodesic.astype(np.float32),
        overlap_matrix=np.zeros((n_robots, alive.size), np.float32),
        edge_counts={},
        overlap_cost=0.0,
    )


ALLOCATION_MODES = (
    "nearest", "flow_spread", "connectivity_aware", "risk_aware_connectivity",
    "predictive_risk_connectivity",
)


def allocate(env, mode: str = "connectivity_aware",
             weights: dict[str, float] | None = None,
             previous_assignments: np.ndarray | None = None) -> ConnectivityResult:
    """Dispatch on mode. Modes match the controller-era baselines plus this
    module's connectivity-aware allocator."""
    if mode not in ALLOCATION_MODES:
        raise ValueError(f"unknown allocation mode {mode!r}")
    if mode == "nearest":
        return nearest_assignment(env)
    if mode == "flow_spread":
        return flow_spread_assignment(env)
    return connectivity_assignment(
        env, weights, previous_assignments,
        risk_aware=(mode in ("risk_aware_connectivity", "predictive_risk_connectivity")),
        predictive=(mode == "predictive_risk_connectivity"),
    )
