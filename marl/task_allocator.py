"""Explicit clot-task allocation for the hierarchical research line."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AllocationResult:
    assignments: np.ndarray
    cost_matrix: np.ndarray
    eta_matrix: np.ndarray
    target_load: np.ndarray
    distinct_targets: int


def clot_cost_matrix(env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized assignment cost, ETA and alive clot indices."""
    alive = np.flatnonzero(
        env.clot_masses[: env.active_clots] > 0
    ).astype(np.int32)
    costs = np.full((env.num_robots, len(alive)), np.inf, dtype=np.float32)
    eta = np.full_like(costs, np.inf)
    if len(alive) == 0:
        return costs, eta, alive

    scale = max(float(env.tree.total_length), 1e-6)
    for column, clot in enumerate(alive):
        distance, _ = env._route(int(clot))
        eta[:, column] = distance[env.robot_stations] / max(env.max_speed, 1e-6)
    eta_norm = eta / max(scale / max(env.max_speed, 1e-6), 1e-6)
    mass = env.clot_masses[alive] / np.maximum(
        env.clot_initial_mass[alive], 1e-8
    )
    priority = 0.25 * (1.0 - mass[None, :])
    costs = eta_norm + priority
    return costs.astype(np.float32), eta.astype(np.float32), alive


def _distinct_matching(costs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use scipy matching when available, with a small-matrix fallback."""
    n_robots, n_targets = costs.shape
    if n_robots == 0 or n_targets == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(costs)
        return rows.astype(np.int32), columns.astype(np.int32)
    except ImportError:
        import itertools

        count = min(n_robots, n_targets)
        best = None
        for rows_tuple in itertools.combinations(range(n_robots), count):
            for columns_tuple in itertools.permutations(range(n_targets), count):
                value = sum(
                    float(costs[row, column])
                    for row, column in zip(rows_tuple, columns_tuple)
                )
                if best is None or value < best[0]:
                    best = (value, rows_tuple, columns_tuple)
        assert best is not None
        return (
            np.asarray(best[1], dtype=np.int32),
            np.asarray(best[2], dtype=np.int32),
        )


def balanced_assignment(env, congestion_penalty: float = 0.35) -> AllocationResult:
    """Assign distinct clots first, then place extra robots by marginal value."""
    costs, eta, alive = clot_cost_matrix(env)
    assignments = np.full((env.num_robots,), -1, dtype=np.int32)
    if len(alive) == 0:
        return AllocationResult(assignments, costs, eta, np.zeros(0, np.int32), 0)

    rows, columns = _distinct_matching(costs)
    for row, column in zip(rows, columns):
        assignments[int(row)] = int(alive[int(column)])

    loads = np.zeros((len(alive),), dtype=np.int32)
    for target in assignments:
        if target >= 0:
            loads[int(np.flatnonzero(alive == target)[0])] += 1

    remaining = np.flatnonzero(assignments < 0)
    for row in remaining:
        score = costs[int(row)] + congestion_penalty * loads
        column = int(np.argmin(score))
        assignments[int(row)] = int(alive[column])
        loads[column] += 1

    return AllocationResult(
        assignments=assignments,
        cost_matrix=costs,
        eta_matrix=eta,
        target_load=loads,
        distinct_targets=int(np.count_nonzero(loads)),
    )


def nearest_assignment(env) -> AllocationResult:
    """Expose the original nearest-clot rule as a measurable baseline."""
    costs, eta, alive = clot_cost_matrix(env)
    assignments = env._assigned_clot().copy()
    loads = np.zeros((len(alive),), dtype=np.int32)
    for target in assignments:
        if target >= 0:
            matches = np.flatnonzero(alive == target)
            if matches.size:
                loads[int(matches[0])] += 1
    return AllocationResult(
        assignments=assignments,
        cost_matrix=costs,
        eta_matrix=eta,
        target_load=loads,
        distinct_targets=int(np.count_nonzero(loads)),
    )
