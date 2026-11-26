"""Small, exhaustive reference for three-target makespan on vessel graph distances."""
from itertools import combinations, permutations

import numpy as np


def geodesic_allocation(distance, starts, targets):
    """Minimize longest tour, then total length. Each deployed cluster visits a target."""
    if len(targets) != 3 or not 1 <= len(starts) <= 3:
        raise ValueError("expected three targets and one to three active clusters")
    best_key = None
    for order in permutations(range(3)):
        for cuts in combinations((1, 2), len(starts) - 1):
            bounds = (0, *cuts, 3)
            tours = tuple(order[a:b] for a, b in zip(bounds, bounds[1:]))
            lengths = []
            for start, tour in zip(starts, tours):
                stations = [start, *(targets[t] for t in tour)]
                lengths.append(sum(distance[a, b] for a, b in zip(stations, stations[1:])))
            key = (max(lengths), sum(lengths), tours)
            if np.isfinite(key[0]) and (best_key is None or key < best_key):
                best_key = key
    if best_key is None:
        raise ValueError("no feasible assignment covers all three targets")
    return [list(tour) for tour in best_key[2]]


def linear_sum_allocation(distance, starts, targets):
    """Assign unused targets by minimum sum each round, updating departure stations."""
    from scipy.optimize import linear_sum_assignment

    positions = list(starts)
    remaining = list(range(len(targets)))
    tours = [[] for _ in positions]
    while remaining:
        costs = distance[np.ix_(positions, [targets[j] for j in remaining])]
        agents, cols = linear_sum_assignment(costs)
        for i, j in zip(agents, cols):
            target = remaining[j]
            tours[i].append(target)
            positions[i] = targets[target]
        assigned = set(cols)
        remaining = [target for j, target in enumerate(remaining) if j not in assigned]
    return tours
