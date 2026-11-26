import itertools

import numpy as np

from scripts.three_target_allocation import geodesic_allocation, linear_sum_allocation


def test_single_cluster_selects_shortest_open_tour():
    coordinates = np.array([0., 8., 1., 4.])
    distance = abs(coordinates[:, None] - coordinates)
    assert geodesic_allocation(distance, [0], [1, 2, 3]) == [[1, 2, 0]]


def test_sum_and_makespan_answer_different_assignment_questions():
    distance = np.zeros((6, 6))
    cost = np.array([[1, 5, 20], [5, 8, 20], [20, 20, 1]])
    distance[:3, 3:] = cost
    starts, targets = range(3), range(3, 6)
    assert linear_sum_allocation(distance, starts, targets) == [[0], [1], [2]]
    assert geodesic_allocation(distance, starts, targets) == [[1], [0], [2]]


def test_allocations_cover_three_targets_once_for_each_cluster_count():
    distance = abs(np.arange(6)[:, None] - np.arange(6))
    for count, planner in itertools.product((1, 2, 3), (linear_sum_allocation, geodesic_allocation)):
        routes = planner(distance, range(count), [3, 4, 5])
        assert len(routes) == count and all(routes)
        assert sorted(itertools.chain.from_iterable(routes)) == [0, 1, 2]
