import json
from pathlib import Path

import numpy as np

from scripts.evaluate_navigation_dataset import aggregate, grouped_aggregates
from scripts.multicluster_protocol import SpacingTracker


def _row(anatomy, flow, success=True):
    return dict(anatomy=anatomy, clusters=3, flow_mode=flow, clot_pattern="historical_sites",
                obstacle_density="low", obstacle_layout="uniform", navigation_success=success,
                strict_success=success, wall_contact_s=0., wall_after_reach_s=0., lost=0,
                particle_collision_events=0., particle_contact_s=0.)


def test_grouped_aggregates_partition_rows():
    rows = [_row("a", "weak"), _row("a", "strong", False), _row("b", "weak")]
    groups = grouped_aggregates(rows)
    assert sum(x["requested"] for x in groups["by_anatomy"].values()) == len(rows)
    assert sum(x["requested"] for x in groups["by_flow_mode"].values()) == len(rows)
    assert sum(x["requested"] for x in groups["by_condition"].values()) == len(rows)
    assert aggregate(rows)["requested"] == len(rows)


def test_spacing_tracker_counts_duration_once():
    tracker = SpacingTracker(np.array([[0., 0., 0.], [2., 0., 0.]]), np.array([True, True]), 2.)
    tracker.begin_step()
    tracker.update(np.array([[0., 0., 0.], [2., 0., 0.]]),
                   np.array([[0., 0., 0.], [1., 0., 0.]]),
                   np.array([True, True]), np.array([1., 1.]), 1.)
    tracker.end_step()
    assert tracker.summary()["spacing_violation_pair_s"] > 0.
    assert tracker.summary()["spacing_violation_steps"] == 1
