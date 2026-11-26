import numpy as np


class FakeCoordinator:
    n = 2
    step = 1.0
    speed = 1.0
    dwell = 0.0
    stop_margin = 0.3
    paths = [np.zeros((4, 3)), np.zeros((4, 3))]
    clot_s = [[], []]
    zones = [{"i": 0, "j": 1, "s0": 1, "s1": 2, "u0": 1, "u1": 2}]


def test_ccbs_fixed_route_search_returns_precedence_metadata():
    from marl.ccbs_adapter import CCBSPlanner
    planner = CCBSPlanner(FakeCoordinator()).plan()
    meta = planner.metadata()
    assert meta["algorithm"] == "CCBS-fixed-route-adapter"
    assert meta["route_policy"] == "fixed"
    assert meta["nodes_expanded"] == 2
    assert meta["conflict_zones"] == 1
    assert set(meta["order"]) == {"0"}

