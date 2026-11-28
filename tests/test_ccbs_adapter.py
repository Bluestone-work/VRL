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



def test_cluster_parked_in_zone_must_enter_last():
    from marl.ccbs_adapter import CCBSPlanner
    c = FakeCoordinator()
    c.zones = [dict(i=0, j=1, s0=1, s1=3, u0=1, u1=2)]
    assert CCBSPlanner(c).plan().order == {0: 1}
    c.zones[0]['u1'] = 3
    planner = CCBSPlanner(c).plan()
    assert planner.infeasible and not planner.has_plan


def test_search_budget_is_distinct_from_proven_infeasibility():
    from marl.ccbs_adapter import CCBSPlanner
    c = FakeCoordinator()
    c.zones = [dict(i=0, j=1, s0=1, s1=3, u0=1, u1=2)]
    planner = CCBSPlanner(c, max_nodes=1).plan()
    assert planner.search_exhausted and not planner.infeasible and not planner.has_plan
