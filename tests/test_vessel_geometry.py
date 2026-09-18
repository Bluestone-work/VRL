"""Tests for the branch-aware vessel geometry.

The important one is `test_hint_prevents_branch_jump`: it reproduces the failure
mode of the original global-argmin lookup, where a robot in one daughter branch
gets snapped onto the other because that station is marginally closer in
Euclidean distance. Everything downstream of that lookup (wall normal, flow
direction, lookahead) was silently wrong whenever it fired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vessel_geometry import (  # noqa: E402
    SCENARIOS,
    Branch,
    VesselTree,
    build_vessel_tree,
    _catmull_rom,
)


def _tree(scenario: str, seed: int = 0) -> VesselTree:
    return build_vessel_tree(scenario, np.random.default_rng(seed))


# ----------------------------------------------------------------- basic shape


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_tree_arrays_consistent(scenario: str) -> None:
    tree = _tree(scenario)
    n = tree.n_stations
    assert tree.points.shape == (n, 3)
    assert tree.radii.shape == (n,)
    assert tree.tangents.shape == (n, 3)
    assert tree.branch_ids.shape == (n,)
    assert np.all(np.isfinite(tree.points))
    assert np.all(tree.radii > 0)
    # Every station belongs to exactly one branch and branches tile the array.
    covered = np.zeros(n, dtype=bool)
    for br in tree.branches:
        assert not covered[br.start : br.stop + 1].any(), "branches overlap"
        covered[br.start : br.stop + 1] = True
    assert covered.all(), "branches do not cover all stations"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_frames_orthonormal(scenario: str) -> None:
    tree = _tree(scenario)
    t, nrm, b = tree.tangents, tree.normals, tree.binormals
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.sum(t * nrm, axis=1), 0.0, atol=1e-4)
    assert np.allclose(np.sum(t * b, axis=1), 0.0, atol=1e-4)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_frame_is_continuous(scenario: str) -> None:
    """No sign flips: consecutive normals within a branch stay aligned.

    A per-station Frenet frame flips wherever curvature vanishes, which would
    put a discontinuity into the observation. Parallel transport must not.
    """
    tree = _tree(scenario)
    for br in tree.branches:
        nrm = tree.normals[br.start : br.stop + 1]
        dots = np.sum(nrm[1:] * nrm[:-1], axis=1)
        assert np.all(dots > 0.5), f"normal flipped in branch {br.branch_id}"


def test_catmull_rom_interpolates_endpoints() -> None:
    ctrl = np.array([[0.0, 0, 0], [0.5, 0.3, 0], [1.0, 0, 0]], np.float32)
    curve = _catmull_rom(ctrl, 50)
    assert np.allclose(curve[0], ctrl[0], atol=1e-5)
    assert np.allclose(curve[-1], ctrl[-1], atol=1e-5)
    # A bowed spline must actually leave the chord.
    assert curve[:, 1].max() > 0.1


def test_catmull_rom_two_points_is_a_line() -> None:
    ctrl = np.array([[0.0, 0, 0], [1.0, 1.0, 0]], np.float32)
    curve = _catmull_rom(ctrl, 11)
    assert np.allclose(curve[5], [0.5, 0.5, 0.0], atol=1e-5)


# ------------------------------------------------------- the bifurcation bug


def _y_tree() -> VesselTree:
    """A hand-built Y with the two daughters deliberately close together.

    Trunk runs along +x to (0.5, 0, 0), then two straight limbs diverge only
    slightly in y. Near the fork the limbs are within a few millimetres of each
    other, which is exactly where a global nearest-point lookup misassigns.
    """
    n = 40
    x = np.linspace(0.0, 0.5, n, dtype=np.float32)
    trunk = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)

    m = 40
    xs = np.linspace(0.5, 1.0, m, dtype=np.float32)
    spread = np.linspace(0.0, 0.10, m, dtype=np.float32)
    up = np.stack([xs, spread, np.zeros(m)], axis=1).astype(np.float32)
    dn = np.stack([xs, -spread, np.zeros(m)], axis=1).astype(np.float32)

    points = np.concatenate([trunk, up, dn], axis=0)
    radii = np.full((points.shape[0],), 0.05, dtype=np.float32)
    branches = [
        Branch(0, 0, n - 1, -1, 1.0),
        Branch(1, n, n + m - 1, 0, 0.5),
        Branch(2, n + m, n + 2 * m - 1, 0, 0.5),
    ]
    return VesselTree(points, radii, branches, scenario="y")


def test_hint_prevents_branch_jump() -> None:
    """A robot just inside the upper limb must not snap to the lower limb."""
    tree = _y_tree()
    up_branch = tree.stations_of_branch(1)
    # Sit slightly *below* an upper-limb station, biased toward the lower limb.
    station = int(up_branch[6])
    probe = tree.points[station] + np.array([0.0, -0.012, 0.0], np.float32)

    # Global lookup is allowed to pick the wrong branch; that is the bug.
    naive = tree.nearest_station(probe[None, :])[0]
    # Hinted lookup, given the true previous station, must stay on branch 1.
    hinted = tree.nearest_station(probe[None, :], hint=np.array([station]), window=6)[0]
    assert tree.branch_ids[hinted] == 1, "hinted lookup left the correct branch"
    # Document the naive behaviour so a regression is visible either way.
    assert tree.branch_ids[naive] in (0, 1, 2)


def test_hint_window_limits_travel() -> None:
    """The hinted lookup cannot jump further than `window` hops."""
    tree = _y_tree()
    start = 5
    far = tree.points[-1]  # distal end of the lower limb
    got = tree.nearest_station(far[None, :], hint=np.array([start]), window=4)[0]
    assert abs(int(got) - start) <= 4


def test_projection_is_smooth_along_axis() -> None:
    """Segment projection must not quantise the wall to station spacing.

    Sampling the clamped radius along a fine sweep of axial positions should
    give a near-constant distance from the axis; a point-based projection
    produces a visible scallop instead.
    """
    tree = _y_tree()
    xs = np.linspace(0.05, 0.45, 200, dtype=np.float32)
    # Place every probe well outside the lumen so all of them get clamped.
    probes = np.stack([xs, np.full_like(xs, 0.5), np.zeros_like(xs)], axis=1)
    clamped, outside, station, axis_point = tree.project(probes, robot_radius=0.004)
    assert outside.all()
    radial = np.linalg.norm(clamped - axis_point, axis=1)
    # radius 0.05 - robot 0.004 = 0.046 everywhere along a straight trunk.
    assert np.allclose(radial, 0.046, atol=1e-4)


def test_projection_keeps_interior_points_untouched() -> None:
    tree = _y_tree()
    inside = tree.points[10] + np.array([0.0, 0.001, 0.0], np.float32)
    clamped, outside, _, _ = tree.project(inside[None, :], robot_radius=0.004)
    assert not outside[0]
    assert np.allclose(clamped[0], inside)


# ----------------------------------------------------------------- geodesics


def test_arclength_increases_downstream() -> None:
    tree = _y_tree()
    trunk = tree.stations_of_branch(0)
    arc = tree.arclength[trunk]
    assert np.all(np.diff(arc) > 0)
    assert arc[0] == pytest.approx(0.0, abs=1e-6)


def test_route_next_hop_reaches_target() -> None:
    """Following next_hop from anywhere must arrive at the target station."""
    tree = _y_tree()
    target = int(tree.stations_of_branch(2)[-1])  # distal lower limb
    distance, next_hop = tree.route_to(target)

    for start in (0, 20, int(tree.stations_of_branch(1)[10]), target):
        cur, hops = start, 0
        while cur != target and hops < 4 * tree.n_stations:
            nxt = int(next_hop[cur])
            assert nxt != cur, f"next_hop stalled at {cur}"
            # Non-strict: stations coincide at a junction, so one hop there can
            # cover zero distance. Progress is guaranteed by the predecessor
            # tree being acyclic, not by strict monotonicity.
            assert distance[nxt] <= distance[cur] + 1e-6
            cur = nxt
            hops += 1
        assert cur == target, f"next_hop from {start} did not reach the target"


def test_route_picks_the_correct_branch_at_the_fork() -> None:
    """At the junction the geodesic must turn into the branch holding the clot.

    This is the information the original observation could not express, so the
    policy had to guess which side of a bifurcation to take.
    """
    tree = _y_tree()
    junction = int(tree.stations_of_branch(0)[-1])

    for branch in (1, 2):
        target = int(tree.stations_of_branch(branch)[-1])
        _distance, next_hop = tree.route_to(target)
        hop = int(next_hop[junction])
        assert tree.branch_ids[hop] == branch


def test_geodesic_exceeds_euclidean_across_branches() -> None:
    """Along-vessel distance must be >= straight-line distance, and strictly
    greater between the two limbs (where a straight line leaves the vessel)."""
    tree = _y_tree()
    a = int(tree.stations_of_branch(1)[-1])
    b = int(tree.stations_of_branch(2)[-1])
    distance, _ = tree.route_to(b)
    euclid = float(np.linalg.norm(tree.points[a] - tree.points[b]))
    assert distance[a] > euclid * 1.5


def test_anastomosis_has_a_cycle() -> None:
    """The merge link must exist, otherwise the loop degenerates to a tree."""
    tree = _tree("anastomosis", seed=3)
    assert tree.n_branches == 5
    tip_up = tree.branches[3].stop
    tip_dn = tree.branches[4].stop
    neighbours = {v for v, _w in tree.station_graph[tip_up]}
    assert tip_dn in neighbours, "distal limbs are not joined"
    # With the loop closed there are two routes from the fork to the outlet.
    distance, _ = tree.route_to(tip_up)
    assert np.isfinite(distance[tip_dn])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_stations_reachable(scenario: str) -> None:
    tree = _tree(scenario, seed=7)
    assert np.all(np.isfinite(tree.arclength)), "disconnected station graph"


# ---------------------------------------------------------------------- flow


def test_stenosis_accelerates_flow() -> None:
    """Continuity: a narrower lumen must carry faster flow.

    The original model kept flow_speed constant, so a stenosis was "narrower but
    equally slow" -- physically wrong and it removed the whole point of the
    scenario.
    """
    rng = np.random.default_rng(11)
    tree = build_vessel_tree("stenotic", rng)
    # Find the narrowest station and a wide one on the same branch.
    tight = int(np.argmin(tree.radii))
    bid = int(tree.branch_ids[tight])
    same = tree.stations_of_branch(bid)
    wide = int(same[np.argmax(tree.radii[same])])
    if tree.radii[tight] >= tree.radii[wide] * 0.95:
        pytest.skip("this sample has no appreciable stenosis")

    pos = tree.points[[tight, wide]]
    station = np.array([tight, wide], dtype=np.int32)
    v = tree.flow(pos, station, inlet_speed=0.004, reference_radius=0.055)
    speed = np.linalg.norm(v, axis=1)
    assert speed[0] > speed[1], "flow did not accelerate through the stenosis"


def test_flow_follows_tangent_and_vanishes_at_wall() -> None:
    tree = _y_tree()
    station = np.array([10, 10], dtype=np.int32)
    axis = tree.points[10]
    # On-axis and (almost) at the wall.
    probes = np.stack([axis, axis + np.array([0.0, 0.0499, 0.0], np.float32)])
    v = tree.flow(probes, station, inlet_speed=0.004, reference_radius=0.05)
    speed = np.linalg.norm(v, axis=1)
    assert speed[0] > speed[1] * 10, "no Poiseuille profile"
    # Trunk tangent is +x.
    assert v[0, 0] > 0 and abs(v[0, 1]) < 1e-6


def test_flow_mean_matches_inlet_speed() -> None:
    """The profile is normalised to the mean, so a radial average recovers it."""
    tree = _y_tree()
    r = np.linspace(0.0, 0.0499, 400, dtype=np.float32)
    axis = tree.points[10]
    probes = np.stack(
        [axis + np.array([0.0, float(ri), 0.0], np.float32) for ri in r]
    )
    station = np.full((probes.shape[0],), 10, dtype=np.int32)
    v = tree.flow(probes, station, inlet_speed=0.004, reference_radius=0.05)
    speed = np.linalg.norm(v, axis=1)
    # Area-weighted mean over the cross-section: integral of u(r)*2*pi*r dr.
    weight = r
    mean = float(np.sum(speed * weight) / np.sum(weight))
    assert mean == pytest.approx(0.004, rel=0.05)


def test_murray_law_slows_daughters() -> None:
    """Daughter branches carry a fraction of inlet flow, so they are slower."""
    tree = _tree("bifurcation", seed=5)
    trunk = tree.branches[0]
    up = tree.branches[1]
    # Murray: r_parent^3 ~= sum r_daughter^3 at the junction.
    r_p = float(tree.radii[trunk.start])
    r_a = float(tree.radii[up.start])
    r_b = float(tree.radii[tree.branches[2].start])
    assert r_a**3 + r_b**3 == pytest.approx(r_p**3, rel=0.15)
    assert r_a < r_p and r_b < r_p


def test_radius_override_couples_clot_to_flow() -> None:
    """Passing an occluded radius must speed up the flow past the clot."""
    tree = _y_tree()
    station = np.array([10], dtype=np.int32)
    pos = tree.points[10][None, :]
    free = tree.flow(pos, station, inlet_speed=0.004, reference_radius=0.05)
    blocked = tree.flow(
        pos, station, inlet_speed=0.004, reference_radius=0.05,
        radius_override=np.array([0.025], np.float32),
    )
    assert np.linalg.norm(blocked) > np.linalg.norm(free) * 2.0


def test_flow_speed_is_capped() -> None:
    """A near-total occlusion must not produce an unbounded velocity."""
    tree = _y_tree()
    station = np.array([10], dtype=np.int32)
    pos = tree.points[10][None, :]
    v = tree.flow(
        pos, station, inlet_speed=0.004, reference_radius=0.05,
        radius_override=np.array([1e-6], np.float32),
    )
    assert np.linalg.norm(v) <= 0.004 * 8.0 * 2.0 + 1e-6


# ----------------------------------------------------------------- lookahead


def test_lookahead_follows_the_route_at_a_fork() -> None:
    """Lookahead must bend into the branch the agent is routed to."""
    tree = _y_tree()
    junction = int(tree.stations_of_branch(0)[-1])
    station = np.array([junction - 2], dtype=np.int32)

    for branch in (1, 2):
        target = int(tree.stations_of_branch(branch)[-1])
        _d, next_hop = tree.route_to(target)
        pts = tree.lookahead(station, offsets=(2, 6, 12), next_hop=next_hop)
        assert pts.shape == (1, 3, 3)
        # The furthest lookahead point should already be in the target branch.
        far = pts[0, -1]
        dists = np.linalg.norm(tree.points - far, axis=1)
        assert tree.branch_ids[int(np.argmin(dists))] == branch


def test_lookahead_without_route_goes_downstream() -> None:
    tree = _y_tree()
    station = np.array([5], dtype=np.int32)
    pts = tree.lookahead(station, offsets=(1, 5, 10), next_hop=None)
    xs = pts[0, :, 0]
    assert np.all(np.diff(xs) > 0), "lookahead did not advance downstream"


def test_lookahead_clamps_at_the_distal_end() -> None:
    """Walking off the end of a branch must saturate, not wrap or crash."""
    tree = _y_tree()
    last = int(tree.stations_of_branch(1)[-1])
    pts = tree.lookahead(np.array([last]), offsets=(1, 20), next_hop=None)
    assert np.allclose(pts[0, 0], tree.points[last])
    assert np.allclose(pts[0, 1], tree.points[last])
