"""Environment contract tests, plus the diagnostics that motivated the rewrite.

`test_geodesic_shaping_agrees_with_geometry` is the load-bearing one: it checks
that a robot moving along the vessel toward its clot is rewarded, and that a
robot moving on the Euclidean straight line toward a clot in the other branch is
not. That distinction is what the original Euclidean shaping got backwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_marl_env import (  # noqa: E402
    NODE_FEATURE_DIM_GEOMETRIC,
    NODE_FEATURE_DIM_DYNAMIC,
    NODE_FEATURE_DIM_PREDICTIVE,
    NODE_FEATURE_DIM_LEGACY,
    OBS_MODES,
    SCENARIOS,
    Vascular3DMARLEnv,
)


def make_env(**kw) -> Vascular3DMARLEnv:
    opts = dict(
        scenario="bifurcation", num_robots=3, num_clots=3, horizon=60,
        seed=0, randomize_scenario=False, randomize_clots=True,
        use_pybullet=False, reward_mode="milestone", obs_mode="geometric",
    )
    opts.update(kw)
    return Vascular3DMARLEnv(**opts)


def test_state_dict_restores_exact_next_transition() -> None:
    env = make_env(
        scenario="mca_m1_lvo", num_clots=2, robot_radius=0.0011, seed=123
    )
    env.reset(seed=123)
    snapshot = env.state_dict()
    action = np.full((3, 3), 0.25, dtype=np.float32)
    obs_a, reward_a, term_a, trunc_a, _ = env.step(action)

    restored = make_env(
        scenario="mca_m1_lvo", num_clots=2, robot_radius=0.0011, seed=999
    )
    restored.load_state_dict(snapshot)
    obs_b, reward_b, term_b, trunc_b, _ = restored.step(action)

    assert reward_a == pytest.approx(reward_b)
    assert term_a == term_b and trunc_a == trunc_b
    for key in obs_a:
        assert np.allclose(obs_a[key], obs_b[key])


# ------------------------------------------------------------------ contracts


@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_observation_matches_space(obs_mode: str) -> None:
    env = make_env(obs_mode=obs_mode)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs), "observation left its declared space"
    expected = (
        NODE_FEATURE_DIM_PREDICTIVE if obs_mode == "geometric_predictive" else NODE_FEATURE_DIM_DYNAMIC if obs_mode == "geometric_dynamic" else 42 if obs_mode == "geometric_v2" else NODE_FEATURE_DIM_GEOMETRIC if obs_mode == "geometric" else NODE_FEATURE_DIM_LEGACY
    )
    assert obs["nodes"].shape == (3, expected)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        if term or trunc:
            break


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenarios_run(scenario: str) -> None:
    env = make_env(scenario=scenario, randomize_scenario=False)
    env.reset(seed=2)
    for _ in range(40):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs["nodes"]))
        assert np.isfinite(r)
        if term or trunc:
            break


def test_observations_stay_in_range() -> None:
    """Every feature must stay in [-1, 1]; an unnormalised feature would let one
    channel dominate the first layer."""
    env = make_env(randomize_scenario=True)
    for ep in range(6):
        obs, _ = env.reset(seed=100 + ep)
        for _ in range(30):
            assert obs["nodes"].min() >= -1.0 - 1e-5, obs["nodes"].min()
            assert obs["nodes"].max() <= 1.0 + 1e-5, obs["nodes"].max()
            obs, r, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break


def test_robots_stay_inside_the_lumen() -> None:
    """Hard invariant: no robot may ever be outside its local vessel radius."""
    env = make_env(num_robots=6, randomize_scenario=True)
    for ep in range(5):
        env.reset(seed=200 + ep)
        for _ in range(50):
            # Drive hard at the wall to stress the projection.
            action = np.tile(np.array([0.2, 1.0, 1.0], np.float32), (6, 1))
            _obs, _r, term, trunc, _info = env.step(action)
            axis_point, radius = env.tree._axis_point(
                env.robot_positions, env.robot_stations
            )
            radial = np.linalg.norm(env.robot_positions - axis_point, axis=1)
            assert np.all(radial <= radius - env.robot_radius + 1e-4)
            if term or trunc:
                break


def test_determinism_under_seed() -> None:
    a, b = make_env(), make_env()
    obs_a, _ = a.reset(seed=7)
    obs_b, _ = b.reset(seed=7)
    assert np.allclose(obs_a["nodes"], obs_b["nodes"])
    for _ in range(15):
        act = a.action_space.sample()
        oa, ra, ta, ua, _ = a.step(act)
        ob, rb, tb, ub, _ = b.step(act)
        assert np.allclose(oa["nodes"], ob["nodes"])
        assert ra == pytest.approx(rb)
        if ta or ua:
            break


def test_agent_rewards_shape_and_reward_consistency() -> None:
    """info must expose the per-agent decomposition the trainer needs."""
    env = make_env()
    env.reset(seed=3)
    _obs, reward, _t, _u, info = env.step(env.action_space.sample())
    assert info["agent_rewards"].shape == (3,)
    # The scalar reward is team + mean(agent), as documented.
    assert reward == pytest.approx(
        info["team_reward"] + float(info["agent_rewards"].mean()), rel=1e-5
    )


def test_truncation_at_horizon() -> None:
    env = make_env(horizon=25)
    env.reset(seed=4)
    for i in range(25):
        _o, _r, term, trunc, _info = env.step(np.zeros((3, 3), np.float32))
        if term:
            pytest.skip("cleared early")
    assert trunc and not term


# ------------------------------------------------------------------- geodesics


def _straight_env() -> Vascular3DMARLEnv:
    env = make_env(scenario="straight", num_robots=1, num_clots=1,
                   randomize_clots=False, horizon=400)
    env.reset(seed=11)
    return env


def test_geodesic_shaping_rewards_moving_along_the_vessel() -> None:
    """Approach reward must be positive while travelling toward the clot.

    Only the approach phase is measured. Driving at full tangent speed does not
    stop on arrival, so continuing past first contact would overshoot the clot and
    grind the distal wall -- a policy failure, not a shaping failure.
    """
    env = _straight_env()
    clot_station = int(env.clot_stations[0])
    # Park the single robot upstream of the clot, on the axis.
    start = int(max(clot_station - 25, 1))
    env.robot_positions = env.tree.points[start][None, :].copy()
    env.robot_stations = np.array([start], np.int32)

    total_approach = 0.0
    steps = 0
    for _ in range(40):
        # Drive along the local tangent.
        t_hat = env.tree.tangents[env.robot_stations[0]]
        _o, _r, _term, _trunc, info = env.step(t_hat[None, :].astype(np.float32))
        steps += 1
        if info["first_contact_step"] >= 0:
            break
        total_approach += float(info["agent_rewards"][0])
        assert info["wall_collisions"] == 0, "following the tangent hit a wall"
    assert info["first_contact_step"] >= 0, "following the vessel never reached the clot"
    assert total_approach > 0, (
        f"approaching the clot was not rewarded (got {total_approach:+.5f})"
    )


def test_shaping_is_continuous_not_a_staircase() -> None:
    """Sub-station resolution: every axial step must earn shaping.

    Taking the geodesic field at the nearest station alone quantises distance to
    the station spacing, so a robot can travel a full step, stay on the same
    station, and receive exactly zero -- a staircase signal that is silent for
    several steps and then jumps.
    """
    env = _straight_env()
    clot_station = int(env.clot_stations[0])
    start = int(max(clot_station - 30, 1))
    env.robot_positions = env.tree.points[start][None, :].copy()
    env.robot_stations = np.array([start], np.int32)

    zero_steps = 0
    measured = 0
    for _ in range(12):
        t_hat = env.tree.tangents[env.robot_stations[0]]
        _o, _r, _t, _u, info = env.step(t_hat[None, :].astype(np.float32))
        if info["first_contact_step"] >= 0:
            break
        measured += 1
        if abs(float(info["agent_rewards"][0])) < 1e-9:
            zero_steps += 1
    assert measured >= 5
    assert zero_steps == 0, (
        f"{zero_steps}/{measured} axial steps earned exactly zero shaping; the "
        "geodesic term is quantised to station spacing"
    )


def test_clots_are_not_placed_in_the_distal_end_cap() -> None:
    """A clot at the very last station is a trap.

    The robot arrives, cannot advance past the terminal wall, and accumulates
    wall penalties while sitting on its own target. Placement must stay clear of
    the end cap.
    """
    for ep in range(20):
        env = make_env(num_clots=3, randomize_scenario=True)
        env.reset(seed=700 + ep)
        arc = env.tree.arclength[env.clot_stations] / max(env.tree.total_length, 1e-8)
        assert np.all(arc <= env.distal_margin + 1e-6), (
            f"clot placed at arclength {arc.max():.3f} > {env.distal_margin}"
        )


def test_euclidean_shortcut_across_branches_is_not_rewarded() -> None:
    """The core fix, stated as a sign disagreement.

    Robot in one daughter branch, clot in the other. Because the two limbs
    diverge, stepping along the straight line to the clot can *shorten* the
    Euclidean distance while *lengthening* the along-vessel distance -- the robot
    is committing further down a branch it will have to back out of.

    The old shaping term paid for exactly that step. Geodesic shaping must charge
    for it. The assertion is on the sign: Euclidean says "closer", the reward must
    say "worse".
    """
    env = make_env(scenario="bifurcation", num_robots=1, num_clots=1,
                   randomize_clots=False, randomize_scenario=False, horizon=400)
    env.reset(seed=0)
    tree = env.tree
    up, dn = tree.stations_of_branch(1), tree.stations_of_branch(2)

    # Clot distally in the lower limb, robot well down the upper limb.
    clot_station = int(dn[48])
    env.clot_stations = np.array([clot_station], np.int32)
    env.clot_positions = tree.points[clot_station][None, :].copy()
    env.clot_masses = np.array([1.0], np.float32)
    env.clot_initial_mass = env.clot_masses.copy()
    env.active_clots = 1
    env._route_cache = {}

    robot_station = int(up[37])
    start = tree.points[robot_station].copy()
    env.robot_positions = start[None, :].copy()
    env.robot_stations = np.array([robot_station], np.int32)

    euclid_before = float(np.linalg.norm(env.clot_positions[0] - start))
    direction = env.clot_positions[0] - start
    direction /= np.linalg.norm(direction)

    _o, _r, _t, _u, info = env.step(direction[None, :].astype(np.float32))

    euclid_after = float(np.linalg.norm(env.clot_positions[0] - env.robot_positions[0]))
    shaping = float(info["agent_rewards"][0]) + (
        env.wall_collision_penalty * info["wall_collisions"]
    )

    assert info["wall_collisions"] == 0, "test wanted a clean step, not a wall grind"
    assert euclid_after < euclid_before, "setup failed: Euclidean distance did not drop"
    assert shaping < 0, (
        "a step that shortens the straight line but lengthens the route was "
        f"rewarded (shaping={shaping:+.6f}); geodesic shaping is not in effect"
    )


def test_geodesic_and_euclidean_disagree_often_enough_to_matter() -> None:
    """Quantifies the bug the geodesic switch fixes.

    Sweeps robot/clot placements across the two daughter branches and counts how
    often "Euclidean closer" and "route shorter" disagree in sign. If this were
    rare, the original shaping would have been a harmless approximation; it is
    not rare, which is why ~45% of episodes never reached a clot.
    """
    disagreements = 0
    trials = 0
    for seed in range(12):
        env = make_env(scenario="bifurcation", num_robots=1, num_clots=1,
                       randomize_clots=False, randomize_scenario=False, horizon=400)
        env.reset(seed=seed)
        tree = env.tree
        up, dn = tree.stations_of_branch(1), tree.stations_of_branch(2)
        # Distal clot: the further apart the limb tips, the more often the
        # straight line and the route disagree.
        clot_station = int(dn[3 * dn.size // 4])
        env.clot_stations = np.array([clot_station], np.int32)
        env.clot_positions = tree.points[clot_station][None, :].copy()
        env.clot_masses = np.array([1.0], np.float32)
        env.clot_initial_mass = env.clot_masses.copy()
        env.active_clots = 1
        env._route_cache = {}

        for ri in range(2, up.size - 2, 7):
            robot_station = int(up[ri])
            start = tree.points[robot_station].copy()
            env.robot_positions = start[None, :].copy()
            env.robot_stations = np.array([robot_station], np.int32)
            euclid_before = float(np.linalg.norm(env.clot_positions[0] - start))
            geo_before = env._geodesic_to_target(env._assigned_clot())[0]
            direction = env.clot_positions[0] - start
            direction /= np.linalg.norm(direction)
            _o, _r, _t, _u, info = env.step(direction[None, :].astype(np.float32))
            if info["wall_collisions"]:
                continue
            trials += 1
            euclid_after = float(
                np.linalg.norm(env.clot_positions[0] - env.robot_positions[0])
            )
            geo_after = env._geodesic_to_target(env._assigned_clot())[0]
            if (euclid_before - euclid_after) > 1e-5 and (geo_before - geo_after) < -1e-5:
                disagreements += 1

    assert trials > 20, "sweep did not produce enough clean steps"
    rate = disagreements / trials
    # Measured at ~64% for a clot 3/4 down the opposite limb; the rate climbs
    # with distality (~30% at the midpoint, ~81% at the tip). The threshold is
    # set well below the measurement so it tracks the phenomenon, not the seed.
    assert rate > 0.4, (
        f"expected the Euclidean/geodesic disagreement to be common, got {rate:.1%}"
    )


def test_assignment_prefers_the_reachable_clot() -> None:
    """Assignment is by along-vessel distance, so a clot in the robot's own
    branch beats a Euclidean-closer clot across the wall."""
    env = make_env(scenario="bifurcation", num_robots=1, num_clots=2,
                   randomize_clots=False, randomize_scenario=False)
    env.reset(seed=17)
    tree = env.tree
    up, dn = tree.stations_of_branch(1), tree.stations_of_branch(2)

    robot_station = int(dn[dn.size // 3])
    env.robot_positions = tree.points[robot_station][None, :].copy()
    env.robot_stations = np.array([robot_station], np.int32)

    far_same_branch = int(dn[-1])
    near_other_branch = int(up[up.size // 3])
    env.clot_stations = np.array([near_other_branch, far_same_branch], np.int32)
    env.clot_positions = tree.points[env.clot_stations].copy()
    env.clot_masses = np.ones(2, np.float32)
    env.clot_initial_mass = env.clot_masses.copy()
    env.active_clots = 2
    env._route_cache = {}

    euclid = np.linalg.norm(env.clot_positions - env.robot_positions, axis=1)
    geo = np.array([
        env._route(c)[0][robot_station] for c in range(2)
    ])
    if not (euclid[0] < euclid[1] and geo[0] > geo[1]):
        pytest.skip("this geometry does not create the ambiguity")
    assert int(env._assigned_clot()[0]) == 1, "assigned the unreachable clot"


# --------------------------------------------------------------------- physics


def test_lysis_saturates_per_clot() -> None:
    """A 5th robot on one clot must not speed it up proportionally."""
    removed = {}
    for n_robots in (2, 12):
        env = make_env(num_robots=n_robots, num_clots=1, randomize_clots=False,
                       scenario="straight", randomize_scenario=False)
        env.reset(seed=21)
        # Stack every robot on the clot.
        env.robot_positions = np.repeat(
            env.clot_positions[:1], n_robots, axis=0
        ).astype(np.float32)
        env.robot_stations = np.repeat(env.clot_stations[:1], n_robots).astype(np.int32)
        _o, _r, _t, _u, info = env.step(np.zeros((n_robots, 3), np.float32))
        removed[n_robots] = info["removed_mass"]
    assert removed[12] < removed[2] * 6.0 * 0.6, "lysis did not saturate"


def test_clot_occludes_and_lysis_reopens_the_vessel() -> None:
    """Clot mass must narrow the lumen, and clearing it must restore the radius."""
    env = make_env(scenario="straight", num_robots=1, num_clots=1,
                   randomize_clots=False, randomize_scenario=False)
    env.reset(seed=23)
    station = env.clot_stations[:1]
    blocked = env._occluded_radius(station)[0]
    geometric = float(env.tree.radii[station[0]])
    assert blocked < geometric * 0.9, "clot did not occlude the lumen"

    env.clot_masses[:] = 0.0
    reopened = env._occluded_radius(station)[0]
    assert reopened == pytest.approx(geometric, rel=1e-5)


def test_wall_contact_removes_normal_velocity() -> None:
    """Hitting the wall must kill the inward-to-outward velocity component,
    not silently preserve momentum through a position clamp."""
    env = make_env(scenario="straight", num_robots=1, randomize_scenario=False)
    env.reset(seed=27)
    hit = False
    for _ in range(40):
        _o, _r, _t, _u, info = env.step(np.array([[0.0, 1.0, 0.0]], np.float32))
        if info["wall_collisions"] > 0:
            hit = True
            axis_point, _r2 = env.tree._axis_point(env.robot_positions, env.robot_stations)
            outward = env.robot_positions - axis_point
            outward /= max(float(np.linalg.norm(outward)), 1e-8)
            vn = float(np.sum(env.robot_velocities * outward))
            assert vn <= 1e-6, f"outward velocity survived wall contact: {vn}"
            break
    assert hit, "never reached the wall"


def test_lubrication_reduces_thrust_near_the_wall() -> None:
    """Closes the 'hug the wall to dodge adverse flow' exploit."""
    env = make_env(scenario="straight", num_robots=2, randomize_scenario=False)
    env.reset(seed=29)
    station = env.robot_stations[:1]
    axis_point, radius = env.tree._axis_point(env.tree.points[station], station)
    axis = env.tree.points[station[0]]
    normal = env.tree.normals[station[0]]
    on_axis = axis[None, :]
    at_wall = (axis + normal * float(radius[0]) * 0.99)[None, :]
    lube_axis = env._lubrication(on_axis, station)[0]
    lube_wall = env._lubrication(at_wall, station)[0]
    assert lube_wall < lube_axis
    assert lube_wall >= env.lubrication_floor - 1e-6
    assert lube_axis == pytest.approx(1.0, abs=1e-5)


def test_robot_separation_pushes_overlaps_apart() -> None:
    """Colliding robots must be separated, not merely penalised."""
    env = make_env(num_robots=2, scenario="straight", randomize_scenario=False)
    env.reset(seed=31)
    station = int(env.robot_stations[0])
    p0 = env.tree.points[station]
    # Place both robots at (nearly) the same point.
    env.robot_positions = np.stack([p0, p0 + np.array([1e-5, 0, 0], np.float32)])
    env.robot_stations = np.array([station, station], np.int32)
    before = float(np.linalg.norm(env.robot_positions[0] - env.robot_positions[1]))
    env.step(np.zeros((2, 3), np.float32))
    after = float(np.linalg.norm(env.robot_positions[0] - env.robot_positions[1]))
    assert after > before, "overlapping robots were not pushed apart"


def test_no_branch_jumping_during_an_episode() -> None:
    """Station tracking must be continuous: a robot cannot switch branch without
    passing through the junction region."""
    env = make_env(scenario="bifurcation", num_robots=4, randomize_scenario=False,
                   horizon=200)
    env.reset(seed=33)
    tree = env.tree
    prev_branch = tree.branch_ids[env.robot_stations].copy()
    for _ in range(150):
        _o, _r, term, trunc, _info = env.step(env.action_space.sample())
        branch = tree.branch_ids[env.robot_stations]
        for i in range(4):
            if branch[i] == prev_branch[i]:
                continue
            # A legal change means the two branches are adjacent in the topology.
            a, b = int(prev_branch[i]), int(branch[i])
            related = (
                tree.branches[a].parent == b
                or tree.branches[b].parent == a
                or tree.branches[a].parent == tree.branches[b].parent
            )
            assert related, f"illegal branch jump {a} -> {b}"
        prev_branch = branch.copy()
        if term or trunc:
            break


# ------------------------------------------------------------------ curriculum


def test_curriculum_starts_with_one_near_clot() -> None:
    env = make_env(num_clots=3, curriculum=True, randomize_clots=True)
    env.set_difficulty(0.0)
    env.reset(seed=41)
    assert env.active_clots == 1
    arc = float(
        env.tree.arclength[env.clot_stations[0]] / max(env.tree.total_length, 1e-8)
    )
    assert arc <= 0.5, f"easiest curriculum clot is not proximal (arc={arc})"


def test_curriculum_full_difficulty_spreads_clots() -> None:
    env = make_env(num_clots=3, curriculum=True, randomize_clots=False)
    env.set_difficulty(1.0)
    env.reset(seed=43)
    assert env.active_clots == 3


def test_difficulty_is_clamped() -> None:
    env = make_env(curriculum=True)
    env.set_difficulty(5.0)
    assert env._difficulty == 1.0
    env.set_difficulty(-2.0)
    assert env._difficulty == 0.0


# ----------------------------------------------------------- contact diagnostic


def test_random_policy_contact_rate_is_measurable() -> None:
    """Sanity check on the metric the whole rewrite targets.

    This does not assert an improvement (a random policy is not a policy), only
    that `first_contact_step` is wired up and that contact is achievable at the
    easiest curriculum setting -- if it were not, no amount of learning would
    help.
    """
    env = make_env(num_robots=6, num_clots=2, curriculum=True, horizon=300,
                   randomize_scenario=True)
    env.set_difficulty(0.0)
    contacted = 0
    episodes = 12
    for ep in range(episodes):
        env.reset(seed=500 + ep)
        for _ in range(300):
            _o, _r, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                break
        if info["first_contact_step"] >= 0:
            contacted += 1
    assert contacted > 0, "even the easiest setting is unreachable by chance"
