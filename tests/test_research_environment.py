"""Research physics/observation/reward regressions; deliberately NumPy-only."""
import numpy as np
import pytest
from environments.vector_env import VectorVascularEnv
from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.contact_geometry import continuous_route_distance
from tests.test_vector_env import _mirror, _make_pair


@pytest.mark.parametrize('mode', ['geodesic', 'euclidean'])
def test_v2_prefix_and_single_vector_parity(mode):
    vec, single = _make_pair('geometric_v2')
    vec.contact_mode = single.contact_mode = mode
    _mirror(single, vec)
    v = vec._observe()['nodes'][0]
    s = single._build_observation()['nodes']
    np.testing.assert_allclose(v, s, atol=1e-6)
    assert single.observation_space['nodes'].contains(s)
    vec.obs_mode, vec.node_feature_dim = 'geometric', 36
    np.testing.assert_array_equal(vec._observe()['nodes'][0], v[:, :36])


def test_flow_features_unsaturated_and_margin_negative():
    env = VectorVascularEnv(n_envs=1, seed=4, obs_mode='geometric_v2')
    station = env.clot_stations[0, 0]
    env.robot_stations[:] = station
    env.robot_positions[:] = env.tree.points[station]
    env.flow_speed = 0.02
    nodes = env._observe()['nodes']
    assert np.all(nodes[..., 36] > 1)
    assert np.all(nodes[..., 38] < 0)
    assert np.isfinite(nodes).all()


def configure_pair(vec, robot, clot):
    vec.clot_alive[:] = False
    vec.clot_masses[:] = 0
    vec.clot_initial[:] = 0
    vec.clot_alive[:, 0] = True
    vec.clot_masses[:, 0] = 1
    vec.clot_initial[:, 0] = 1
    vec.clot_stations[:, 0] = clot
    vec.clot_positions[:, 0] = vec.tree.points[clot]
    vec.robot_stations[:] = robot
    vec.robot_positions[:] = vec.tree.points[robot]
    vec.flow_speed = vec.brownian_sigma = 0
    vec._route_cache.clear()


@pytest.mark.parametrize('legal', [False, True])
def test_cross_wall_and_legal_parent_child_contact(legal):
    vec = VectorVascularEnv(n_envs=1, num_robots=1, seed=0, randomize_scenario=False)
    tree = vec.tree
    candidates = []
    for c in tree.stations_of_branch(1):
        d, _ = tree.route_to(int(c))
        euclidean = np.linalg.norm(tree.points - tree.points[c], axis=1)
        source = tree.stations_of_branch(0 if legal else 2)
        mask = (euclidean[source] < .035) & ((d[source] < .03) if legal else (d[source] > .04))
        candidates.extend((int(a), int(c)) for a in source[mask])
    assert candidates, 'geometry must contain the required counterexample'
    robot, clot = candidates[0]
    configure_pair(vec, robot, clot)
    single = Vascular3DMARLEnv(num_robots=1, seed=0, randomize_scenario=False)
    single.reset(seed=0)
    _mirror(single, vec)
    single.flow_speed = single.brownian_sigma = 0
    for env in (vec, single):
        nodes = env._observe()['nodes'][0] if env is vec else env._build_observation()['nodes']
        assert bool(nodes[0, 30]) == legal
    _, _, _, _, vi = vec.step(np.zeros((1, 1, 3)))
    _, _, _, _, si = single.step(np.zeros((1, 3)))
    assert bool(vi['clots_engaged'][0]) == legal
    assert bool(si['clots_engaged']) == legal
    if not legal:
        configure_pair(vec, robot, clot)
        vec.contact_mode = 'euclidean'
        assert vec._observe()['nodes'][0, 0, 30] == 1
        assert vec.step(np.zeros((1, 1, 3)))[4]['clots_engaged'][0] == 1


def test_continuous_correction_and_same_branch_foldback():
    from types import SimpleNamespace
    points = np.array([[0,0,0], [.02,0,0], [.02,.1,0], [0,.01,0]], dtype=np.float32)
    tree = SimpleNamespace(points=points, tangents=np.tile([1.,0,0], (4,1)), branch_ids=np.zeros(4))
    distance = np.array([.21,.19,.09,0])
    hop = np.array([1,2,3,3])
    d = continuous_route_distance(tree, np.array([[.005,0,0]]), np.array([0]), distance, hop)
    assert d[0] == pytest.approx(.205)
    assert np.linalg.norm(points[0]-points[3]) < .035
    assert d[0] > .035  # same branch, but not reachable through the local lumen


def test_reward_ablation_changes_actual_ppo_targets():
    vec = VectorVascularEnv(n_envs=1, num_robots=2, seed=3)
    configure_pair(vec, int(vec.clot_stations[0,0]), int(vec.clot_stations[0,0]))
    vec.clot_masses[:,0] = .001
    snapshot = vec.state_dict()
    on = vec.step(np.zeros((1,2,3)))[4]
    vec.load_state_dict(snapshot)
    vec.reward_double_count = 'off'
    off = vec.step(np.zeros((1,2,3)))[4]
    target_on = on['agent_rewards'] + on['team_reward'][:,None]/2
    target_off = off['agent_rewards'] + off['team_reward'][:,None]/2
    assert on['success'][0] and off['success'][0]
    assert (target_on-target_off).sum() == pytest.approx(vec.success_bonus + .001*vec.progress_scale, abs=1e-5)
    np.testing.assert_allclose(on['team_reward'], off['team_reward'])
