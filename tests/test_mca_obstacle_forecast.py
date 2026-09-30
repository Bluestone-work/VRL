import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import MCAPhysicalEnv, DynamicsConfig
from environments.mca_obstacle_forecast import forecast_particles, trajectory_features, HORIZONS_S
from marl.mca_physical_policy import make_physical_agent, initialize_expanded_obstacle_policy, physical_context
from tests.test_mca_physical_dynamics import pipe


def config(**kwargs):
    cfg = replace(DynamicsConfig(), contact_model='localized_point', obstacle_observation='trajectory_four',
                  num_robots=1, particle_count=2, inlet_flow_ml_min=.06*np.pi,
                  initial_radius_fraction=1., lysis_mass_per_s=0., episode_duration_s=10., control_dt_s=.1)
    return replace(cfg, **kwargs)


def make(cls=CompiledMCAPhysicalEnv, mode='trajectory_four'):
    env = cls(config(obstacle_observation=mode), tree=pipe(), clot_stations=[10])
    obs, _ = env.reset(seed=31, options=dict(robot_positions_mm=np.array([[5., 0., 0.]]),
        particle_positions_mm=np.array([[7., 0., 0.], [19.5, 0., 0.]])))
    return env, obs


@pytest.mark.parametrize('cls', [MCAPhysicalEnv, CompiledMCAPhysicalEnv])
def test_forecast_is_causal_nonmutating_and_tracks_exact_pipe_exit(cls):
    env, obs = make(cls)
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    fields = {k:copy.deepcopy(getattr(env, k)) for k in
        ('positions_mm','edges','active','masses','velocity_mm_s','path_mm','exit_time_s','exit_node','elapsed_s','steps')}
    solution = copy.deepcopy(env.solution)
    predicted, valid, exits = forecast_particles(env, np.array([1, 2]))
    for k, value in fields.items():np.testing.assert_equal(getattr(env, k), value)
    for k, value in solution.items():np.testing.assert_equal(env.solution[k], value)
    assert env.np_random.bit_generator.state == rng
    np.testing.assert_allclose(predicted[:, 0, 0], 7+2*np.asarray(HORIZONS_S), atol=1e-9)
    np.testing.assert_allclose(predicted[:, 1, 0], 20., atol=1e-9)
    assert np.all(valid[:, 0]) and not np.any(valid[:, 1])
    assert exits[1] == pytest.approx(.25, abs=1e-9) and np.isinf(exits[0])
    assert obs['nodes'].shape == (1, 172) and env.observation_space.contains(obs)


def test_legacy_prefix_and_masked_ablation_preserve_original_observations():
    old, original = make(mode='predictive_four')
    forecast, expanded = make()
    masked, hidden = make(mode='trajectory_four_masked')
    np.testing.assert_array_equal(original['nodes'], expanded['nodes'][:, :76])
    np.testing.assert_array_equal(original['nodes'], hidden['nodes'][:, :76])
    assert np.any(expanded['nodes'][:, 76:]) and not np.any(hidden['nodes'][:, 76:])
    for env in (forecast, masked):
        np.testing.assert_array_equal(env.positions_mm, old.positions_mm)
        np.testing.assert_array_equal(env.clot_positions_mm, old.clot_positions_mm)
    for _ in range(20):
        results = [env.step(np.zeros((1, 3))) for env in (old, forecast, masked)]
        for env in (forecast, masked):
            np.testing.assert_array_equal(env.positions_mm, old.positions_mm)
            np.testing.assert_array_equal(env.masses, old.masses)
            np.testing.assert_array_equal(env.active, old.active)
        assert results[0][1:4] == results[1][1:4] == results[2][1:4]


def test_swept_risk_prioritizes_future_crossing_over_near_receding_body():
    positions = np.array([[1.,0.,0.],[.2,0.,0.]])
    velocities = np.array([[-1.4,0.,0.],[1.,0.,0.]])
    pred = positions[None]+np.asarray(HORIZONS_S)[:,None,None]*velocities[None]
    features = trajectory_features(np.zeros((1,3)), np.zeros((1,3)), np.eye(3)[None],
        positions, velocities, pred, np.ones((3,2),bool), np.full(2,np.inf), .08,.02,1.,.15)[0]
    assert features[0] == pytest.approx(1/1.5)  # farther but approaching first
    assert features[21] == pytest.approx(-.1/.15)
    assert features[22] == pytest.approx((1/1.4)/2)
    assert features[23] == 1 and features[47] == 1
    assert np.all(features[48:] == 0)


def test_exited_future_points_are_masked_without_phantom_post_exit_collision():
    features = trajectory_features(np.zeros((1,3)), np.zeros((1,3)), np.eye(3)[None],
        np.array([[.3,0.,0.]]), np.array([[-1.,0.,0.]]), np.array([[[.2,0.,0.]]]*3),
        np.zeros((3,1),bool), np.array([.1]), .08,.02,1.,.15)[0]
    np.testing.assert_array_equal(features[6:21], np.zeros(15))
    assert features[21] == pytest.approx(.1/.15) and features[23] == 1


def test_no_obstacles_and_inactive_robots_have_zero_forecast_slots():
    env = CompiledMCAPhysicalEnv(config(particle_count=0), tree=pipe(), clot_stations=[10])
    obs, _ = env.reset(seed=1)
    assert np.all(obs['nodes'][:,76:] == 0)
    env, _ = make();env.active[0] = False;env._sync_public_state()
    assert np.all(env._observation()['nodes'][0] == 0)


def test_expanded_actor_and_critic_preserve_parent_policy_before_training():
    old_env, old_obs = make(mode='predictive_four')
    new_env, new_obs = make()
    old = make_physical_agent(old_env, hidden_dim=32)
    new = make_physical_agent(new_env, hidden_dim=32)
    checkpoint = dict(meta=old.meta, actor=old.actor.state_dict(), critic=old.critic.state_dict())
    keys = initialize_expanded_obstacle_policy(new, checkpoint)
    assert len(keys) == 2 and new.meta['observation_schema']=='mca_point_trajectories_172_v5'
    for name, net in [('actor',new.actor), ('critic',new.critic)]:
        for key,value in net.state_dict().items():
            if f'{name}.{key}' in keys:
                torch.testing.assert_close(value[:,:76],checkpoint[name][key])
                assert torch.count_nonzero(value[:,76:]) == 0
            else:torch.testing.assert_close(value,checkpoint[name][key])
    a = old.act(old_obs['nodes'], physical_context(old_env,old_obs), old_obs['clot_state'].reshape(-1), deterministic=True)
    b = new.act(new_obs['nodes'], physical_context(new_env,new_obs), new_obs['clot_state'].reshape(-1), deterministic=True)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)


def test_curved_mca_forecast_matches_independent_frozen_flow_rollout():
    base = DynamicsConfig.from_json('configs/experiments/EXP_0029_MCA_ALL_RANDOM_DYNAMICS.json')
    cfg = replace(base, obstacle_observation='trajectory_four', lysis_mass_per_s=0.)
    env = CompiledMCAPhysicalEnv(cfg);env.reset(seed=832000000)
    ids = np.arange(env.num_robots,len(env.active))
    predicted, valid, exits = forecast_particles(env, ids)
    actual = copy.deepcopy(env)
    for k,horizon in enumerate(HORIZONS_S):
        while actual.elapsed_s < horizon-1e-9:actual.step(np.zeros((env.num_robots,3)))
        np.testing.assert_array_equal(valid[k], actual.active[ids])
        np.testing.assert_allclose(predicted[k], actual.positions_mm[ids], rtol=0, atol=.01)
    # The reference forecast must agree with the compiled predictor too.
    ref = MCAPhysicalEnv(cfg);ref.reset(seed=832000000)
    expected, expected_valid, _ = forecast_particles(ref, ids)
    np.testing.assert_allclose(predicted,expected,rtol=0,atol=.01)
    np.testing.assert_array_equal(valid,expected_valid)
