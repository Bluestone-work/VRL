from dataclasses import replace

import numpy as np
import pytest

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from environments.mca_obstacle_forecast import (
    HORIZONS_S, trajectory_features, linear_particle_predictions, bound_trajectory_features)
from marl.mca_physical_policy import make_physical_agent, initialize_expanded_obstacle_policy, physical_context
from tests.test_mca_obstacle_forecast import make


MODES = ('bounded_trajectory_four', 'bounded_linear_four', 'anchored_linear_four')


def test_relative_linear_extrapolation_and_fixed_current_robot_reference():
    robot = np.array([[10., 2., 3.]])
    velocity = np.array([[1., 0., 0.]])
    particle = robot+np.array([[2., 0., 0.]])
    particle_velocity = np.array([[-.2, 0., 0.]])
    points, valid, exits = linear_particle_predictions(particle, particle_velocity)
    args = (robot, velocity, np.eye(3)[None], particle, particle_velocity, points, valid, exits, .08, .02, 1., .15)
    relative = trajectory_features(*args)
    anchored = trajectory_features(*args, reference_velocities=np.zeros_like(velocity))
    for k, horizon in enumerate(HORIZONS_S):
        offset = 6+5*k
        assert relative[0, offset] == pytest.approx((2-1.2*horizon)/1.5)
        assert anchored[0, offset] == pytest.approx((2-.2*horizon)/1.5)
    # Anchoring changes the forecast reference only, not observed relative speed.
    assert relative[0,3] == anchored[0,3] == pytest.approx(-1.2)
    assert relative[0,21] < 0 and anchored[0,21] > 0


def test_linear_prediction_is_translation_invariant_and_does_not_read_future_state():
    particle = np.array([[.3,.4,.2]])
    robot = np.zeros((1,3)); velocity = np.array([[.1,-.2,.3]])
    robot_velocity = np.array([[.4,.1,-.1]])
    features = []
    for shift in (np.zeros(3), np.array([80.,-40.,20.])):
        predicted, valid, exits = linear_particle_predictions(particle+shift, velocity)
        features.append(trajectory_features(robot+shift, robot_velocity, np.eye(3)[None],
            particle+shift, velocity, predicted, valid, exits, .08,.02,1.,.15))
    np.testing.assert_allclose(features[0], features[1], atol=1e-6)
    assert np.all(valid) and np.all(np.isinf(exits))  # no oracle future-exit mask


def test_softsign_preserves_masks_time_order_and_does_not_mutate_raw_features():
    x = np.zeros((2,96), np.float32)
    x[:, :24] = np.arange(24)-12
    for k in (10,15,20,23):x[:,k]=1
    x[:,22]=.7
    saved=x.copy();bounded=bound_trajectory_features(x)
    np.testing.assert_array_equal(x,saved)
    for k in (10,15,20,22,23):np.testing.assert_array_equal(bounded[:,k],x[:,k])
    assert np.abs(bounded).max() <= 1
    assert np.all(np.diff(bounded[0,:10]) > 0)
    np.testing.assert_array_equal(bounded[:,24:],0)


@pytest.mark.parametrize('mode',MODES)
def test_observation_change_preserves_physics_rewards_and_original76(mode):
    cfg=DynamicsConfig.from_json('configs/experiments/EXP_0032_TRAJECTORY_DYNAMICS.json')
    old=CompiledMCAPhysicalEnv(cfg);new=CompiledMCAPhysicalEnv(replace(cfg,obstacle_observation=mode))
    a,_=old.reset(seed=940000011);b,_=new.reset(seed=940000011)
    rng=np.random.default_rng(25)
    for _ in range(30):
        np.testing.assert_array_equal(a['nodes'][:,:76],b['nodes'][:,:76])
        assert np.max(np.abs(b['nodes'][:,76:]))<=1
        command=rng.uniform(-.5,.5,(cfg.num_robots,3))
        x=old.step(command);y=new.step(command);a=x[0];b=y[0]
        assert x[1:4]==y[1:4]
        for field in ('positions_mm','active','masses','velocity_mm_s'):
            np.testing.assert_array_equal(getattr(old,field),getattr(new,field))
        assert old.np_random.bit_generator.state==new.np_random.bit_generator.state


@pytest.mark.parametrize('mode',('bounded_linear_four','anchored_linear_four'))
def test_linear_observation_does_not_run_transport_prediction(monkeypatch,mode):
    env,_=make(mode=mode)
    def forbidden(*args,**kwargs):raise AssertionError('Linear prediction must not call transport')
    monkeypatch.setattr(env,'_advance_particle_prediction',forbidden)
    obs=env._observation()
    assert obs['nodes'].shape==(1,172) and np.any(obs['nodes'][:,76:])


@pytest.mark.parametrize('mode',MODES)
def test_expansion_preserves_initial_actor_and_critic_and_tags_new_semantics(mode):
    old_env,a=make(mode='predictive_four');new_env,b=make(mode=mode)
    old=make_physical_agent(old_env,hidden_dim=32);new=make_physical_agent(new_env,hidden_dim=32)
    checkpoint=dict(meta=old.meta,actor=old.actor.state_dict(),critic=old.critic.state_dict())
    initialize_expanded_obstacle_policy(new,checkpoint)
    assert new.meta['observation_schema'] != 'mca_point_trajectories_172_v5'
    x=old.act(a['nodes'],physical_context(old_env,a),a['clot_state'].reshape(-1),deterministic=True)
    y=new.act(b['nodes'],physical_context(new_env,b),b['clot_state'].reshape(-1),deterministic=True)
    for u,v in zip(x,y):np.testing.assert_allclose(u,v,rtol=1e-5,atol=1e-6)
