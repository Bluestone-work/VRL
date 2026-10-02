"""Own-route-bearing action prior (structured policy, NOT pure RL).

Executed command = own route bearing + action_residual_scale * policy command,
then the usual bounded/unit actuator. With action_prior='none' nothing changes.
"""
from dataclasses import replace

import numpy as np

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from marl.geometric_control import direct_local_action
from scripts.train_mca_physical import reset_with_valid_particles

CONFIG = 'configs/experiments/EXP_0040_OWN_BEARING_DYNAMICS.json'
SEED = 940000000


def configs():
    base = DynamicsConfig.from_json(CONFIG)
    return base, replace(base, action_prior='own_route_bearing', action_residual_scale=.5)


def test_default_is_off_and_invalid_prior_rejected():
    base, _ = configs()
    assert base.action_prior == 'none'
    env = CompiledMCAPhysicalEnv(base)
    reset_with_valid_particles(env, SEED)
    a = np.random.default_rng(0).uniform(-1, 1, (base.num_robots, 3))
    assert np.array_equal(env._apply_action_prior(a), a)
    import pytest
    with pytest.raises(ValueError):
        replace(base, action_prior='witness')


def test_zero_policy_follows_the_observed_own_bearing_exactly():
    # The prior is the same vector the policy observes in columns 112:115.
    base, prior = configs()
    runs = []
    for cfg, use_obs in ((prior, False), (base, True)):
        env = CompiledMCAPhysicalEnv(cfg)
        obs, _ = reset_with_valid_particles(env, SEED)
        trace = []
        for _ in range(60):
            if use_obs:
                act = direct_local_action(obs['nodes'][:, 112:115].astype(np.float64), env)
                act[~env.active[:cfg.num_robots]] = 0
            else:
                act = np.zeros((cfg.num_robots, 3))
            obs, _, term, trunc, _ = env.step(act)
            trace.append(env.positions_mm[:cfg.num_robots].copy())
            if term or trunc:
                break
        runs.append(np.array(trace))
    assert runs[0].shape == runs[1].shape
    assert np.allclose(runs[0], runs[1], atol=1e-4)


def test_residual_is_scaled_and_backends_agree():
    _, prior = configs()
    out = []
    for cls in (CompiledMCAPhysicalEnv, MCAPhysicalEnv):
        env = cls(prior)
        reset_with_valid_particles(env, SEED)
        rng = np.random.default_rng(3)
        trace = []
        for _ in range(10):
            env.step(rng.uniform(-1, 1, (prior.num_robots, 3)))
            trace.append(env.positions_mm[:prior.num_robots].copy())
        out.append(np.array(trace))
    assert np.allclose(out[0], out[1], atol=1e-6)
    env = CompiledMCAPhysicalEnv(prior)
    reset_with_valid_particles(env, SEED)
    a = np.ones((prior.num_robots, 3))
    assert np.allclose(env._apply_action_prior(a)-env._apply_action_prior(np.zeros_like(a)), .5*a)


def test_inactive_and_unassigned_robots_get_no_prior():
    _, prior = configs()
    env = CompiledMCAPhysicalEnv(prior)
    reset_with_valid_particles(env, SEED)
    env.active[0] = False
    assert not env._apply_action_prior(np.zeros((prior.num_robots, 3)))[0].any()
    env.masses[:] = 0.
    assert not env._apply_action_prior(np.zeros((prior.num_robots, 3))).any()


def test_avoid_prior_matches_the_observation_only_probe():
    import importlib.util
    base, _ = configs()
    cfg = replace(base, action_prior='own_route_bearing_avoid', action_residual_scale=.5, action_avoid_gain=6.)
    spec = importlib.util.spec_from_file_location('probe', 'research/validation/EXP0041_ROUTE_PRIOR_PROBE_20261001/probe_avoid.py')
    from environments.mca_physical_env import observed_particle_repulsion
    runs = []
    for env_cfg, manual in ((cfg, False), (base, True)):
        env = CompiledMCAPhysicalEnv(env_cfg)
        obs, _ = reset_with_valid_particles(env, SEED)
        trace = []
        for _ in range(60):
            if manual:
                nodes = obs['nodes'].astype(np.float64)
                act = direct_local_action(nodes[:, 112:115]+6.*observed_particle_repulsion(nodes, base), env)
                act[~env.active[:base.num_robots]] = 0
            else:
                act = np.zeros((base.num_robots, 3))
            obs, _, term, trunc, _ = env.step(act)
            trace.append(env.positions_mm[:base.num_robots].copy())
            if term or trunc:
                break
        runs.append(np.array(trace))
    assert runs[0].shape == runs[1].shape and np.allclose(runs[0], runs[1], atol=1e-4)


def test_priors_require_the_observed_features():
    import pytest
    base, _ = configs()
    with pytest.raises(ValueError, match='own_v9'):
        CompiledMCAPhysicalEnv(replace(base, target_observation='routed_assigned', action_prior='own_route_bearing'))


def test_gated_prior_zero_policy_is_the_avoid_prior_and_policy_can_stop():
    base, _ = configs()
    avoid = replace(base, action_prior='own_route_bearing_avoid', action_residual_scale=1., action_avoid_gain=6.)
    gated = replace(avoid, action_prior='own_route_bearing_avoid_gated', action_stop_deadzone=.35)
    a, b = CompiledMCAPhysicalEnv(avoid), CompiledMCAPhysicalEnv(gated)
    reset_with_valid_particles(a, SEED); reset_with_valid_particles(b, SEED)
    z = np.zeros((base.num_robots, 3))
    prior = a._apply_action_prior(z)
    assert np.allclose(b._apply_action_prior(z)[np.linalg.norm(prior, axis=1) >= .35], prior[np.linalg.norm(prior, axis=1) >= .35])
    # Cancelling the prior executes an exact stop, which the unit actuator keeps at zero.
    from environments.mca_physical_env import bound_command
    stop = b._apply_action_prior(-prior)
    assert not stop.any() and not bound_command(stop, 'unit').any()
    before = b.positions_mm[:base.num_robots].copy()
    b.step(-prior)
    assert np.linalg.norm(b.positions_mm[:base.num_robots]-before, axis=1).max() < base.robot_speed_mm_s*base.control_dt_s*.5


def test_wait_prior_drops_the_prior_only_for_forecast_conflicts():
    base, _ = configs()
    avoid = replace(base, action_prior='own_route_bearing_avoid', action_residual_scale=1., action_avoid_gain=6.)
    wait = replace(avoid, action_prior='own_route_bearing_avoid_wait', action_wait_clearance=.3, action_wait_horizon_s=.5)
    a, b = CompiledMCAPhysicalEnv(avoid), CompiledMCAPhysicalEnv(wait)
    reset_with_valid_particles(a, SEED); reset_with_valid_particles(b, SEED)
    z = np.zeros((base.num_robots, 3))
    nodes = b._observation()['nodes'].astype(np.float64)
    conflict = np.zeros(base.num_robots, bool)
    for k in range(4):
        s = 36+10*k
        conflict |= (nodes[:, s+9] > 0) & (nodes[:, s+8] < .3) & (nodes[:, s+7]*base.particle_prediction_horizon_s < .5)
    pa, pb = a._apply_action_prior(z), b._apply_action_prior(z)
    assert np.allclose(pb[~conflict], pa[~conflict]) and not pb[conflict].any()
