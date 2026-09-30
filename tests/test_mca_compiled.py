"""Reference parity: 0.001 mm positions (50x tighter than original .05 mm gate),
exact edges/exits/masks, 2e-5 mass/time and 1e-4 contact/agent-reward bounds.
Bit identity is not expected for compiled adaptive integration near events.
"""
import copy
from dataclasses import replace
import numpy as np
import pytest
from environments.mca_physical_env import MCAPhysicalEnv, DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_physical import reset_with_valid_particles
from tests.test_mca_physical_dynamics import pipe


def pair(env, seed=42, options=None):
    if options is None: reset_with_valid_particles(env,seed)
    else: env.reset(seed=seed,options=options)
    fast=copy.deepcopy(env);fast.__class__=CompiledMCAPhysicalEnv
    return env,fast


def compare(ref,fast,action):
    ro,rr,rt,rx,ri=ref.step(action);fo,fr,ft,fx,fi=fast.step(action)
    np.testing.assert_allclose(fast.positions_mm,ref.positions_mm,rtol=0,atol=1e-3)
    np.testing.assert_array_equal(fast.edges,ref.edges)
    np.testing.assert_array_equal(fast.active,ref.active)
    np.testing.assert_array_equal(fast.exit_node,ref.exit_node)
    np.testing.assert_allclose(fast.exit_time_s,ref.exit_time_s,atol=2e-5,rtol=0,equal_nan=True)
    np.testing.assert_allclose(fast.masses,ref.masses,atol=2e-5,rtol=0)
    np.testing.assert_allclose(fast.path_mm,ref.path_mm,atol=1e-3,rtol=0)
    assert (rt,rx)==(ft,fx)
    assert fr==pytest.approx(rr,abs=1e-4)
    for k in ro: np.testing.assert_allclose(fo[k],ro[k],atol=1e-2,rtol=1e-5)
    for k in ('removed_mass','contact_s','wall_contact_s','blocked_s','particle_contact_s','robot_pair_contact_s','agent_rewards'):
        np.testing.assert_allclose(fi[k],ri[k],atol=1e-4,rtol=0)
    return rt or rx


@pytest.mark.parametrize('seed',[42,43,44,520000000,530000000,540000000])
def test_nominal_full_episode_matches_reference(seed):
    ref,fast=pair(MCAPhysicalEnv(),seed)
    rng=np.random.default_rng(seed)
    for _ in range(20):
        if compare(ref,fast,rng.uniform(-1,1,(5,3))): break
    assert ref._done


@pytest.mark.parametrize('sign',[-1,0,1])
def test_pipe_and_wall_dynamics_match(sign):
    cfg=replace(DynamicsConfig.from_json(),num_robots=1,particle_count=0,
                inlet_flow_ml_min=1e-10,episode_duration_s=.5,robot_speed_mm_s=12.)
    ref,fast=pair(MCAPhysicalEnv(cfg,tree=pipe(),clot_stations=[10]),options={'robot_positions_mm':np.array([[10.,.2,0.]])})
    for _ in range(10):
        if compare(ref,fast,np.array([[sign,.2,0.]])): break


def test_sustained_lysis_and_time_limit_match():
    cfg=replace(DynamicsConfig.from_json(),num_robots=1,particle_count=0,
                inlet_flow_ml_min=1e-12,episode_duration_s=2.,initial_radius_fraction=.35)
    ref,fast=pair(MCAPhysicalEnv(cfg,tree=pipe(),clot_stations=[10]),options={'robot_positions_mm':np.array([[10.,0.,0.]])})
    for _ in range(40):
        if compare(ref,fast,np.zeros((1,3))): break
    assert ref.masses[0]==pytest.approx(.28,abs=1e-9)


def test_budget_error_is_transactional():
    cfg=replace(DynamicsConfig.from_json(),max_substeps_per_control=1)
    ref,fast=pair(MCAPhysicalEnv(cfg))
    before=fast.positions_mm.copy();mass=fast.masses.copy()
    with pytest.raises(RuntimeError,match='budget'): fast.step(np.zeros((5,3)))
    np.testing.assert_array_equal(fast.positions_mm,before)
    np.testing.assert_array_equal(fast.masses,mass)
    assert fast.elapsed_s==0


def test_cached_nominal_reset_preserves_episode_rng_and_observations():
    ref=MCAPhysicalEnv();fast=CompiledMCAPhysicalEnv()
    for seed in (42,43,44,520000000,530000000,540000000):
        ro,ri=reset_with_valid_particles(ref,seed)
        fo,fi=reset_with_valid_particles(fast,seed)
        assert ri==fi
        assert ref.np_random.bit_generator.state==fast.np_random.bit_generator.state
        np.testing.assert_array_equal(ref.positions_mm,fast.positions_mm)
        for key in ro: np.testing.assert_array_equal(ro[key],fo[key])


def test_trace_recording_does_not_change_dynamics():
    ref=CompiledMCAPhysicalEnv();reset_with_valid_particles(ref,42)
    traced=copy.deepcopy(ref);traced.trace_dt_s=.001
    compare(ref,traced,np.zeros((5,3)))
    np.testing.assert_array_equal(ref.positions_mm,traced.positions_mm)
    assert len(traced.last_trace['time_s'])>20
