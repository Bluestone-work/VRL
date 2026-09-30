"""Restore local point reaction; moving obstacle count and avoidance feedback."""
from dataclasses import replace
import numpy as np
import pytest
from environments.mca_physical_env import MCAPhysicalEnv,DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_physical import reset_with_valid_particles
from tests.test_mca_physical_dynamics import pipe
from tests.test_mca_compiled import pair,compare

CONFIG='configs/experiments/EXP_0027_MCA_POINT_DYNAMICS.json'


def small_config(**kwargs):
    data=dict(num_robots=1,particle_count=0,inlet_flow_ml_min=1e-10,
              contact_model='localized_point',contact_distance_mm=.12,episode_duration_s=5.)
    data.update(kwargs)
    return replace(DynamicsConfig(),**data)


@pytest.mark.parametrize('backend',[MCAPhysicalEnv,CompiledMCAPhysicalEnv])
@pytest.mark.parametrize('position,contact',[
    ([10.,0.,0.],True),([10.,.11,0.],True),([10.,.13,0.],False),
    ([10.,.25,0.],False),([3.5,0.,0.],False)])
def test_only_the_local_point_region_can_dissolve(backend,position,contact):
    env=backend(small_config(),tree=pipe(),clot_stations=[10])
    env.reset(seed=1,options={'robot_positions_mm':np.array([position])})
    *_,info=env.step(np.zeros((1,3)))
    assert bool(info['removed_mass']>0)==contact
    if not contact:assert env.masses[0]==1.


def test_full_point_clearance_and_contact_reward_match_reference():
    ref,fast=pair(MCAPhysicalEnv(small_config(progress_reward_scale=.3),tree=pipe(),clot_stations=[10]),
                  options={'robot_positions_mm':np.array([[10.,0.,0.]])})
    for _ in range(100):
        if compare(ref,fast,np.zeros((1,3))):break
    assert ref._info()['success'] and fast._info()['success']
    assert ref.elapsed_s>=1/ref.config.lysis_mass_per_s


def test_32_particles_are_seeded_moving_and_cached_resets_match_reference():
    cfg=DynamicsConfig.from_json(CONFIG)
    assert cfg.particle_count==32
    ref,fast=MCAPhysicalEnv(cfg),CompiledMCAPhysicalEnv(cfg)
    previous=None
    for seed in (42,43):
        ro,ri=reset_with_valid_particles(ref,seed);fo,fi=reset_with_valid_particles(fast,seed)
        assert ri==fi
        np.testing.assert_array_equal(ref.positions_mm,fast.positions_mm)
        starts=fast.positions_mm[5:].copy()
        assert starts.shape==(32,3) and fast.active[5:].sum()==32
        assert len(np.unique(starts,axis=0))==32
        if previous is not None:assert not np.array_equal(previous,starts)
        for _ in range(20):compare(ref,fast,np.zeros((5,3)))
        assert np.max(np.linalg.norm(fast.positions_mm[5:]-starts,axis=1))>1e-4
        previous=starts


def test_particle_overlap_changes_avoidance_reward_in_both_backends():
    cfg=small_config(particle_count=1,particle_contact_penalty_per_s=2.)
    ref,fast=pair(MCAPhysicalEnv(cfg,tree=pipe(),clot_stations=[10]),
                  options={'robot_positions_mm':np.array([[5.,0.,0.]]),
                           'particle_positions_mm':np.array([[5.04,0.,0.]])})
    plain=CompiledMCAPhysicalEnv(replace(cfg,particle_contact_penalty_per_s=0.),tree=pipe(),clot_stations=[10])
    plain.reset(seed=42,options={'robot_positions_mm':ref.positions_mm[:1].copy(),
                               'particle_positions_mm':ref.positions_mm[1:].copy()})
    compare(ref,fast,np.zeros((1,3)))
    *_,pi=plain.step(np.zeros((1,3)))
    # Cost has no hidden impulse or auto-avoidance controller.
    np.testing.assert_array_equal(fast.positions_mm,plain.positions_mm)
    assert pi['particle_contact_s'][0]==pytest.approx(cfg.control_dt_s)
    *_,fi=fast.step(np.zeros((1,3)))
    assert fi['particle_penalty'][0]==pytest.approx(2*cfg.control_dt_s)
    assert pi['agent_rewards'][0]-fi['agent_rewards'][0]==pytest.approx(2*cfg.control_dt_s)


def test_point_observation_and_potential_aim_at_centre_not_wall():
    cfg=small_config(progress_reward_scale=.3)
    env=CompiledMCAPhysicalEnv(cfg,tree=pipe(),clot_stations=[10])
    env.reset(seed=1,options={'robot_positions_mm':np.array([[10.,.25,0.]])})
    assert env.observation_schema=='mca_point_36_v3'
    outside=env._reward_potential()[0]
    env.reset(seed=1,options={'robot_positions_mm':np.array([[10.,0.,0.]])})
    assert env._reward_potential()[0]==0. and outside<0.
