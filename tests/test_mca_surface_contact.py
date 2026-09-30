"""Physical contact regression: no centerline lysis, moving wall and parity."""
from dataclasses import replace
import numpy as np
import pytest
from environments.mca_physical_env import MCAPhysicalEnv,DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from tests.test_mca_physical_dynamics import pipe
from tests.test_mca_compiled import pair,compare


def config(**kwargs):
    return replace(DynamicsConfig(),num_robots=1,particle_count=0,
                   inlet_flow_ml_min=1e-10,episode_duration_s=8.,
                   contact_model='stenosis_surface',contact_distance_mm=.04,**kwargs)


@pytest.mark.parametrize('backend',[MCAPhysicalEnv,CompiledMCAPhysicalEnv])
def test_centerline_is_not_the_clot_surface(backend):
    env=backend(config(),tree=pipe(),clot_stations=[10])
    env.reset(seed=1,options={'robot_positions_mm':np.array([[10.,0.,0.]])})
    for _ in range(20):
        *_,info=env.step(np.zeros((1,3)))
        assert info['removed_mass']==0
    assert env.masses[0]==1.


def test_contact_clears_only_by_following_the_receding_surface_with_bounded_actions():
    ref,fast=pair(MCAPhysicalEnv(config(),tree=pipe(),clot_stations=[10]),
                 options={'robot_positions_mm':np.array([[10.,.25,0.]])})
    seconds=0.
    for _ in range(160):
        # Same bounded action in both exact simulators; no direct state changes.
        radius=ref.solution['radius_mm'][10]
        goal=np.array([[10.,radius-ref.config.robot_radius_mm-.02,0.]])
        action=(goal-ref.positions_mm)*8
        action/=np.maximum(1,np.linalg.norm(action,axis=1,keepdims=True))
        done=compare(ref,fast,action);seconds+=ref.config.control_dt_s
        if done:break
    assert ref._info()['success'] and fast._info()['success']
    assert seconds>=1/ref.config.lysis_mass_per_s
    assert ref.positions_mm[0,1]>.85


def test_stationary_robot_loses_contact_as_the_surface_recedes():
    env=CompiledMCAPhysicalEnv(config(),tree=pipe(),clot_stations=[10])
    env.reset(seed=1,options={'robot_positions_mm':np.array([[10.,.25,0.]])})
    for _ in range(30):env.step(np.zeros((1,3)))
    assert .9<env.masses[0]<1.
    assert env._surface_contacts(env.positions_mm,env.edges,env.solution,env.masses).sum()==0


def test_overlapping_stenoses_do_not_multiply_robot_lysis_capacity():
    env=CompiledMCAPhysicalEnv(config(),tree=pipe(),clot_stations=[10,11])
    env.reset(seed=1,options={'robot_positions_mm':np.array([[10.,.25,0.]])})
    *_,info=env.step(np.zeros((1,3)))
    assert info['removed_mass']<=env.config.lysis_mass_per_s*env.config.control_dt_s+1e-12
    assert info['contact_s'].sum()<=env.config.control_dt_s+1e-12
    assert env.masses[1]==1.


def test_historical_configuration_retains_point_contact_schema():
    old=MCAPhysicalEnv(DynamicsConfig.from_json())
    new=MCAPhysicalEnv(config())
    assert old.config.contact_model=='point_target'
    assert old.observation_schema=='mca_physical_36_v1'
    assert new.observation_schema=='mca_surface_36_v2'


def test_navigation_shaping_telescopes_without_altering_lysis_or_success():
    shaped=CompiledMCAPhysicalEnv(config(progress_reward_scale=.3,reward_discount=.999),
                                 tree=pipe(),clot_stations=[10])
    plain=CompiledMCAPhysicalEnv(config(),tree=pipe(),clot_stations=[10])
    options={'robot_positions_mm':np.array([[5.,0.,0.]])}
    shaped.reset(seed=1,options=options);plain.reset(seed=1,options=options)
    start=shaped._reward_potential();total=np.zeros(1)
    for step in range(20):
        action=np.array([[.5,0.,0.]])
        *_,si=shaped.step(action);*_,pi=plain.step(action)
        np.testing.assert_array_equal(shaped.positions_mm,plain.positions_mm)
        np.testing.assert_array_equal(shaped.masses,plain.masses)
        assert si['success']==pi['success']
        total+=shaped.config.reward_discount**step*si['shaping_rewards']
    np.testing.assert_allclose(total,shaped.config.reward_discount**20*shaped._reward_potential()-start,atol=1e-12)
