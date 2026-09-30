"""All task objects vary independently; collisions cannot masquerade as safe clearance."""
from dataclasses import replace
import numpy as np
import pytest
from environments.mca_physical_env import MCAPhysicalEnv,DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from tests.test_mca_physical_dynamics import pipe
from tests.test_mca_compiled import compare


def config(**kwargs):
    cfg=DynamicsConfig.from_json('configs/experiments/EXP_0028_MCA_RANDOMIZED_DYNAMICS.json')
    return replace(cfg,clot_initialization='random_branches',obstacle_observation='predictive_four',
                   particle_collision_event_penalty=6.,particle_contact_penalty_per_s=20.,
                   particle_near_penalty_per_s=1.,**kwargs)


def test_all_three_layouts_vary_in_cached_resets_and_same_seed_reproduces():
    env=CompiledMCAPhysicalEnv(config());env.reset(seed=800)
    first=env.initialization_record();geometry=env.tree.points.copy()
    layouts=[set(),set(),set()]
    for _ in range(100):
        obs,_=env.reset()
        for bucket,points in zip(layouts,(env.clot_positions_mm,env.positions_mm[:5],env.positions_mm[5:])):
            bucket.add(points.tobytes())
        assert len(set(env.tree.branch_ids[env.clot_stations]))==4
        gap=np.linalg.norm(env.positions_mm[:5,None]-env.positions_mm[None,5:],axis=-1)
        assert np.all(gap>env.config.robot_radius_mm+env.config.particle_radius_mm)
        np.testing.assert_array_equal(env.tree.points,geometry)
        assert obs['nodes'].shape==(5,76) and env.observation_space.contains(obs)
        assert np.isfinite(obs['nodes']).all()
    assert [len(bucket) for bucket in layouts]==[100,100,100]
    env.reset(seed=800);assert env.initialization_record()==first


def test_random_clots_rebuild_hydraulics_routes_and_reference_rng():
    ref,fast=MCAPhysicalEnv(config()),CompiledMCAPhysicalEnv(config())
    for seed in (80,81,82,80):
        ro,_=ref.reset(seed=seed);fo,_=fast.reset(seed=seed)
        assert ref.initialization_record()==fast.initialization_record()
        assert ref.np_random.bit_generator.state==fast.np_random.bit_generator.state
        np.testing.assert_array_equal(ref._occlusion_bump,fast._occlusion_bump)
        for a,b in zip(ref.routes,fast.routes):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(ref.solution['radius_mm'],fast.solution['radius_mm'])
        for _ in range(6):compare(ref,fast,np.zeros((5,3)))


def small(**kwargs):
    return replace(DynamicsConfig(),num_robots=1,particle_count=1,inlet_flow_ml_min=1e-10,
                   contact_model='localized_point',control_dt_s=.1,episode_duration_s=5.,
                   obstacle_observation='predictive_four',particle_collision_event_penalty=6.,
                   particle_contact_penalty_per_s=20.,particle_near_penalty_per_s=1.,**kwargs)


def test_collisions_charge_on_entry_not_every_substep_and_remove_success_bonus():
    ref=MCAPhysicalEnv(small(),tree=pipe(),clot_stations=[10])
    fast=CompiledMCAPhysicalEnv(small(),tree=pipe(),clot_stations=[10])
    options={'robot_positions_mm':np.array([[10.,0.,0.]]),'particle_positions_mm':np.array([[10.04,0.,0.]])}
    ref.reset(seed=42,options=options);fast.reset(seed=42,options=options)
    compare(ref,fast,np.zeros((1,3)))
    assert fast.episode_particle_collision_events==1
    for _ in range(40):
        if compare(ref,fast,np.zeros((1,3))):break
    assert fast._info()['success'] and not fast._info()['collision_free_success']
    assert fast.episode_particle_collision_events==1
    assert fast.episode_particle_contact_s>2.
    # The completion bonus is withheld even though all masses reached zero.
    solo=CompiledMCAPhysicalEnv(replace(small(),particle_count=0),tree=pipe(),clot_stations=[10])
    solo.reset(seed=42,options={'robot_positions_mm':np.array([[10.,0.,0.]])})
    while not solo._done:
        *_,info=solo.step(np.zeros((1,3)))
    assert info['team_reward']==30. and info['collision_free_success']


def test_contact_event_and_near_miss_cost_exist_before_lysis():
    def run(distance):
        env=CompiledMCAPhysicalEnv(small(),tree=pipe(),clot_stations=[10])
        env.reset(seed=1,options={'robot_positions_mm':np.array([[5.,0.,0.]]),
                                 'particle_positions_mm':np.array([[5.+distance,0.,0.]])})
        *_,first=env.step(np.zeros((1,3)));*_,second=env.step(np.zeros((1,3)))
        return first,second
    collision,next_step=run(.04);near,_=run(.18);far,_=run(.4)
    assert collision['particle_collision_events'][0]==1
    assert next_step['particle_collision_events'][0]==0
    assert collision['particle_penalty'][0]>8.
    assert 0<near['particle_penalty'][0]<1.
    assert far['particle_penalty'][0]==0.
    assert collision['removed_mass']==near['removed_mass']==far['removed_mass']==0.


def test_predictive_features_prioritize_an_approaching_particle(monkeypatch):
    env=MCAPhysicalEnv(replace(small(),particle_count=2),tree=pipe(),clot_stations=[10])
    obs,_=env.reset(seed=1,options={'robot_positions_mm':np.array([[5.,0.,0.]]),
                                  'particle_positions_mm':np.array([[5.4,0.,0.],[5.15,0.,0.]])})
    def velocity(positions,edges,solution):
        result=np.zeros_like(positions)
        result[:,0]=np.where(positions[:,0]>5.3,-.4,np.where(positions[:,0]>5.1,1.,0.))
        return result
    monkeypatch.setattr(env.transport,'velocity_mm_s',velocity)
    obs=env._observation()['nodes'][0]
    assert obs[36]==pytest.approx(.4/1.5)
    assert obs[43]==pytest.approx(1.)
    assert obs[44]<0 and obs[45]==1.
    assert obs[46]==pytest.approx(.15/1.5)
    assert np.all(obs[56:]==0.)  # Missing neighbors are masked, not duplicated.


def test_fixed_explicit_target_override_still_works():
    env=CompiledMCAPhysicalEnv(config(),clot_stations=[8,14,21,48])
    for seed in (4,5):
        env.reset(seed=seed);np.testing.assert_array_equal(env.clot_stations,[8,14,21,48])
