"""Reset diversity and physical consistency across randomized flow/layouts."""
from dataclasses import replace
import numpy as np
import pytest
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from environments.mca_compiled import CompiledMCAPhysicalEnv
from tests.test_mca_compiled import compare


def config(**kwargs):
    return replace(DynamicsConfig.from_json('configs/experiments/EXP_0027_MCA_POINT_DYNAMICS.json'),
                   particle_initialization='mixed_branch_density', **kwargs)


def test_random_resets_keep_geometry_and_cover_different_distributions():
    env=CompiledMCAPhysicalEnv(config(inlet_flow_multiplier_min=1.5,inlet_flow_multiplier_max=2.5))
    layouts=set();robots=set();particles=set();counts=set()
    env.reset(seed=123);geometry=env.tree.points.copy();first=env.initialization_record()
    for _ in range(100):
        env.reset()  # No repeated seed: the episode RNG must keep advancing.
        r=env.initialization_record();layouts.add(r['particle_layout'])
        robots.add(env.positions_mm[:5].tobytes());particles.add(env.positions_mm[5:].tobytes())
        counts.add(tuple(r['particle_branch_counts']))
        np.testing.assert_array_equal(env.tree.points,geometry)
        _,radius,distance,_=env.transport.coordinates(env.positions_mm,env.edges,env.solution)
        assert np.all(distance+env.body_radius<=radius+1e-7)
        assert 1.5<=r['flow_multiplier']<=2.5
    assert len(layouts)==3 and len(robots)==len(particles)==100 and len(counts)>90
    env.reset(seed=123)
    assert env.initialization_record()==first


def test_cached_flow_changes_match_reference_after_stepping():
    cfg=config(inlet_flow_multiplier_min=1.5,inlet_flow_multiplier_max=2.5)
    ref,fast=MCAPhysicalEnv(cfg),CompiledMCAPhysicalEnv(cfg)
    for seed in (10,11,12,10):
        ref.reset(seed=seed);fast.reset(seed=seed)
        assert ref.initialization_record()==fast.initialization_record()
        assert ref.np_random.bit_generator.state==fast.np_random.bit_generator.state
        for _ in range(8):compare(ref,fast,np.zeros((5,3)))


@pytest.mark.parametrize('multiplier',[1.5,2.5])
def test_faster_pressure_scales_both_robot_drift_and_particles(multiplier):
    base=CompiledMCAPhysicalEnv(config());base.reset(seed=44)
    fast=CompiledMCAPhysicalEnv(config(inlet_flow_multiplier_min=multiplier,inlet_flow_multiplier_max=multiplier))
    for _ in range(2):
        fast.reset(seed=44)
        np.testing.assert_array_equal(fast.positions_mm,base.positions_mm)
        v0=base.transport.velocity_mm_s(base.positions_mm,base.edges,base.solution)
        v1=fast.transport.velocity_mm_s(fast.positions_mm,fast.edges,fast.solution)
        np.testing.assert_allclose(v1,v0*multiplier,rtol=1e-12,atol=1e-12)
        fast.step(np.zeros((5,3)))


def test_invalid_randomization_config_rejected():
    with pytest.raises(ValueError,match='range'):
        config(inlet_flow_multiplier_min=3,inlet_flow_multiplier_max=2)
    with pytest.raises(ValueError,match='particle_initialization'):
        replace(DynamicsConfig(),particle_initialization='typo')
