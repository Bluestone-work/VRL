"""Analytical and geometry-based checks for EXP_0022 physical calibration."""
from types import SimpleNamespace

import numpy as np
import pytest

from environments.mca_physiology import (
    PhysicalUnits, PressureDrivenTreeFlow, mean_speed_mm_s,
    optimistic_wall_upstream_margin,
)
from environments.vessel_anatomy import build_territory


def pipe_tree(edges=((0, 1, 1.0),), radii=(1.0, 1.0)):
    graph = [[] for _ in radii]
    for a, b, length in edges:
        graph[a].append((b, length))
        graph[b].append((a, length))
    return SimpleNamespace(inlet_station=0, n_stations=len(radii),
                           station_graph=graph, radii=np.asarray(radii))


def test_volume_flow_conversion_uses_area_not_doppler_peak():
    assert mean_speed_mm_s(146, 3) == pytest.approx(344.246247, rel=1e-7)
    assert mean_speed_mm_s(0, 3) == 0
    assert mean_speed_mm_s(146, 6) == pytest.approx(mean_speed_mm_s(146, 3) / 4)


@pytest.mark.parametrize('bad', [None, -1, np.nan, np.inf, 'invalid'])
def test_invalid_flow_rejected(bad):
    with pytest.raises(ValueError):
        mean_speed_mm_s(bad, 3)


@pytest.mark.parametrize('name', ['mm_per_unit', 'control_dt_s', 'robot_speed_mm_s'])
@pytest.mark.parametrize('bad', [None, 0, -1, np.nan, np.inf])
def test_units_require_explicit_positive_finite_parameters(name, bad):
    kwargs = dict(mm_per_unit=70, control_dt_s=.05, robot_speed_mm_s=1,
                  provenance='literature sensitivity, not measured')
    kwargs[name] = bad
    with pytest.raises(ValueError):
        PhysicalUnits(**kwargs)


def test_units_roundtrip_and_substeps_do_not_cap_velocity():
    units = PhysicalUnits(70, .05, 1, 'test')
    speeds = np.array([0, .25, 1, 700.])
    np.testing.assert_allclose(units.speed_from_step_displacement(
        units.displacement_per_step(speeds)), speeds)
    assert units.robot_action_scale == pytest.approx(.05 / 70)
    assert units.required_substeps(700, .3) == 1167
    assert units.required_substeps(0, .3) == 1


def test_uniform_narrowing_obeys_fixed_pressure_r_to_fourth():
    model = PressureDrivenTreeFlow(pipe_tree(), 1, inlet_flow_ml_min=60)
    healthy = model.solve()
    narrow = model.solve(np.full(2, .5))
    assert healthy['inlet_flow_ml_min'] == pytest.approx(60)
    assert narrow['inlet_flow_ml_min'] == pytest.approx(60 / 16)
    np.testing.assert_allclose(narrow['mean_speed_mm_s'],
                               healthy['mean_speed_mm_s'] / 4)
    assert narrow['maximum_conservation_residual_mm3_s'] < 1e-10


def test_bifurcation_split_blockage_and_conservation():
    tree = pipe_tree(((0, 1, 1.), (1, 2, 1.), (1, 3, 2.)), (1.,) * 4)
    model = PressureDrivenTreeFlow(tree, 1, 60)
    normal = model.solve()['station_inflow_mm3_s']
    assert normal[2] / normal[3] == pytest.approx(2)
    blocked = model.solve(np.array([1., 1., 0., 1.]))
    flows = blocked['station_inflow_mm3_s']
    assert flows[2] == 0
    assert flows[0] == pytest.approx(flows[3])
    assert 0 < blocked['inlet_flow_ml_min'] < 60
    assert blocked['maximum_conservation_residual_mm3_s'] < 1e-10


def test_total_occlusion_has_zero_flow_without_nan():
    model = PressureDrivenTreeFlow(pipe_tree(), 1, 60)
    for radii in (np.array([0., 1.]), np.zeros(2)):
        result = model.solve(radii)
        assert result['inlet_flow_ml_min'] == 0
        assert np.isfinite(result['mean_speed_mm_s']).all()
        assert np.count_nonzero(result['station_inflow_mm3_s']) == 0


def test_distal_load_preserves_healthy_flow_but_changes_obstructed_flow():
    tree = pipe_tree(((0, 1, 1.), (1, 2, 1.), (1, 3, 2.)), (1.,) * 4)
    bare = PressureDrivenTreeFlow(tree, 1, 60)
    loaded = PressureDrivenTreeFlow(tree, 1, 60, distal_resistance_ratio=9)
    np.testing.assert_allclose(loaded.solve()['station_inflow_mm3_s'],
                               bare.solve()['station_inflow_mm3_s'])
    radii = np.array([1., .5, 1., 1.])
    assert bare.solve(radii)['inlet_flow_ml_min'] < loaded.solve(radii)['inlet_flow_ml_min'] < 60


@pytest.mark.parametrize('tree', [
    pipe_tree(((0, 1, 1.),), (1., 1., 1.)),
    pipe_tree(((0, 1, 1.), (1, 2, 1.), (2, 0, 1.)), (1.,) * 3),
    pipe_tree(((0, 1, 0.),), (1., 1.)),
])
def test_invalid_or_degenerate_graphs_rejected(tree):
    with pytest.raises(ValueError):
        PressureDrivenTreeFlow(tree, 1)


@pytest.mark.parametrize('radii', [[1.], [1., -1.], [np.nan, 1.], [1., np.inf]])
def test_bad_occluded_radii_rejected(radii):
    model = PressureDrivenTreeFlow(pipe_tree(), 1)
    with pytest.raises(ValueError):
        model.solve(radii)


def test_optimistic_finite_robot_margin_and_no_fit():
    speed = mean_speed_mm_s(146, 3)
    margin = optimistic_wall_upstream_margin(speed, 1.5, .08, 1.)
    assert margin < -70
    assert optimistic_wall_upstream_margin(0, 1.5, .08, 1.) == 1
    with pytest.raises(ValueError):
        optimistic_wall_upstream_margin(speed, .08, .08, 1.)


@pytest.fixture
def mca():
    return build_territory('mca_m1_lvo', rng=np.random.default_rng(42),
                           variation=0, min_radius=0)


def test_real_geometry_inlet_conservation_and_no_artificial_cap(mca):
    model = PressureDrivenTreeFlow(mca, mca.mm_per_unit)
    result = model.solve()
    assert result['inlet_flow_ml_min'] == pytest.approx(146)
    assert result['maximum_conservation_residual_mm3_s'] < 1e-8
    root = model.root
    v = model.velocity_mm_s(mca.points[[root]], np.array([root]), result)
    assert np.linalg.norm(v[0]) == pytest.approx(
        2 * result['mean_speed_mm_s'][root], rel=1e-5)
    assert np.linalg.norm(v[0]) > 600


def test_velocity_requires_same_radius_as_flow_solution(mca):
    model = PressureDrivenTreeFlow(mca, mca.mm_per_unit)
    result = model.solve()
    with pytest.raises(ValueError, match='match'):
        model.velocity_mm_s(mca.points[[0]], np.array([0]), result,
                            radius_mm=result['radius_mm'] * .5)


def test_reported_radius_does_not_alias_internal_geometry(mca):
    model = PressureDrivenTreeFlow(mca, mca.mm_per_unit)
    expected = model.healthy_radius_mm.copy()
    model.solve()['radius_mm'][:] = 0
    np.testing.assert_array_equal(model.healthy_radius_mm, expected)


def test_physical_length_scale_survives_fit_and_radius_floor():
    from environments.vessel_tree_generator import VesselSegment, segments_to_vessel_tree
    seg = VesselSegment(start=np.array([0., 0., 0.]), end=np.array([18., 0., 0.]),
                        radius_prox=1.2, radius_dist=.8, parent_idx=None, generation=0)
    tree = segments_to_vessel_tree([seg], min_radius=0)
    scale = tree.source_units_per_unit
    assert np.linalg.norm(tree.points[-1] - tree.points[0]) * scale == pytest.approx(18, rel=1e-6)
    assert tree.radii[0] * scale == pytest.approx(1.2, rel=1e-6)
    floored = segments_to_vessel_tree([seg], min_radius=.1)
    assert floored.source_units_per_unit == scale


def test_sampled_anatomy_is_not_rescaled_to_fixed_nominal_diameter():
    trees = [build_territory('mca_m1_lvo', rng=np.random.default_rng(seed),
                             variation=1, min_radius=0) for seed in (42, 43, 44)]
    diameters = [2 * t.radii[t.inlet_station] * t.physical_mm_per_unit for t in trees]
    assert np.ptp(diameters) > .1
    for t in trees:
        assert 2 * t.radii.max() * t.mm_per_unit == pytest.approx(3)


def test_audit_reports_all_cases_and_serializes_without_success_claim():
    import json
    from scripts.validate_mca_physiology import DEFAULT_CONFIG, run_audit, report_markdown
    config = json.loads(DEFAULT_CONFIG.read_text())
    config['geometry_validation'].update(seeds=[42], variation_levels=[0.])
    config['reference_inlet']['sensitivity_values_ml_min'] = [146.]
    config['occlusion_sensitivity'].update(radius_fractions=[0., .35, 1.],
                                          distal_resistance_ratios=[0., 9.])
    summary, rows, margins, numerics = run_audit(config)
    assert summary['engineering_checks_passed']
    assert summary['flow_cases'] == len(rows) == 24
    assert len(margins) == 24 * 3 * 3
    assert len(numerics) == 24 * 3
    assert summary['training_ready'] is False
    assert summary['success_rate'] is None
    assert summary['nominal_optimistic_upstream_margin_mm_s'] < 0
    assert all(type(v) is bool for v in summary['engineering_checks'].values())
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert '未训练新策略' in report_markdown(summary, config, rows)


@pytest.mark.parametrize('key,value', [('flow_speed_cap', 10), ('radius_floor', .0045)])
def test_audit_rejects_unregistered_easier_physics(key, value):
    import json
    from scripts.validate_mca_physiology import DEFAULT_CONFIG, run_audit
    config = json.loads(DEFAULT_CONFIG.read_text())
    block = config['numerics'] if key == 'flow_speed_cap' else config['geometry_validation']
    block[key] = value
    with pytest.raises(ValueError, match='disallows'):
        run_audit(config)
