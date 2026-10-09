import numpy as np

from marl.paper_navigation import CBFConfig, MPPIConfig, _project_halfspaces
from scripts.audit_paper_method_privilege import audit_source
from scripts.evaluate_paper_navigation import build_configurations


def test_target_selection_accepts_observed_clot_status():
    from scripts.benchmark_multicluster import PlanTargets

    class Env:
        masses = np.array([1., 1.])
        clot_positions_mm = np.array([[0., 0., 0.], [1., 0., 0.]])

    targets = PlanTargets([[0, 1]])
    np.testing.assert_array_equal(targets.targets(Env(), np.zeros((1, 3)), np.array([False, True])), [1])


def test_cbf_projection_preserves_feasible_nominal():
    command, residual = _project_halfspaces(np.array([1., 0., 0.]),
                                             np.array([[1., 0., 0.]]),
                                             np.array([.5]))
    np.testing.assert_allclose(command, [1., 0., 0.])
    assert residual <= 0.


def test_cbf_projection_moves_command_into_measured_halfspace():
    command, residual = _project_halfspaces(np.array([1., 0., 0.]),
                                             np.array([[1., 0., 0.]]),
                                             np.array([.5]))
    assert command[0] >= .5
    assert residual <= 1e-8


def test_paper_configs_are_finite():
    assert CBFConfig().max_obstacles > 0
    assert MPPIConfig().samples >= 8


def test_paper_methods_do_not_read_simulator_truth_attributes():
    assert audit_source() == []


def test_two_method_evaluation_uses_independent_configurations():
    configurations = build_configurations(('cbf_qp', 'mppi'))
    assert 'alpha' in configurations['cbf_qp']
    assert 'horizon_steps' in configurations['mppi']
    assert 'alpha' not in configurations['mppi']
