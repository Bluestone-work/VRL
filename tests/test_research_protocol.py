"""Protocol regressions: semantic identity, environment drift, and split leakage."""
import copy

import numpy as np
import pytest

from environments.vessel_geometry import build_vessel_tree, VesselTree
from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from scripts.research_protocol import geometry_hash, initial_identity, validate_protocol, overlaps
from scripts.research_evaluate import environment_settings
from tests.test_research_reproduction import config


def test_geometry_hash_distinguishes_connectivity_and_flow_with_same_points():
    original = build_vessel_tree("bifurcation", np.random.default_rng(123))
    changed_flow = copy.deepcopy(original)
    changed_flow.branches[-1].flow_fraction *= .5
    changed_graph = VesselTree(original.points.copy(), original.radii.copy(), copy.deepcopy(original.branches),
                               extra_links=[(0, original.n_stations - 1)])
    assert np.array_equal(original.points, changed_graph.points)
    assert np.array_equal(original.radii, changed_flow.radii)
    assert len({geometry_hash(t) for t in (original, changed_flow, changed_graph)}) == 3
    assert geometry_hash(original) == geometry_hash(copy.deepcopy(original))


def test_reset_identity_reproducible_and_seed_sensitive():
    env = Vascular3DMARLEnv(scenario="bifurcation", randomize_scenario=False)
    try:
        first = initial_identity(env, 51)
        assert first == initial_identity(env, 51)
        assert first["initial_state_sha256"] != initial_identity(env, 52)["initial_state_sha256"]
    finally:
        env.close()


@pytest.mark.parametrize("field,value", [("robot_radius", .0045), ("reward_mode", "baseline"), ("flow_speed", .008)])
def test_protocol_rejects_silent_environment_drift(field, value):
    cfg = config()
    cfg.update(hidden_dim=128, num_layers=2, num_heads=4, dropout=0.)
    meta = {**cfg, "n_agents": 5, "obs_dim": 36, "state_dim": 24, "action_dim": 3}
    env = Vascular3DMARLEnv(**environment_settings(cfg))
    try:
        defaults = {"flow_speed": .004}
        validate_protocol(meta, cfg, env, defaults)
        setattr(env, field, value)
        with pytest.raises(ValueError):
            validate_protocol(meta, cfg, env, defaults)
    finally:
        env.close()


def test_split_leakage_detected_even_when_labels_differ():
    result = overlaps({"train": ["geometry-A"], "validation": ["geometry-B"], "test": ["geometry-A"]})
    assert result["train__test"] == ["geometry-A"]
    assert result["train__validation"] == []
