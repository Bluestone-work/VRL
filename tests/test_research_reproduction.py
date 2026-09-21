"""Protocol checks: mismatched environments and censored metrics must not hide failures."""
import pytest
from scripts.research_evaluate import environment_settings, validate_checkpoint, summarize, METRICS


def config():
    return dict(robots=5, clots=3, horizon=300, robot_radius=.0011, obs_mode="geometric",
                reward_mode="milestone", contact_mode="geodesic", coverage_bonus=.2, step_cost=0.,
                approach_scale=.1, reward_double_count="on", no_control_margin=False,
                architecture="gat", control_mode="flow_guided", residual_scale=.2,
                guidance_speed=.65, critic_value_mode="v")


def test_configuration_prevents_contact_mode_drift():
    c = config()
    assert environment_settings(c)["contact_mode"] == "geodesic"
    meta = {**c, "n_agents": 5}
    validate_checkpoint(meta, c)
    meta["contact_mode"] = "euclidean"
    with pytest.raises(ValueError, match="contact_mode"):
        validate_checkpoint(meta, c)


def test_failed_completion_is_censored_not_zero_time():
    first = {key: 0.0 for key in METRICS}
    first.update(scenario="a", success=1., completion_steps=100., first_contact_steps=3.)
    failure = {key: 0.0 for key in METRICS}
    failure.update(scenario="b", success=0., completion_steps=None, first_contact_steps=None)
    result = summarize([first, failure])
    assert result["macro"]["success"] == .5
    assert result["macro"]["completion_steps"] == 100.
    assert result["macro"]["first_contact_steps"] == 3.
    assert result["per_territory"]["b"]["completion_steps"] is None
