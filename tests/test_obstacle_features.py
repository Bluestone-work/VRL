import numpy as np
from types import SimpleNamespace

from marl.obstacle_features import forecast_features


def test_obstacle_features_are_compact_and_observation_only():
    ep = SimpleNamespace(env=SimpleNamespace(config=SimpleNamespace(robot_radius_mm=.08, particle_radius_mm=.02)))
    est = SimpleNamespace(particles=[[(np.array([1., 0., 0.]), np.array([-1., 0., 0.]))]])
    x = forecast_features(ep, est, 0)
    assert x.shape == (6,)
    assert x[-1] == 1.
    assert np.all(np.isfinite(x))
