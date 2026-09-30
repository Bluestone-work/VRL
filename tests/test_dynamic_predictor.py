import numpy as np

from marl.dynamic_predictor import predict_obstacle_risk


def test_predictor_is_causal_and_finite():
    position = np.asarray([[0.02, 0.0, 0.0]], dtype=np.float32)
    velocity = np.asarray([[-0.04, 0.0, 0.0]], dtype=np.float32)
    out = predict_obstacle_risk(position, velocity, np.asarray([0.01]), 0.004)
    assert np.isfinite(out["predicted_relative_position"]).all()
    assert np.isfinite(out["collision_probability"]).all()
    assert 0.0 <= float(out["collision_probability"][0]) <= 1.0
    assert float(out["predicted_relative_position"][0, 0]) < float(position[0, 0])


def test_predictor_does_not_turn_stationary_obstacle_into_collision():
    out = predict_obstacle_risk(
        np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.asarray([0.19], dtype=np.float32),
        0.004,
    )
    assert float(out["ttc"][0]) > 0.45
    assert float(out["collision_probability"][0]) < 0.5
