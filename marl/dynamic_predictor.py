"""Causal short-horizon prediction for currently observed moving obstacles.

The predictor is deliberately model-light.  It extrapolates the *measured*
relative position and relative velocity for a short horizon and adds a drift
uncertainty term.  It never receives the simulator's next particle position,
future random drift, or episode outcome.  This makes the feature suitable for
both a policy observation and a closed-loop task allocator.
"""
from __future__ import annotations

import numpy as np


def predict_obstacle_risk(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    clearance: np.ndarray,
    contact_distance: float,
    horizon: float = 0.45,
    drift_sigma: float = 0.0,
) -> dict[str, np.ndarray]:
    """Predict one nearest obstacle from current relative measurements.

    ``relative_position`` is obstacle minus robot and ``relative_velocity`` is
    the corresponding derivative.  The returned probability is a conservative
    Gaussian-radius approximation, not a calibrated simulator oracle.
    """
    pos = np.asarray(relative_position, dtype=np.float32)
    vel = np.asarray(relative_velocity, dtype=np.float32)
    horizon = float(max(horizon, 1e-4))
    predicted = pos + horizon * vel
    predicted_distance = np.linalg.norm(predicted, axis=-1)
    predicted_clearance = predicted_distance - float(contact_distance)
    closing_speed = np.maximum(
        -np.sum(pos * vel, axis=-1) / np.maximum(np.linalg.norm(pos, axis=-1), 1e-8),
        0.0,
    )
    # Linearised time-to-contact, clipped to the prediction window.
    numerator = np.maximum(np.linalg.norm(pos, axis=-1) - float(contact_distance), 0.0)
    ttc = np.full_like(numerator, horizon * 2.0, dtype=np.float32)
    np.divide(numerator, closing_speed, out=ttc, where=closing_speed > 1e-8)
    ttc = np.minimum(ttc, horizon * 2.0)
    sigma = np.asarray(max(float(drift_sigma), 1e-6), dtype=np.float32)
    # A smooth probability-like risk score.  It is intentionally bounded and
    # monotonic in predicted clearance; no threshold is tuned on test episodes.
    scale = np.maximum(sigma, 0.25 * float(contact_distance))
    probability = 1.0 / (1.0 + np.exp(np.clip(predicted_clearance / scale, -30.0, 30.0)))
    probability = np.where(ttc <= horizon, np.maximum(probability, 0.5), probability)
    uncertainty = np.broadcast_to(sigma * np.sqrt(horizon), predicted_clearance.shape)
    return {
        "predicted_relative_position": predicted,
        "predicted_clearance": predicted_clearance.astype(np.float32),
        "closing_speed": closing_speed.astype(np.float32),
        "ttc": ttc.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
        "collision_probability": probability.astype(np.float32),
    }
