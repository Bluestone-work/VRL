"""Compact deployable dynamic-obstacle forecast features.

Only particle detections attached to the current delayed image estimate are
used.  The feature vector is deliberately small: predicted clearance at four
times, the nearest relative speed, and a detection-valid flag.  Simulator
particle positions and collision labels are never consulted.
"""
import numpy as np


def forecast_features(ep, est, i, times=(0., .2, .5, 1.), history=None):
    rows = list(est.particles[i] if hasattr(est, 'particles') and i < len(est.particles) else [])
    # Short deployable memory: retain only recent detections and inflate their
    # uncertainty through the age-dependent radius. The caller supplies rows
    # as (relative position, relative velocity, age_s).
    aged=[]
    for row in (history or []):
        rel, vel, age = row
        aged.append((np.asarray(rel)+np.asarray(vel)*float(age), np.asarray(vel)))
    rows.extend(aged)
    radius = float(getattr(ep.env.config, 'robot_radius_mm', .08)) + float(getattr(ep.env.config, 'particle_radius_mm', .02))
    clear = np.full(len(times), 6., dtype=np.float32)
    rel_speed = 0.
    for rel, vel in rows:
        rel, vel = np.asarray(rel, float), np.asarray(vel, float)
        rel_speed = max(rel_speed, float(np.linalg.norm(vel)))
        d = np.linalg.norm(rel[None] + np.asarray(times)[:, None] * vel[None], axis=1) - radius
        clear = np.minimum(clear, d)
    # Normalize to a stable millimetre scale and retain whether the estimate
    # actually contained a particle; absence is different from far away.
    return np.r_[np.clip(clear / 2., -2., 3.), np.clip(rel_speed / 2., 0., 3.), float(bool(rows))].astype(np.float32)
