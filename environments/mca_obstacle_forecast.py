"""Causal frozen-flow trajectory observations, not a learned world model.

Only current particle positions, current flow/lumen and known vessel geometry
are used. Future lysis and future robot actions are unknown. Robot-relative
risk therefore assumes the robot keeps its currently observed velocity.
"""
import numpy as np

HORIZONS_S = (.5, 1., 2.)
FORECAST_SLOTS = 4
FEATURES_PER_SLOT = 24


def forecast_particles(env, particle_ids):
    ids = np.asarray(particle_ids, dtype=np.int64)
    positions = env.positions_mm[ids].copy()
    edges = env.edges[ids].copy()
    active = env.active[ids].copy()
    body = env.body_radius[ids].copy()
    predictions, valid = [], []
    exits = np.full(len(ids), np.inf)
    previous = 0.
    for horizon in HORIZONS_S:
        result = env._advance_particle_prediction(positions, edges, body, active, horizon-previous)
        newly_exited = np.isfinite(result.exit_time_s)
        exits[newly_exited] = previous + result.exit_time_s[newly_exited]
        positions, edges, active = result.positions_mm, result.edge, result.active
        predictions.append(positions.copy());valid.append(active.copy())
        previous = horizon
    return np.asarray(predictions), np.asarray(valid), exits


def trajectory_features(robot_positions, robot_velocities, frame, particle_positions,
                        particle_velocities, predictions, valid, exit_times,
                        robot_radius, particle_radius, speed_scale, safety_margin):
    """Four risk-ranked trajectories: current state, 3 future points, swept risk.

    Per-slot layout: position3, velocity3; 3*(future relative position3,
    clearance1, alive1); minimum swept clearance1, time of minimum1, present1.
    Exited particles contribute only until their predicted exit time.
    """
    n, count = len(robot_positions), len(particle_positions)
    features = np.zeros((n, FORECAST_SLOTS*FEATURES_PER_SLOT), np.float32)
    if not count:
        return features
    current = particle_positions[None] - robot_positions[:, None]
    rel_velocity = particle_velocities[None] - robot_velocities[:, None]
    radius = robot_radius + particle_radius
    closest = np.linalg.norm(current, axis=-1) - radius
    closest_time = np.zeros((n, count))
    previous_positions = particle_positions.copy()
    previous_time = 0.
    for k, horizon in enumerate(HORIZONS_S):
        end_time = np.minimum(horizon, exit_times)
        duration = np.maximum(end_time-previous_time, 0.)
        start_relative = previous_positions[None] - (robot_positions[:, None] + robot_velocities[:, None]*previous_time)
        end_relative = predictions[k][None] - (robot_positions[:, None] + robot_velocities[:, None]*end_time[None, :, None])
        delta = end_relative-start_relative
        fraction = np.clip(-np.sum(start_relative*delta, axis=-1)/np.maximum(np.sum(delta*delta, axis=-1), 1e-15), 0., 1.)
        gap = np.linalg.norm(start_relative+fraction[:, :, None]*delta, axis=-1)-radius
        improve = (duration[None] > 0) & (gap < closest)
        closest_time = np.where(improve, previous_time+fraction*duration[None], closest_time)
        closest = np.where(improve, gap, closest)
        previous_positions = predictions[k]
        previous_time = horizon
    selected = np.argsort(closest, axis=1, kind='stable')[:, :FORECAST_SLOTS]
    for i in range(n):
        for slot, j in enumerate(selected[i]):
            offset = slot*FEATURES_PER_SLOT
            features[i, offset:offset+3] = frame[i] @ current[i, j] / 1.5
            features[i, offset+3:offset+6] = frame[i] @ rel_velocity[i, j] / speed_scale
            for k, horizon in enumerate(HORIZONS_S):
                start = offset+6+5*k
                if valid[k, j]:
                    relative = predictions[k, j] - robot_positions[i] - robot_velocities[i]*horizon
                    features[i, start:start+3] = frame[i] @ relative / 1.5
                    features[i, start+3] = (np.linalg.norm(relative)-radius)/safety_margin
                    features[i, start+4] = 1.
            features[i, offset+21] = closest[i, j]/safety_margin
            features[i, offset+22] = closest_time[i, j]/HORIZONS_S[-1]
            features[i, offset+23] = 1.
    return features
