from dataclasses import replace

import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.image_sensing import ImageSensor
from marl.teacher import TEACHER_CONFIG
from scripts.multicluster_protocol import paired_environment


def _env(n=1):
    cfg = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy='mca_m1_lvo', junction_model='union')
    return paired_environment(cfg, n, 2600000000)[0]


def test_detection_localises_cluster_from_pixels():
    env = _env()
    s = ImageSensor(env, seed=0)
    p = env.positions_mm.astype(np.float64)
    dets = s._detections(p[0], p, p)
    big = [q for q, a, f in dets if s._is_cluster(a, f)]
    assert len(big) == 1, len(dets)
    assert np.linalg.norm(big[0]-p[0]) < .02          # one pixel
    env.close()


def test_track_survives_motion_without_drift():
    env = _env()
    s = ImageSensor(env, seed=0, latency_steps=0)
    for _ in range(40):
        est = s.observe()
        env.step(np.zeros((1, 3)))          # cluster drifts with the flow
    assert s.lost[0] == 0
    assert np.linalg.norm(est.pos[0]-env.positions_mm[0]) < .05
    env.close()


def test_frame_timestamp_tracks_delayed_acquisition_and_held_frame():
    env = _env(); sensor = ImageSensor(env, seed=0, latency_steps=2)
    try:
        dt = env.config.control_dt_s
        stamps = []
        for _ in range(5):
            stamps.append(sensor.observe().frame_time_s)
            env.step(np.zeros((1, 3)))
        assert np.allclose(stamps, [0., 0., 0., dt, 2*dt])
        sensor.cam = replace(sensor.cam, fps=0.)
        assert sensor.observe().frame_time_s == stamps[-1]
    finally:
        env.close()
