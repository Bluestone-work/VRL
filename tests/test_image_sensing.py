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
    big = [q for q, a in dets if a >= s.cam.cluster_min_area_px]
    assert len(big) == 1
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
