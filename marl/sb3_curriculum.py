"""Causal obstacle beliefs and CNN/attention features for the SB3 experiment.

All tracks are associated from delayed image detections, never simulator IDs.
The first predictor is an explicit constant-velocity model with heuristic
uncertainty, not a learned or calibrated collision probability estimator.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

FORECAST_TIMES = np.array([.2, .5, 1.], dtype=np.float32)
OBSTACLE_SLOTS, OBSTACLE_DIM = 8, 25
DIFFICULTIES = (
    dict(name="easy", obstacles=4, flows=[.01, .025], latencies=[0, 1]),
    dict(name="medium", obstacles=8, flows=[.025, .05], latencies=[1, 2]),
    dict(name="hard", obstacles=16, flows=[.05, .1], latencies=[2, 3]),
    dict(name="stress", obstacles=32, flows=[.1, .15], latencies=[3, 4]),
)


class ObstacleBelief:
    """One-to-one gated tracks, short occlusion memory, multi-horizon forecast."""
    def __init__(self, robots, prediction=True):
        self.tracks = [[] for _ in range(robots)]
        self.last_frame = None
        self.prediction = prediction
        self.error_count = np.zeros(3, np.int64)
        self.squared_error = np.zeros(3, float)
        self.stationary_squared_error = np.zeros(3, float)

    def diagnostics(self):
        return {str(float(h)): dict(matched_detection_targets=int(n),
                forecast_rmse_mm=float(np.sqrt(self.squared_error[k] / n)) if n else None,
                stationary_rmse_mm=float(np.sqrt(self.stationary_squared_error[k] / n)) if n else None)
                for k, (h, n) in enumerate(zip(FORECAST_TIMES, self.error_count))}

    def observe(self, est, frames, now, contact_radius):
        timestamp = float(getattr(est, "frame_time_s", now))
        fresh = timestamp != self.last_frame
        data = np.zeros((len(self.tracks), OBSTACLE_SLOTS, OBSTACLE_DIM), np.float32)
        valid = np.zeros((len(self.tracks), OBSTACLE_SLOTS), np.float32)
        for i, old in enumerate(self.tracks):
            if fresh:
                detections = [np.asarray(r, float) + est.pos[i] for r, _ in est.particles[i]]
                old = [t for t in old if timestamp - t["t"] <= .5]
                matched = set()
                if old and detections:
                    dt = np.array([max(timestamp - t["t"], 1e-6) for t in old])
                    expected = np.array([t["p"] + t["v"] * d for t, d in zip(old, dt)])
                    cost = np.linalg.norm(expected[:, None] - np.array(detections)[None], axis=-1)
                    a, b = linear_sum_assignment(cost)
                    for j, k in zip(a, b):
                        if cost[j, k] > .25:
                            continue
                        t, q = old[j], detections[k]
                        measured = np.clip((q - t["p"]) / dt[j], -3., 3.)
                        remaining = []
                        for origin_t, origin_p, origin_v, k in t.get("pending", []):
                            elapsed = timestamp - origin_t
                            if elapsed + 1e-6 >= float(FORECAST_TIMES[k]):
                                if abs(elapsed - float(FORECAST_TIMES[k])) <= .051:
                                    self.error_count[k] += 1
                                    self.squared_error[k] += float(np.sum((q - origin_p - origin_v * elapsed) ** 2))
                                    self.stationary_squared_error[k] += float(np.sum((q - origin_p) ** 2))
                            else:
                                remaining.append((origin_t, origin_p, origin_v, k))
                        t.update(p=q, v=.6 * t["v"] + .4 * measured, t=timestamp,
                                 hits=t["hits"] + 1)
                        t["pending"] = remaining + [(timestamp, q.copy(), t["v"].copy(), k) for k in range(3)]
                        matched.add(k)
                old.extend(dict(p=q, v=np.zeros(3), t=timestamp, hits=1)
                           for k, q in enumerate(detections) if k not in matched)
                self.tracks[i] = old
            rows = []
            for track in old:
                age = max(now - track["t"], 0.)
                if age > 1. or not est.active[i]:
                    continue
                # Robot coasts under its measured velocity; candidate actions
                # remain the policy's decision, not an oracle trajectory.
                frame_age = max(now - timestamp, 0.)
                robot = est.pos[i] + est.vel[i] * frame_age
                rel = track["p"] + track["v"] * age - robot
                velocity = track["v"] - est.vel[i]
                F = frames[i]
                local, local_v = F @ rel, F @ velocity
                future = local[None] + FORECAST_TIMES[:, None] * local_v[None]
                clearance = np.linalg.norm(future, axis=-1) - contact_radius
                sigma = .03 + .12 * (age + FORECAST_TIMES) + .06 / np.sqrt(track["hits"])
                closing = -float(rel @ velocity) / max(float(np.linalg.norm(rel)), 1e-6)
                token = np.r_[local / 2., local_v / 2., future.ravel() / 2.,
                              clearance / 2., sigma / 2., age,
                              float(track["t"] == timestamp),
                              (np.linalg.norm(rel) - contact_radius) / 2., closing / 2.]
                if not self.prediction:
                    # Preserve current detections/velocity and masks; remove
                    # every explicit forecast, including the old ctx forecast.
                    token[6:21] = 0.
                rows.append((np.linalg.norm(rel), token))
            for j, (_, token) in enumerate(sorted(rows, key=lambda r: r[0])[:OBSTACLE_SLOTS]):
                data[i, j] = np.clip(token, -10., 10.)
                valid[i, j] = 1.
        self.last_frame = timestamp
        return data, valid


class PredictiveTransformerExtractor(BaseFeaturesExtractor):
    """Shared CNN/GRU, robot-obstacle cross attention, joint robot attention.

    Obstacle tokens share the robot's current route frame. Peer tokens use
    pairwise delayed positions transformed into the querying robot's frame.
    There are no arbitrary slot embeddings on obstacle or peer tokens.
    """
    def __init__(self, observation_space):
        n = observation_space["ctx"].shape[0]
        super().__init__(observation_space, features_dim=64 * n)
        self.n = n
        self.cnn = nn.Sequential(nn.Conv2d(4, 16, 5, 2, 2), nn.ReLU(),
                                 nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(96, 32), nn.Tanh())
        self.history = nn.GRU(11, 32, batch_first=True)
        self.robot = nn.Sequential(nn.Linear(96, 64), nn.LayerNorm(64), nn.GELU())
        self.obstacle = nn.Sequential(nn.Linear(OBSTACLE_DIM, 64), nn.GELU())
        self.peer = nn.Sequential(nn.Linear(8, 64), nn.GELU())
        self.cross = nn.MultiheadAttention(64, 4, dropout=0., batch_first=True)
        self.null = nn.Parameter(torch.zeros(1, 1, 64))
        self.norm = nn.LayerNorm(64)
        self.ff = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
        self.final_norm = nn.LayerNorm(64)

    def forward(self, obs):
        b, n = obs["ctx"].shape[:2]
        hist = obs["hist"].reshape(b * n, 24, 11)
        mask = obs["hmask"].reshape(b * n, 24).bool()
        h = self.history(hist.masked_fill(mask[..., None], 0.))[0][:, -1]
        image = self.cnn(obs["image"].reshape(b * n, 4, 32, 32))
        robot = self.robot(torch.cat([h, image, obs["ctx"].reshape(b * n, 32)], -1))
        obstacle = self.obstacle(obs["obstacles"].reshape(b * n, OBSTACLE_SLOTS, OBSTACLE_DIM))
        peer = self.peer(obs["peers"].reshape(b * n, n, 8))
        tokens = torch.cat([self.null.expand(b * n, -1, -1), obstacle, peer], 1)
        padding = torch.cat([torch.zeros((b * n, 1), device=robot.device, dtype=torch.bool),
                             ~obs["obstacle_valid"].reshape(b * n, OBSTACLE_SLOTS).bool(),
                             ~obs["peer_valid"].reshape(b * n, n).bool()], 1)
        z = self.cross(robot[:, None], tokens, tokens, key_padding_mask=padding,
                       need_weights=False)[0][:, 0]
        z = self.norm(robot + z)
        z = self.final_norm(z + self.ff(z))
        return z.reshape(b, -1)
