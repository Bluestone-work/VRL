"""Candidate shared tracker: remove known input history before estimating drift.

This is an observation-processing ablation, not a learning contribution or a
certified observer. It uses actuator commands and delayed detected centroids;
the independently calibrated command-to-speed relation remains an assumption.
It is NOT installed in frozen EXP0051 runs.
"""
from bisect import bisect_right
from collections import deque
import numpy as np

from marl.tracked_sensors import MeasurementProcessor, TrackedSensorAdapter, SimulatedLocalImager


class CommandHistory:
    """Exact integral of logged, piecewise-constant requested commands."""
    def __init__(self, n):
        self.n = n
        self.times, self.commands, self.prefix = [], [], []

    def record(self, timestamp, command):
        command = np.asarray(command, float)
        timestamp = float(timestamp)
        if command.shape != (self.n, 3) or not np.isfinite(command).all() or not np.isfinite(timestamp):
            raise ValueError('Finite timestamp and [n,3] commands required')
        if self.times and timestamp < self.times[-1]:
            raise ValueError('Cannot retroactively insert commands')
        if self.times and timestamp == self.times[-1]:
            self.commands[-1] = command.copy()
            return
        cumulative = self.integral_at(timestamp)
        self.times.append(timestamp); self.commands.append(command.copy()); self.prefix.append(cumulative)

    def at(self, timestamp):
        i = bisect_right(self.times, float(timestamp))-1
        return np.zeros((self.n,3)) if i < 0 else self.commands[i].copy()

    def integral_at(self, timestamp):
        i = bisect_right(self.times, float(timestamp))-1
        if i < 0:
            return np.zeros((self.n,3))
        return self.prefix[i]+(float(timestamp)-self.times[i])*self.commands[i]

    def integral(self, start, end):
        if end < start:
            raise ValueError('Negative interval')
        return self.integral_at(end)-self.integral_at(start)


class CommandAlignedProcessor(MeasurementProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = CommandHistory(self.n)
        self.drift = {}
        self.samples = {}

    def record_command(self, timestamp, commands):
        self.commands.record(timestamp, commands)
        self.last_commands = np.asarray(commands, float).copy()

    def ingest(self, frame):
        if frame.time_s <= self.last_frame_time:
            return
        super().ingest(frame)
        for i, position in enumerate(frame.robot_centroids):
            if not np.isfinite(position).all():
                continue
            previous = self.samples.get(i)
            if previous is not None:
                stamp, old = previous
                elapsed = frame.time_s-stamp
                applied = self.speed*self.commands.integral(stamp, frame.time_s)[i]
                residual = (position-old-applied)/elapsed
                old_drift = self.drift.get(i)
                self.drift[i] = residual if old_drift is None else (
                    self.spec.velocity_alpha*residual+(1-self.spec.velocity_alpha)*old_drift)
            self.samples[i] = (frame.time_s, position.copy())
            velocity = self.drift.get(i, np.zeros(3))+self.speed*self.commands.at(frame.time_s)[i]
            self.tracks[('robot', i)] = (frame.time_s, position.copy(), velocity)

    def observe(self, now):
        # The original timestamps are restored, and expired tracks stay expired.
        # No true velocity, flow, position, route or body-edge access occurs.
        saved = {}
        current_commands = self.commands.at(now)
        for i in range(self.n):
            key = ('robot', i)
            if key not in self.tracks:
                continue
            stamp, position, _ = self.tracks[key]
            age = now-stamp
            if age < 0 or age > self.spec.track_lifetime_s+1e-9:
                continue
            saved[key] = self.tracks[key]
            drift = self.drift.get(i, np.zeros(3))
            predicted = position+age*drift+self.speed*self.commands.integral(stamp,now)[i]
            velocity = drift+self.speed*current_commands[i]
            self.tracks[key] = (now,predicted,velocity)
        try:
            return super().observe(now)
        finally:
            self.tracks.update(saved)


class CommandAlignedSensorAdapter(TrackedSensorAdapter):
    def reset(self, seed):
        self.imager = SimulatedLocalImager(self.env, self.spec, seed)
        self.processor = CommandAlignedProcessor(self.env.num_robots,self.imager.preoperative_targets,
            self.spec,speed=self.env.config.robot_speed_mm_s,
            body_radius=self.env.config.robot_radius_mm,duration=self.env.config.episode_duration_s)
        self.queue = deque()

    def execute(self, commands):
        requested = super().execute(commands)
        self.processor.record_command(float(self.env.elapsed_s),requested)
        return requested
