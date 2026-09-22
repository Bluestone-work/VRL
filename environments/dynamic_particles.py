"""Blood-cell-inspired dynamic intravascular obstacles.

First-version model, deliberately scoped:

  * Particles are advected by the SAME local flow field the robots
    experience (``tree.flow`` with the occluded-radius override), plus a
    small random drift representing the off-axis jitter of a deformable
    cell tumbling in shear. They are NOT a red-blood-cell model: no
    haemorheology, no deformability, no near-wall lift. Calling them
    "blood-cell-inspired" is the honest description.
  * Size is set relative to the robot radius (dimensionless ratio), so the
    obstacle scale follows the device scale rather than absolute metres.
  * Robots receive a separation impulse out of overlap (same trick as the
    robot-robot collision resolution); per-step collision counts, clearance
    and relative robot-obstacle velocity are recorded for the caller to put
    in `info`.

The class is used from both the single and vector envs; the single env is
the n_envs=1 case. Off by default: ``dynamic_intravascular_particles=False``
reproduces legacy behaviour exactly -- no RNG draws, no force terms, no info
keys -- because the particle RNG is a dedicated generator and every code
path is behind the flag.
"""
from __future__ import annotations

import numpy as np


DEFAULT_RADIUS_RATIO = 1.6       # obstacle radius = ratio * robot radius
DEFAULT_LATERAL_DRIFT = 0.15     # lateral jitter, fraction of flow speed
DEFAULT_COUNT = 24               # particles per env


class DynamicIntravascularParticles:
    """State + dynamics for one batch of obstacle particles.

    Parameters
    ----------
    n_envs, robot_radius: batch size and device scale.
    count: particles per env.
    radius_ratio: obstacle radius in units of robot radius (dimensionless).
    lateral_drift: per-step random drift as a fraction of the local flow
        speed, applied perpendicular to the local tangent.
    seed: RNG for initial placement and drift; a DEDICATED generator so
        enabling particles does not perturb the env's own RNG stream.
    """

    def __init__(
        self,
        n_envs: int,
        robot_radius: float,
        count: int = DEFAULT_COUNT,
        radius_ratio: float = DEFAULT_RADIUS_RATIO,
        lateral_drift: float = DEFAULT_LATERAL_DRIFT,
        seed: int | None = None,
    ) -> None:
        self.n_envs = int(n_envs)
        self.robot_radius = float(robot_radius)
        self.count = int(count)
        self.radius_ratio = float(radius_ratio)
        self.lateral_drift = float(lateral_drift)
        self.radius = self.radius_ratio * self.robot_radius
        # Combined contact distance: robot radius + obstacle radius.
        self.contact_distance = self.radius + self.robot_radius
        self._rng = np.random.default_rng(seed)

        self.positions = np.zeros((self.n_envs, self.count, 3), np.float32)
        # Last-step velocity, filled by step(); zero before the first step.
        self.velocities = np.zeros((self.n_envs, self.count, 3), np.float32)

    # ------------------------------------------------------------------ state

    def reset(self, tree) -> None:
        """Scatter particles along the tree, inside the lumen.

        Placement is uniform over stations (station count is proportional to
        branch length, so every branch gets particles in proportion to its
        length), with a radial offset drawn inside the local lumen so a
        particle never starts embedded in the wall.
        """
        n = tree.n_stations
        if n == 0 or self.count == 0:
            return
        stations = self._rng.integers(0, n, size=(self.n_envs, self.count))
        anchor = tree.points[stations]                                  # [E, P, 3]
        radii = tree.radii[stations]
        # Radial offset within (lumen - obstacle radius) so it starts legal.
        room = np.maximum(radii - self.radius, 0.0)
        direction = self._rng.normal(size=(self.n_envs, self.count, 3)).astype(np.float32)
        direction /= np.maximum(np.linalg.norm(direction, axis=2, keepdims=True), 1e-8)
        magnitude = (room * np.sqrt(
            self._rng.uniform(size=(self.n_envs, self.count))
        ).astype(np.float32))[:, :, None]
        self.positions = (anchor + direction * magnitude).astype(np.float32)
        self.velocities[:] = 0.0

    # --------------------------------------------------------------- dynamics

    def step(self, tree, occluded_radius: np.ndarray) -> np.ndarray:
        """Advect one step; updates positions and velocities in place.

        Advection uses the env's flow field evaluated at each particle's own
        position, with the same occluded radius the robots see, so obstacle
        motion and robot motion share one flow model. On top of the axial
        advection there is a small random DRIFT isotropic in the lateral
        plane -- the "inspired" part, standing in for the off-axis tumbling
        a real cell would show in shear. The result is projected back into
        the lumen exactly like a robot, so a particle cannot tunnel through
        a wall.
        """
        if self.count == 0:
            return self.positions
        # Flow at each particle position, projected to its nearest station.
        flat = self.positions.reshape(-1, 3)
        st = tree.nearest_station(flat)
        # `occluded_radius` may arrive per-station (the env's [n_stations]
        # occlusion array) or per-particle; index it by the station.
        occluded = np.asarray(occluded_radius)
        if occluded.shape == (tree.n_stations,):
            occluded = occluded[st]
        elif occluded.shape != flat.shape[:1]:
            raise ValueError(
                f"occluded_radius must be [n_stations] or [n_particles], got {occluded.shape}"
            )
        flow = tree.flow(
            flat, st, self.flow_speed, self.reference_radius,
            radius_override=occluded,
        ).reshape(self.n_envs, self.count, 3)

        drift_scale = self.lateral_drift * np.linalg.norm(
            flow, axis=2, keepdims=True
        )
        drift = self._rng.normal(size=flow.shape).astype(np.float32) * drift_scale

        proposed = self.positions + flow + drift
        clamped, _out, st2, _ax = tree.project(
            proposed.reshape(-1, 3), self.radius, hint=st,
        )
        clamped = clamped.reshape(self.n_envs, self.count, 3)
        self.velocities = (clamped - self.positions).astype(np.float32)
        self.positions = clamped.astype(np.float32)
        return self.positions

    def configure_flow(self, flow_speed: float, reference_radius: float) -> None:
        """Capture the env's flow constants (same values the robots use)."""
        self.flow_speed = float(flow_speed)
        self.reference_radius = float(reference_radius)

    # ------------------------------------------------------------- interaction

    def nearest_stats(
        self, robot_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(overlap count, clearance, nearest-particle index) per robot.

        * overlap: [n_envs, n_robots] int -- particles the robot overlaps.
        * clearance: [n_envs, n_robots] float -- distance to the nearest
          particle minus the contact distance; negative == overlap.
        * nearest: [n_envs, n_robots] int -- index of that particle.
        """
        dist = self._distances(robot_positions)
        overlap = (dist < self.contact_distance).sum(axis=2).astype(np.int32)
        nearest = np.argmin(dist, axis=2)
        nearest_dist = np.take_along_axis(dist, nearest[:, :, None], axis=2)[:, :, 0]
        return overlap, nearest_dist - self.contact_distance, nearest

    def relative_speed(
        self, robot_positions: np.ndarray, robot_velocities: np.ndarray
    ) -> np.ndarray:
        """|robot velocity - nearest particle velocity|, [n_envs, n_robots]."""
        _, _, nearest = self.nearest_stats(robot_positions)
        particle_vel = np.take_along_axis(
            self.velocities, nearest[:, :, None], axis=1
        )                                            # [E, R, 3]
        return np.linalg.norm(robot_velocities - particle_vel, axis=2).astype(
            np.float32
        )

    def separation_impulse(self, robot_positions: np.ndarray) -> np.ndarray:
        """Push robots out of particle overlaps; [n_envs, n_robots, 3].

        Same convention as the robot-robot term: a robot wedged between two
        particles is pushed by both (sum over particles).
        """
        delta = robot_positions[:, :, None, :] - self.positions[:, None, :, :]
        dist = np.linalg.norm(delta, axis=3)
        overlap = dist < self.contact_distance
        depth = np.where(overlap, self.contact_distance - dist, 0.0)
        direction = delta / np.maximum(dist, 1e-8)[..., None]
        impulse = (direction * depth[..., None]).sum(axis=2)
        return impulse.astype(np.float32)

    def _distances(self, robot_positions: np.ndarray) -> np.ndarray:
        """[n_envs, n_robots, n_particles] robot-particle distances."""
        return np.linalg.norm(
            robot_positions[:, :, None, :] - self.positions[:, None, :, :],
            axis=3,
        )
