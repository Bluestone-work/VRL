"""Helical microrobot cluster rendering (visual only).

Real magnetically actuated microrobots for endovascular work are not spheres.
The workhorse morphology is the artificial bacterial flagellum (ABF): a rigid
helical filament that converts a rotating magnetic field into forward thrust via
rotation-translation coupling. Clinically relevant doses are administered as a
*cluster* of many such helices, not a single body.

This module renders each agent as a small cluster of helices whose common axis is
aligned with the agent's instantaneous velocity, with a spin phase that advances
with distance travelled so the swarm visibly corkscrews along the vessel.

IMPORTANT: this is presentation only. The action space, dynamics, observation and
reward are untouched, so training throughput and all learning results are
unaffected -- PyBullet bodies are created only when rendering is enabled. Making
the actuation model genuinely helical (field-frequency actions, step-out
frequency, non-holonomic heading) is a separate, much larger change to the
dynamics; keeping the two apart means the visual upgrade cannot invalidate the
policy comparisons already collected.

Geometry of one helix, in its own frame with the axis along +z:

    x(t) = r_h * cos(2*pi*turns*t + phase)
    y(t) = r_h * sin(2*pi*turns*t + phase)
    z(t) = (t - 0.5) * length,          t in [0, 1]

The curve is drawn as a chain of small spheres, because PyBullet has no swept
generalized-cylinder primitive and a per-frame mesh rebuild would be far more
expensive than moving a fixed pool of bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HelixStyle:
    """Appearance of one helical cluster.

    Attributes:
        n_helices: Helices per agent (the "cluster" the clinician injects).
        beads_per_helix: Spheres used to draw one helix; the cost driver.
        turns: Number of full turns along the filament.
        helix_radius: Radius of the helical coil, in world units.
        length: Filament length along its own axis, in world units.
        bead_radius: Radius of one drawn sphere.
        cluster_spread: Lateral scatter of the helices within one cluster.
        color: RGBA for the filaments.
    """

    n_helices: int = 3
    beads_per_helix: int = 9
    turns: float = 2.5
    helix_radius: float = 0.0035
    length: float = 0.016
    bead_radius: float = 0.0016
    cluster_spread: float = 0.0055
    color: tuple[float, float, float, float] = (0.10, 0.45, 0.98, 1.0)

    @property
    def beads_per_agent(self) -> int:
        return self.n_helices * self.beads_per_helix


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return fallback
    return (v / n).astype(np.float32)


def axis_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed frame (u, v, w) with w along `axis`.

    Returns three unit vectors; u and v span the plane perpendicular to w. The
    seed is switched when the axis is near-parallel to z so the cross product
    never degenerates.
    """
    w = _unit(np.asarray(axis, dtype=np.float32), np.array([1.0, 0.0, 0.0], np.float32))
    seed = np.array([0.0, 0.0, 1.0], np.float32)
    if abs(float(np.dot(w, seed))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], np.float32)
    u = _unit(np.cross(seed, w), np.array([0.0, 1.0, 0.0], np.float32))
    v = np.cross(w, u).astype(np.float32)
    return u, v, w


def helix_points(
    center: np.ndarray,
    axis: np.ndarray,
    phase: float,
    style: HelixStyle,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Bead positions for a single helix.

    Args:
        center: Midpoint of the filament axis in world coordinates.
        axis: Direction the filament points (need not be normalized).
        phase: Spin phase in radians; advancing it rotates the coil.
        style: Geometry and appearance.
        offset: Optional lateral displacement within the cluster.

    Returns:
        `[beads_per_helix, 3]` world positions.
    """
    u, v, w = axis_frame(axis)
    t = np.linspace(0.0, 1.0, style.beads_per_helix, dtype=np.float32)
    ang = 2.0 * np.pi * style.turns * t + float(phase)
    radial = style.helix_radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
    along = ((t - 0.5) * style.length)[:, None] * w
    pts = np.asarray(center, dtype=np.float32)[None, :] + radial + along
    if offset is not None:
        pts = pts + np.asarray(offset, dtype=np.float32)[None, :]
    return pts.astype(np.float32)


def cluster_points(
    center: np.ndarray,
    axis: np.ndarray,
    phase: float,
    style: HelixStyle,
    spread_seed: int = 0,
) -> np.ndarray:
    """Bead positions for one agent's whole helical cluster.

    The helices are placed on a small circle perpendicular to the axis and given
    staggered phases, which reads as a swarm rather than one thick filament.

    Returns:
        `[beads_per_agent, 3]` world positions.
    """
    u, v, _w = axis_frame(axis)
    n = max(int(style.n_helices), 1)
    out = np.empty((n * style.beads_per_helix, 3), dtype=np.float32)
    # Deterministic in-cluster layout: a fixed rosette, rotated per agent so
    # neighbouring agents do not look like clones.
    base = 2.0 * np.pi * (spread_seed % max(n, 1)) / max(n, 1)
    for k in range(n):
        if n == 1:
            offset = np.zeros(3, np.float32)
        else:
            a = base + 2.0 * np.pi * k / n
            offset = style.cluster_spread * (np.cos(a) * u + np.sin(a) * v)
        # Stagger the spin so the cluster does not pulse in unison.
        ph = phase + 2.0 * np.pi * k / max(n, 1)
        lo = k * style.beads_per_helix
        out[lo : lo + style.beads_per_helix] = helix_points(
            center, axis, ph, style, offset=offset.astype(np.float32)
        )
    return out


class HelixClusterRenderer:
    """Fixed pool of PyBullet spheres that draws every agent as a helix cluster.

    Bodies are allocated once (`n_agents * beads_per_agent` spheres) and only
    repositioned afterwards, so the per-frame cost is a batch of
    `resetBasePositionAndOrientation` calls with no shape churn.
    """

    def __init__(self, pybullet_module, client_id: int, n_agents: int,
                 style: HelixStyle | None = None) -> None:
        self._p = pybullet_module
        self._cid = int(client_id)
        self.n_agents = int(n_agents)
        self.style = style or HelixStyle()
        self._bodies: list[int] = []
        # Spin phase per agent, advanced by distance travelled so the corkscrew
        # rate tracks speed instead of wall-clock frames.
        self._phase = np.zeros((self.n_agents,), dtype=np.float32)
        self._allocate()

    # ------------------------------------------------------------------ setup

    def _allocate(self) -> None:
        p, cid = self._p, self._cid
        need = self.n_agents * self.style.beads_per_agent
        # One visual shape reused by every body: PyBullet stores it once.
        vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.style.bead_radius,
            rgbaColor=list(self.style.color),
            physicsClientId=cid,
        )
        while len(self._bodies) < need:
            self._bodies.append(
                p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=vis,
                    basePosition=[0.0, 0.0, -5.0],
                    physicsClientId=cid,
                )
            )

    # ----------------------------------------------------------------- update

    def update(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        fallback_axes: np.ndarray | None = None,
        spin_gain: float = 900.0,
    ) -> None:
        """Move the beads to match the current agent states.

        Args:
            positions: `[n_agents, 3]` cluster centers.
            velocities: `[n_agents, 3]` per-step displacement; sets the axis and
                advances the spin phase.
            fallback_axes: `[n_agents, 3]` axis to use where the velocity is
                ~zero (pass the local vessel tangent so a stalled agent still
                points sensibly down the lumen).
            spin_gain: Radians of spin per world unit travelled. Purely
                cosmetic; only affects how fast the coil appears to rotate.
        """
        p, cid = self._p, self._cid
        pos = np.asarray(positions, dtype=np.float32).reshape(self.n_agents, 3)
        vel = np.asarray(velocities, dtype=np.float32).reshape(self.n_agents, 3)
        speed = np.linalg.norm(vel, axis=1)
        # Rotation-translation coupling: a real ABF advances one pitch per turn,
        # so tying phase to distance (not to frame count) is the honest cosmetic
        # analogue of that constraint.
        self._phase = (self._phase + spin_gain * speed).astype(np.float32)

        style = self.style
        per_agent = style.beads_per_agent
        for i in range(self.n_agents):
            axis = vel[i]
            if speed[i] < 1e-7:
                if fallback_axes is not None:
                    axis = np.asarray(fallback_axes, dtype=np.float32)[i]
                else:
                    axis = np.array([1.0, 0.0, 0.0], np.float32)
            beads = cluster_points(
                pos[i], axis, float(self._phase[i]), style, spread_seed=i
            )
            lo = i * per_agent
            for k in range(per_agent):
                p.resetBasePositionAndOrientation(
                    self._bodies[lo + k],
                    beads[k].tolist(),
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=cid,
                )

    def hide(self) -> None:
        """Park every bead out of frame (used when an agent pool shrinks)."""
        p, cid = self._p, self._cid
        for body in self._bodies:
            p.resetBasePositionAndOrientation(
                body, [0.0, 0.0, -5.0], [0.0, 0.0, 0.0, 1.0], physicsClientId=cid
            )
