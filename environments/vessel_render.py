"""Vessel and microrobot rendering for PyBullet.

Replaces two things that made the old view unreadable:

1. The vessel was a chain of separate spheres, one every third station, at
   alpha 0.22. Against the light default background those barely registered,
   and where they did they read as a string of beads rather than a lumen. Here
   each pair of consecutive stations is joined by a capsule whose radius is the
   local lumen radius, so the wall is continuous and the taper, the branching
   and any stenosis are all visible as vessel shape.

2. Each microrobot was a cluster of 27 tiny beads (3 helices x 9, bead radius
   0.0016 in normalised units -- about one pixel at 640x480). At that size the
   swarm was invisible and, when it wasn't, it looked like scattered dots. Here
   one robot is one body, sized to be legible, coloured by what it is doing.

Capsules are the right primitive for a centerline: `GEOM_CAPSULE` in PyBullet
is a cylinder with hemispherical caps, so consecutive segments join without a
visible seam or gap at bends -- which a plain cylinder chain would show at
every station.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _quat_from_z_to(direction: np.ndarray) -> list[float]:
    """Quaternion rotating +Z onto `direction`.

    PyBullet capsules and cylinders are defined along their local +Z, so every
    segment needs the rotation taking +Z to the centerline tangent.
    """
    d = np.asarray(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    d = d / n
    z = np.array([0.0, 0.0, 1.0])

    dot = float(np.dot(z, d))
    if dot > 1.0 - 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -1.0 + 1e-9:
        # Antiparallel: rotate pi about any axis orthogonal to Z.
        return [1.0, 0.0, 0.0, 0.0]

    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(angle / 2.0)
    return [float(axis[0] * s), float(axis[1] * s), float(axis[2] * s),
            float(np.cos(angle / 2.0))]


@dataclass
class VesselStyle:
    """Appearance knobs.

    `wall_alpha` is the one that matters most: the lumen has to be see-through
    enough to follow robots inside it, but opaque enough to read as tissue.
    0.22 (the old value) is too faint; ~0.35 keeps both.
    """

    wall_alpha: float = 0.35
    wall_color: tuple[float, float, float] = (0.80, 0.30, 0.30)
    segment_stride: int = 1          # capsule per Nth station pair; 1 = every
    robot_radius: float = 0.010
    robot_color: tuple[float, float, float, float] = (0.10, 0.45, 1.00, 1.0)
    robot_contact_color: tuple[float, float, float, float] = (0.10, 0.95, 0.45, 1.0)
    clot_color: tuple[float, float, float] = (0.45, 0.32, 0.10)
    clot_alpha_min: float = 0.55


@dataclass
class VesselRenderer:
    """Owns the PyBullet bodies for one vessel tree plus its robots and clots."""

    p: object
    cid: int
    style: VesselStyle = field(default_factory=VesselStyle)

    vessel_bodies: list[int] = field(default_factory=list)
    robot_bodies: list[int] = field(default_factory=list)
    clot_bodies: list[int] = field(default_factory=list)
    _clot_radii: list[float] = field(default_factory=list)
    _contact_state: list[bool] = field(default_factory=list)

    # ---------------------------------------------------------------- vessel

    def build_vessel(self, points: np.ndarray, radii: np.ndarray,
                     branch_ids: np.ndarray | None = None) -> None:
        """(Re)build the lumen as a chain of capsules along the centerline."""
        p, cid = self.p, self.cid

        for b in self.vessel_bodies:
            try:
                p.removeBody(b, physicsClientId=cid)
            except Exception:
                pass
        self.vessel_bodies = []

        pts = np.asarray(points, dtype=np.float64)
        rad = np.asarray(radii, dtype=np.float64)
        stride = max(1, int(self.style.segment_stride))
        r, g, b_ = self.style.wall_color
        rgba = [r, g, b_, self.style.wall_alpha]

        for i in range(0, len(pts) - stride, stride):
            a, c = pts[i], pts[i + stride]
            seg = c - a
            length = float(np.linalg.norm(seg))
            if length < 1e-9:
                continue

            # Mean of endpoint radii, so Murray-law taper and stenoses show up
            # as an actual change in vessel calibre.
            seg_r = float(0.5 * (rad[i] + rad[i + stride]))

            # A capsule's `length` is the cylindrical part only; the caps add
            # seg_r at each end. Subtracting keeps total extent correct rather
            # than bulging past every junction.
            h = max(1e-4, length - 2.0 * seg_r)

            vis = p.createVisualShape(
                p.GEOM_CAPSULE, radius=seg_r, length=h,
                rgbaColor=rgba, physicsClientId=cid,
            )
            body = p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis,
                basePosition=(0.5 * (a + c)).tolist(),
                baseOrientation=_quat_from_z_to(seg),
                physicsClientId=cid,
            )
            self.vessel_bodies.append(body)

    # ---------------------------------------------------------------- robots

    def _ensure_robots(self, n: int) -> None:
        p, cid = self.p, self.cid
        while len(self.robot_bodies) < n:
            vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=self.style.robot_radius,
                rgbaColor=list(self.style.robot_color), physicsClientId=cid,
            )
            self.robot_bodies.append(p.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis,
                basePosition=[0, 0, 0], physicsClientId=cid,
            ))
            self._contact_state.append(False)

    def update_robots(self, positions: np.ndarray,
                      contacting: np.ndarray | None = None) -> None:
        """Move robot bodies; recolour the ones currently lysing a clot.

        Colour is the cheapest way to make the behaviour the reward actually
        pays for -- contact with a clot -- visible in a single still frame.
        `changeVisualShape` is only called on transitions, since it is far more
        expensive than a pose reset and would otherwise run every step.
        """
        p, cid = self.p, self.cid
        pos = np.asarray(positions, dtype=np.float64)
        self._ensure_robots(len(pos))

        for i, xyz in enumerate(pos):
            body = self.robot_bodies[i]
            p.resetBasePositionAndOrientation(
                body, xyz.tolist(), [0, 0, 0, 1], physicsClientId=cid,
            )
            if contacting is None:
                continue
            want = bool(contacting[i])
            if want != self._contact_state[i]:
                color = (self.style.robot_contact_color if want
                         else self.style.robot_color)
                try:
                    p.changeVisualShape(body, -1, rgbaColor=list(color),
                                        physicsClientId=cid)
                    self._contact_state[i] = want
                except Exception:
                    pass

    # ----------------------------------------------------------------- clots

    def update_clots(self, positions: np.ndarray, mass_frac: np.ndarray,
                     base_radius: float) -> None:
        """Draw clots with radius scaled by remaining mass.

        Shrinking geometry (not just fading colour) is what makes lysis
        progress readable: a clot being dissolved visibly gets smaller, and a
        fully lysed one disappears.
        """
        p, cid = self.p, self.cid
        pos = np.asarray(positions, dtype=np.float64)
        frac = np.clip(np.asarray(mass_frac, dtype=np.float64), 0.0, 1.0)

        live = [i for i, f in enumerate(frac) if f > 1e-6]
        want = [base_radius * (0.40 + 0.60 * float(frac[i])) for i in live]

        # Radius is baked into the visual shape, so a changed radius needs a new
        # body. Rebuild only when the set or a radius moved materially.
        need_rebuild = (
            len(self.clot_bodies) != len(want)
            or any(abs(a - b) > 2e-4 for a, b in zip(self._clot_radii, want))
        )

        if need_rebuild:
            for b in self.clot_bodies:
                try:
                    p.removeBody(b, physicsClientId=cid)
                except Exception:
                    pass
            self.clot_bodies, self._clot_radii = [], []
            cr, cg, cb = self.style.clot_color
            for idx, rr in zip(live, want):
                a = self.style.clot_alpha_min
                alpha = a + (1.0 - a) * float(frac[idx])
                vis = p.createVisualShape(
                    p.GEOM_SPHERE, radius=max(1e-4, rr),
                    rgbaColor=[cr, cg, cb, alpha], physicsClientId=cid,
                )
                self.clot_bodies.append(p.createMultiBody(
                    baseMass=0, baseVisualShapeIndex=vis,
                    basePosition=pos[idx].tolist(), physicsClientId=cid,
                ))
                self._clot_radii.append(rr)
        else:
            for body, idx in zip(self.clot_bodies, live):
                p.resetBasePositionAndOrientation(
                    body, pos[idx].tolist(), [0, 0, 0, 1], physicsClientId=cid,
                )

    # ----------------------------------------------------------------- camera

    @staticmethod
    def frame_camera(p, points: np.ndarray, width: int, height: int,
                     yaw: float = 50.0, pitch: float = -22.0):
        """View/projection that frames the whole tree.

        The old camera hardcoded target [0.5,0.5,0.5] and distance 1.1. Those
        sit near the synthetic tree's centre by luck; any scenario with a
        different extent (or a real geometry) would fall partly outside the
        frustum. Fitting to the actual bounding sphere keeps every scenario in
        frame with no per-scenario tuning.
        """
        pts = np.asarray(points, dtype=np.float64)
        centre = pts.mean(axis=0)
        extent = float(np.linalg.norm(pts - centre, axis=1).max())

        fov = 55.0
        dist = extent / np.tan(np.radians(fov / 2.0)) * 1.30

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=centre.tolist(), distance=dist,
            yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=fov, aspect=width / height,
            nearVal=max(0.01, dist - extent * 2.5),
            farVal=dist + extent * 2.5,
        )
        return view, proj
