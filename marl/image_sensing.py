"""Image-based sensing: render camera images, detect bodies, and estimate state from detections only.

The deployable controller must work from what a camera sees. Following the two NMI references (Turbo:
camera + YOLO detections; Medany et al.: microscope images + segmentation/tracking), the simulator renders
two orthogonal camera views of the in-vitro phantom (biplane: top view sees x/y, side view sees x/z) and
the controller only receives what a detector extracts from those pixels.

Per control step and per view
  1. Tracking window. A `window_px` square crop centred on each cluster's PREDICTED position (from its
     own previous estimate + velocity), i.e. no ground truth is used to place the crop. A cluster outside
     its window is missed (track lost -> re-detected next frame in a doubled window).
  2. Rendering. Every cluster and particle whose projection falls in the crop is drawn as a Gaussian spot
     (sigma = physical radius blurred by the optics PSF), with intensity proportional to its projected
     area. The lumen is drawn as a faint background (contrast-filled phantom). Pixel noise: Gaussian read
     noise + shot noise. Motion blur over the exposure.
  3. Detection. Background subtraction, threshold at `k_sigma` x noise, connected components, intensity-
     weighted centroid and area. Blobs are classified cluster / particle by area. This stands in for the
     learned detectors of the references; a CNN can be swapped in with the same interface.
  4. 3-D estimate. The two views share the x axis; a detection in the top view is paired with the side
     view detection of nearest x (within tolerance) -> (x, y, z). Clusters are associated to tracks by
     nearest predicted position; particles near a cluster become its local obstacle list, with velocity
     from nearest-neighbour association between frames.
The resulting `Estimate` has the same fields as marl.deployable_sensing, so every deployable controller
runs unchanged; map matching, frames and the pre-operative map are re-used from DeployableSensor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from marl.deployable_sensing import DeployableSensor, Estimate


@dataclass(frozen=True)
class CameraConfig:
    pixel_mm: float = .02            # 20 um/px (microscope-class in-vitro imaging)
    window_px: int = 160             # 3.2 mm tracking window per cluster and view
    psf_px: float = 1.2              # optics blur (Gaussian sigma, px)
    read_noise: float = .03          # relative to a cluster's peak intensity
    shot_noise: float = .02
    background: float = .15          # contrast-filled lumen intensity
    exposure_steps: int = 3          # motion blur samples over the exposure
    k_sigma: float = 4.              # detection threshold in noise sigmas
    cluster_min_area_px: int = 80    # clusters ~200 px; overlapping particles reach ~30 px
    min_area_px: int = 4             # smaller blobs are pixel noise
    pair_tol_mm: float = .06         # x-agreement for pairing the two views


def scaled_camera(pixel_mm=.02, **kw):
    """Camera with a different pixel size; window and area thresholds keep the same physical meaning."""
    s = (.02/pixel_mm)**2
    return CameraConfig(pixel_mm=pixel_mm, window_px=int(round(3.2/pixel_mm)), cluster_min_area_px=max(int(80*s), 6),
                        min_area_px=max(int(round(4*s)), 1), **kw)


def random_camera(rng):
    """Imaging-level domain randomisation (cf. Turbo's multi-level randomisation): resolution, optics blur,
    contrast and noise are drawn per episode, so results do not hinge on one camera setting."""
    return scaled_camera(pixel_mm=float(rng.uniform(.015, .035)), psf_px=float(rng.uniform(.8, 2.)),
                         read_noise=float(rng.uniform(.02, .06)), shot_noise=float(rng.uniform(.01, .04)),
                         background=float(rng.uniform(.05, .3)), exposure_steps=int(rng.integers(1, 5)))


class ImageSensor(DeployableSensor):
    def __init__(self, env, cam=CameraConfig(), seed=0, latency_steps=1):
        from marl.deployable_sensing import DeployableConfig
        super().__init__(env, DeployableConfig(latency_steps=latency_steps), seed)
        self.cam = cam
        n = env.num_robots
        self.track = None                                   # [n, 3] last estimates
        self.track_vel = np.zeros((n, 3))
        self.lost = np.zeros(n, int)
        self.frames_buf = []
        self.err_log = []                                   # (true - est) for evaluation only, never fed back

    # ---------- rendering (simulator side) ----------
    def _render(self, centre_uv, axes, prev_pos, pos):
        """One window of one view. axes: indices of the two world axes seen by this view."""
        c = self.cam; W = c.window_px; px = c.pixel_mm
        img = np.zeros((W, W))
        origin = centre_uv - W*px/2
        env = self.env; n = env.num_robots
        radii = env.body_radius
        for s in range(c.exposure_steps):
            frac = (s+.5)/c.exposure_steps
            P = prev_pos+(pos-prev_pos)*frac
            uv = (P[:, axes]-origin)/px
            inside = (uv[:, 0] > -8) & (uv[:, 0] < W+8) & (uv[:, 1] > -8) & (uv[:, 1] < W+8) & env.active
            for k in np.flatnonzero(inside):
                sig = np.hypot(radii[k]/px/1.5, c.psf_px); h = int(np.ceil(4*sig))
                amp = 1. if k < n else (radii[k]/radii[0])**2*4.     # particles are faint
                u0, v0 = int(round(uv[k, 0])), int(round(uv[k, 1]))
                a0, a1, b0, b1 = max(u0-h, 0), min(u0+h+1, W), max(v0-h, 0), min(v0+h+1, W)
                if a0 >= a1 or b0 >= b1:
                    continue
                du = np.arange(a0, a1)-uv[k, 0]; dv = np.arange(b0, b1)-uv[k, 1]
                img[a0:a1, b0:b1] += amp/c.exposure_steps*np.exp(-du[:, None]**2/(2*sig**2))*np.exp(-dv[None, :]**2/(2*sig**2))
        img += c.background
        img += self.rng.normal(0., c.read_noise, img.shape)+self.rng.normal(0., 1., img.shape)*np.sqrt(np.maximum(img, 0))*c.shot_noise
        return img, origin

    # ---------- detection (controller side: pixels only) ----------
    def _detect(self, img, origin):
        c = self.cam
        fg = ndimage.gaussian_filter(img-c.background, 1.)
        # noise of the smoothed image: a unit Gaussian filter divides white noise by 2*sqrt(pi)
        mask = fg > c.k_sigma*c.read_noise/(2*np.sqrt(np.pi))
        lab, k = ndimage.label(mask)
        out = []
        for j, sl in enumerate(ndimage.find_objects(lab), 1):
            ii, jj = np.nonzero(lab[sl] == j); w = np.maximum(fg[sl][ii, jj], 1e-9)
            ii = ii+sl[0].start; jj = jj+sl[1].start
            u = float((ii*w).sum()/w.sum()); v = float((jj*w).sum()/w.sum())
            if len(ii) >= c.min_area_px:
                out.append((origin+np.array([u, v])*c.pixel_mm, len(ii)))
        return out

    def _detections(self, centre, prev_pos, pos):
        top, o1 = self._render(centre[[0, 1]], [0, 1], prev_pos, pos)
        side, o2 = self._render(centre[[0, 2]], [0, 2], prev_pos, pos)
        A = self._detect(top, o1); B = self._detect(side, o2)
        # Epipolar pairing on the shared x axis. Two bodies far apart in depth can overlap in one view
        # (merged blob), so a blob may pair with several blobs of the other view; the resulting ghost
        # hypotheses are rejected later by the gated track assignment.
        big = self.cam.cluster_min_area_px
        pts = []
        for xy, area in A:
            for xz, area2 in B:
                if (area >= big) != (area2 >= big):
                    continue
                tol = self.cam.pair_tol_mm*(2. if area >= big else 1.)
                if abs(xy[0]-xz[0]) < tol:
                    p = np.array([(xy[0]+xz[0])/2, xy[1], xz[1]])
                    if all(np.linalg.norm(p-q) > .02 for q, _ in pts):
                        pts.append((p, max(area, area2)))
        return pts

    def observe(self):
        env = self.env; n = env.num_robots
        truth = env.positions_mm.astype(np.float64).copy()
        if not hasattr(self, '_prev_truth'):
            self._prev_truth = truth.copy()
        if self.track is None:
            # acquisition: the operator marks each cluster once at the start (initial detection window)
            self.track = truth[:n]+self.rng.normal(0., .02, (n, 3))
        pred = self.track+self.track_vel*self.dt
        cl_est = pred.copy(); parts_world = [[] for _ in range(n)]
        live = [i for i in range(n) if env.active[i]]
        big = self.cam.cluster_min_area_px
        views = (((0, 1), [0, 1]), ((0, 2), [0, 2]))         # (world axes, render axes): top x/y, side x/z
        per_view = []
        for axes, rax in views:
            blobs, small = [], []
            for i in live:
                img, o = self._render(pred[i][list(axes)], rax, self._prev_truth, truth)
                for q, a in self._detect(img, o):
                    (blobs if a >= big else small).append(q)
            # tracking windows overlap: merge duplicate detections of the same blob
            uniq = []
            for q in blobs:
                if all(np.linalg.norm(q-u) > .05 for u in uniq):
                    uniq.append(q)
            per_view.append((axes, uniq, small))
        # Per-view 2-D association (Hungarian, gated by the predicted projection), then 3-D fusion. Doing
        # it per view avoids biplane ghosts: two clusters with the same x but different y are separate in
        # the top view even when they overlap in the side view, and vice versa.
        # A blob is NOT exclusive: two clusters overlapping in one projection form one merged blob, and both
        # tracks take it (their separation in the other view keeps them distinct), instead of one track
        # coasting blindly and swapping identity when the blob splits again.
        # 3-D hypotheses: every top/side blob pair agreeing on the shared x axis (merged blobs pair with
        # several partners, so ghosts are included). A ghost combines one cluster's y with another's z and
        # lies far from both tracks in 3-D, so the gate rejects it.
        # Joint assignment over all tracks (n <= 3, exhaustive): each track takes one hypothesis or none, and
        # a blob may be shared by two tracks only if their predicted projections overlap in that view
        # (a merged blob). This resolves the biplane ambiguity when two clusters share x and one other
        # coordinate: as they separate, each keeps the blob consistent with its own prediction.
        import itertools
        (_, A, _), (_, B, _) = per_view
        tol = 2*self.cam.pair_tol_mm
        H = [(ia, ib, np.array([(a[0]+b[0])/2, a[1], b[1]])) for ia, a in enumerate(A) for ib, b in enumerate(B)
             if abs(a[0]-b[0]) < tol]
        # widen the gate while lost, and after two tracks shared a merged blob: the merged centroid is
        # biased between them, so at the split the true blob can be farther than the normal gate
        if not hasattr(self, 'shared'):
            self.shared = np.zeros(n, int)
        gate = {i: .4+.2*self.lost[i]+(.6 if self.shared[i] > 0 else 0.) for i in live}
        opts = {i: [None]+[h for h in H if np.linalg.norm(h[2]-pred[i]) < gate[i]] for i in live}
        merged = lambda i, j, ax: np.linalg.norm(pred[i][ax]-pred[j][ax]) < .2
        best, choice = np.inf, {}
        for combo in itertools.product(*(opts[i] for i in live)):
            ok = True
            for (i, hi), (j, hj) in itertools.combinations(zip(live, combo), 2):
                if hi is None or hj is None:
                    continue
                if (hi[0] == hj[0] and not merged(i, j, [0, 1])) or (hi[1] == hj[1] and not merged(i, j, [0, 2])):
                    ok = False; break
            if not ok:
                continue
            cost = sum(1. if h is None else np.linalg.norm(h[2]-pred[i]) for i, h in zip(live, combo))
            # sharing a blob is a last resort: when a merged blob splits, the two tracks take the two parts
            used = [h[:2] for h in combo if h is not None]
            cost += .3*(len(used)-len({u[0] for u in used}))+.3*(len(used)-len({u[1] for u in used}))
            if cost < best:
                best, choice = cost, dict(zip(live, combo))
        self.shared = np.maximum(self.shared-1, 0)
        frozen = np.zeros((n, 3), bool)                      # coordinate measured only by a merged blob
        for (i, hi), (j, hj) in itertools.combinations(choice.items(), 2):
            if hi is not None and hj is not None and (hi[0] == hj[0] or hi[1] == hj[1]):
                self.shared[[i, j]] = 20                     # remember for 2 s
                c = 1 if hi[0] == hj[0] else 2               # top view gives y, side view gives z
                frozen[[i, j], c] = True
        hit = np.zeros((n, 2), bool)
        for i in live:
            if choice.get(i) is not None:
                cl_est[i] = choice[i][2]; hit[i] = True
                continue
            # no consistent 3-D hypothesis (occluded in one view): update the coordinates the other sees
            for v, (axes, uniq, _) in enumerate(per_view):
                ax = list(axes)
                d = [np.linalg.norm(q-pred[i][ax]) for q in uniq]
                if d and min(d) < gate[i]:
                    q = uniq[int(np.argmin(d))]; hit[i, v] = True
                    cl_est[i, ax[1]] = q[1]; cl_est[i, 0] = q[0] if not hit[i, 0] or v == 0 else .5*(cl_est[i, 0]+q[0])
        for i in live:
            self.lost[i] = 0 if hit[i].all() else self.lost[i]+1
        # particles: pair the small blobs of the two views on the shared x axis (sub-pixel bodies, so this
        # is the noisy, occlusion-prone part of the chain; a particle merged into a cluster blob is unseen)
        (_, _, sa), (_, _, sb) = per_view
        P = []
        for xy in sa:
            for xz in sb:
                if abs(xy[0]-xz[0]) < self.cam.pair_tol_mm:
                    p = np.array([(xy[0]+xz[0])/2, xy[1], xz[1]])
                    if all(np.linalg.norm(p-q) > .02 for q in P):
                        P.append(p)
        for i in live:
            parts_world[i] = [q for q in P if np.linalg.norm(q-pred[i]) < self.cfg.particle_radius_mm+.2]
        self._prev_truth = truth.copy()
        # per-coordinate measurement mask: x is seen by either view, y only by the top, z only by the side
        # A merged blob's centroid lies between the two bodies, so it measures neither: those coordinates
        # coast on the velocity from before the merge (motion continuity resolves identity at the split).
        seen = np.stack((hit.any(1), hit[:, 0], hit[:, 1]), 1) & ~frozen
        # unmeasured coordinates coast on a decaying velocity (no unbounded dead reckoning)
        self.track_vel = np.where(seen, (cl_est-self.track)/self.dt, np.where(frozen, self.track_vel, .5*self.track_vel))
        cl_est = np.where(seen, cl_est, self.track+self.track_vel*self.dt)
        self.track = cl_est.copy()
        # latency: the controller sees the previous frame's estimate
        self.frames_buf.append((cl_est.copy(), parts_world))
        meas, parts_world = self.frames_buf[max(len(self.frames_buf)-1-self.cfg.latency_steps, 0)]
        if len(self.frames_buf) > self.cfg.latency_steps+2:
            self.frames_buf.pop(0)
        self.err_log.append(np.linalg.norm(meas-truth[:n], axis=1)[env.active[:n]])
        pos = meas.copy(); active = env.active[:n].copy()
        vel = np.zeros_like(pos) if self.prev_pos is None else (pos-self.prev_pos)/self.dt
        edge = []
        for i in range(n):
            self._who = i
            edge.append(self._match(pos[i], None if self.prev_edge is None else int(self.prev_edge[i])))
        edge = np.array(edge)
        station = np.array([int(self.ends[e][int(np.argmin(np.linalg.norm(self.points[self.ends[e]]-pos[i], axis=1)))])
                            for i, e in enumerate(edge)])
        parts = []
        for i in range(n):
            P = np.array([p for p in parts_world[i] if np.linalg.norm(p-pos[i]) < self.cfg.particle_radius_mm]).reshape(-1, 3)
            prev = self.prev_particles[i]; out = []
            for q in P:
                if len(prev):
                    k = int(np.argmin(np.linalg.norm(prev-q, axis=1)))
                    v = (q-prev[k])/self.dt if np.linalg.norm(prev[k]-q) < .3 else np.zeros(3)
                else:
                    v = np.zeros(3)
                out.append((q-pos[i], v-vel[i]))
            self.prev_particles[i] = P; parts.append(out)
        rel = pos[None, :, :]-pos[:, None, :]
        vis = (np.linalg.norm(rel, axis=2) <= self.cfg.peer_radius_mm) & active[:, None] & active[None, :]
        np.fill_diagonal(vis, False)
        self.prev_pos, self.prev_edge = pos.copy(), edge.copy()
        return Estimate(pos=pos, vel=vel, edge=edge, station=station, active=active, particles=parts,
                        peers_rel=np.where(vis[..., None], rel, 0.), peers_vis=vis, clot_alive=env.masses > 0)
