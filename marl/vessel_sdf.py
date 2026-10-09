"""Signed distance to the vessel wall from the pre-operative map (deployable), and a wall-safety projection.

Map = union of tapered tubes (one per centreline edge, healthy radius interpolated along the edge), the same
geometry the simulator uses with junction_model='union'. Inside distance of one tube:
    d_e(p) = r_e(t) - |p - axis_e(t)|,   t = clipped projection of p on the edge
Union: D(p) = smooth-max_e d_e(p) (log-sum-exp, temperature TAU), so D and its gradient stay continuous at
bends and bifurcations. grad D points from the wall toward the lumen interior.
wall_safe_projection removes only the dangerous wall-normal part of a command: when the body clearance
D(p) - body < MARGIN and the command points outward (u . n_out > 0, n_out = -grad D / |grad D|), the outward
component is scaled down by beta = clip(1 - clearance / MARGIN, 0, 1) (fully removed at contact). The tangential
part is kept; nothing pushes toward the centreline. Inputs: estimated position + map only.
"""
from __future__ import annotations
import numpy as np

TAU, MARGIN = .05, .25


class VesselSDF:
    def __init__(self, sensor):
        self.a = np.asarray(sensor.a, float); self.ab = np.asarray(sensor.ab, float)
        self.l2 = np.maximum(np.einsum('ij,ij->i', self.ab, self.ab), 1e-12)
        h = np.asarray(sensor.healthy, float); ends = np.asarray(sensor.ends, int)
        self.r0, self.r1 = h[ends[:, 0]], h[ends[:, 1]]
        tp = getattr(sensor.env, 'transport', None)          # map topology: edges sharing a junction
        self.neigh = None
        if tp is not None and hasattr(tp, 'candidates'):
            self.neigh = [np.asarray(tp.candidates[e][tp.candidate_valid[e]], int) for e in range(len(self.a))]

    def __call__(self, p, edge=None):
        """D(p) (mm, > 0 inside) and grad D (unit-ish vector toward the interior). With `edge` (the matched map
        edge), only that edge and the edges sharing one of its junctions count, as in the simulator's union
        lumen: tubes of other branches that merely overlap in space are not reachable from here, and counting
        them made D report > 1 mm clearance while the body was pressed against the wall (2026-10-07 diagnosis)."""
        p = np.asarray(p, float)
        idx = self.neigh[int(edge)] if (edge is not None and self.neigh is not None) else slice(None)
        a, ab, l2, r0, r1 = self.a[idx], self.ab[idx], self.l2[idx], self.r0[idx], self.r1[idx]
        t = np.clip(((p-a)*ab).sum(1)/l2, 0, 1)
        ax = a+t[:, None]*ab; off = p-ax; dist = np.linalg.norm(off, axis=1)
        d = (1-t)*r0+t*r1-dist
        near = d > d.max()-6*TAU                              # only tubes that matter for the smooth max
        w = np.exp((d[near]-d[near].max())/TAU); w /= w.sum()
        D = float(d[near].max()+TAU*np.log(np.exp((d[near]-d[near].max())/TAU).sum()))
        g = -(w[:, None]*off[near]/np.maximum(dist[near], 1e-9)[:, None]).sum(0)
        return D, g


def wall_safe_projection(u_world, pos, sdf, body, margin=MARGIN, edge=None):
    """Remove the outward wall-normal component of a world command near the wall; keep the tangential part."""
    u = np.asarray(u_world, float).copy()
    if not np.any(u):
        return u
    D, g = sdf(pos, edge); clear = D-body
    if clear >= margin or np.linalg.norm(g) < 1e-9:
        return u
    n_out = -g/np.linalg.norm(g); c = float(u@n_out)
    if c > 0:
        u = u-float(np.clip(1-clear/margin, 0, 1))*c*n_out
    return u
