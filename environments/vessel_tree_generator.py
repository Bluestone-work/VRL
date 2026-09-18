"""Multi-generation vessel tree with Murray's law, tortuosity, and smooth bifurcations.

This replaces the hard-coded 3-segment trees in `vessel_geometry.py::build_vessel_tree`
with a recursive branching generator. The old scenarios stay as presets; new ones
(multi-level trees, anatomical templates) are added here.

Key features:
- N generations of symmetric or asymmetric branching
- Murray's law: r_parent³ = Σ r_daughter³
- Tortuosity: each segment is a curved spline, not a straight line
- Smooth bifurcations: Bézier blending at junctions, no hard corners
- Anatomical presets: ICA siphon + MCA stroke scenario (M1 → M2 superior/inferior)
- Centerline loader stub for real patient data (VMR / CT segmentation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class VesselSegment:
    """One branch segment with its proximal/distal radii and control points."""
    start: np.ndarray          # [3] xyz
    end: np.ndarray            # [3] xyz
    radius_prox: float
    radius_dist: float
    parent_idx: int | None     # None for root
    generation: int
    tortuosity: float = 0.0    # 0=straight, >0=curved
    control_points: list[np.ndarray] | None = None  # spline interior points
    # A fully precomputed centerline, used instead of splining start/controls/end.
    # The anatomical builder needs this: its curves come from named landmarks and
    # measured curvature indices, not from a tortuosity scalar.
    curve: np.ndarray | None = None
    # Station count for this segment alone. Anatomical segments differ in length
    # by more than 10x (a 47mm pulmonary trunk against a 117mm right PA), so one
    # global count would make station spacing -- and with it the geodesic edge
    # weights and lookahead offsets -- mean different things per segment.
    n_stations: int | None = None


# Murray's law generalises to r_parent^k = sum(r_daughter^k). k=3 is the
# classical minimum-work result and holds well in the systemic arterial tree,
# but it is not universal: morphometry of the pulmonary arteries gives k ~= 2.3,
# and the coronary tree is fitted better by the Huo-Kassab relation than by
# Murray. Territories therefore carry their own exponent rather than inheriting
# a hardcoded cube.
MURRAY_EXPONENT = 3.0
PULMONARY_EXPONENT = 2.3


def murray_split(r_parent: float, n_daughters: int, asymmetry: float = 0.0,
                 exponent: float = MURRAY_EXPONENT) -> list[float]:
    """Daughter radii from the generalised Murray relation.

    r_parent^k = sum(r_daughter^k), with k = `exponent`.

    Args:
        r_parent: parent lumen radius
        n_daughters: number of daughters (typically 2)
        asymmetry: [0,1) — 0=symmetric, >0=one daughter dominates
        exponent: the k above. 3.0 is Murray's law; use ~2.3 for pulmonary
            arteries, where measured branching does not follow the cube law.

    Returns:
        list of daughter radii
    """
    if n_daughters == 1:
        return [r_parent]
    k = float(exponent)

    if asymmetry < 1e-6:
        return [r_parent / (n_daughters ** (1.0 / k))] * n_daughters

    # Asymmetric: redistribute the power-sum while keeping one daughter larger.
    # Let the dominant daughter take fraction (1+asymmetry)/2 of the sum.
    power_sum = r_parent ** k
    frac_dom = 0.5 * (1.0 + asymmetry)
    frac_sub = (1.0 - frac_dom) / max(n_daughters - 1, 1)

    r_dom = (frac_dom * power_sum) ** (1.0 / k)
    r_sub = (frac_sub * power_sum) ** (1.0 / k)

    return [r_dom] + [r_sub] * (n_daughters - 1)


def add_tortuosity(
    start: np.ndarray,
    end: np.ndarray,
    tortuosity: float,
    n_control: int = 2,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Insert control points between start/end to create a curved spline.

    Returns:
        List of interior control points (excludes start/end).
    """
    if tortuosity < 1e-6 or n_control < 1:
        return []

    rng = rng or np.random.default_rng()
    vec = end - start
    length = np.linalg.norm(vec)
    if length < 1e-8:
        return []

    tangent = vec / length
    # Perpendicular plane
    if abs(tangent[2]) < 0.9:
        perp1 = np.cross(tangent, [0, 0, 1])
    else:
        perp1 = np.cross(tangent, [1, 0, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(tangent, perp1)

    controls = []
    for i in range(n_control):
        t = (i + 1) / (n_control + 1)
        base = start + t * vec
        # Offset perpendicular to the axis by ~tortuosity * length
        amp = tortuosity * length * (0.5 + 0.5 * rng.standard_normal())
        angle = rng.uniform(0, 2 * np.pi)
        offset = amp * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
        controls.append(base + offset)
    return controls


def catmull_rom_spline(
    points: Sequence[np.ndarray],
    n_samples: int = 20,
    tension: float = 0.5,
    arclength_uniform: bool = True,
) -> np.ndarray:
    """Sample a Catmull-Rom spline through the given points.

    Args:
        points: control points the curve interpolates, [>=2, 3].
        n_samples: number of output stations.
        tension: Catmull-Rom tension; 0.5 is the uniform (centripetal-free) form.
        arclength_uniform: resample so stations are evenly spaced *along the
            curve*. Without this, spacing is uniform in the curve parameter,
            which is not the same thing: a spline through 8 control points came
            out with steps ranging 0.020-0.033 (a 1.6x spread), and across
            different control-point counts the spread was 16x. Station spacing
            is not cosmetic here -- it sets the resolution of wall projection,
            the geodesic edge weights and the lookahead offsets, so uneven
            spacing makes those quantities mean different things in different
            parts of the same vessel.

    Returns:
        [n_samples, 3] array of interpolated positions.
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return points
    n_samples = max(int(n_samples), 2)

    if len(points) == 2:
        t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[:, None]
        return ((1.0 - t) * points[0] + t * points[1]).astype(np.float32)

    # Pad endpoints so the curve interpolates both ends.
    p = np.vstack([points[0:1], points, points[-1:]])
    n_spans = len(p) - 3

    # Sample densely first, then redistribute by arclength. Oversampling is
    # cheap and makes the arclength estimate accurate enough that the
    # redistribution does not itself introduce a bias.
    dense_per_span = max(16, int(np.ceil(4.0 * n_samples / n_spans)))
    us = np.linspace(0.0, 1.0, dense_per_span, endpoint=False, dtype=np.float64)
    u2, u3 = us * us, us * us * us
    b0 = -tension * us + 2 * tension * u2 - tension * u3
    b1 = 1 + (tension - 3) * u2 + (2 - tension) * u3
    b2 = tension * us + (3 - 2 * tension) * u2 + (tension - 2) * u3
    b3 = -tension * u2 + tension * u3

    chunks = [
        (b0[:, None] * p[s] + b1[:, None] * p[s + 1]
         + b2[:, None] * p[s + 2] + b3[:, None] * p[s + 3])
        for s in range(n_spans)
    ]
    dense = np.vstack(chunks + [points[-1][None, :]]).astype(np.float64)

    if not arclength_uniform:
        idx = np.linspace(0, dense.shape[0] - 1, n_samples).round().astype(int)
        return dense[idx].astype(np.float32)

    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-12:
        return np.repeat(dense[:1], n_samples, axis=0).astype(np.float32)

    want = np.linspace(0.0, total, n_samples)
    out = np.empty((n_samples, 3), dtype=np.float64)
    for k in range(3):
        out[:, k] = np.interp(want, s, dense[:, k])
    return out.astype(np.float32)


def smooth_bifurcation(
    parent_end: np.ndarray,
    parent_tangent: np.ndarray,
    daughter_dirs: list[np.ndarray],
    radius_parent: float,
    blend_length: float = 0.05,
) -> list[np.ndarray]:
    """Generate blended start points for daughters so the junction is smooth.

    Returns:
        List of daughter start positions (slightly offset from parent_end).
    """
    # For now, a simple offset along the daughter directions scaled by blend_length.
    # A full Bézier patch would connect parent curvature to daughter curvatures;
    # this is a placeholder that at least eliminates the hard corner.
    return [parent_end + blend_length * d for d in daughter_dirs]


def generate_tree(
    root_pos: np.ndarray,
    root_dir: np.ndarray,
    root_radius: float,
    generations: int,
    branch_factor: int = 2,
    asymmetry: float = 0.0,
    tortuosity: float = 0.1,
    segment_length: float = 0.15,
    bifurcation_angle: float = 30.0,
    rng: np.random.Generator | None = None,
) -> list[VesselSegment]:
    """Recursively build a multi-generation branching tree.

    Args:
        root_pos: starting position [3]
        root_dir: initial tangent direction [3] (will be normalized)
        root_radius: lumen radius at root
        generations: how many branching levels (0=single segment)
        branch_factor: daughters per parent (typically 2)
        asymmetry: [0,1) for Murray asymmetry
        tortuosity: curvature magnitude
        segment_length: approximate length of each segment
        bifurcation_angle: degrees between daughter branches
        rng: random generator

    Returns:
        List of VesselSegment in breadth-first order (generation 0, then 1, etc.)
    """
    rng = rng or np.random.default_rng()
    root_dir = np.asarray(root_dir, dtype=np.float32)
    root_dir = root_dir / max(np.linalg.norm(root_dir), 1e-8)

    segments: list[VesselSegment] = []
    queue: list[tuple[np.ndarray, np.ndarray, float, int, int | None]] = [
        (root_pos, root_dir, root_radius, 0, None)
    ]

    while queue:
        pos, direction, radius, gen, parent_idx = queue.pop(0)

        # Segment end
        end = pos + segment_length * direction
        controls = add_tortuosity(pos, end, tortuosity, n_control=2, rng=rng)

        seg = VesselSegment(
            start=pos.copy(),
            end=end.copy(),
            radius_prox=radius,
            radius_dist=radius,  # Will taper if we add that later
            parent_idx=parent_idx,
            generation=gen,
            tortuosity=tortuosity,
            control_points=controls,
        )
        idx = len(segments)
        segments.append(seg)

        # Branch if not terminal
        if gen < generations:
            daughters = murray_split(radius, branch_factor, asymmetry)
            angle_rad = np.deg2rad(bifurcation_angle)

            # Daughter directions: spread symmetrically around parent direction
            # For 2 daughters, one veers left, one right; for N, distribute evenly.
            for i, r_d in enumerate(daughters):
                # Rotation axis perpendicular to parent direction
                if abs(direction[2]) < 0.9:
                    axis = np.cross(direction, [0, 0, 1])
                else:
                    axis = np.cross(direction, [1, 0, 0])
                axis = axis / max(np.linalg.norm(axis), 1e-8)

                # Angle offset for daughter i
                theta = angle_rad * (i - (branch_factor - 1) / 2.0) / max(branch_factor - 1, 1)
                # Rodrigues rotation
                c, s = np.cos(theta), np.sin(theta)
                rot = (
                    c * np.eye(3)
                    + s * np.array([
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ])
                    + (1 - c) * np.outer(axis, axis)
                )
                d_dir = rot @ direction
                d_dir = d_dir / max(np.linalg.norm(d_dir), 1e-8)

                # Smooth bifurcation offset (stub for now)
                d_start = end + 0.02 * d_dir

                queue.append((d_start, d_dir, r_d, gen + 1, idx))

    return segments


# --------------------------------------------------------------- tree assembly


def _fit_to_unit_cube(
    points: np.ndarray, radii: np.ndarray, margin: float = 0.06
) -> tuple[np.ndarray, np.ndarray]:
    """Rigidly scale+translate a tree so it fits inside the unit cube.

    The observation encodes absolute position as `pos * 2 - 1` and the renderer
    aims its camera at (0.5, 0.5, 0.5), so every tree must live in [0,1]^3
    regardless of the units the generator worked in. Radii are scaled by the
    same isotropic factor, which keeps the geometry similar (Murray's law and
    the radius/lumen ratios are scale invariant).
    """
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = float(np.max(hi - lo))
    if extent < 1e-8:
        return points.astype(np.float32), radii.astype(np.float32)

    # Leave room for the lumen wall itself, not just the centerline.
    r_max = float(radii.max())
    scale = (1.0 - 2.0 * margin) / (extent + 2.0 * r_max)
    centre = 0.5 * (lo + hi)
    out = (points - centre) * scale + 0.5
    return out.astype(np.float32), (radii * scale).astype(np.float32)


def segments_to_vessel_tree(
    segments: list[VesselSegment],
    n_per_segment: int = 20,
    scenario: str = "multilevel",
    min_radius: float = 0.0045,
    fit_unit_cube: bool = True,
):
    """Convert a VesselSegment list into a fully built `VesselTree`.

    This must go through `VesselTree.__init__`: the constructor is what builds
    the rotation-minimising Frenet frames, the station connectivity graph and
    the Dijkstra arclength field. Assembling the attributes by hand would leave
    `station_graph` missing and every geodesic query would fail.

    Flow fractions come from Murray's law applied to the assembled radii: a
    branch carries flow in proportion to r^3 among its siblings.
    """
    from environments.vessel_geometry import Branch, VesselTree

    points_list: list[np.ndarray] = []
    radii_list: list[np.ndarray] = []
    raw_branches: list[tuple[int, int, int]] = []  # (start, stop, parent)

    cursor = 0
    for seg in segments:
        n_want = int(seg.n_stations or n_per_segment)
        if seg.curve is not None:
            # Already a centerline: resample to the requested station count so
            # spacing stays uniform, but do not re-spline it.
            curve = np.asarray(seg.curve, dtype=np.float32)
            if curve.shape[0] != n_want and curve.shape[0] >= 2:
                curve = catmull_rom_spline(curve, n_samples=n_want)
        elif seg.control_points:
            waypoints = [seg.start, *seg.control_points, seg.end]
            curve = catmull_rom_spline(waypoints, n_samples=n_want)
        else:
            curve = np.linspace(seg.start, seg.end, n_want).astype(np.float32)

        n = int(curve.shape[0])
        points_list.append(curve.astype(np.float32))
        radii_list.append(
            np.linspace(seg.radius_prox, seg.radius_dist, n).astype(np.float32)
        )
        raw_branches.append((cursor, cursor + n - 1, -1 if seg.parent_idx is None else int(seg.parent_idx)))
        cursor += n

    points = np.concatenate(points_list, axis=0)
    radii = np.concatenate(radii_list, axis=0)
    if fit_unit_cube:
        points, radii = _fit_to_unit_cube(points, radii)
    radii = np.maximum(radii, min_radius)

    # Murray flow fractions: split the parent's flow among siblings by r^3.
    n_branches = len(raw_branches)
    children: dict[int, list[int]] = {}
    for bid, (_s, _e, parent) in enumerate(raw_branches):
        children.setdefault(parent, []).append(bid)

    flow = np.zeros((n_branches,), dtype=np.float64)
    roots = children.get(-1, [])
    for bid in roots:
        flow[bid] = 1.0 / max(len(roots), 1)
    # Breadth-first so a parent's flow is known before its children are split.
    queue = list(roots)
    while queue:
        parent = queue.pop(0)
        kids = children.get(parent, [])
        if not kids:
            continue
        # Radius at each child's proximal station drives the split.
        cubes = np.array([float(radii[raw_branches[k][0]]) ** 3 for k in kids])
        total = float(cubes.sum())
        for k, cube in zip(kids, cubes):
            flow[k] = flow[parent] * (cube / total if total > 0 else 1.0 / len(kids))
            queue.append(k)

    branches = [
        Branch(branch_id=bid, start=s, stop=e, parent=p, flow_fraction=float(flow[bid]))
        for bid, (s, e, p) in enumerate(raw_branches)
    ]

    return VesselTree(points, radii, branches, extra_links=[], scenario=scenario)


# ------------------------------------------------------------------ presets


def preset_mca_stroke(
    base_radius: float = 0.055,
    rng: np.random.Generator | None = None,
    min_radius: float = 0.0045,
):
    """Middle cerebral artery stroke scenario: ICA siphon -> M1 -> M2 bifurcation.

    Anatomical landmarks, in the order a device would traverse them:
      * ICA siphon: the carotid siphon's S-bend, the hardest part to navigate,
      * M1: the horizontal segment, the classic large-vessel-occlusion site,
      * M2 superior / inferior divisions: the bifurcation the policy must choose,
      * lenticulostriate perforator: a small high-resistance branch off M1.

    Radii use Murray's law at the M2 split, with the perforator taking its
    share of the M1 flow according to its own r^3.
    """
    rng = rng if rng is not None else np.random.default_rng()

    def jit(scale: float) -> np.ndarray:
        return rng.uniform(-scale, scale, size=3).astype(np.float32)

    # --- ICA siphon: an S-bend built from four control points ---------------
    ica_pts = [
        np.array([0.00, 0.00, 0.00], np.float32),
        np.array([0.07, 0.06, 0.02], np.float32) + jit(0.01),
        np.array([0.11, -0.03, 0.05], np.float32) + jit(0.01),
        np.array([0.17, 0.01, 0.07], np.float32) + jit(0.01),
    ]
    r_ica = base_radius * float(rng.uniform(1.02, 1.18))

    # --- M1: near-horizontal run, mildly tortuous --------------------------
    m1_start = ica_pts[-1]
    m1_end = m1_start + np.array([0.24, 0.01, 0.0], np.float32) + jit(0.015)
    r_m1_prox = r_ica * float(rng.uniform(0.80, 0.90))
    r_m1_dist = r_m1_prox * float(rng.uniform(0.86, 0.94))
    m1_ctrl = add_tortuosity(m1_start, m1_end, 0.06, n_control=1, rng=rng)

    # --- M2 bifurcation ----------------------------------------------------
    asym = float(rng.uniform(0.05, 0.30))
    r_sup, r_inf = murray_split(r_m1_dist, 2, asymmetry=asym)
    m2_sup_end = m1_end + np.array([0.15, 0.11, 0.06], np.float32) + jit(0.02)
    m2_inf_end = m1_end + np.array([0.15, -0.10, -0.03], np.float32) + jit(0.02)

    # --- lenticulostriate perforator off mid-M1 ----------------------------
    lenti_root = 0.5 * (m1_start + m1_end)
    lenti_end = lenti_root + np.array([0.01, 0.07, 0.09], np.float32) + jit(0.01)
    r_lenti = max(r_m1_prox * float(rng.uniform(0.22, 0.34)), min_radius * 1.5)

    segments = [
        VesselSegment(
            start=ica_pts[0], end=ica_pts[-1],
            radius_prox=r_ica, radius_dist=r_ica * 0.97,
            parent_idx=None, generation=0, tortuosity=0.0,
            control_points=[ica_pts[1], ica_pts[2]],
        ),
        VesselSegment(
            start=m1_start, end=m1_end,
            radius_prox=r_m1_prox, radius_dist=r_m1_dist,
            parent_idx=0, generation=1, tortuosity=0.06,
            control_points=m1_ctrl,
        ),
        VesselSegment(
            start=m1_end, end=m2_sup_end,
            radius_prox=r_sup, radius_dist=r_sup * 0.9,
            parent_idx=1, generation=2, tortuosity=0.05,
            control_points=add_tortuosity(m1_end, m2_sup_end, 0.05, 1, rng),
        ),
        VesselSegment(
            start=m1_end, end=m2_inf_end,
            radius_prox=r_inf, radius_dist=r_inf * 0.9,
            parent_idx=1, generation=2, tortuosity=0.05,
            control_points=add_tortuosity(m1_end, m2_inf_end, 0.05, 1, rng),
        ),
        VesselSegment(
            start=lenti_root, end=lenti_end,
            radius_prox=r_lenti, radius_dist=r_lenti * 0.85,
            parent_idx=1, generation=2, tortuosity=0.08,
            control_points=add_tortuosity(lenti_root, lenti_end, 0.08, 1, rng),
        ),
    ]

    return segments_to_vessel_tree(
        segments, n_per_segment=26, scenario="mca_stroke", min_radius=min_radius
    )


# --------------------------------------------------------- real-data loader


def load_centerline_csv(path, scenario: str = "patient", min_radius: float = 0.0045):
    """Load a centerline table into a `VesselTree`.

    Expected columns (header required): x, y, z, r, branch, parent.
    One row per station, rows of the same `branch` contiguous and ordered
    proximal -> distal; `parent` is the parent branch id, -1 for the root.
    This is the format both the Vascular Model Repository `.vtp` centerlines
    and a CT segmentation skeleton can be exported to.
    """
    import csv
    from pathlib import Path

    from environments.vessel_geometry import Branch, VesselTree

    rows: list[tuple[float, float, float, float, int, int]] = []
    with Path(path).open(newline="") as fh:
        for rec in csv.DictReader(fh):
            rows.append((
                float(rec["x"]), float(rec["y"]), float(rec["z"]),
                float(rec["r"]), int(rec["branch"]), int(rec["parent"]),
            ))
    if not rows:
        raise ValueError(f"no rows in {path}")

    points = np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.float32)
    radii = np.array([r[3] for r in rows], dtype=np.float32)
    bids = np.array([r[4] for r in rows], dtype=np.int32)
    parents = {int(r[4]): int(r[5]) for r in rows}

    points, radii = _fit_to_unit_cube(points, radii)
    radii = np.maximum(radii, min_radius)

    branches: list[Branch] = []
    for bid in sorted(set(bids.tolist())):
        idx = np.flatnonzero(bids == bid)
        if idx.size < 2:
            continue
        if int(idx[-1] - idx[0] + 1) != idx.size:
            raise ValueError(f"branch {bid} rows are not contiguous")
        branches.append(
            Branch(branch_id=len(branches), start=int(idx[0]), stop=int(idx[-1]),
                   parent=parents[bid], flow_fraction=1.0)
        )
    if not branches:
        raise ValueError("no branch had at least 2 stations")

    # Remap parent ids to the compacted branch indices.
    id_map = {int(bids[br.start]): br.branch_id for br in branches}
    for br in branches:
        raw_parent = parents[int(bids[br.start])]
        br.parent = id_map.get(raw_parent, -1) if raw_parent >= 0 else -1

    return VesselTree(points, radii, branches, extra_links=[], scenario=scenario)
