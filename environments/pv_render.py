"""VTK/PyVista renderer for the vascular swarm, replacing the PyBullet view.

Why not PyBullet
----------------
PyBullet's TinyRenderer does not composite transparency. An `rgbaColor` alpha
below 1 writes the depth buffer without blending what is behind it, so a robot
inside a translucent lumen is not dimmed -- it is *gone*. Measured on the old
pipeline, at 640x480:

    vessel wall     19,600 px
    robots              13 px      <- three agents, total
    lysing robots        0 px      <- the green contact state never showed

VTK does order-independent transparency via depth peeling, so the same scene
renders the wall as tissue *and* the swarm inside it. Same test scene, wall at
alpha 0.35 with one robot inside: PyBullet 0 robot pixels, VTK 738.

Rendering is off-screen through VTK's OpenGL window with `DISPLAY` unset (the
X server on :1 is an Intel iGPU that returns wrong colours for this scene;
unsetting falls back to the Mesa software rasteriser, which is correct and
still renders this scene at ~600 fps -- far faster than the env steps).

What a frame shows
------------------
The 3D scene: translucent lumen, opaque clots sized by remaining mass, robots
coloured by state, and a geodesic from each robot to the clot it is assigned to
-- so "is it heading the right way at the fork" is answerable from a still
frame.

One frame can hold several cameras (see `VIEW_SETS`), tiled into a grid. The top
view earns its place: these trees branch mostly within a plane, and looking down
that plane's normal separates the branches, whereas any oblique view stacks them
along the line of sight so a robot in the near branch and one in the far branch
land on the same pixels. The obliques are what make the frame read as
three-dimensional, so the useful sets pair the two.

A HUD built with matplotlib and pasted in as pixels. Per-agent geodesic distance
to target over time is the panel that actually answers "how is the navigation
doing": a policy that navigates shows monotone descent, one that mills around
shows flat noise, and one that takes the wrong branch shows a rise that only
recovers after it backtracks through the junction.

This module is presentation only. It never touches dynamics, observations or
reward, and nothing here is imported during training.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# VTK picks its render window at import time, so the environment has to be set
# before pyvista is pulled in. See the module docstring: the X display present
# in this setup renders the scene with wrong colours, and unsetting DISPLAY is
# what selects the correct (software) path.
_PV = None


def _pyvista(off_screen: bool = True):
    """Import pyvista once, with the render environment already set.

    `off_screen` drops DISPLAY, which is what selects the correct software
    rasteriser here (see the module docstring). An interactive window needs the
    display, so the live viewer passes False -- and must therefore be the first
    caller, since VTK fixes its render-window class at import time.
    """
    global _PV
    if _PV is None:
        if off_screen:
            os.environ.pop("DISPLAY", None)
            os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        import pyvista as pv

        pv.OFF_SCREEN = bool(off_screen)
        _PV = pv
    return _PV


# ----------------------------------------------------------------------- style


# Sentinel azimuths, resolved per tree relative to the fitted azimuth. Using
# these instead of fixed angles keeps a view set from pointing a camera down a
# tree's degenerate axis.
ORTHOGONAL = "orthogonal"   # fitted azimuth + 90
OBLIQUE = "oblique"         # fitted azimuth + 40


@dataclass
class View:
    """One camera in a multi-view frame.

    `azimuth` may be a number (degrees), None to fit this tree's shape, or one
    of the ORTHOGONAL / OBLIQUE sentinels, which offset the fitted azimuth.
    `pitch` is degrees above horizontal, so 90 is a top-down view.

    `span` is how many grid columns the tile occupies. A top view of these trees
    has a wide silhouette, so giving it a full-width row shows far more of the
    vessel than squeezing it into a narrow column.
    """

    label: str
    azimuth: float | str | None = None
    pitch: float = 17.0
    span: int = 1


# Named view sets. Each tuple renders as one row of tiles.
#
# The top view is the one that answers "which branch is the swarm in": these
# trees branch mostly within a plane, and looking down that plane's normal
# separates the branches instead of stacking them along the line of sight, where
# an oblique view makes a robot in the near branch and one in the far branch
# land on the same pixels.
VIEW_SETS: dict[str, tuple[View, ...]] = {
    "single": (View("fitted", None, 17.0),),
    # Anatomical trio. "front" is fitted to the tree and "side" is locked 90
    # from it, so the pair always spans two independent in-plane axes while
    # avoiding the degenerate direction: a fixed azimuth=0 side view of
    # `multilevel` looks along its 0.027-thick axis and shows a bare line
    # (measured: 12% of the tile width).
    "triple": (View("top", 90.0, 90.0, span=2),
               View("front", None, 8.0),
               View("side", ORTHOGONAL, 8.0)),
    # Adds an oblique, which is the one that reads as three-dimensional; the
    # orthogonal three are easier to measure against but flatter to look at.
    "quad": (View("top", 90.0, 90.0),
             View("front", None, 8.0),
             View("side", ORTHOGONAL, 8.0),
             View("oblique", OBLIQUE, 26.0)),
    # Two tiles: a top view for branch identity beside the fitted oblique for
    # depth. The cheapest set that answers both questions.
    "top_oblique": (View("top", 90.0, 90.0),
                    View("oblique", None, 22.0)),
}


def _grid(views: tuple) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Place views into a grid, honouring each view's column span.

    Returns (cols, rows, cells) where each cell is (col, row, span). Views are
    laid out in order, wrapping to a new row when the current one cannot fit the
    next span. The column count is the widest row's total span, so a full-width
    tile and a pair of half-width tiles coexist without leaving a hole.
    """
    spans = [max(1, int(getattr(v, "span", 1))) for v in views]
    total = sum(spans)
    # Width must fit the widest single tile, and otherwise wants to be the
    # factor of the total span that gives the squarest grid. Taking only the max
    # span would put four unspanned views in one column.
    candidates = [c for c in range(max(spans), total + 1) if total % c == 0]
    cols = min(candidates or [max(spans)],
               key=lambda c: abs(c - total / c))
    row_used = 0
    # Greedy row packing at that width.
    placed: list[tuple[int, int, int]] = []
    row = 0
    for v in views:
        span = max(1, int(getattr(v, "span", 1)))
        if row_used and row_used + span > cols:
            row += 1
            row_used = 0
        placed.append((row_used, row, span))
        row_used += span
        if row_used >= cols:
            row += 1
            row_used = 0
    rows = max(r for _c, r, _s in placed) + 1
    return cols, rows, placed


def _label_font(size: int):
    """A truetype font for the tile labels, falling back to PIL's bitmap one.

    PIL's default font ignores `size`, so the labels would be tiny on a 1280px
    frame; DejaVu ships with matplotlib, which is already a dependency here.
    """
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        import matplotlib

        return ImageFont.truetype(
            str(Path(matplotlib.get_data_path()) / "fonts" / "ttf"
                / "DejaVuSans-Bold.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def resolve_views(spec) -> tuple[View, ...]:
    """Turn a VIEW_SETS key, a single View, or a sequence into a view tuple."""
    if isinstance(spec, str):
        try:
            return VIEW_SETS[spec]
        except KeyError:
            raise ValueError(
                f"unknown view set {spec!r}; choose from {sorted(VIEW_SETS)}"
            ) from None
    if isinstance(spec, View):
        return (spec,)
    views = tuple(spec)
    if not views:
        raise ValueError("need at least one view")
    return views


@dataclass
class PVStyle:
    """Appearance knobs for the 3D scene and the HUD.

    `wall_opacity` is the one that matters. Too high and the swarm vanishes
    again; too low and there is no vessel to navigate. 0.22-0.30 reads as
    tissue while leaving the lumen legible, because depth peeling means the
    front and back wall both contribute without hiding the contents.
    """

    width: int = 1280
    height: int = 720
    # "bottom" gives the 3D view the full frame width, which is what these flat
    # vessel trees need; "right" stacks the panels vertically instead. Set
    # either dimension to 0 to drop the HUD.
    hud_position: str = "bottom"
    hud_width: int = 420          # used when hud_position == "right"
    hud_height: int = 250         # used when hud_position == "bottom"

    background: str = "#080b14"
    wall_color: str = "#e8544f"
    # Measured: at 0.26 the swarm is ~100 px and half-hidden; at 0.16 it is
    # ~300 px and every state colour survives being seen through tissue. Below
    # ~0.10 the vessel stops reading as a wall at all.
    wall_opacity: float = 0.16
    wall_sides: int = 26          # tube cross-section resolution
    lumen_edge_color: str = "#ff9b96"
    lumen_edge_opacity: float = 0.35

    # Deliberately larger than the physical robot radius (0.0045). At true
    # scale an agent is ~4 px and unreadable; the swarm's position relative to
    # the lumen and the clots is what the frame has to convey, and that
    # survives the exaggeration. The physics is untouched either way.
    robot_radius: float = 0.016
    robot_color: str = "#3ea8ff"          # transiting
    robot_contact_color: str = "#31e07a"  # lysing a clot
    robot_wall_color: str = "#ffd23f"     # scraping the wall this step
    robot_specular: float = 0.45
    robot_ambient: float = 0.5

    # Warm ochre against the red wall: distinct in hue from both the tissue and
    # every robot state, so a still frame never leaves "what am I looking at"
    # ambiguous.
    clot_color: str = "#c8821e"
    clot_color_dying: str = "#e8c88a"
    clot_opacity: float = 1.0
    clot_ambient: float = 0.55

    route_color: str = "#7ce8ff"
    route_opacity: float = 0.75
    route_width: float = 2.0

    hud_bg: str = "#0d1220"
    hud_fg: str = "#dfe6f2"
    hud_grid: str = "#243049"

    n_peels: int = 14
    camera_azimuth_per_frame: float = 0.0   # slow orbit; 0 holds still
    # None picks the azimuth that fills the viewport for this tree's shape;
    # set a number to pin the view.
    camera_azimuth: float | None = None
    # Degrees above the horizontal. 90 is straight down (a top view); the
    # up-vector construction stays well conditioned there, so no special case
    # is needed.
    camera_pitch: float = 17.0
    fov_degrees: float = 30.0

    # Which cameras to render per frame. A key of VIEW_SETS, or an explicit
    # tuple of View. Multiple views are tiled in one row across the frame.
    views: str | tuple = "single"
    view_label_size: int = 15
    view_divider_color: str = "#243049"
    # 1.0 frames the centerline box exactly. The margin covers what that box
    # does not: robots and clots are drawn at exaggerated radii and sit off the
    # centerline, so a tight fit clips them at the vessel's extremities.
    camera_margin: float = 1.04


# --------------------------------------------------------------- vessel mesh


def centerline_tube(points: np.ndarray, radii: np.ndarray,
                    branch_ids: np.ndarray, n_sides: int):
    """Swept surface of the lumen, one tube per branch.

    Tubing each branch separately matters: a single polyline through the
    concatenated station array would sweep a spurious tube from the end of one
    branch back to the start of the next, drawing vessel where there is none.

    The tube radius follows the per-station lumen radius, so Murray-law taper
    and any stenosis appear as a real change in calibre rather than as a
    uniform pipe.
    """
    pv = _pyvista()
    pts = np.asarray(points, dtype=np.float64)
    rad = np.asarray(radii, dtype=np.float64)
    bid = np.asarray(branch_ids, dtype=np.int32)

    meshes = []
    for b in np.unique(bid):
        idx = np.flatnonzero(bid == b)
        if idx.size < 2:
            continue
        line = pv.PolyData(pts[idx])
        line.lines = np.hstack([[idx.size], np.arange(idx.size)])
        line["radius"] = rad[idx]
        # `absolute=True` makes the scalar the radius in world units instead of
        # a multiplier, and capping off keeps junctions from showing a disc.
        meshes.append(line.tube(scalars="radius", absolute=True,
                                radius=float(rad[idx].mean()),
                                n_sides=int(n_sides), capping=False))
    if not meshes:
        return None
    out = meshes[0]
    for m in meshes[1:]:
        out = out.merge(m)
    return out


def centerline_polyline(points: np.ndarray, branch_ids: np.ndarray):
    """The centerline itself, as one polyline per branch."""
    pv = _pyvista()
    pts = np.asarray(points, dtype=np.float64)
    bid = np.asarray(branch_ids, dtype=np.int32)
    segs = []
    for b in np.unique(bid):
        idx = np.flatnonzero(bid == b)
        if idx.size < 2:
            continue
        line = pv.PolyData(pts[idx])
        line.lines = np.hstack([[idx.size], np.arange(idx.size)])
        segs.append(line)
    if not segs:
        return None
    out = segs[0]
    for s in segs[1:]:
        out = out.merge(s)
    return out


def _polyline(points: np.ndarray):
    """A single open polyline through `points` (>=2)."""
    pv = _pyvista()
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return None
    line = pv.PolyData(pts)
    line.lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])])
    return line


# ------------------------------------------------------------------ HUD panel


class HudPainter:
    """Draws the right-hand instrument panel with matplotlib.

    Kept as an Agg figure that is reused across frames: creating a figure per
    frame dominates the frame time, while `set_data` on existing artists does
    not. The figure is rasterised to RGB and pasted into the frame.
    """

    def __init__(self, style: PVStyle, n_robots: int, horizon: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.style = style
        self.n_robots = int(n_robots)
        self.horizon = int(horizon)
        self._plt = plt

        dpi = 100.0
        self.horizontal = style.hud_position == "bottom"
        if self.horizontal:
            w_in, h_in = style.width / dpi, style.hud_height / dpi
        else:
            w_in, h_in = style.hud_width / dpi, style.height / dpi
        self.fig = plt.figure(figsize=(w_in, h_in), dpi=dpi,
                              facecolor=style.hud_bg)

        if self.horizontal:
            # Text block then three panels side by side. A wide strip suits the
            # vessel trees, which are flat slabs: giving the 3D view the full
            # frame width is what lets the lumen fill it.
            self.ax_text = self.fig.add_axes([0.005, 0.0, 0.185, 1.0])
            self.ax_dist = self.fig.add_axes([0.215, 0.20, 0.185, 0.66])
            self.ax_rate = self.fig.add_axes([0.425, 0.20, 0.145, 0.66])
            self.ax_reward = self.fig.add_axes([0.595, 0.20, 0.145, 0.66])
            self.ax_clot = self.fig.add_axes([0.765, 0.20, 0.21, 0.66])
        else:
            self.ax_text = self.fig.add_axes([0.0, 0.74, 1.0, 0.26])
            self.ax_dist = self.fig.add_axes([0.17, 0.475, 0.78, 0.235])
            self.ax_rate = self.fig.add_axes([0.17, 0.265, 0.78, 0.175])
            self.ax_reward = self.fig.add_axes([0.17, 0.375, 0.78, 0.10])
            self.ax_clot = self.fig.add_axes([0.17, 0.055, 0.78, 0.175])

        self.ax_text.axis("off")
        self._text = self.ax_text.text(
            0.02, 0.97, "", va="top", ha="left", family="monospace",
            fontsize=7.2 if self.horizontal else 9.5,
            color=style.hud_fg, linespacing=1.5,
        )

        cmap = plt.get_cmap("turbo")
        self._agent_colors = [cmap(0.12 + 0.76 * i / max(self.n_robots - 1, 1))
                              for i in range(self.n_robots)]

        for ax, ylab in ((self.ax_dist, "geodesic → target"),
                         (self.ax_rate, "removal rate"),
                         (self.ax_reward, "step reward"),
                         (self.ax_clot, "clot mass")):
            ax.set_facecolor(style.hud_bg)
            ax.grid(True, color=style.hud_grid, lw=0.6, alpha=0.9)
            ax.set_ylabel(ylab, color=style.hud_fg, fontsize=8.5)
            ax.tick_params(colors=style.hud_fg, labelsize=7.5)
            for sp in ax.spines.values():
                sp.set_color(style.hud_grid)
            ax.set_xlim(0, max(self.horizon, 10))

        self._dist_lines = [
            self.ax_dist.plot([], [], lw=1.5, color=c)[0]
            for c in self._agent_colors
        ]
        self._rate_line, = self.ax_rate.plot([], [], lw=1.8, color="#31e07a")
        self._reward_line, = self.ax_reward.plot([], [], lw=1.5, color="#ffb347")
        self.ax_rate.set_ylim(-0.02, 1.02)
        self._clot_lines: list = []
        self.ax_dist.set_ylim(0, 1)
        self.ax_clot.set_ylim(0, 1)
        if self.horizontal:
            # Side by side, every panel needs its own x label.
            for ax in (self.ax_dist, self.ax_rate, self.ax_reward, self.ax_clot):
                ax.set_xlabel("step", color=style.hud_fg, fontsize=8.0)
        else:
            self.ax_clot.set_xlabel("step", color=style.hud_fg, fontsize=8.5)

    def _ensure_clot_lines(self, n: int) -> None:
        cmap = self._plt.get_cmap("copper")
        while len(self._clot_lines) < n:
            i = len(self._clot_lines)
            c = cmap(0.35 + 0.5 * i / max(n - 1, 1))
            self._clot_lines.append(
                self.ax_clot.plot([], [], lw=1.4, color=c)[0]
            )

    def draw(self, hist: "History", header: str) -> np.ndarray:
        """Rasterise the panel for the current history. Returns HxWx3 uint8."""
        self._text.set_text(header)

        steps = np.asarray(hist.steps, dtype=np.float64)
        if steps.size:
            xmax = max(float(steps[-1]), 10.0)
            for ax in (self.ax_dist, self.ax_rate, self.ax_clot):
                ax.set_xlim(0, max(xmax, 10.0))

            dist = np.asarray(hist.geodesic, dtype=np.float64)  # [T, n_robots]
            for i, line in enumerate(self._dist_lines):
                if i < dist.shape[1]:
                    line.set_data(steps, dist[:, i])
            finite = dist[np.isfinite(dist)]
            if finite.size:
                self.ax_dist.set_ylim(0, max(float(finite.max()) * 1.1, 1e-3))

            self._rate_line.set_data(steps, np.asarray(hist.removal))
            rewards = np.asarray(hist.reward, dtype=np.float64)
            self._reward_line.set_data(steps, rewards)
            if rewards.size:
                low = float(rewards.min())
                high = float(rewards.max())
                pad = max((high - low) * 0.15, 0.05)
                self.ax_reward.set_ylim(low - pad, high + pad)

            mass = np.asarray(hist.clot_mass, dtype=np.float64)  # [T, n_clots]
            if mass.size:
                self._ensure_clot_lines(mass.shape[1])
                for i, line in enumerate(self._clot_lines):
                    if i < mass.shape[1]:
                        line.set_data(steps, mass[:, i])
                self.ax_clot.set_ylim(0, max(float(mass.max()) * 1.1, 1e-3))

        canvas = self.fig.canvas
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        return buf.copy()

    def close(self) -> None:
        self._plt.close(self.fig)


@dataclass
class History:
    """Per-step traces the HUD plots."""

    steps: list = field(default_factory=list)
    geodesic: list = field(default_factory=list)   # per-robot distance to target
    removal: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    clot_mass: list = field(default_factory=list)

    def push(self, step: int, geodesic: np.ndarray, removal: float,
             clot_mass: np.ndarray, reward: float) -> None:
        self.steps.append(int(step))
        self.geodesic.append(np.asarray(geodesic, dtype=np.float32).copy())
        self.removal.append(float(removal))
        self.reward.append(float(reward))
        self.clot_mass.append(np.asarray(clot_mass, dtype=np.float32).copy())


# -------------------------------------------------------------- the renderer


class PVRenderer:
    """Off-screen VTK scene for one vessel tree, its robots and its clots.

    Actors are created once per episode and only repositioned afterwards, with
    the exception of the clots (whose radius encodes remaining mass, and a VTK
    sphere source has to be re-tuned rather than rescaled) and the route lines
    (whose geometry changes when the assignment changes).
    """

    def __init__(self, style: PVStyle | None = None,
                 off_screen: bool = True) -> None:
        self.style = style or PVStyle()
        self.off_screen = bool(off_screen)
        self.pv = _pyvista(off_screen=self.off_screen)
        self.plotter = None
        self._robot_actors: list = []
        self._robot_meshes: list = []
        self._robot_state: list = []
        self._clot_actors: list = []
        self._route_actors: list = []
        self._hud: HudPainter | None = None
        self._frame = 0
        self._content: np.ndarray | None = None
        # Largest radius a clot is ever drawn at, needed by the camera fit
        # before any clot exists. Set by `build` from the env's contact radius.
        self._clot_draw_radius = 0.0

    # ------------------------------------------------------------------ scene

    def build(self, tree, n_robots: int, horizon: int,
              clot_draw_radius: float = 0.0) -> None:
        """(Re)create the scene for a new episode.

        `clot_draw_radius` is the largest radius a clot will be drawn at; the
        camera fit needs it up front, before any clot actor exists.
        """
        pv = self.pv
        st = self.style
        self._clot_draw_radius = float(clot_draw_radius)
        self.close_plotter()

        self._views = resolve_views(st.views)

        bottom = st.hud_position == "bottom"
        scene_w = st.width if bottom else max(st.width - st.hud_width, 320)
        scene_h = max(st.height - st.hud_height, 240) if bottom else st.height
        # One render window sized to a single tile, reused for every view: the
        # scene's actors are identical across views, so re-rendering the same
        # plotter from different cameras is both simpler and cheaper than
        # building one plotter (and one copy of every mesh) per view.
        # Grid, not a single row. Splitting 3-4 views across one row leaves each
        # tile a tall sliver (426x470), and the views that matter most here have
        # wide silhouettes -- the top view filled only 16-37% of its height that
        # way. Two rows keep every tile closer to the silhouettes' proportions.
        self._cols, self._rows, self._cells = _grid(self._views)
        self._scene_size = (scene_w, scene_h)
        col_w = scene_w // self._cols
        row_h = scene_h // self._rows
        # Pixel rect per view, in frame coordinates. Spans make the tiles
        # different sizes, so each view carries its own.
        self._rects = [
            (c * col_w, r * row_h, max(span * col_w, 200), max(row_h, 160))
            for (c, r, span) in self._cells
        ]
        # The window is resized per view, which measured at ~6ms per resize and
        # keeps depth peeling intact, so each tile is rendered at exactly its
        # own size rather than scaled or cropped.
        self._tile = self._rects[0][2:]
        self.plotter = pv.Plotter(off_screen=self.off_screen,
                                  window_size=list(self._tile))
        self.plotter.set_background(st.background)
        # Order-independent transparency. Without this the wall either hides
        # the swarm or the swarm draws through the wall, depending on actor
        # order -- which is exactly the failure the PyBullet view had.
        self.plotter.enable_depth_peeling(number_of_peels=st.n_peels,
                                          occlusion_ratio=0.0)

        wall = centerline_tube(tree.points, tree.radii, tree.branch_ids,
                               st.wall_sides)
        if wall is not None:
            self.plotter.add_mesh(wall, color=st.wall_color,
                                  opacity=st.wall_opacity,
                                  smooth_shading=True, specular=0.25,
                                  diffuse=0.9, ambient=0.35)
        axis = centerline_polyline(tree.points, tree.branch_ids)
        if axis is not None:
            # The centerline is the navigational reference; drawing it faintly
            # makes "how far off-axis is this robot" readable.
            self.plotter.add_mesh(axis, color=st.lumen_edge_color,
                                  opacity=st.lumen_edge_opacity, line_width=1.2)

        self._robot_actors, self._robot_meshes, self._robot_state = [], [], []
        for _ in range(int(n_robots)):
            mesh = pv.Sphere(radius=st.robot_radius, theta_resolution=20,
                             phi_resolution=20)
            # Same reasoning as the clots: ambient carries the colour through
            # the wall so the state a frame is trying to show survives being
            # seen from behind tissue.
            actor = self.plotter.add_mesh(mesh, color=st.robot_color,
                                          smooth_shading=True,
                                          specular=st.robot_specular,
                                          ambient=st.robot_ambient,
                                          diffuse=0.8)
            self._robot_actors.append(actor)
            self._robot_meshes.append(mesh)
            self._robot_state.append("")

        self._clot_actors, self._route_actors = [], []

        # Frame on the bounding box, not the bounding sphere. These trees are
        # flat slabs -- mca_stroke spans 0.73 in x but only 0.26 in y and 0.21
        # in z -- so a sphere fit is driven by the long axis and leaves the
        # vessel as a thin band across an otherwise empty frame.
        pts = np.asarray(tree.points, dtype=np.float64)
        rad = np.asarray(tree.radii, dtype=np.float64)
        # Grow the box by the local lumen radius rather than the global maximum:
        # `straight` is nearly one-dimensional (0.09 in y, 0.03 in z) and its
        # contents sit up to 0.04 off the centerline, so a box fitted to the
        # centerline alone clips them.
        # Point cloud the camera fit measures. The lumen surface is sampled as
        # the centerline offset by +/- the local radius along each axis, which
        # bounds the tube without meshing it; robots and clots stay inside that
        # envelope once it is grown by the largest radius either is drawn at.
        pad = max(st.robot_radius, self._clot_draw_radius)
        shell = [pts + s * (rad[:, None] + pad) * np.eye(3)[ax]
                 for ax in range(3) for s in (-1.0, 1.0)]
        self._content = np.vstack(shell)

        lo = self._content.min(axis=0)
        hi = self._content.max(axis=0)
        self._centre = 0.5 * (lo + hi)
        self._half = np.maximum(0.5 * (hi - lo), 1e-6)
        # Already folded into the box above; kept at zero so the silhouette is
        # not padded twice.
        self._radii_max = 0.0
        # The camera fit is per tile, not per frame, and spans make tiles differ
        # in shape, so the aspect is set from the tile being rendered.
        self._aspects = [w / max(h, 1) for _x, _y, w, h in self._rects]
        self._aspect = self._aspects[0]
        # Resolve each view's azimuth once. A view with azimuth None is fitted
        # to this tree's shape at that view's own pitch, since the silhouette
        # that best fills a tile depends on both angles.
        def fitted(pitch: float) -> float:
            return (st.camera_azimuth if st.camera_azimuth is not None
                    else self._best_azimuth(pitch))

        azimuths = []
        for i, v in enumerate(self._views):
            # Fitting depends on the tile's aspect, so evaluate each view with
            # its own tile in effect.
            self._aspect = self._aspects[i]
            if v.azimuth is None:
                azimuths.append(fitted(v.pitch))
            elif v.azimuth == ORTHOGONAL:
                azimuths.append(self._second_azimuth(v.pitch, fitted(v.pitch)))
            elif v.azimuth == OBLIQUE:
                azimuths.append(fitted(v.pitch) + 40.0)
            else:
                azimuths.append(float(v.azimuth))
        self._azimuths = tuple(azimuths)
        self._aspect = self._aspects[0]
        # Kept for the orbit, which spins the first view.
        self._azimuth = self._azimuths[0]
        self._set_camera(self._azimuth, self._views[0].pitch)

        hud_size = st.hud_height if bottom else st.hud_width
        self._hud = (HudPainter(self.style, int(n_robots), int(horizon))
                     if hud_size > 0 else None)
        self._frame = 0

    def _view_frame(self, azimuth: float,
                    pitch: float | None = None) -> tuple[np.ndarray, ...]:
        """Orthonormal (view_dir, right, up) for a given azimuth and pitch.

        `pitch` is degrees above horizontal; 90 looks straight down. `right`
        stays horizontal, so a top view keeps a level horizon rather than
        rolling, and `up` never degenerates because it is built from a cross
        product with a vector that is orthogonal to the vertical by
        construction.
        """
        if pitch is None:
            pitch = self.style.camera_pitch
        a, e = np.radians(azimuth), np.radians(float(pitch))
        view_dir = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e),
                             np.sin(e)])
        view_dir /= np.linalg.norm(view_dir)
        right = np.array([-np.sin(a), np.cos(a), 0.0])
        # cross(view_dir, right), not the reverse: the other order points the
        # up vector below the horizon and renders the scene upside down.
        up = np.cross(view_dir, right)
        return view_dir, right, up

    def _silhouette(self, azimuth: float,
                    pitch: float | None = None) -> tuple[float, float]:
        """Half-width and half-height of the tree's silhouette at this view."""
        _vd, right, up = self._view_frame(azimuth, pitch)
        h = np.abs(self._half)
        return (float(h @ np.abs(right)) + self._radii_max,
                float(h @ np.abs(up)) + self._radii_max)

    def _best_azimuth(self, pitch: float | None = None) -> float:
        """Azimuth whose silhouette best matches the viewport's aspect ratio.

        The scenarios are flat slabs with very different proportions -- the
        silhouette aspect at a fixed azimuth ranges from 0.52 (multilevel) to
        1.51 (mca_stroke) -- so one hardcoded angle leaves a third of the frame
        empty for some of them. Choosing the angle that fills the viewport puts
        the vessel across as much of the frame as the geometry allows, which is
        also the angle at which the swarm is largest on screen.
        """
        best, best_fill = 0.0, -1.0
        for az in np.arange(0.0, 180.0, 5.0):
            hw, hh = self._silhouette(float(az), pitch)
            # Fitting is limited by whichever axis binds first, so the fraction
            # of the viewport actually covered is the ratio of the silhouette's
            # aspect to the viewport's, whichever way round that lands.
            ratio = hw / max(hh, 1e-9)
            fill = min(ratio, self._aspect) / max(ratio, self._aspect)
            if fill > best_fill:
                best, best_fill = float(az), fill
        return best

    def _second_azimuth(self, pitch: float, first: float,
                        min_separation: float = 50.0) -> float:
        """A useful second viewpoint, at least `min_separation` from `first`.

        A plain +90 from the fitted azimuth is the textbook orthogonal view, but
        on a near-planar tree that direction is the thin one: `multilevel` is
        0.027 thick, and its +90 view collapsed to 6% of the tile width, showing
        a line rather than a vessel. Maximising the silhouette subject to a
        minimum angular separation keeps the second view genuinely different
        from the first while still showing something.
        """
        best, best_score = (first + 90.0) % 360.0, -1.0
        for az in np.arange(0.0, 360.0, 5.0):
            sep = abs((az - first + 180.0) % 360.0 - 180.0)
            if sep < min_separation:
                continue
            hw, hh = self._silhouette(float(az), pitch)
            ratio = hw / max(hh, 1e-9)
            fill = min(ratio, self._aspect) / max(ratio, self._aspect)
            # Reward separation from the first view, but peaking at 90 rather
            # than growing to 180: the opposite azimuth is a mirror of the
            # first view, so it fills the tile just as well while showing the
            # same thing. sin(sep) is 1 at 90 and 0 at both 0 and 180.
            score = fill + 0.35 * float(np.sin(np.radians(sep)))
            if score > best_score:
                best, best_score = float(az), score
        return best

    def _set_camera(self, azimuth: float, pitch: float | None = None) -> None:
        """Frame the tree tightly from a given azimuth and pitch.

        The distance is solved from the projected extent of the bounding box as
        seen from this view, so the vessel fills the frame at every angle
        instead of only at the one the fit was made for. Hardcoding a target
        (the old camera used [0.5,0.5,0.5]) suits exactly one topology; the
        bounding-sphere fit that replaced it wastes most of the frame on these
        flat trees.
        """
        c, hx = self._centre, self._half
        view_dir, right, up = self._view_frame(azimuth, pitch)
        half_w, half_h = self._silhouette(azimuth, pitch)

        fov = np.radians(self.style.fov_degrees)
        # Fit whichever axis binds: vertical FOV is `fov`, horizontal is wider
        # by the aspect ratio.
        d_h = half_h / np.tan(fov / 2.0)
        d_w = half_w / (np.tan(fov / 2.0) * self._aspect)
        dist = max(d_h, d_w) * self.style.camera_margin

        eye = c + dist * view_dir
        # Set the camera directly rather than via `camera_position`, which
        # resets the view angle and would discard the FOV the fit assumed.
        cam = self.plotter.camera
        cam.SetViewAngle(float(np.degrees(fov)))
        cam.SetPosition(*eye.tolist())
        cam.SetFocalPoint(*c.tolist())
        cam.SetViewUp(*up.tolist())
        cam.SetClippingRange(max(dist - 3.0 * float(hx.max()), 1e-3),
                             dist + 3.0 * float(hx.max()))

        # The box fit above is only approximate: the content is not symmetric
        # about the box centre once robots and clots are included, and a
        # perspective projection is not linear in depth, so the fitted view can
        # still sit off-centre and clip. Re-centre and re-scale from the actual
        # projected extent, which is exact.
        self._refit(view_dir, right, up, fov)

    def _refit(self, view_dir: np.ndarray, right: np.ndarray, up: np.ndarray,
               fov: float) -> None:
        """Pan and zoom so the projected content is centred and inside frame."""
        if self._content is None:
            return
        cam = self.plotter.camera
        lim_y = np.tan(fov / 2.0)
        lim_x = lim_y * self._aspect

        # Aim at the centre of the content's own extent in the view frame. An
        # earlier version derived the focal point from depths measured at the
        # current eye; when the content is asymmetric along the view direction
        # that pushes the focal point far outside the vessel (measured: 1.5
        # units past it on `multilevel`, which emptied the frame entirely).
        # Projecting the content onto the three view axes and taking the
        # midpoint of each cannot do that.
        c = self._content
        focal = (view_dir * 0.5 * ((c @ view_dir).min() + (c @ view_dir).max())
                 + right * 0.5 * ((c @ right).min() + (c @ right).max())
                 + up * 0.5 * ((c @ up).min() + (c @ up).max()))

        # With the axis fixed, solve the distance directly. For each point the
        # constraint is |lateral| <= lim * (depth_to_eye), and depth depends on
        # the distance being solved for, so take the max over points of the
        # distance each one requires.
        rel = self._content - focal
        along = rel @ view_dir          # +ve is behind the focal plane
        lx = np.abs(rel @ right)
        ly = np.abs(rel @ up)
        # dist_needed: |lx| <= lim_x * (dist + along)  ->  dist >= lx/lim_x - along
        need = max(float(np.max(lx / lim_x - along)),
                   float(np.max(ly / lim_y - along)))
        dist = max(need, 1e-4) * self.style.camera_margin

        eye = focal - dist * view_dir
        cam.SetPosition(*eye.tolist())
        cam.SetFocalPoint(*focal.tolist())
        span = float(np.abs(self._half).max())
        cam.SetClippingRange(max(dist - 3.0 * span, 1e-4), dist + 3.0 * span)

    # ----------------------------------------------------------------- update

    def update(self, robot_positions: np.ndarray, robot_states: list[str],
               clot_positions: np.ndarray, clot_frac: np.ndarray,
               clot_base_radius: float,
               routes: list[np.ndarray] | None = None) -> None:
        """Move/recolour the per-step actors."""
        st = self.style
        pv = self.pv
        pos = np.asarray(robot_positions, dtype=np.float64)

        for i, actor in enumerate(self._robot_actors):
            if i >= pos.shape[0]:
                break
            # The sphere mesh sits at the origin, so the actor's position *is*
            # the world position -- no mesh rewrite per frame.
            actor.SetPosition(*pos[i].tolist())
            state = robot_states[i] if i < len(robot_states) else "transit"
            if state != self._robot_state[i]:
                color = {"contact": st.robot_contact_color,
                         "wall": st.robot_wall_color}.get(state, st.robot_color)
                actor.GetProperty().SetColor(*pv.Color(color).float_rgb)
                self._robot_state[i] = state

        # Clots: radius encodes remaining mass, so lysis is visible as the clot
        # shrinking and finally vanishing. Radius lives in the sphere source,
        # hence the rebuild.
        for actor in self._clot_actors:
            self.plotter.remove_actor(actor, render=False)
        self._clot_actors = []
        cpos = np.asarray(clot_positions, dtype=np.float64)
        frac = np.clip(np.asarray(clot_frac, dtype=np.float64), 0.0, 1.0)
        for i in range(cpos.shape[0]):
            f = float(frac[i])
            if f <= 1e-6:
                continue
            r = clot_base_radius * (0.42 + 0.58 * f)
            mesh = pv.Sphere(radius=r, center=tuple(cpos[i]),
                             theta_resolution=22, phi_resolution=22)
            col = st.clot_color if f > 0.5 else st.clot_color_dying
            # High ambient keeps a clot readable through the wall in front of
            # it. With ordinary lighting the wall's own shading dims it into
            # the tissue colour and the target stops being findable.
            self._clot_actors.append(self.plotter.add_mesh(
                mesh, color=col, opacity=st.clot_opacity,
                smooth_shading=True, specular=0.15,
                ambient=st.clot_ambient, diffuse=0.75, render=False,
            ))

        # Assignment lines. This is what makes navigation legible: the line
        # follows the vessel geodesic, so a robot heading into the wrong branch
        # shows a line doubling back through the junction.
        for actor in self._route_actors:
            self.plotter.remove_actor(actor, render=False)
        self._route_actors = []
        if routes:
            for path in routes:
                line = _polyline(path)
                if line is None:
                    continue
                self._route_actors.append(self.plotter.add_mesh(
                    line, color=st.route_color, opacity=st.route_opacity,
                    line_width=st.route_width, render=False,
                ))

    # ------------------------------------------------------------------ frame

    def _tile_scenes(self, tiles: list[np.ndarray]) -> np.ndarray:
        """Lay the view tiles out in a row, labelled and divided."""
        if len(tiles) == 1:
            return tiles[0]
        from PIL import Image, ImageDraw

        scene_w, scene_h = self._scene_size
        out = np.empty((scene_h, scene_w, 3), dtype=np.uint8)
        out[:, :] = np.asarray(self.pv.Color(self.style.background).int_rgb,
                               dtype=np.uint8)
        for t, (x, y, w, h) in zip(tiles, self._rects):
            th, tw = min(t.shape[0], scene_h - y), min(t.shape[1], scene_w - x)
            out[y: y + th, x: x + tw] = t[:th, :tw]

        img = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        divider = self.pv.Color(self.style.view_divider_color).int_rgb
        font = _label_font(self.style.view_label_size)
        for view, (x, y, w, h) in zip(self._views, self._rects):
            if x:
                draw.line([(x, y), (x, y + h)], fill=divider, width=1)
            if y:
                draw.line([(x, y), (x + w, y)], fill=divider, width=1)
            # Naming each tile matters more than it sounds: "top" and "front"
            # are indistinguishable for a tree that happens to be symmetric
            # about the viewing plane, and the reader has no other cue.
            draw.text((x + 12, y + 8), view.label,
                      fill=self.style.hud_fg, font=font)
        return np.asarray(img)

    def _render_tile(self, azimuth: float, pitch: float,
                     index: int = 0) -> np.ndarray:
        """Point the camera at one view and grab the pixels."""
        x, y, w, h = self._rects[index]
        if tuple(self.plotter.window_size) != (w, h):
            self.plotter.window_size = [w, h]
        self._aspect = self._aspects[index]
        self._set_camera(azimuth, pitch)
        # `screenshot` reuses whatever is already in the framebuffer, so it must
        # be preceded by an explicit render. Without this every frame is a copy
        # of the first one: actors move, the camera moves, and the output does
        # not change.
        self.plotter.render()
        return np.asarray(self.plotter.screenshot(return_img=True))[:, :, :3]

    def frame(self, hist: History, header: str) -> np.ndarray:
        """Render every view, tile them, paste the HUD, return HxWx3 uint8."""
        st = self.style
        spin = self._frame * st.camera_azimuth_per_frame
        self._frame += 1

        tiles = []
        for i, (view, az) in enumerate(zip(self._views, self._azimuths)):
            # The orbit spins every view together, so a multi-view frame stays
            # internally consistent rather than drifting out of relative angle.
            # A top-down view is exempt: spinning it just rotates the picture.
            drift = 0.0 if view.pitch >= 88.0 else spin
            tiles.append(self._render_tile(az + drift, view.pitch, i))
        scene = self._tile_scenes(tiles)

        if self._hud is None:
            return scene
        hud = self._hud.draw(hist, header)

        pad = np.asarray(self.pv.Color(st.hud_bg).int_rgb, dtype=np.uint8)
        if st.hud_position == "bottom":
            w = max(scene.shape[1], hud.shape[1])
            out = np.empty((scene.shape[0] + hud.shape[0], w, 3), np.uint8)
            out[:, :] = pad
            out[: scene.shape[0], : scene.shape[1]] = scene
            out[scene.shape[0]:, : hud.shape[1]] = hud
        else:
            h = max(scene.shape[0], hud.shape[0])
            out = np.empty((h, scene.shape[1] + hud.shape[1], 3), np.uint8)
            out[:, :] = pad
            out[: scene.shape[0], : scene.shape[1]] = scene
            out[: hud.shape[0], scene.shape[1]:] = hud
        return out

    # ----------------------------------------------------------------- teardown

    def show_interactive(self) -> None:
        """Open the scene in a live window (needs a real display).

        Off-screen mode is what the GIF/MP4 path uses; this is for watching an
        episode play. `interactive_update` keeps `show` from blocking so the
        caller can keep stepping the environment.
        """
        if self.plotter is None:
            return
        # A live window shows one camera, which you steer with the mouse; the
        # multi-view tiling only applies to the off-screen frames.
        self.plotter.show(auto_close=False, interactive=False,
                          interactive_update=True)

    def pump(self) -> None:
        """Redraw a live window and process its input events."""
        if self.plotter is None:
            return
        st = self.style
        if st.camera_azimuth_per_frame:
            self._set_camera(self._azimuth
                             + self._frame * st.camera_azimuth_per_frame,
                             self._views[0].pitch)
        self._frame += 1
        self.plotter.update()
        self.plotter.render()

    def close_plotter(self) -> None:
        if self.plotter is not None:
            try:
                self.plotter.close()
                self.plotter.deep_clean()
            except Exception:
                pass
            self.plotter = None
        if self._hud is not None:
            self._hud.close()
            self._hud = None
        # Drop our references to the meshes as well. PyVista's PolyData
        # finaliser touches the import system, so anything still alive at
        # interpreter shutdown raises "sys.meta_path is None" from __del__ --
        # harmless, but it buries the run's actual output in tracebacks.
        self._robot_meshes = []
        self._robot_actors = []
        self._clot_actors = []
        self._route_actors = []
        self._content = None

    def close(self) -> None:
        self.close_plotter()
