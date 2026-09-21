"""Multi-agent vascular thrombolysis environment with per-agent action authority.

Every microrobot receives its OWN 3-D velocity command instead of a single
shared magnetic-field command broadcast to the whole swarm. The magnetic
actuation constraint is deliberately dropped, which is what makes genuine
multi-agent RL possible -- with a shared field the agents cannot be addressed
independently and "collision avoidance between agents" is not a learnable
behaviour, only an emergent side effect of the field geometry.

What the agents must learn:
  * navigate a randomly generated curved 3-D vessel tree without hitting a wall,
  * pick the correct branch at a bifurcation,
  * avoid colliding with each other (finite robot radius, explicit pair check),
  * split across MULTIPLE clots rather than piling onto one, because lysis is
    contact-driven and saturates per clot.

Modelling notes (what changed and why)
--------------------------------------
The first version of this environment had a diagnosed failure mode: ~45% of
episodes ended with the swarm never touching a single clot. That was not a
reward-shaping problem, it was an observability problem, plus three physics
bugs. Specifically:

1. *The observation contained no vessel geometry.* Of 20 features exactly one
   (a scalar radial clearance) said anything about the vessel, so an agent knew
   "a wall is near" but not which way it faced, which way the vessel ran, or
   that a bifurcation existed. The only usable navigation signal was the
   Euclidean direction to the clot -- which in a branching vessel frequently
   points through a wall. `obs_mode="geometric"` adds the Frenet frame, the
   routed lookahead and the local flow; `obs_mode="legacy"` reproduces the old
   20-D layout so the two can be compared on identical physics.

2. *Nearest-centerline lookup was a global argmin*, so a robot in one daughter
   branch could be snapped onto the other and handed the wrong wall and the
   wrong flow direction. Now hinted by the previous station (see
   `vessel_geometry.VesselTree.nearest_station`).

3. *Shaping used Euclidean distance to the clot.* Replaced by geodesic
   (along-vessel) distance, so the shaping gradient never points into a wall.

4. *Flow was one global constant.* A stenosis was "narrower but equally slow",
   which is not what continuity says and removed the whole point of the
   scenario. Flow now obeys continuity and Murray's law, and clots occlude the
   lumen, so lysing a clot opens up the vessel it is blocking.

There was also a reward exploit: the Poiseuille profile drops to ~0 at the wall,
so hugging the wall was the cheapest way to escape adverse flow, at a cost of
one small penalty. That is now removed *physically* rather than by penalty --
near-wall lubrication drag reduces how much thrust a robot can develop, so
wall-hugging is slow instead of free.

PyBullet is used only for geometry construction and rendering; the swarm
dynamics stay in vectorized numpy because the per-step cost dominates training
throughput.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from environments.contact_geometry import continuous_route_distance, append_flow_features
from gymnasium import spaces

from environments.vessel_geometry import (
    ALL_SCENARIOS,
    SCENARIOS,
    resolve_pool,
    build_vessel_tree,
)

# Per-agent egocentric feature layouts. See `_build_observation`.
NODE_FEATURE_DIM_GEOMETRIC = 36
NODE_FEATURE_DIM_LEGACY = 20

# Station offsets sampled ahead of each robot along its route. Spread over
# roughly half a branch so the agent sees the next junction before reaching it.
LOOKAHEAD_OFFSETS = (4, 12, 26)

OBS_MODES = ("geometric", "geometric_v2", "legacy")


class Vascular3DMARLEnv(gym.Env):
    """Per-agent multi-agent thrombolysis navigation in a 3-D vessel tree."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        scenario: str = "bifurcation",
        num_robots: int = 12,
        num_clots: int = 3,
        horizon: int = 300,
        seed: int | None = None,
        render_mode: str | None = None,
        randomize_scenario: bool = True,
        randomize_clots: bool = True,
        use_pybullet: bool = False,
        reward_mode: str = "milestone",
        obs_mode: str = "geometric",
        curriculum: bool = False,
        robot_radius: float = 0.0045,
        scenario_pool: str | Sequence[str] = "legacy",
        initialization_mode: str = "legacy",
        contact_mode: str = "geodesic",
        coverage_bonus: float = 0.2,
        step_cost: float = 0.0,
        approach_scale: float = 0.1,
        reward_double_count: str = "on",
        control_margin: bool = True,
    ) -> None:
        """
        Args:
            scenario: Vessel topology; one of SCENARIOS. Used as the fixed
                topology when `randomize_scenario` is False, otherwise it is
                only the fallback for the first reset.
            num_robots: Number of independently actuated microrobots (agents).
            num_clots: Nominal clot count. With `randomize_clots` the actual
                count per episode is sampled in [max(1, n-1), n+1] and the
                observation is zero-padded to `num_clots + 1`.
            horizon: Max steps per episode.
            randomize_scenario: Resample the vessel topology every reset.
            randomize_clots: Resample clot count / position / mass every reset.
            use_pybullet: Build a PyBullet scene. Forced on when `render_mode`
                is "rgb_array" or "human". Not needed for training.
            obs_mode: "geometric" for the 36-D layout that includes vessel
                geometry, "legacy" for the original 20-D layout. Keeping both
                makes the observation change an A/B rather than a rewrite.
            curriculum: Start with one clot on a short route and widen as the
                success rate rises. Driven externally via `set_difficulty`.
        """
        super().__init__()
        if scenario not in ALL_SCENARIOS:
            raise ValueError(
                f"scenario must be one of {ALL_SCENARIOS}, got {scenario!r}"
            )
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {obs_mode!r}")
        if initialization_mode not in ("legacy", "stratified", "random"):
            raise ValueError(
                "initialization_mode must be 'legacy', 'stratified', or 'random'"
            )
        if reward_mode not in ("baseline", "milestone"):
            raise ValueError(
                f"reward_mode must be 'baseline' or 'milestone', got {reward_mode!r}"
            )

        self.scenario = scenario
        self.num_robots = int(num_robots)
        self.num_clots = int(num_clots)
        self.horizon = int(horizon)
        self.render_mode = render_mode
        self.randomize_scenario = bool(randomize_scenario)
        # Which scenarios `randomize_scenario` draws from. Defaults to the
        # legacy four so existing runs are bit-identical; an anatomical pool is
        # what a new run would use.
        self.scenario_pool = resolve_pool(scenario_pool)
        self.randomize_clots = bool(randomize_clots)
        self.use_pybullet = bool(use_pybullet) or render_mode in ("rgb_array", "human")
        if contact_mode not in ("geodesic", "euclidean"):
            raise ValueError("unknown contact_mode")
        if obs_mode not in ("legacy", "geometric", "geometric_v2"):
            raise ValueError("unknown obs_mode")
        if reward_double_count not in ("on", "off"):
            raise ValueError("unknown reward_double_count")
        self.contact_mode = contact_mode
        self.reward_double_count = reward_double_count
        self.control_margin = control_margin
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.curriculum = bool(curriculum)
        self.initialization_mode = initialization_mode

        # --- physical scale ---------------------------------------------------
        # Domain is the unit cube. Vessel radius and robot radius are chosen so
        # that `num_robots` fit in a cross-section with room to manoeuvre: a
        # collision radius comparable to the tube radius would saturate the
        # avoidance penalty into a constant and kill the learning signal.
        #
        # The device radius sets the narrowest lumen the geometry may contain,
        # because `build_vessel_tree` floors every radius at 3x it. At the
        # original 0.0045 that floor was wider than a real stenosis: a 65%
        # diameter lesion on a 3mm coronary was clipped away entirely (measured
        # residual 0.97 against a declared 0.38, and 1.00 on the femoral), so
        # the anatomical territories could not express the lesion that makes
        # their clot site thrombogenic in the first place. 0.0045 also
        # corresponds to a ~0.5mm device at coronary scale, an order of
        # magnitude larger than the microrobots being modelled.
        self.tube_radius = 0.055
        self.robot_radius = float(robot_radius)
        self.collision_distance = 2.0 * self.robot_radius
        self.max_speed = 0.018
        self.flow_speed = 0.004
        self.clot_contact_radius = 0.035
        self.lysis_rate = 0.018
        # Lysis saturates at this many robots per clot, so an extra robot on the
        # same clot contributes nothing and the swarm is pushed to split up.
        self.lysis_saturation = 4.0

        # Fraction of the local lumen a full-mass clot occludes. This is what
        # couples lysis back into the flow field.
        self.clot_occlusion = 0.65
        # Clots are never placed beyond this fraction of the vessel's arclength,
        # keeping them out of the distal end cap where an arriving robot has
        # nowhere left to go.
        self.distal_margin = 0.92
        # Brownian displacement per step. At micron scale Re << 1 so motion is
        # overdamped and "velocity = command" is the right model; this term is
        # the thermal component on top of it.
        # Brownian displacement scales with the device, not with the vessel: for
        # a sphere in the overdamped limit the diffusion coefficient goes as 1/a,
        # so a smaller device diffuses *more* per unit time, and the displacement
        # per step as sqrt(D) ~ 1/sqrt(a). Holding the number fixed while
        # shrinking the device would have quietly removed the thermal term's
        # relative influence.
        self.brownian_sigma = 0.0007 * float(np.sqrt(0.0045 / self.robot_radius))
        # Near-wall lubrication: thrust available falls off within this many
        # robot radii of the wall. Removes the "hug the wall to dodge flow"
        # exploit physically rather than with a penalty term.
        self.lubrication_range = 4.0 * self.robot_radius
        self.lubrication_floor = 0.35

        self.wall_collision_penalty = 0.1
        self.robot_collision_penalty = 0.1
        self.step_cost = float(step_cost)
        self.progress_scale = 10.0
        self.approach_scale = float(approach_scale)
        self.coverage_bonus = float(coverage_bonus)
        self.success_bonus = 30.0
        self._milestones = ((0.5, 2.0), (0.75, 3.0), (0.9, 5.0), (0.99, 10.0))
        self.clot_cleared_bonus = 3.0

        if reward_mode == "baseline":
            # run_2's original settings, kept as an ablation control.
            self.success_bonus = 10.0
            self._milestones = ()
            self.clot_cleared_bonus = 0.0

        self.neighbor_radius = 0.06  # graph adjacency + crowding features

        # Curriculum state: 0 = easiest. Consumed in _sample_clots.
        self._difficulty = 0.0 if self.curriculum else 1.0

        self._max_clot_slots = self.num_clots + 1
        self._rng = np.random.default_rng(seed)
        self._milestones_hit: set[float] = set()

        self.node_feature_dim = (
            42 if obs_mode == "geometric_v2" else NODE_FEATURE_DIM_GEOMETRIC
            if obs_mode == "geometric"
            else NODE_FEATURE_DIM_LEGACY
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_robots, 3), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "nodes": spaces.Box(
                    low=np.array([-1.0] * 36 + [0.0, -np.inf, -np.inf, 0.0, 0.0, 0.0], np.float32)[None, :].repeat(self.num_robots, 0) if obs_mode == "geometric_v2" else -1.0,
                    high=np.array([1.0] * 36 + [np.inf, np.inf, 1.0, 1.0, 1.0, 1.0], np.float32)[None, :].repeat(self.num_robots, 0) if obs_mode == "geometric_v2" else 1.0,
                    shape=(self.num_robots, self.node_feature_dim), dtype=np.float32,
                ),
                "adjacency": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self.num_robots, self.num_robots), dtype=np.float32,
                ),
                "clot_state": spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(self._max_clot_slots, 6), dtype=np.float32,
                ),
            }
        )

        # State placeholders, populated in reset().
        self.tree = None
        self.robot_positions = np.zeros((self.num_robots, 3), dtype=np.float32)
        self.robot_velocities = np.zeros((self.num_robots, 3), dtype=np.float32)
        self.robot_stations = np.zeros((self.num_robots,), dtype=np.int32)
        self.clot_positions = np.zeros((0, 3), dtype=np.float32)
        self.clot_stations = np.zeros((0,), dtype=np.int32)
        self.clot_masses = np.zeros((0,), dtype=np.float32)
        self.clot_initial_mass = np.zeros((0,), dtype=np.float32)
        self.active_clots = 0
        self.task_assignments: np.ndarray | None = None
        self.steps = 0
        self.path_length = 0.0
        self._pb = None
        self._route_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # -------------------------------------------------------------- curriculum

    def set_difficulty(self, difficulty: float) -> None:
        """Set curriculum difficulty in [0, 1].

        At 0 a single clot is placed on the trunk, close in. At 1 the full
        `num_clots` are spread across distal branches. The diagnosed bottleneck
        was exploration -- agents that never make contact never see the lysis
        reward at all -- so starting from a reachable target gives the value
        function something to bootstrap from.
        """
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ------------------------------------------------------------------ scene

    def _sample_scene(self) -> None:
        scenario = (
            str(self._rng.choice(self.scenario_pool))
            if self.randomize_scenario else self.scenario
        )
        self.active_scenario = scenario
        self.tree = build_vessel_tree(
            scenario,
            self._rng,
            base_radius=self.tube_radius,
            min_radius=3.0 * self.robot_radius,
        )
        self._route_cache = {}

    def _sample_clots(self) -> None:
        """Place clots at centerline stations, spread over branches.

        Placement is driven by *geodesic* distance from the inlet rather than by
        raw station index: index order is an artefact of how branches were
        concatenated, so "index > 0.25 * n" did not mean "distal" for anything
        but the trunk.
        """
        rng = self._rng
        tree = self.tree

        if self.randomize_clots:
            n = int(rng.integers(max(1, self.num_clots - 1), self.num_clots + 2))
        else:
            n = self.num_clots
        n = min(n, self._max_clot_slots)

        if self.curriculum:
            # Ramp count with difficulty: 1 clot at d=0, full spread at d=1.
            n = int(round(1 + (n - 1) * self._difficulty))
            n = max(1, min(n, self._max_clot_slots))

        # Anatomical territories declare where thrombus actually lodges, so use
        # those sites instead of sampling arclength. This is the whole point of
        # the territory: a clot at a random station is a navigation exercise,
        # while a clot at a bifurcation apex or a valve sinus is the lesion the
        # policy would meet clinically.
        if getattr(tree, "territory", None) is not None:
            sites = self._sample_clot_sites(tree, n)
            if sites is not None:
                return

        arc = tree.arclength / max(tree.total_length, 1e-8)
        # Curriculum shifts the eligible band nearer the inlet, so the first
        # contact is reachable inside the horizon. The distal cap is excluded:
        # a clot at the very last station sits inside the vessel's end cap, where
        # a robot that has arrived cannot advance any further and just grinds
        # against the terminal wall collecting penalties.
        lo = 0.15 + 0.15 * self._difficulty
        hi = min(0.45 + 0.55 * self._difficulty, self.distal_margin)
        # Also require the clot to sit at least a contact radius short of the end
        # of its own branch, expressed in absolute arclength: a fractional cap
        # alone is not enough on a short branch.
        end_arc = np.zeros((tree.n_stations,), np.float32)
        for br in tree.branches:
            end_arc[br.start : br.stop + 1] = tree.arclength[br.stop]
        room = end_arc - tree.arclength
        clear_of_terminus = room >= self.clot_contact_radius
        eligible = np.flatnonzero((arc >= lo) & (arc <= hi) & clear_of_terminus)
        if eligible.size < n:
            eligible = np.flatnonzero(
                (arc >= lo) & (arc <= self.distal_margin) & clear_of_terminus
            )
        if eligible.size == 0:
            eligible = np.flatnonzero(clear_of_terminus)
        if eligible.size == 0:
            eligible = np.arange(tree.n_stations, dtype=np.int32)

        branches = tree.branch_ids[eligible]
        chosen: list[int] = []
        # One clot per branch first, so the swarm has to split up.
        for branch in rng.permutation(np.unique(branches)):
            if len(chosen) >= n:
                break
            pool = eligible[branches == branch]
            chosen.append(int(rng.choice(pool)))
        # Then fill, keeping clots apart in arclength so contact spheres do not
        # overlap into a single blob.
        guard = 0
        while len(chosen) < n and guard < 200:
            guard += 1
            cand = int(rng.choice(eligible))
            if all(
                abs(float(tree.arclength[cand] - tree.arclength[c]))
                > 1.5 * self.clot_contact_radius
                for c in chosen
            ):
                chosen.append(cand)
        while len(chosen) < n:  # degenerate geometry: accept duplicates
            chosen.append(int(rng.choice(eligible)))

        idx = np.asarray(chosen[:n], dtype=np.int32)
        self.clot_stations = idx
        # Small radial offset, scaled to the local lumen so a clot never starts
        # outside the vessel it is supposed to be occluding.
        radial = rng.normal(0.0, 0.12, size=(n, 3)).astype(np.float32)
        radial *= tree.radii[idx][:, None]
        self.clot_positions = (tree.points[idx] + radial).astype(np.float32)
        self.clot_positions, _, _, _ = tree.project(
            self.clot_positions, robot_radius=0.0, hint=idx
        )
        self.clot_masses = rng.uniform(0.7, 1.3, size=n).astype(np.float32)
        self.clot_initial_mass = self.clot_masses.copy()
        self.active_clots = n

    def _sample_clot_sites(self, tree, n: int) -> bool | None:
        """Place clots at the territory's declared thrombogenic sites.

        Sites are drawn without replacement, weighted by how often each is
        implicated clinically. Returns True on success, or None if the territory
        could not supply enough usable sites -- in which case the caller falls
        back to arclength sampling rather than producing a degenerate scene.
        """
        rng = self._rng
        sites = list(tree.territory.clot_sites)
        if not sites:
            return None

        weights = np.array([max(s.weight, 0.0) for s in sites], dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        take = min(n, len(sites))
        order = rng.choice(len(sites), size=take, replace=False, p=weights)

        stations: list[int] = []
        occl: list[float] = []
        for i in order:
            site = sites[int(i)]
            bid = tree.segment_index.get(site.segment)
            if bid is None:
                continue
            br = tree.branches[bid]
            k = int(round(np.clip(site.position, 0.0, 1.0) * (br.size - 1)))
            station = int(np.clip(br.start + k, br.start, br.stop))
            # Keep the clot clear of the branch's distal cap for the same reason
            # the arclength path does: a robot that arrives there has nowhere
            # left to go and just grinds against the terminal wall.
            room = tree.arclength[br.stop] - tree.arclength[station]
            if room < self.clot_contact_radius and br.size > 2:
                back = min(br.size - 1, int(np.ceil(
                    self.clot_contact_radius
                    / max(tree.total_length / max(tree.n_stations, 1), 1e-8)
                )))
                station = int(np.clip(br.stop - back, br.start, br.stop))
            stations.append(station)
            occl.append(float(site.occlusion))

        if not stations:
            return None

        idx = np.asarray(stations, dtype=np.int32)
        self.clot_stations = idx
        radial = rng.normal(0.0, 0.10, size=(idx.size, 3)).astype(np.float32)
        radial *= tree.radii[idx][:, None]
        self.clot_positions = (tree.points[idx] + radial).astype(np.float32)
        self.clot_positions, _, _, _ = tree.project(
            self.clot_positions, robot_radius=0.0, hint=idx
        )
        # Mass tracks the site's stated occlusion, so a site described as
        # subtotal does not silently become a total occlusion.
        base = np.asarray(occl, dtype=np.float32) / 0.8
        self.clot_masses = (base * rng.uniform(0.85, 1.15, size=idx.size)
                            ).astype(np.float32)
        self.clot_initial_mass = self.clot_masses.copy()
        self.active_clots = int(idx.size)
        return True

    # -------------------------------------------------------------- kinematics

    def _route(self, clot_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Cached (geodesic distance, next-hop) field toward one clot.

        Routes depend only on geometry, so they are computed once per clot per
        episode. Without the cache this would be a Dijkstra per agent per step.
        """
        if clot_index not in self._route_cache:
            station = int(self.clot_stations[clot_index])
            self._route_cache[clot_index] = self.tree.route_to(station)
        return self._route_cache[clot_index]

    def _contact_geodesic(self):
        out = np.full((self.num_robots, self.active_clots), np.inf, np.float32)
        for c in range(self.active_clots):
            if self.clot_masses[c] > 0:
                distance, hop = self._route(c)
                out[:, c] = continuous_route_distance(
                    self.tree, self.robot_positions, self.robot_stations, distance, hop)
        return out

    def _occluded_radius(self, station: np.ndarray) -> np.ndarray:
        """Local lumen radius including the narrowing caused by live clots.

        A clot is no longer a massless point: while it has mass it constricts the
        vessel, which (via continuity) accelerates flow past it. Lysing it both
        removes the target and reopens the vessel, so the clot and the flow field
        are actually coupled.
        """
        radius = self.tree.radii[station].copy()
        if self.active_clots == 0:
            return radius
        alive = self.clot_masses > 0
        if not np.any(alive):
            return radius

        arc_station = self.tree.arclength[station]
        arc_clot = self.tree.arclength[self.clot_stations]
        same_branch = (
            self.tree.branch_ids[station][:, None]
            == self.tree.branch_ids[self.clot_stations][None, :]
        )
        # Gaussian narrowing centred on the clot, in arclength.
        d_arc = np.abs(arc_station[:, None] - arc_clot[None, :])
        width = max(self.clot_contact_radius, 1e-6)
        bump = np.exp(-0.5 * np.square(d_arc / width)) * same_branch
        frac = self.clot_masses / np.maximum(self.clot_initial_mass, 1e-8)
        block = (bump * (frac * alive)[None, :]).max(axis=1)
        return np.maximum(radius * (1.0 - self.clot_occlusion * block), 1e-4)

    def _lubrication(self, positions: np.ndarray, station: np.ndarray) -> np.ndarray:
        """Thrust scaling near the wall, in [lubrication_floor, 1].

        Physically, drag on a sphere rises steeply as it approaches a wall, so a
        robot pinned against the wall cannot develop full thrust. This closes the
        exploit where the Poiseuille profile made wall-hugging the cheapest way
        to escape adverse flow.
        """
        axis_point, radius = self.tree._axis_point(positions, station)
        gap = np.maximum(radius - np.linalg.norm(positions - axis_point, axis=1), 0.0)
        t = np.clip(gap / max(self.lubrication_range, 1e-8), 0.0, 1.0)
        return (self.lubrication_floor + (1.0 - self.lubrication_floor) * t).astype(
            np.float32
        )

    def _robot_collisions(self) -> tuple[np.ndarray, int, np.ndarray]:
        """Per-agent overlap count, unique pair count, and separation impulse."""
        n = self.num_robots
        if n < 2:
            return np.zeros((n,), np.float32), 0, np.zeros((n, 3), np.float32)
        delta = self.robot_positions[:, None, :] - self.robot_positions[None, :, :]
        d = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(d, np.inf)
        overlap = d < self.collision_distance
        # Push overlapping pairs apart instead of only penalising them, so
        # robots cannot occupy the same point in space.
        depth = np.where(overlap, self.collision_distance - d, 0.0)
        direction = delta / np.maximum(d, 1e-8)[:, :, None]
        impulse = 0.5 * (direction * depth[:, :, None]).sum(axis=1)
        return (
            overlap.sum(axis=1).astype(np.float32),
            int(overlap.sum() // 2),
            impulse.astype(np.float32),
        )

    def _assigned_clot(self) -> np.ndarray:
        """Nearest ACTIVE clot per robot by GEODESIC distance; -1 when cleared.

        Euclidean nearest is wrong in a branching vessel: the closest clot in
        straight-line terms is regularly in a different branch, reachable only by
        going back through the junction. Assigning by along-vessel distance means
        the target the agent is shaped toward is the one it can actually reach.
        """
        alive = self.clot_masses > 0
        if self.task_assignments is not None:
            assigned = np.asarray(self.task_assignments, dtype=np.int32).copy()
            if assigned.shape != (self.num_robots,):
                raise ValueError("task assignments must have one entry per robot")
            valid = (assigned >= 0) & (assigned < self.active_clots)
            if self.active_clots > 0:
                valid &= alive[np.clip(assigned, 0, self.active_clots - 1)]
            else:
                valid[:] = False
            return np.where(valid, assigned, -1).astype(np.int32)
        if not np.any(alive):
            return np.full((self.num_robots,), -1, np.int32)
        dists = np.full((self.num_robots, self.active_clots), np.inf, np.float32)
        for c in range(self.active_clots):
            if not alive[c]:
                continue
            distance, _ = self._route(c)
            dists[:, c] = distance[self.robot_stations]
        # A robot off the graph (shouldn't happen) falls back to Euclidean.
        bad = ~np.isfinite(dists).any(axis=1)
        if np.any(bad):
            euc = np.linalg.norm(
                self.robot_positions[:, None, :] - self.clot_positions[None, :, :], axis=2
            )
            euc[:, ~alive] = np.inf
            dists[bad] = euc[bad]
        return np.argmin(dists, axis=1).astype(np.int32)

    def set_task_assignments(self, assignments: np.ndarray | Sequence[int]) -> None:
        """Set explicit per-robot clot tasks for the hierarchical controller.

        A value in ``[0, active_clots)`` selects a clot; ``-1`` means idle or
        transit. The default ``None`` mode preserves the original nearest-clot
        assignment behavior exactly.
        """
        values = np.asarray(assignments, dtype=np.int32).reshape(-1)
        if values.shape != (self.num_robots,):
            raise ValueError("task assignments must have one entry per robot")
        if np.any(values < -1) or np.any(values >= self.active_clots):
            raise ValueError("task assignment contains an invalid clot index")
        self.task_assignments = values.copy()

    def clear_task_assignments(self) -> None:
        """Return to the original nearest-active-clot assignment mode."""
        self.task_assignments = None

    def _geodesic_to_target(
        self, target: np.ndarray, positions: np.ndarray | None = None,
        stations: np.ndarray | None = None,
    ) -> np.ndarray:
        """Continuous along-vessel distance from each robot to its assigned clot.

        Taking the distance field at the nearest station alone quantises the
        result to the station spacing: a robot can move a full step and land on
        the same station, making the shaping term exactly zero and turning the
        signal into a staircase that is zero for several steps and then jumps.

        The sub-station correction projects the robot's offset from its station
        onto the direction of the next hop toward the target, so axial progress is
        credited continuously while purely radial motion (which makes no progress
        along the vessel) correctly earns nothing.
        """
        positions = self.robot_positions if positions is None else positions
        stations = self.robot_stations if stations is None else stations
        out = np.zeros((positions.shape[0],), np.float32)
        for c in range(self.active_clots):
            mask = target == c
            if not np.any(mask):
                continue
            distance, next_hop = self._route(c)
            st = stations[mask]
            base = distance[st]
            hop = next_hop[st]
            step_vec = self.tree.points[hop] - self.tree.points[st]
            step_len = np.linalg.norm(step_vec, axis=1)
            # A zero-length hop happens at the target itself and at coincident
            # junction stations; there is no direction to project onto.
            valid = step_len > 1e-8
            unit = np.where(
                valid[:, None], step_vec / np.maximum(step_len, 1e-8)[:, None], 0.0
            )
            offset = positions[mask] - self.tree.points[st]
            along = np.sum(offset * unit, axis=1)
            # Clamp to the hop length so the correction cannot overshoot past the
            # next station and make the distance non-monotonic.
            along = np.clip(along, -step_len, step_len)
            out[mask] = (base - np.where(valid, along, 0.0)).astype(np.float32)
        return out

    # ------------------------------------------------------------ observations

    def _peer_features(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.num_robots
        d = np.linalg.norm(
            self.robot_positions[:, None, :] - self.robot_positions[None, :, :], axis=2
        )
        np.fill_diagonal(d, np.inf)
        if n > 1:
            peer = np.argmin(d, axis=1)
            peer_delta = self.robot_positions[peer] - self.robot_positions
            peer_dist = np.linalg.norm(peer_delta, axis=1)
            crowding = (d < self.neighbor_radius).sum(axis=1) / max(n - 1, 1)
        else:
            peer_delta = np.zeros((n, 3), np.float32)
            peer_dist = np.ones((n,), np.float32)
            crowding = np.zeros((n,), np.float32)
        return d, peer_delta.astype(np.float32), peer_dist.astype(np.float32), crowding.astype(np.float32)

    def _build_observation(self) -> dict[str, np.ndarray]:
        n = self.num_robots
        tree = self.tree
        nodes = np.zeros((n, self.node_feature_dim), np.float32)
        target = self._assigned_clot()
        station = self.robot_stations

        axis_point, lumen_radius = tree._axis_point(self.robot_positions, station)
        radial_vec = self.robot_positions - axis_point
        radial = np.linalg.norm(radial_vec, axis=1)
        clearance = 1.0 - np.clip(radial / np.maximum(lumen_radius, 1e-8), 0.0, 1.0)

        has_target = target >= 0
        safe_target = np.where(has_target, target, 0)
        delta = self.clot_positions[safe_target] - self.robot_positions
        cdist = np.linalg.norm(delta, axis=1)
        delta = np.where(has_target[:, None], delta, 0.0)
        cdist = np.where(has_target, cdist, 1.0)

        d, peer_delta, peer_dist, crowding = self._peer_features()

        total = float(self.clot_initial_mass.sum()) or 1.0
        remaining_frac = float(self.clot_masses.sum()) / total
        time_progress = self.steps / self.horizon

        touching = (cdist <= self.clot_contact_radius) & has_target
        if self.contact_mode == "geodesic":
            touching &= self._contact_geodesic()[np.arange(n), safe_target] <= self.clot_contact_radius

        if self.obs_mode == "legacy":
            nodes[:, 0:3] = self.robot_positions * 2.0 - 1.0
            nodes[:, 3:6] = np.clip(self.robot_velocities / self.max_speed, -1.0, 1.0)
            nodes[:, 6:9] = np.clip(delta / 0.5, -1.0, 1.0)
            nodes[:, 9] = np.clip(cdist / 0.5, 0.0, 1.0)
            nodes[:, 10] = touching
            nodes[:, 11] = np.where(
                has_target,
                self.clot_masses[safe_target]
                / np.maximum(self.clot_initial_mass[safe_target], 1e-8),
                0.0,
            )
            nodes[:, 12:15] = np.clip(peer_delta / self.neighbor_radius, -1.0, 1.0)
            nodes[:, 15] = np.clip(peer_dist / self.neighbor_radius, 0.0, 1.0)
            nodes[:, 16] = clearance
            nodes[:, 17] = time_progress
            nodes[:, 18] = crowding
            nodes[:, 19] = remaining_frac
        else:
            # --- egocentric frame ---------------------------------------------
            # Everything directional is expressed in the local Frenet frame
            # (tangent, normal, binormal). A world-frame vector would force the
            # network to learn "which way is downstream" separately at every
            # point in the vessel; in the local frame "forward" is always +t.
            t_hat = tree.tangents[station]
            n_hat = tree.normals[station]
            b_hat = tree.binormals[station]

            def to_local(vec: np.ndarray) -> np.ndarray:
                return np.stack(
                    [
                        np.sum(vec * t_hat, axis=1),
                        np.sum(vec * n_hat, axis=1),
                        np.sum(vec * b_hat, axis=1),
                    ],
                    axis=1,
                ).astype(np.float32)

            # Routed lookahead: where the vessel goes, toward THIS agent's clot.
            # At a fork this bends into the correct branch, which is the single
            # piece of information the old observation could not express.
            look = np.zeros((n, len(LOOKAHEAD_OFFSETS), 3), np.float32)
            for c in range(self.active_clots):
                mask = target == c
                if not np.any(mask):
                    continue
                _dist, next_hop = self._route(c)
                look[mask] = tree.lookahead(
                    station[mask], LOOKAHEAD_OFFSETS, next_hop
                )
            no_route = ~has_target
            if np.any(no_route):
                look[no_route] = tree.lookahead(
                    station[no_route], LOOKAHEAD_OFFSETS, None
                )
            # Relative to the robot, in the local frame, scaled by lookahead span.
            look_rel = look - self.robot_positions[:, None, :]
            look_local = np.stack(
                [to_local(look_rel[:, k, :]) for k in range(look.shape[1])], axis=1
            )
            look_scale = 0.25
            look_local = np.clip(look_local / look_scale, -1.0, 1.0)

            geo = self._geodesic_to_target(target)
            geo_norm = np.where(
                has_target, np.clip(geo / max(tree.total_length, 1e-8), 0.0, 1.0), 1.0
            )

            occluded = self._occluded_radius(station)
            flow = tree.flow(
                self.robot_positions, station, self.flow_speed,
                self.tube_radius, radius_override=occluded,
            )
            lube = self._lubrication(self.robot_positions, station)

            # Wall direction: unit outward radial, in the local frame. Tells the
            # agent WHICH WAY the wall is, not merely that it is close.
            wall_dir = radial_vec / np.maximum(radial, 1e-8)[:, None]

            nodes[:, 0:3] = self.robot_positions * 2.0 - 1.0
            nodes[:, 3:6] = np.clip(to_local(self.robot_velocities) / self.max_speed, -1.0, 1.0)
            # 6:15 -- routed lookahead, 3 stations x 3 axes.
            nodes[:, 6:15] = look_local.reshape(n, -1)
            nodes[:, 15:18] = to_local(wall_dir)
            nodes[:, 18] = clearance
            nodes[:, 19] = np.clip(lumen_radius / self.tube_radius, 0.0, 1.0)
            nodes[:, 20] = np.clip(occluded / np.maximum(lumen_radius, 1e-8), 0.0, 1.0)
            nodes[:, 21:24] = np.clip(to_local(flow) / max(self.max_speed, 1e-8), -1.0, 1.0)
            nodes[:, 24] = lube
            # Clot, both as a local-frame direction and as a geodesic distance.
            nodes[:, 25:28] = np.clip(to_local(delta) / 0.5, -1.0, 1.0)
            nodes[:, 28] = geo_norm
            nodes[:, 29] = np.clip(cdist / 0.5, 0.0, 1.0)
            nodes[:, 30] = touching
            nodes[:, 31] = np.where(
                has_target,
                self.clot_masses[safe_target]
                / np.maximum(self.clot_initial_mass[safe_target], 1e-8),
                0.0,
            )
            nodes[:, 32:35] = np.clip(to_local(peer_delta) / self.neighbor_radius, -1.0, 1.0)
            nodes[:, 35] = np.clip(peer_dist / self.neighbor_radius, 0.0, 1.0)

        if self.obs_mode == "geometric_v2":
            append_flow_features(nodes, flow, look_rel[:, 0], lube, self.max_speed,
                                 time_progress, remaining_frac, crowding, self.control_margin)

        adjacency = (d <= self.neighbor_radius).astype(np.float32)
        np.fill_diagonal(adjacency, 1.0)  # self-loops, as a GAT layer expects

        clot_state = np.zeros((self._max_clot_slots, 6), np.float32)
        for i in range(self.active_clots):
            clot_state[i, 0:3] = self.clot_positions[i] * 2.0 - 1.0
            clot_state[i, 3] = self.clot_masses[i] / max(
                float(self.clot_initial_mass[i]), 1e-8
            )
            clot_state[i, 4] = 1.0 if self.clot_masses[i] > 0 else 0.0
            # Arclength position, so the critic can tell proximal from distal.
            clot_state[i, 5] = float(
                self.tree.arclength[self.clot_stations[i]]
                / max(self.tree.total_length, 1e-8)
            )

        return {"nodes": nodes, "adjacency": adjacency, "clot_state": clot_state}

    # ------------------------------------------------------------------- gym API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._sample_scene()
        self._sample_clots()

        stations = self._initial_robot_stations()
        anchor = self.tree.points[stations]
        radial = self._rng.normal(0.0, 0.25, size=(self.num_robots, 3)).astype(np.float32)
        radial *= self.tree.radii[stations][:, None]
        self.robot_positions = (anchor + radial).astype(np.float32)
        self.robot_positions, _, self.robot_stations, _ = self.tree.project(
            self.robot_positions, self.robot_radius, hint=stations
        )
        self.robot_velocities = np.zeros((self.num_robots, 3), np.float32)

        self.steps = 0
        self.path_length = 0.0
        self._first_contact_step = -1
        self._milestones_hit = set()
        self.task_assignments = None

        if self.use_pybullet:
            self._sync_pybullet(rebuild=True)
        return self._build_observation(), {
            "scenario": self.active_scenario,
            "difficulty": self._difficulty,
        }

    def _initial_robot_stations(self) -> np.ndarray:
        """Choose legacy, branch-stratified, or random starting stations."""
        trunk = self.tree.stations_of_branch(0)
        if self.initialization_mode == "legacy":
            span = max(int(0.25 * trunk.size), 2)
            picks = np.linspace(0, span - 1, self.num_robots).astype(np.int32)
            return trunk[picks]

        branches = [branch for branch in self.tree.branches if branch.size >= 4]
        if not branches:
            return trunk[np.linspace(0, max(int(0.25 * trunk.size) - 1, 1), self.num_robots).astype(np.int32)]
        if self.initialization_mode == "random":
            selected = self._rng.integers(0, len(branches), size=self.num_robots)
        else:
            order = self._rng.permutation(len(branches))
            selected = order[np.arange(self.num_robots) % len(order)]
        stations = []
        for index in selected:
            branch = branches[int(index)]
            low = max(1, int(0.12 * (branch.size - 1)))
            high = max(low + 1, int(0.65 * (branch.size - 1)))
            offset = int(self._rng.integers(low, min(high, branch.size - 1) + 1))
            stations.append(branch.start + offset)
        return np.asarray(stations, dtype=np.int32)

    def state_dict(self) -> dict[str, Any]:
        """Serializable simulator state for exact training continuation."""
        return {
            "tree": self.tree,
            "active_scenario": self.active_scenario,
            "clot_stations": self.clot_stations.copy(),
            "clot_positions": self.clot_positions.copy(),
            "clot_masses": self.clot_masses.copy(),
            "clot_initial_mass": self.clot_initial_mass.copy(),
            "active_clots": self.active_clots,
            "robot_positions": self.robot_positions.copy(),
            "robot_stations": self.robot_stations.copy(),
            "robot_velocities": self.robot_velocities.copy(),
            "task_assignments": None if self.task_assignments is None else self.task_assignments.copy(),
            "steps": self.steps,
            "path_length": self.path_length,
            "first_contact_step": self._first_contact_step,
            "milestones_hit": set(self._milestones_hit),
            "difficulty": self._difficulty,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`."""
        self.tree = state["tree"]
        self.active_scenario = state["active_scenario"]
        self.clot_stations = state["clot_stations"].copy()
        self.clot_positions = state["clot_positions"].copy()
        self.clot_masses = state["clot_masses"].copy()
        self.clot_initial_mass = state["clot_initial_mass"].copy()
        self.active_clots = int(state["active_clots"])
        self.robot_positions = state["robot_positions"].copy()
        self.robot_stations = state["robot_stations"].copy()
        self.robot_velocities = state["robot_velocities"].copy()
        assignments = state.get("task_assignments")
        self.task_assignments = None if assignments is None else assignments.copy()
        self.steps = int(state["steps"])
        self.path_length = float(state["path_length"])
        self._first_contact_step = int(state["first_contact_step"])
        self._milestones_hit = set(state["milestones_hit"])
        self._difficulty = float(state["difficulty"])
        self._rng.bit_generator.state = state["rng_state"]
        self._route_cache = {}
        if self.use_pybullet:
            self._sync_pybullet(rebuild=True)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.num_robots, 3)
        action = np.clip(action, -1.0, 1.0)

        prev_positions = self.robot_positions.copy()
        prev_stations = self.robot_stations.copy()
        prev_target = self._assigned_clot()
        # Shaping uses GEODESIC distance. The old Euclidean version pointed
        # straight through vessel walls at a bifurcation, so the shaping gradient
        # actively fought the geometry.
        prev_geo = self._geodesic_to_target(prev_target)

        # Per-agent commanded velocity. No cohesion term and no shared field:
        # each agent owns its own intent, which is the point of the MARL setup.
        norm = np.linalg.norm(action, axis=1, keepdims=True)
        magnitude = np.clip(norm, 0.0, 1.0)
        direction = action / np.maximum(norm, 1e-8)
        commanded = direction * self.max_speed * magnitude
        # Near-wall lubrication drag: less thrust available when pinned.
        commanded *= self._lubrication(prev_positions, prev_stations)[:, None]

        occluded = self._occluded_radius(prev_stations)
        flow = self.tree.flow(
            prev_positions, prev_stations, self.flow_speed,
            self.tube_radius, radius_override=occluded,
        )
        noise = self._rng.normal(
            0.0, self.brownian_sigma, prev_positions.shape
        ).astype(np.float32)
        _hits, _pairs, separation = self._robot_collisions()
        proposed = prev_positions + commanded + flow + noise + separation

        self.robot_positions, wall_hits, self.robot_stations, axis_point = (
            self.tree.project(proposed, self.robot_radius, hint=prev_stations)
        )
        # Wall contact removes the normal velocity component instead of silently
        # teleporting the robot back into the lumen with its momentum intact.
        self.robot_velocities = (self.robot_positions - prev_positions).astype(np.float32)
        if np.any(wall_hits):
            radial_vec = self.robot_positions - axis_point
            outward = radial_vec / np.maximum(
                np.linalg.norm(radial_vec, axis=1, keepdims=True), 1e-8
            )
            vn = np.sum(self.robot_velocities * outward, axis=1, keepdims=True)
            self.robot_velocities = np.where(
                wall_hits[:, None],
                self.robot_velocities - np.maximum(vn, 0.0) * outward,
                self.robot_velocities,
            ).astype(np.float32)
        self.path_length += float(np.linalg.norm(self.robot_velocities, axis=1).mean())

        peer_hits, collision_pairs, _ = self._robot_collisions()

        # --- contact-driven lysis, saturating per clot -----------------------
        prev_masses = self.clot_masses.copy()
        alive = self.clot_masses > 0
        removed_per_clot = np.zeros_like(self.clot_masses)
        agent_lysis = np.zeros((self.num_robots,), np.float32)
        contacts = 0
        if np.any(alive):
            d = np.linalg.norm(
                self.robot_positions[:, None, :] - self.clot_positions[None, :, :], axis=2
            )
            contact = (d <= self.clot_contact_radius) & alive[None, :]
            if self.contact_mode == "geodesic":
                contact &= self._contact_geodesic() <= self.clot_contact_radius
            weight = np.exp(-np.square(d / self.clot_contact_radius)) * contact
            # Saturation: the marginal value of piling more robots onto one clot
            # decays, so spreading across clots clears mass faster.
            per_clot = weight.sum(axis=0)
            damp = np.where(
                per_clot > 0,
                np.minimum(1.0, self.lysis_saturation / np.maximum(per_clot, 1e-8)),
                0.0,
            )
            effective = weight * damp[None, :]
            removed_per_clot = np.minimum(
                self.lysis_rate * effective.sum(axis=0), self.clot_masses
            )
            share = effective / np.maximum(effective.sum(axis=0, keepdims=True), 1e-8)
            agent_lysis = (share * removed_per_clot[None, :]).sum(axis=1).astype(np.float32)
            self.clot_masses = np.maximum(self.clot_masses - removed_per_clot, 0.0)
            contacts = int(contact.sum())
            if contacts > 0 and self._first_contact_step < 0:
                self._first_contact_step = self.steps

        # --- reward ----------------------------------------------------------
        new_target = self._assigned_clot()
        new_geo = self._geodesic_to_target(new_target)
        same_target = (prev_target >= 0) & (prev_target == new_target)
        approach = np.where(same_target, prev_geo - new_geo, 0.0)

        removed_mass = float(removed_per_clot.sum())
        # Parallel work on distinct clots is what "maximum efficiency" means
        # here, so reward the number of clots being attacked simultaneously.
        clots_engaged = int((removed_per_clot > 0).sum())

        # Milestone-style nonlinearity; per-step progress stays linear. A concave
        # per-step transform would make "spread the work over many steps" pay
        # better than "finish fast", which rewards stalling.
        total_mass = float(self.clot_initial_mass.sum()) or 1.0
        removal_rate = 1.0 - float(self.clot_masses.sum()) / total_mass
        milestone_bonus = 0.0
        for level, bonus in self._milestones:
            if removal_rate >= level and level not in self._milestones_hit:
                self._milestones_hit.add(level)
                milestone_bonus += bonus
        newly_cleared = int(((self.clot_masses <= 0) & (prev_masses > 0)).sum())

        team = (
            self.progress_scale * removed_mass
            + milestone_bonus
            + self.clot_cleared_bonus * newly_cleared
            + self.coverage_bonus * clots_engaged
            - self.step_cost
        )
        agent_rewards = (
            self.progress_scale * agent_lysis
            + self.approach_scale * approach
            - self.wall_collision_penalty * wall_hits
            - self.robot_collision_penalty * peer_hits
        ).astype(np.float32)

        self.steps += 1
        success = bool(np.all(self.clot_masses <= 0))
        terminated = success
        truncated = self.steps >= self.horizon
        if success:
            team += self.success_bonus
            agent_rewards += self.success_bonus / self.num_robots

        if self.reward_double_count == "off":
            agent_rewards -= self.progress_scale * agent_lysis + self.success_bonus / self.num_robots * success
        reward = float(team + agent_rewards.mean())

        if self.use_pybullet:
            self._sync_pybullet(rebuild=False)

        info = {
            "scenario": self.active_scenario,
            "success": success,
            "agent_rewards": agent_rewards,
            "team_reward": float(team),
            "removed_mass": removed_mass,
            "remaining_mass": float(self.clot_masses.sum()),
            "removal_rate": removal_rate,
            "active_clots": int((self.clot_masses > 0).sum()),
            "clots_engaged": clots_engaged,
            "active_contacts": contacts,
            "wall_collisions": int(wall_hits.sum()),
            "robot_collisions": collision_pairs,
            "collision_rate": collision_pairs / max(self.num_robots, 1),
            "path_length": self.path_length,
            "first_contact_step": self._first_contact_step,
            "difficulty": self._difficulty,
        }
        return self._build_observation(), reward, terminated, truncated, info

    # -------------------------------------------------------------- pybullet

    def _sync_pybullet(self, rebuild: bool) -> None:
        """Create/refresh the PyBullet scene. Import is lazy and optional."""
        try:
            import pybullet as p
        except ImportError:
            self.use_pybullet = False
            return

        if self._pb is None:
            # render_mode="human" opens an interactive OpenGL window (local
            # machine only -- a headless server has no GL context and must use
            # DIRECT, which renders off-screen via the software rasterizer).
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 0, physicsClientId=self._client
                )
                p.resetDebugVisualizerCamera(
                    cameraDistance=1.1, cameraYaw=45, cameraPitch=-25,
                    cameraTargetPosition=[0.5, 0.5, 0.5],
                    physicsClientId=self._client,
                )
            else:
                self._client = p.connect(p.DIRECT)
            p.setGravity(0, 0, 0, physicsClientId=self._client)
            self._pb = p
            self._robot_bodies: list[int] = []
            self._clot_bodies: list[int] = []
            self._vessel_bodies: list[int] = []
            self._helix_renderer = None

        p, cid = self._pb, self._client

        if rebuild:
            # Continuous capsule lumen instead of a sparse chain of translucent
            # spheres: the wall reads as a vessel, and the taper, branching and
            # any stenosis are visible as shape rather than as bead spacing.
            from environments.vessel_render import VesselRenderer, VesselStyle

            if self._renderer is None:
                self._renderer = VesselRenderer(p, cid, style=VesselStyle())
            self._renderer.build_vessel(
                self.tree.points, self.tree.radii, self.tree.branch_ids
            )
            self._vessel_bodies = self._renderer.vessel_bodies

        if self._renderer is None:
            from environments.vessel_render import VesselRenderer, VesselStyle
            self._renderer = VesselRenderer(p, cid, style=VesselStyle())
            self._renderer.build_vessel(
                self.tree.points, self.tree.radii, self.tree.branch_ids
            )
            self._vessel_bodies = self._renderer.vessel_bodies

        # Which robots are close enough to a live clot to be lysing it. Same
        # radius the lysis model uses, so the colour cannot disagree with the
        # reward.
        contacting = np.zeros((self.num_robots,), dtype=bool)
        if self.active_clots > 0:
            alive = self.clot_masses[: self.active_clots] > 0
            if np.any(alive):
                d = np.linalg.norm(
                    self.robot_positions[:, None, :]
                    - self.clot_positions[: self.active_clots][None, :, :],
                    axis=2,
                )
                d[:, ~alive] = np.inf
                contacting = d.min(axis=1) <= self.clot_contact_radius

        self._renderer.update_robots(self.robot_positions, contacting)

        if self.active_clots > 0:
            frac = self.clot_masses[: self.active_clots] / np.maximum(
                self.clot_initial_mass[: self.active_clots], 1e-8
            )
            self._renderer.update_clots(
                self.clot_positions[: self.active_clots], frac,
                base_radius=self.clot_contact_radius * 0.7,
            )
        self._clot_bodies = self._renderer.clot_bodies

    def render(self) -> np.ndarray | None:
        if self.render_mode not in ("rgb_array", "human"):
            return None
        self._sync_pybullet(rebuild=False)
        if not self.use_pybullet:
            return None
        # In human mode the GUI window draws itself; there is no buffer to hand
        # back, but the sync above is what actually advances what you see.
        if self.render_mode == "human":
            return None
        p, cid = self._pb, self._client
        width, height = 640, 480
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0.5, 0.5], distance=1.1,
            yaw=45, pitch=-25, roll=0, upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=width / height, nearVal=0.05, farVal=3.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=cid,
        )
        return np.reshape(np.asarray(rgb, np.uint8), (height, width, 4))[:, :, :3]

    def close(self) -> None:
        if self._pb is not None:
            try:
                self._pb.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._pb = None
