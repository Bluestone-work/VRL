"""Batched vectorized environment: many episodes advanced in lockstep.

The single-environment loop spends its time in Python and per-episode numpy calls
on arrays of shape [n_robots, 3] -- far too small to keep any modern CPU, let
alone a 4090, busy. Because the dynamics are already pure numpy, adding a leading
batch dimension turns every per-step operation into one call on
[n_envs, n_robots, 3], which is where the vectorization actually pays.

What is and is not shared
-------------------------
Each env keeps its own clots, robots and step counter, but the whole batch shares
ONE vessel tree. That is a deliberate trade: per-env geometry would force the
station lookups back into a Python loop, which is exactly the cost being removed.
Topology randomisation still happens -- `reset_all` resamples the shared tree --
it is just synchronised across the batch rather than independent per env. For
on-policy-style data collection that is fine; if you need fully independent
geometry, run several `VectorVascularEnv` instances.

Auto-reset semantics follow the usual vector-env convention: when an env
terminates or truncates, the observation returned for that slot is the FIRST
observation of the next episode, and the final observation of the episode that
just ended is preserved in `info["final_observation"]`. Bootstrapping code needs
that distinction to avoid learning across an episode boundary.

This module is intentionally dependency-free numpy. The next step up in
throughput is porting `step` to a Warp kernel (`pip install warp-lang`), at which
point the same interface can back onto the GPU; see README.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from environments.vessel_geometry import (
    SCENARIOS,
    build_vessel_tree,
    resolve_pool,
)


class VectorVascularEnv:
    """Vectorized thrombolysis environment sharing one vessel tree per batch."""

    def __init__(
        self,
        n_envs: int = 64,
        scenario: str = "bifurcation",
        num_robots: int = 3,
        num_clots: int = 3,
        horizon: int = 300,
        seed: int | None = None,
        randomize_scenario: bool = True,
        reward_mode: str = "milestone",
        obs_mode: str = "geometric",
        robot_radius: float = 0.0045,
        scenario_pool: str | Sequence[str] = "legacy",
        tree_resample_interval: int = 0,
    ) -> None:
        from environments.vascular_3d_marl_env import (
            LOOKAHEAD_OFFSETS,
            NODE_FEATURE_DIM_GEOMETRIC,
            NODE_FEATURE_DIM_LEGACY,
        )

        self.n_envs = int(n_envs)
        self.scenario = scenario
        self.num_robots = int(num_robots)
        self.num_clots = int(num_clots)
        self.horizon = int(horizon)
        self.randomize_scenario = bool(randomize_scenario)
        self.scenario_pool = resolve_pool(scenario_pool)
        if reward_mode not in ("baseline", "milestone"):
            raise ValueError(
                f"reward_mode must be 'baseline' or 'milestone', got {reward_mode!r}"
            )
        self.reward_mode = reward_mode
        # Steps between resampling the shared tree. 0 keeps the historical
        # behaviour (sample once at construction). A few multiples of the
        # horizon is the useful range: long enough that most episodes finish
        # inside one topology, short enough to cycle the pool during a run.
        self.tree_resample_interval = int(tree_resample_interval)
        self._steps_since_tree = 0
        self.obs_mode = obs_mode
        self._lookahead_offsets = LOOKAHEAD_OFFSETS
        self.node_feature_dim = (
            NODE_FEATURE_DIM_GEOMETRIC if obs_mode == "geometric"
            else NODE_FEATURE_DIM_LEGACY
        )

        # Physical constants mirror the single env; see that module for rationale.
        self.tube_radius = 0.055
        self.robot_radius = float(robot_radius)
        self.collision_distance = 2.0 * self.robot_radius
        self.max_speed = 0.018
        self.flow_speed = 0.004
        self.clot_contact_radius = 0.035
        self.lysis_rate = 0.018
        self.lysis_saturation = 4.0
        self.clot_occlusion = 0.65
        # See the single env: diffusion scales as 1/sqrt(device radius), so the
        # thermal term has to be rescaled with the device or its relative
        # influence changes silently.
        self.brownian_sigma = 0.0007 * float(np.sqrt(0.0045 / self.robot_radius))
        self.lubrication_range = 4.0 * self.robot_radius
        self.lubrication_floor = 0.35
        self.distal_margin = 0.92
        self.neighbor_radius = 0.06

        self.wall_collision_penalty = 0.1
        self.robot_collision_penalty = 0.1
        self.step_cost = 0.0
        self.progress_scale = 10.0
        self.approach_scale = 0.1
        self.coverage_bonus = 0.2
        self.success_bonus = 30.0
        self.clot_cleared_bonus = 3.0
        self._milestones = ((0.5, 2.0), (0.75, 3.0), (0.9, 5.0), (0.99, 10.0))
        if reward_mode == "baseline":
            self.success_bonus = 10.0
            self._milestones = ()
            self.clot_cleared_bonus = 0.0

        self._max_clot_slots = self.num_clots + 1
        self._rng = np.random.default_rng(seed)
        self._difficulty = 1.0
        self.tree = None
        self._geometry_generation = 0

        self.reset_all()

    # ------------------------------------------------------------------ setup

    def set_difficulty(self, difficulty: float) -> None:
        self._difficulty = float(np.clip(difficulty, 0.0, 1.0))

    def _sample_tree(self) -> None:
        scenario = (
            str(self._rng.choice(self.scenario_pool))
            if self.randomize_scenario else self.scenario
        )
        self.active_scenario = scenario
        self._geometry_generation += 1
        self.active_geometry_id = self._geometry_generation
        self.tree = build_vessel_tree(
            scenario, self._rng,
            base_radius=self.tube_radius, min_radius=3.0 * self.robot_radius,
        )
        # Route fields are per clot-station; cache keyed by station so envs that
        # happen to share a station share the Dijkstra result.
        self._route_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        arc = self.tree.arclength / max(self.tree.total_length, 1e-8)
        end_arc = np.zeros((self.tree.n_stations,), np.float32)
        for br in self.tree.branches:
            end_arc[br.start : br.stop + 1] = self.tree.arclength[br.stop]
        self._clot_eligible_arc = arc
        self._clot_room = end_arc - self.tree.arclength

    def _territory_clot_candidates(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return declared anatomical clot stations, weights and occlusions."""
        territory = getattr(self.tree, "territory", None)
        sites = [] if territory is None else list(territory.clot_sites)
        if not sites:
            return None

        stations: list[int] = []
        weights: list[float] = []
        occlusions: list[float] = []
        for site in sites:
            bid = self.tree.segment_index.get(site.segment)
            if bid is None:
                continue
            branch = self.tree.branches[bid]
            k = int(round(np.clip(site.position, 0.0, 1.0) * (branch.size - 1)))
            station = int(np.clip(branch.start + k, branch.start, branch.stop))
            room = self.tree.arclength[branch.stop] - self.tree.arclength[station]
            if room < self.clot_contact_radius and branch.size > 2:
                back = min(
                    branch.size - 1,
                    int(np.ceil(
                        self.clot_contact_radius
                        / max(
                            self.tree.total_length / max(self.tree.n_stations, 1),
                            1e-8,
                        )
                    )),
                )
                station = int(np.clip(branch.stop - back, branch.start, branch.stop))
            stations.append(station)
            weights.append(max(float(site.weight), 0.0))
            occlusions.append(float(site.occlusion))

        if not stations:
            return None
        probs = np.asarray(weights, dtype=np.float64)
        if probs.sum() <= 0:
            probs = np.ones_like(probs)
        probs /= probs.sum()
        return (
            np.asarray(stations, dtype=np.int32),
            probs,
            np.asarray(occlusions, dtype=np.float32),
        )

    def _route(self, station: int) -> tuple[np.ndarray, np.ndarray]:
        if station not in self._route_cache:
            self._route_cache[station] = self.tree.route_to(int(station))
        return self._route_cache[station]

    def reset_all(self) -> dict[str, np.ndarray]:
        """Resample the shared tree and reset every env."""
        self._sample_tree()
        n, r = self.n_envs, self.num_robots
        self.clot_stations = np.zeros((n, self._max_clot_slots), np.int32)
        self.clot_positions = np.zeros((n, self._max_clot_slots, 3), np.float32)
        self.clot_masses = np.zeros((n, self._max_clot_slots), np.float32)
        self.clot_initial = np.zeros((n, self._max_clot_slots), np.float32)
        self.clot_alive = np.zeros((n, self._max_clot_slots), bool)
        self.robot_positions = np.zeros((n, r, 3), np.float32)
        self.robot_stations = np.zeros((n, r), np.int32)
        self.robot_velocities = np.zeros((n, r, 3), np.float32)
        self.steps = np.zeros((n,), np.int32)
        self.first_contact = np.full((n,), -1, np.int32)
        self._milestone_hit = np.zeros((n, len(self._milestones)), bool)
        self._reset_envs(np.arange(n))
        self._steps_since_tree = 0
        return self._observe()

    def _reset_envs(self, idx: np.ndarray) -> None:
        """Reset the given env indices in place."""
        if idx.size == 0:
            return
        rng, tree = self._rng, self.tree

        # --- clots ---
        eligible = np.flatnonzero(
            (self._clot_eligible_arc >= 0.15 + 0.15 * self._difficulty)
            & (self._clot_eligible_arc <= min(0.45 + 0.55 * self._difficulty,
                                              self.distal_margin))
            & (self._clot_room >= self.clot_contact_radius)
        )
        if eligible.size == 0:
            eligible = np.flatnonzero(self._clot_room >= self.clot_contact_radius)
        if eligible.size == 0:
            eligible = np.arange(tree.n_stations, dtype=np.int32)

        counts = rng.integers(
            max(1, self.num_clots - 1), self.num_clots + 2, size=idx.size
        )
        counts = np.minimum(counts, self._max_clot_slots)
        if self._difficulty < 1.0:
            counts = np.maximum(
                1, np.round(1 + (counts - 1) * self._difficulty).astype(int)
            )

        self.clot_stations[idx] = 0
        self.clot_positions[idx] = 0.0
        self.clot_masses[idx] = 0.0
        self.clot_initial[idx] = 0.0
        self.clot_alive[idx] = False

        territory = self._territory_clot_candidates()
        if territory is not None:
            site_stations, site_probs, site_occlusions = territory
            for row, requested in zip(idx, counts):
                take = min(int(requested), site_stations.size, self._max_clot_slots)
                chosen = rng.choice(
                    site_stations.size, size=take, replace=False, p=site_probs
                )
                stations = site_stations[chosen]
                mass = (
                    site_occlusions[chosen] / 0.8
                    * rng.uniform(0.85, 1.15, size=take)
                ).astype(np.float32)
                radial = rng.normal(0.0, 0.10, size=(take, 3)).astype(np.float32)
                radial *= tree.radii[stations, None]
                positions = tree.points[stations] + radial
                positions, _out, stations, _ax = tree.project(
                    positions, robot_radius=0.0, hint=stations
                )
                self.clot_stations[row, :take] = stations
                self.clot_positions[row, :take] = positions
                self.clot_masses[row, :take] = mass
                self.clot_initial[row, :take] = mass
                self.clot_alive[row, :take] = True
        else:
            picks = rng.choice(eligible, size=(idx.size, self._max_clot_slots))
            self.clot_stations[idx] = picks
            for slot in range(self._max_clot_slots):
                active = counts > slot
                rows = idx[active]
                if rows.size == 0:
                    continue
                mass = rng.uniform(0.7, 1.3, size=rows.size).astype(np.float32)
                self.clot_masses[rows, slot] = mass
                self.clot_initial[rows, slot] = mass
                self.clot_alive[rows, slot] = True
            stations = self.clot_stations[idx]
            self.clot_positions[idx] = tree.points[stations]

        # --- robots: spread along the proximal trunk ---
        trunk = tree.stations_of_branch(0)
        span = max(int(0.25 * trunk.size), 2)
        sel = trunk[np.linspace(0, span - 1, self.num_robots).astype(np.int32)]
        anchor = tree.points[sel]                               # [R, 3]
        radial = rng.normal(0.0, 0.25, size=(idx.size, self.num_robots, 3))
        radial = radial.astype(np.float32) * tree.radii[sel][None, :, None]
        pos = anchor[None, :, :] + radial
        flat_hint = np.tile(sel, (idx.size, 1)).reshape(-1)
        clamped, _out, st, _ax = tree.project(
            pos.reshape(-1, 3), self.robot_radius, hint=flat_hint
        )
        self.robot_positions[idx] = clamped.reshape(idx.size, self.num_robots, 3)
        self.robot_stations[idx] = st.reshape(idx.size, self.num_robots)
        self.robot_velocities[idx] = 0.0
        self.steps[idx] = 0
        self.first_contact[idx] = -1
        self._milestone_hit[idx] = False

    # ------------------------------------------------------------- mechanics

    def _assign(self) -> np.ndarray:
        """Geodesic-nearest live clot per robot; -1 if the env is cleared."""
        n, r = self.n_envs, self.num_robots
        best = np.full((n, r), -1, np.int32)
        best_d = np.full((n, r), np.inf, np.float32)
        for slot in range(self._max_clot_slots):
            live = self.clot_alive[:, slot] & (self.clot_masses[:, slot] > 0)
            rows = np.flatnonzero(live)
            if rows.size == 0:
                continue
            # Group envs by clot station so each Dijkstra is reused.
            stations = self.clot_stations[rows, slot]
            for station in np.unique(stations):
                sub = rows[stations == station]
                dist, _hop = self._route(int(station))
                d = dist[self.robot_stations[sub]]
                better = d < best_d[sub]
                idx = sub[:, None]
                best_d[idx, np.arange(r)[None, :]] = np.where(
                    better, d, best_d[sub]
                )
                best[idx, np.arange(r)[None, :]] = np.where(
                    better, slot, best[sub]
                )
        return best

    def _geodesic(self, target: np.ndarray) -> np.ndarray:
        """Continuous along-vessel distance to each robot's assigned clot."""
        out = np.zeros((self.n_envs, self.num_robots), np.float32)
        tree = self.tree
        for slot in range(self._max_clot_slots):
            rows, cols = np.nonzero(target == slot)
            if rows.size == 0:
                continue
            stations = self.clot_stations[rows, slot]
            for station in np.unique(stations):
                m = stations == station
                rr, cc = rows[m], cols[m]
                dist, hop = self._route(int(station))
                st = self.robot_stations[rr, cc]
                nxt = hop[st]
                step_vec = tree.points[nxt] - tree.points[st]
                step_len = np.linalg.norm(step_vec, axis=1)
                valid = step_len > 1e-8
                unit = np.where(
                    valid[:, None], step_vec / np.maximum(step_len, 1e-8)[:, None], 0.0
                )
                offset = self.robot_positions[rr, cc] - tree.points[st]
                along = np.clip(np.sum(offset * unit, axis=1), -step_len, step_len)
                out[rr, cc] = dist[st] - np.where(valid, along, 0.0)
        return out

    def _occluded_radius(self, station: np.ndarray) -> np.ndarray:
        """[n_envs, n_robots] lumen radius including live clot narrowing."""
        tree = self.tree
        radius = tree.radii[station]                      # [E, R]
        arc_r = tree.arclength[station]                   # [E, R]
        arc_c = tree.arclength[self.clot_stations]        # [E, S]
        same = (
            tree.branch_ids[station][:, :, None]
            == tree.branch_ids[self.clot_stations][:, None, :]
        )
        d_arc = np.abs(arc_r[:, :, None] - arc_c[:, None, :])
        bump = np.exp(-0.5 * np.square(d_arc / self.clot_contact_radius)) * same
        frac = self.clot_masses / np.maximum(self.clot_initial, 1e-8)
        block = (bump * (frac * self.clot_alive)[:, None, :]).max(axis=2)
        return np.maximum(radius * (1.0 - self.clot_occlusion * block), 1e-4)

    def _flow_and_axis(self, positions, station, occluded):
        """Batched flow lookup; returns (flow, axis_point, lumen_radius)."""
        tree = self.tree
        e, r = station.shape
        flat_pos = positions.reshape(-1, 3)
        flat_st = station.reshape(-1)
        axis_point, lumen = tree._axis_point(flat_pos, flat_st)
        flow = tree.flow(
            flat_pos, flat_st, self.flow_speed, self.tube_radius,
            radius_override=occluded.reshape(-1),
        )
        return (
            flow.reshape(e, r, 3),
            axis_point.reshape(e, r, 3),
            lumen.reshape(e, r),
        )

    def _lubrication(self, positions, axis_point, lumen) -> np.ndarray:
        gap = np.maximum(
            lumen - np.linalg.norm(positions - axis_point, axis=2), 0.0
        )
        t = np.clip(gap / max(self.lubrication_range, 1e-8), 0.0, 1.0)
        return (self.lubrication_floor + (1.0 - self.lubrication_floor) * t).astype(
            np.float32
        )

    def _pair_terms(self):
        """Overlap counts and separation impulses, batched."""
        delta = self.robot_positions[:, :, None, :] - self.robot_positions[:, None, :, :]
        d = np.linalg.norm(delta, axis=3)
        eye = np.eye(self.num_robots, dtype=bool)[None, :, :]
        d = np.where(eye, np.inf, d)
        overlap = d < self.collision_distance
        depth = np.where(overlap, self.collision_distance - d, 0.0)
        direction = delta / np.maximum(d, 1e-8)[:, :, :, None]
        impulse = 0.5 * (direction * depth[:, :, :, None]).sum(axis=2)
        return d, overlap.sum(axis=2).astype(np.float32), impulse.astype(np.float32)

    # ---------------------------------------------------------------- gym-ish

    def step(self, action: np.ndarray):
        """Advance every env one step.

        Returns (obs, reward, terminated, truncated, info) with leading batch dim.
        `reward` is [n_envs]; `info["agent_rewards"]` is [n_envs, n_robots].
        """
        tree = self.tree
        e, r = self.n_envs, self.num_robots
        action = np.clip(
            np.asarray(action, np.float32).reshape(e, r, 3), -1.0, 1.0
        )

        prev_pos = self.robot_positions.copy()
        prev_st = self.robot_stations.copy()
        prev_target = self._assign()
        prev_geo = self._geodesic(prev_target)

        norm = np.linalg.norm(action, axis=2, keepdims=True)
        commanded = action / np.maximum(norm, 1e-8) * self.max_speed * np.clip(
            norm, 0.0, 1.0
        )

        occluded = self._occluded_radius(prev_st)
        flow, axis_point, lumen = self._flow_and_axis(prev_pos, prev_st, occluded)
        commanded = commanded * self._lubrication(prev_pos, axis_point, lumen)[:, :, None]

        noise = self._rng.normal(0.0, self.brownian_sigma, prev_pos.shape).astype(
            np.float32
        )
        _d, _hits, separation = self._pair_terms()
        proposed = prev_pos + commanded + flow + noise + separation

        clamped, outside, st, ax = tree.project(
            proposed.reshape(-1, 3), self.robot_radius, hint=prev_st.reshape(-1)
        )
        self.robot_positions = clamped.reshape(e, r, 3)
        self.robot_stations = st.reshape(e, r)
        wall_hits = outside.reshape(e, r).astype(np.float32)
        self.robot_velocities = (self.robot_positions - prev_pos).astype(np.float32)

        # Kill outward velocity on wall contact.
        axis = ax.reshape(e, r, 3)
        radial = self.robot_positions - axis
        outward = radial / np.maximum(
            np.linalg.norm(radial, axis=2, keepdims=True), 1e-8
        )
        vn = np.sum(self.robot_velocities * outward, axis=2, keepdims=True)
        self.robot_velocities = np.where(
            wall_hits[:, :, None] > 0,
            self.robot_velocities - np.maximum(vn, 0.0) * outward,
            self.robot_velocities,
        ).astype(np.float32)

        _d2, peer_hits, _imp = self._pair_terms()

        # --- lysis ---
        prev_mass = self.clot_masses.copy()
        dist = np.linalg.norm(
            self.robot_positions[:, :, None, :] - self.clot_positions[:, None, :, :],
            axis=3,
        )                                                   # [E, R, S]
        live = self.clot_alive & (self.clot_masses > 0)
        contact = (dist <= self.clot_contact_radius) & live[:, None, :]
        weight = np.exp(-np.square(dist / self.clot_contact_radius)) * contact
        per_clot = weight.sum(axis=1)                       # [E, S]
        damp = np.where(
            per_clot > 0,
            np.minimum(1.0, self.lysis_saturation / np.maximum(per_clot, 1e-8)),
            0.0,
        )
        effective = weight * damp[:, None, :]
        removed = np.minimum(
            self.lysis_rate * effective.sum(axis=1), self.clot_masses
        )
        share = effective / np.maximum(
            effective.sum(axis=1, keepdims=True), 1e-8
        )
        agent_lysis = (share * removed[:, None, :]).sum(axis=2).astype(np.float32)
        self.clot_masses = np.maximum(self.clot_masses - removed, 0.0)

        contacts = contact.any(axis=(1, 2))
        newly = contacts & (self.first_contact < 0)
        self.first_contact[newly] = self.steps[newly]

        # --- reward ---
        new_target = self._assign()
        new_geo = self._geodesic(new_target)
        same = (prev_target >= 0) & (prev_target == new_target)
        approach = np.where(same, prev_geo - new_geo, 0.0)

        removed_mass = removed.sum(axis=1)
        engaged = (removed > 0).sum(axis=1)
        total = np.maximum(self.clot_initial.sum(axis=1), 1e-8)
        removal_rate = 1.0 - self.clot_masses.sum(axis=1) / total

        milestone = np.zeros((e,), np.float32)
        for k, (level, bonus) in enumerate(self._milestones):
            hit = (removal_rate >= level) & ~self._milestone_hit[:, k]
            milestone += bonus * hit
            self._milestone_hit[:, k] |= hit
        cleared = ((self.clot_masses <= 0) & (prev_mass > 0)).sum(axis=1)

        team = (
            self.progress_scale * removed_mass
            + milestone
            + self.clot_cleared_bonus * cleared
            + self.coverage_bonus * engaged
            - self.step_cost
        ).astype(np.float32)
        agent_rewards = (
            self.progress_scale * agent_lysis
            + self.approach_scale * approach
            - self.wall_collision_penalty * wall_hits
            - self.robot_collision_penalty * peer_hits
        ).astype(np.float32)

        self.steps += 1
        success = (self.clot_masses.sum(axis=1) <= 0) & (
            self.clot_initial.sum(axis=1) > 0
        )
        terminated = success
        truncated = (self.steps >= self.horizon) & ~terminated
        team = team + self.success_bonus * success
        agent_rewards = agent_rewards + (
            self.success_bonus / r * success[:, None]
        )
        reward = (team + agent_rewards.mean(axis=1)).astype(np.float32)

        info: dict[str, Any] = {
            "agent_rewards": agent_rewards,
            "team_reward": team,
            "success": success,
            "removal_rate": removal_rate.astype(np.float32),
            "wall_collisions": wall_hits.sum(axis=1).astype(np.int32),
            "clots_engaged": engaged.astype(np.int32),
            "first_contact_step": self.first_contact.copy(),
            "contact_miss": self.first_contact < 0,
            "scenario": self.active_scenario,
            "geometry_id": self.active_geometry_id,
        }

        done = terminated | truncated
        # Periodic topology resampling. Without this the shared tree is sampled
        # once at construction and never again -- `_reset_envs` deliberately
        # keeps it, and training only calls `reset_all` at start-up and at
        # curriculum stage changes. That means a whole run trains on ONE
        # topology, so a scenario pool of 14 territories would contribute
        # exactly one of them. Resampling on a step schedule is what turns the
        # pool into actual variety.
        self._steps_since_tree += 1
        if (self.tree_resample_interval > 0
                and self._steps_since_tree >= self.tree_resample_interval):
            # Envs still mid-episode are cut short, so they must be reported as
            # truncated rather than terminated: the learner has to bootstrap
            # their value instead of treating the cut as a real episode end.
            truncated = truncated | ~done
            done = terminated | truncated
            info["final_observation"] = self._observe()
            info["final_context"] = {
                "positions": self.robot_positions.copy(),
                "velocities": self.robot_velocities.copy(),
            }
            info["tree_resampled"] = True
            self.reset_all()
            return self._observe(), reward, terminated, truncated, info

        if np.any(done):
            info["final_observation"] = self._observe()
            info["final_context"] = {
                "positions": self.robot_positions.copy(),
                "velocities": self.robot_velocities.copy(),
            }
            self._reset_envs(np.flatnonzero(done))
        return self._observe(), reward, terminated, truncated, info

    def state_dict(self) -> dict[str, Any]:
        """Serializable batched simulator state for exact continuation."""
        return {
            "tree": self.tree,
            "active_scenario": self.active_scenario,
            "active_geometry_id": self.active_geometry_id,
            "geometry_generation": self._geometry_generation,
            "steps_since_tree": self._steps_since_tree,
            "clot_eligible_arc": self._clot_eligible_arc.copy(),
            "clot_room": self._clot_room.copy(),
            "clot_stations": self.clot_stations.copy(),
            "clot_positions": self.clot_positions.copy(),
            "clot_masses": self.clot_masses.copy(),
            "clot_initial": self.clot_initial.copy(),
            "clot_alive": self.clot_alive.copy(),
            "robot_positions": self.robot_positions.copy(),
            "robot_stations": self.robot_stations.copy(),
            "robot_velocities": self.robot_velocities.copy(),
            "steps": self.steps.copy(),
            "first_contact": self.first_contact.copy(),
            "milestone_hit": self._milestone_hit.copy(),
            "difficulty": self._difficulty,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.tree = state["tree"]
        self.active_scenario = state["active_scenario"]
        self.active_geometry_id = int(state["active_geometry_id"])
        self._geometry_generation = int(state["geometry_generation"])
        self._steps_since_tree = int(state["steps_since_tree"])
        self._clot_eligible_arc = state["clot_eligible_arc"].copy()
        self._clot_room = state["clot_room"].copy()
        for name in (
            "clot_stations", "clot_positions", "clot_masses", "clot_initial",
            "clot_alive", "robot_positions", "robot_stations",
            "robot_velocities", "steps", "first_contact",
        ):
            setattr(self, name, state[name].copy())
        self._milestone_hit = state["milestone_hit"].copy()
        self._difficulty = float(state["difficulty"])
        self._rng.bit_generator.state = state["rng_state"]
        self._route_cache = {}

    # ------------------------------------------------------------ observation

    def _observe(self) -> dict[str, np.ndarray]:
        tree = self.tree
        e, r = self.n_envs, self.num_robots
        nodes = np.zeros((e, r, self.node_feature_dim), np.float32)
        target = self._assign()
        st = self.robot_stations

        occluded = self._occluded_radius(st)
        flow, axis_point, lumen = self._flow_and_axis(
            self.robot_positions, st, occluded
        )
        radial_vec = self.robot_positions - axis_point
        radial = np.linalg.norm(radial_vec, axis=2)
        clearance = 1.0 - np.clip(radial / np.maximum(lumen, 1e-8), 0.0, 1.0)

        has = target >= 0
        safe = np.where(has, target, 0)
        rows = np.arange(e)[:, None]
        clot_pos = self.clot_positions[rows, safe]
        delta = np.where(has[:, :, None], clot_pos - self.robot_positions, 0.0)
        cdist = np.where(has, np.linalg.norm(clot_pos - self.robot_positions, axis=2), 1.0)

        d, _hits, _imp = self._pair_terms()
        peer = np.argmin(d, axis=2)
        peer_delta = np.take_along_axis(
            self.robot_positions, peer[:, :, None], axis=1
        ) - self.robot_positions
        peer_dist = np.linalg.norm(peer_delta, axis=2)
        crowd = (d < self.neighbor_radius).sum(axis=2) / max(r - 1, 1)

        total = np.maximum(self.clot_initial.sum(axis=1), 1e-8)
        remaining = (self.clot_masses.sum(axis=1) / total)[:, None]
        tprog = (self.steps / self.horizon)[:, None]

        mass_frac = np.where(
            has,
            self.clot_masses[rows, safe]
            / np.maximum(self.clot_initial[rows, safe], 1e-8),
            0.0,
        )

        if self.obs_mode == "legacy":
            nodes[:, :, 0:3] = self.robot_positions * 2.0 - 1.0
            nodes[:, :, 3:6] = np.clip(self.robot_velocities / self.max_speed, -1, 1)
            nodes[:, :, 6:9] = np.clip(delta / 0.5, -1, 1)
            nodes[:, :, 9] = np.clip(cdist / 0.5, 0, 1)
            nodes[:, :, 10] = (cdist <= self.clot_contact_radius) & has
            nodes[:, :, 11] = mass_frac
            nodes[:, :, 12:15] = np.clip(peer_delta / self.neighbor_radius, -1, 1)
            nodes[:, :, 15] = np.clip(peer_dist / self.neighbor_radius, 0, 1)
            nodes[:, :, 16] = clearance
            nodes[:, :, 17] = tprog
            nodes[:, :, 18] = crowd
            nodes[:, :, 19] = remaining
        else:
            t_hat, n_hat, b_hat = tree.tangents[st], tree.normals[st], tree.binormals[st]

            def to_local(vec):
                return np.stack(
                    [
                        np.sum(vec * t_hat, axis=2),
                        np.sum(vec * n_hat, axis=2),
                        np.sum(vec * b_hat, axis=2),
                    ],
                    axis=2,
                ).astype(np.float32)

            look = np.zeros((e, r, len(self._lookahead_offsets), 3), np.float32)
            # Group by (clot slot, station) so each route is fetched once.
            for slot in range(self._max_clot_slots):
                rr, cc = np.nonzero(target == slot)
                if rr.size == 0:
                    continue
                stations = self.clot_stations[rr, slot]
                for station in np.unique(stations):
                    m = stations == station
                    _dist, hop = self._route(int(station))
                    look[rr[m], cc[m]] = tree.lookahead(
                        st[rr[m], cc[m]], self._lookahead_offsets, hop
                    )
            rr, cc = np.nonzero(~has)
            if rr.size:
                look[rr, cc] = tree.lookahead(
                    st[rr, cc], self._lookahead_offsets, None
                )

            look_rel = look - self.robot_positions[:, :, None, :]
            look_local = np.stack(
                [to_local(look_rel[:, :, k, :]) for k in range(look.shape[2])], axis=2
            )
            look_local = np.clip(look_local / 0.25, -1.0, 1.0)

            geo = self._geodesic(target)
            geo_norm = np.where(
                has, np.clip(geo / max(tree.total_length, 1e-8), 0.0, 1.0), 1.0
            )
            wall_dir = radial_vec / np.maximum(radial, 1e-8)[:, :, None]
            lube = self._lubrication(self.robot_positions, axis_point, lumen)

            nodes[:, :, 0:3] = self.robot_positions * 2.0 - 1.0
            nodes[:, :, 3:6] = np.clip(
                to_local(self.robot_velocities) / self.max_speed, -1, 1
            )
            nodes[:, :, 6:15] = look_local.reshape(e, r, -1)
            nodes[:, :, 15:18] = to_local(wall_dir)
            nodes[:, :, 18] = clearance
            nodes[:, :, 19] = np.clip(lumen / self.tube_radius, 0, 1)
            nodes[:, :, 20] = np.clip(occluded / np.maximum(lumen, 1e-8), 0, 1)
            nodes[:, :, 21:24] = np.clip(to_local(flow) / self.max_speed, -1, 1)
            nodes[:, :, 24] = lube
            nodes[:, :, 25:28] = np.clip(to_local(delta) / 0.5, -1, 1)
            nodes[:, :, 28] = geo_norm
            nodes[:, :, 29] = np.clip(cdist / 0.5, 0, 1)
            nodes[:, :, 30] = (cdist <= self.clot_contact_radius) & has
            nodes[:, :, 31] = mass_frac
            nodes[:, :, 32:35] = np.clip(to_local(peer_delta) / self.neighbor_radius, -1, 1)
            nodes[:, :, 35] = np.clip(peer_dist / self.neighbor_radius, 0, 1)

        adjacency = (d <= self.neighbor_radius).astype(np.float32)
        idx = np.arange(r)
        adjacency[:, idx, idx] = 1.0

        clot_state = np.zeros((e, self._max_clot_slots, 6), np.float32)
        clot_state[:, :, 0:3] = self.clot_positions * 2.0 - 1.0
        clot_state[:, :, 3] = self.clot_masses / np.maximum(self.clot_initial, 1e-8)
        clot_state[:, :, 4] = (self.clot_masses > 0).astype(np.float32)
        clot_state[:, :, 5] = (
            tree.arclength[self.clot_stations] / max(tree.total_length, 1e-8)
        )
        clot_state *= self.clot_alive[:, :, None]

        return {"nodes": nodes, "adjacency": adjacency, "clot_state": clot_state}

    def close(self) -> None:
        pass
