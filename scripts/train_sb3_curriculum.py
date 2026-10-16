"""EXP0114: matched SB3 curriculum/forecast ablations on training anatomies."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from marl.sb3_curriculum import (DIFFICULTIES, OBSTACLE_DIM, OBSTACLE_SLOTS,
                                 ObstacleBelief, PredictiveTransformerExtractor)
from scripts.train_sb3_vascular import ALGORITHMS, RunCallback, VascularSB3Env, build_model

GAMMA = .9995


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


class CurriculumVascularEnv(VascularSB3Env):
    def __init__(self, seed=41, clusters=3, horizon=300., budget=131072,
                 curriculum=True, prediction=True, shaping=True, reward_mode="potential",
                 uniform_only=False, episode_log=None, navigation_guidance=0.,
                 navigation_slow=False, zero_latency=False,
                 preserve_action_magnitude=False, direct_action=False):
        super().__init__(seed=seed, clusters=clusters, horizon=horizon,
                         preserve_action_magnitude=preserve_action_magnitude,
                         direct_action=direct_action)
        self.episode_log_path = Path(episode_log) if episode_log else None
        self.budget, self.curriculum = budget, curriculum
        self.curriculum_scene_base = 2840000000 + seed * 100000
        self.prediction, self.shaping = prediction, shaping
        if reward_mode not in ("potential", "drone_nav", "navigation"):
            raise ValueError("reward_mode must be potential, drone_nav or navigation")
        self.reward_mode = reward_mode
        self.uniform_only = bool(uniform_only)
        self.navigation_guidance = float(np.clip(navigation_guidance, 0., 1.))
        # The route follower's slow mode adds an observable-map centerline
        # recovery term and reduces speed in narrow lumen sections.  It is a
        # deployable low-level safety primitive; PPO still learns the joint
        # residual and obstacle response above it.
        self.navigation_slow = bool(navigation_slow)
        self.zero_latency = bool(zero_latency)
        self.frontier = 0
        self.recent = deque(maxlen=6)
        self.transitions = []
        self.anatomy_bag = []
        self.difficulty = 0
        self.last_task_return = self.last_shaping_return = 0.
        self.last_action = np.zeros(clusters * 3, np.float32)
        self.drone_return = 0.
        self.drone_components = dict(progress=0., goal=0., collision=0., near=0., wall=0.,
                                     lost=0., smooth=0., time=0.)
        self.nav_targets = np.full(clusters, -1, np.int64)
        self.nav_reached = np.zeros(clusters, bool)
        self.nav_previous_distance = np.full(clusters, np.inf)
        self.nav_return = 0.
        self.nav_goal_tolerance_mm = .30
        n = clusters
        self.observation_space.spaces.update({
            "obstacles": spaces.Box(-10., 10., (n, OBSTACLE_SLOTS, OBSTACLE_DIM), np.float32),
            "obstacle_valid": spaces.Box(0., 1., (n, OBSTACLE_SLOTS), np.float32),
            "peers": spaces.Box(-10., 10., (n, n, 8), np.float32),
            "peer_valid": spaces.Box(0., 1., (n, n), np.float32),
        })

    def _promote(self):
        if not self.curriculum:
            return
        scheduled = min(3, int(4 * self.total_environment_steps / max(self.budget, 1)))
        if self.reward_mode == "navigation":
            # Navigation has no lysis signal.  Require navigation mastery
            # before promoting so difficulty is earned by the actual task.
            # ``zero_latency`` only changes the sensing/command latency; it
            # must not change the success definition or curriculum semantics
            # of the potential/removal task.
            mastered = (len(self.recent) == 6 and
                        np.mean([bool(r.get("navigation_success", r.get("task_success", False)))
                                 for r in self.recent]) >= .8 and
                        all(int(r.get("lost", 0)) == 0 for r in self.recent) and
                        np.mean([float(r.get("wall_contact_s", 0.)) for r in self.recent]) <= 20. and
                        np.mean([float(r.get("particle_collision_events", 0.)) for r in self.recent]) <= .5)
            next_stage = min(3, self.frontier + int(mastered))
        else:
            mastered = (len(self.recent) == 6 and
                        np.mean([r["removal"] for r in self.recent]) >= [.1, .3, .6, .9][self.frontier]
                        and all(r["lost"] == 0 for r in self.recent)
                        and np.mean([r["particle_contact_s"] for r in self.recent]) <= .1)
            next_stage = min(3, max(scheduled, self.frontier + int(mastered)))
        if next_stage > self.frontier:
            self.transitions.append(dict(step=self.total_environment_steps, previous=self.frontier,
                                         stage=next_stage, reason="mastery" if mastered else "budget_exposure"))
            self.frontier = next_stage
            self.recent.clear()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.anatomy_bag = []
        self._promote()
        options = dict(options or {})
        if "difficulty" in options:
            difficulty = int(options.pop("difficulty"))
        elif not self.curriculum:
            difficulty = int(self.rng.integers(4))
        else:
            draw = self.rng.random()
            difficulty = (int(self.rng.integers(4)) if draw < .1 else
                          int(self.rng.integers(self.frontier + 1)) if draw < .3 else self.frontier)
        if not 0 <= difficulty < len(DIFFICULTIES):
            raise ValueError("difficulty must be 0..3")
        self.difficulty = difficulty
        self.scenario_family = "uniform" if self.uniform_only else (options.pop("scenario_family", None) or str(self.rng.choice(["uniform", "route_encounter"])))
        if self.scenario_family not in ("uniform", "route_encounter"):
            raise ValueError("unknown scenario family")
        self.scene_initialized = False
        level = DIFFICULTIES[difficulty]
        self.dynamic_obstacle_count = level["obstacles"]
        if not options.get("anatomy"):
            if not self.anatomy_bag:
                self.anatomy_bag = list(self.rng.permutation(self.train_anatomies))
            options["anatomy"] = str(self.anatomy_bag.pop())
        options.setdefault("flow", float(self.rng.choice(level["flows"])))
        if self.reward_mode == "navigation" or self.zero_latency:
            # Navigation studies isolate flow and moving-obstacle effects.
            # Image/command latency is held at zero in every navigation
            # scene; the original latency curriculum remains available for
            # the lysis/drone reward experiments.
            options["latency"] = 0
        else:
            options.setdefault("latency", int(self.rng.choice(level["latencies"])))
        self.belief = ObstacleBelief(self.clusters, self.prediction)
        self.task_return = self.shaping_return = 0.
        self.drone_return = 0.
        self.last_action = np.zeros(self.clusters * 3, np.float32)
        self.drone_components = dict(progress=0., goal=0., collision=0., near=0., wall=0.,
                                     lost=0., smooth=0., time=0.)
        # Avoid reseeding parent after sampling from the same seeded RNG.
        if seed is not None:
            self.scene_base = 2840000000 + seed * 100000
            self.episode_index = 0
        else:
            # Separate EXP0114 training scene namespace from EXP0113.
            self.scene_base = self.curriculum_scene_base
        self.curriculum_scene_base = self.scene_base
        obs, info = super().reset(options=options)
        if self.reward_mode == "navigation":
            self.ep.ctl.slow = self.navigation_slow
        self.scene.update(difficulty=difficulty, difficulty_name=level["name"],
                          dynamic_obstacles=level["obstacles"], frontier=self.frontier,
                          scenario_family=self.scenario_family,
                          localized_obstacles=self.localized_obstacles)
        self.potential = self._potential()
        if self.reward_mode == "navigation":
            # Freeze the episode assignment and remove lysis from this task:
            # success means reaching the assigned target points.
            self.nav_targets = np.asarray(self.current[1], np.int64).copy()
            if np.any(self.nav_targets < 0):
                # The deployable schedule can intentionally leave a slot
                # unassigned when a previous task has no live clot. A fresh
                # navigation episode still has all targets alive; fall back
                # to the physical geodesic assignment rather than failing
                # the reset contract.
                self.nav_targets = np.asarray(self.ep.env._assigned_targets(), np.int64).copy()
            if np.any(self.nav_targets < 0):
                raise RuntimeError("navigation reset produced an unassigned target")
            self.ep.env.config = replace(self.ep.env.config, lysis_mass_per_s=0.)
            self.nav_reached[:] = False
            self.nav_previous_distance = self._navigation_distances()
            self.nav_return = 0.
        info["scene"] = dict(self.scene)
        return obs, info

    def _navigation_distances(self):
        matrix = self.ep.env._target_distances()
        ids = np.clip(self.nav_targets, 0, max(matrix.shape[1] - 1, 0))
        distance = matrix[np.arange(self.clusters), ids].astype(float)
        distance[~self.ep.env.active[:self.clusters]] = np.inf
        return distance

    def _observe(self):
        if not self.scene_initialized:
            self._initialize_encounters()
            self.scene_initialized = True
        obs = super()._observe()
        est, targets, hold, frames = self.current
        if not self.prediction:
            obs["ctx"][:, 19:25] = 0.
        radius = float(self.ep.env.config.robot_radius_mm + self.ep.env.config.particle_radius_mm)
        obstacle, valid = self.belief.observe(est, frames, float(self.ep.env.elapsed_s), radius)
        n = self.clusters
        peers = np.zeros((n, n, 8), np.float32)
        peer_valid = np.asarray(est.peers_vis, np.float32).copy()
        for i in range(n):
            for j in range(n):
                if peer_valid[i, j]:
                    peers[i, j] = np.r_[frames[i] @ (est.pos[j] - est.pos[i]) / 2.,
                                         frames[i] @ (est.vel[j] - est.vel[i]) / 2.,
                                         est.active[j], hold[j]]
        obs.update(obstacles=obstacle, obstacle_valid=valid,
                   peers=np.clip(peers, -10., 10.), peer_valid=peer_valid)
        return obs

    def _initialize_encounters(self):
        """Simulator-side scene generation, before the first image is acquired.

        Localized particles are finite non-overlapping bodies placed near robot
        starts on map centreline segments. Physics and the image pipeline are
        unchanged; this does not give the actor the scene's true positions.
        This is a synthetic encounter distribution, not a clinical claim.
        """
        from scripts.multicluster_protocol import digest
        self.localized_obstacles = 0
        if self.scenario_family == "uniform":
            return
        ep, n = self.ep, self.clusters
        env = ep.env
        rng = np.random.default_rng([ep.seed, 114, 1])
        starts = env.positions_mm[:n].copy()
        positions = env.positions_mm[n:].copy()
        points = env.transport.points
        candidates = []
        for i in range(n):
            dist = np.linalg.norm(points - starts[i], axis=1)
            choices = np.flatnonzero((dist >= .3) & (dist <= 1.4))
            candidates.append(list(rng.permutation(choices)))
        for k in range(min(len(positions), 4 * n)):
            choices = candidates[k % n]
            while choices:
                point = points[choices.pop()] + rng.normal(0., .015, 3)
                if np.linalg.norm(starts - point, axis=1).min() < .25:
                    continue
                if len(positions) > 1 and np.linalg.norm(np.delete(positions, k, axis=0) - point, axis=1).min() < .05:
                    continue
                edge = env.transport.nearest_edges(point[None])
                _, lumen, radial, _ = env.transport.coordinates(point[None], edge, env.solution)
                if radial[0] + env.config.particle_radius_mm >= lumen[0] - 1e-6:
                    continue
                positions[k] = point
                self.localized_obstacles += 1
                break
        env.positions_mm[n:] = positions
        env.edges[n:] = env.transport.nearest_edges(positions)
        env._sync_public_state()
        ep.manifest = dict(ep.manifest, original_scenario_hash=ep.manifest["scenario_hash"],
                           scenario_family=self.scenario_family, localized_obstacles=self.localized_obstacles)
        ep.manifest["scenario_hash"] = digest(dict(parent=ep.manifest["original_scenario_hash"],
                                                   family=self.scenario_family, particle_positions=positions.tolist()))

    def _potential(self):
        """Observable remaining route distance, Phi(terminal)=0, no action prior."""
        est, targets, hold, frames = self.current
        distance = 0.
        for i, target in enumerate(targets):
            if not est.active[i] or target < 0:
                continue
            route = self.ep.ctl.route[i]
            if route is None:
                continue
            points = self.ep.ctl.pts[route]
            k = int(self.ep.ctl.prog[i])
            d = np.linalg.norm(est.pos[i] - points[k])
            d += np.linalg.norm(np.diff(points[k:], axis=0), axis=1).sum()
            distance += min(float(d), 60.)
        return -.2 * distance / self.clusters

    def step(self, action):
        if self.reward_mode == "navigation":
            before_distance = self._navigation_distances()
            active_before = self.ep.env.active[:self.clusters].copy()
            action = np.asarray(action, np.float32).reshape(-1)
            if self.navigation_guidance > 0.:
                # A deployable route follower supplies a bounded direction;
                # PPO learns a residual around it. This keeps the navigation
                # task identifiable while retaining learned collision-aware
                # corrections.
                rule = self.ep.ctl.act(self.nav_targets, self.current[0])
                action = ((1.-self.navigation_guidance) * action.reshape(self.clusters, 3)
                          + self.navigation_guidance * np.asarray(rule, np.float32)).reshape(-1)
            obs, task_reward, done, truncated, info = super().step(action)
            after_distance = self._navigation_distances()
            finite = np.isfinite(before_distance) & np.isfinite(after_distance)
            delta = np.zeros_like(before_distance)
            delta[finite] = before_distance[finite] - after_distance[finite]
            progress = np.clip(delta, -0.5, 0.5)
            newly_reached = (~self.nav_reached) & (after_distance <= self.nav_goal_tolerance_mm)
            self.nav_reached |= newly_reached
            lost = active_before & ~self.ep.env.active[:self.clusters]
            wall_vec = np.asarray(self.ep.info.get("wall_contact_s", np.zeros(self.clusters)), dtype=float).reshape(-1)
            particle_contact = np.asarray(self.ep.info.get("particle_contact_s", np.zeros(self.clusters)), dtype=float).reshape(-1)
            particle_events = np.asarray(self.ep.info.get("particle_collision_events", np.zeros(self.clusters)), dtype=float).reshape(-1)
            particle_near = np.asarray(self.ep.info.get("particle_near_s", np.zeros(self.clusters)), dtype=float).reshape(-1)
            smooth_vec = np.mean((action.reshape(self.clusters, 3) - self.last_action.reshape(self.clusters, 3)) ** 2, axis=1)
            # Per-cluster deployable navigation reward for IPPO.  Keep the
            # legacy scalar navigation reward below unchanged so existing
            # SB3 records remain comparable; IPPO consumes this decomposition.
            agent_reward = (1.5 * progress + 8. * newly_reached.astype(float)
                            - 2. * wall_vec - 8. * lost.astype(float)
                            - 2. * particle_contact - 5. * particle_events
                            - .5 * particle_near - .02 * smooth_vec - .01)
            wall = float(wall_vec.sum())
            smooth = float(np.mean((action - self.last_action) ** 2))
            nav_success = bool(np.all(self.nav_reached) and not np.any(lost))
            if nav_success:
                # Shared terminal credit is visible to every local IPPO
                # learner; without it, reaching the final target has no
                # reward beyond the clipped progress increments.
                agent_reward += 20.
            reward = (1.5 * float(progress.sum()) + 8. * float(newly_reached.sum())
                      + (20. if nav_success else 0.) - 2. * wall - 8. * float(lost.sum())
                      - .02 * smooth - .01)
            self.nav_return += reward
            self.return_sum += reward - task_reward
            self.nav_previous_distance = after_distance
            if nav_success and not done:
                done, truncated = True, False
                self.ended = True
            info.update(navigation_success=nav_success,
                        navigation_reached=int(self.nav_reached.sum()),
                        navigation_targets=self.nav_targets.copy(),
                        navigation_distances=after_distance.copy(),
                        navigation_return=float(self.nav_return),
                        reward_mode="navigation", difficulty=self.difficulty,
                        frontier=self.frontier,
                        observed_obstacles=int(obs["obstacle_valid"].sum()),
                        agent_rewards=agent_reward.astype(np.float32),
                        team_reward=0.0,
                        agent_wall_contact_s=wall_vec.astype(np.float32),
                        agent_particle_contact_s=particle_contact.astype(np.float32),
                        agent_particle_collision_events=particle_events.astype(np.float32),
                        agent_particle_near_s=particle_near.astype(np.float32),
                        agent_lost=lost.astype(np.float32))
            if done:
                row = info.get("episode_metrics") or self.ep.row(
                    "sb3_curriculum_navigation", return_sum=self.return_sum,
                    flow=self.scene["flow"], latency=self.scene["latency"],
                    environment_steps=self.ep.steps, agent_steps=self.episode_agent_steps,
                    wall_projection_changes=self.safety_changed)
                row.update(method="sb3_curriculum_navigation", task_mode="navigation",
                           difficulty=self.difficulty,
                           difficulty_name=DIFFICULTIES[self.difficulty]["name"],
                           frontier=self.frontier, curriculum=self.curriculum,
                           prediction=self.prediction, shaping=False,
                           scenario_family=self.scenario_family,
                           localized_obstacles=self.localized_obstacles,
                           navigation_success=nav_success,
                           navigation_reached=int(self.nav_reached.sum()),
                           navigation_targets=self.nav_targets.tolist(),
                           navigation_goal_tolerance_mm=self.nav_goal_tolerance_mm,
                           navigation_return=float(self.nav_return), reward_mode="navigation",
                           task_success=nav_success,
                           strict_success=bool(nav_success and row.get("wall_contact_s", 0.) < 1.),
                           relaxed_success=bool(nav_success and row.get("wall_contact_s", 0.) < 5.),
                           return_sum=float(self.return_sum))
                info["episode_metrics"] = row
                info["is_success"] = nav_success
                # Keep navigation episodes in the same rolling curriculum
                # buffer as the original lysis task.  Without this append,
                # navigation could never satisfy a task-specific mastery
                # rule and would always fall back to budget-based promotion.
                if self.difficulty == self.frontier:
                    self.recent.append(row)
                if self.episode_log_path:
                    with self.episode_log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.last_action = action.copy()
            return obs, float(reward), bool(done), bool(truncated), info
        before = self.potential
        action = np.asarray(action, np.float32).reshape(-1)
        mass_before = float(self.ep.env.masses.sum())
        active_before = self.ep.env.active[:self.clusters].copy()
        obs, task_reward, done, truncated, info = super().step(action)
        after = 0. if done else self._potential()
        shaping = GAMMA * after - before if (self.shaping and self.reward_mode == "potential") else 0.
        self.potential = after
        if self.reward_mode == "drone_nav":
            # Dense progress and terminal incentives follow common UAV/local
            # navigation rewards. Every term is observable or comes from the
            # environment's collision accounting; no future truth is exposed.
            progress_mm = float(np.clip((before - after) * self.clusters / .2, -2., 2.))
            removed = max(0., mass_before - float(self.ep.env.masses.sum())) / max(self.ep.initial, 1e-9)
            goal = 50. * removed
            if done and info.get("episode_metrics", {}).get("task_success", False):
                goal += 25.
            particle_contact = float(np.asarray(self.ep.info.get("particle_contact_s", 0.)).sum())
            particle_events = float(np.asarray(self.ep.info.get("particle_collision_events", 0.)).sum())
            near = float(np.asarray(self.ep.info.get("particle_near_s", 0.)).sum())
            wall = float(np.asarray(self.ep.info.get("wall_contact_s", 0.)).sum())
            lost = float(np.sum(active_before & ~self.ep.env.active[:self.clusters]))
            smooth = float(np.mean((action - self.last_action) ** 2))
            time_cost = .02
            # Collision terms dominate progress, as in drone collision
            # avoidance, while near-obstacle exposure is a softer warning.
            collision = -8. * particle_contact - 25. * particle_events
            near_cost = -1.5 * near
            reward = (1.5 * progress_mm + goal + collision + near_cost - 4. * wall
                      - 12. * lost - .05 * smooth - time_cost)
            self.drone_components = dict(progress=1.5 * progress_mm, goal=goal,
                collision=collision, near=near_cost, wall=-4. * wall,
                lost=-12. * lost, smooth=-.05 * smooth, time=-time_cost)
            self.drone_return += reward
            # The parent class has already added task_reward to return_sum.
            self.return_sum += reward - task_reward
            shaping = 0.
        else:
            self.return_sum += shaping
            reward = float(task_reward + shaping)
        self.task_return += task_reward
        self.shaping_return += shaping
        self.last_action = action.copy()
        info.update(difficulty=self.difficulty, frontier=self.frontier,
                    task_reward=task_reward, shaping_reward=shaping,
                    reward_mode=self.reward_mode, drone_reward=float(reward),
                    drone_components=dict(self.drone_components),
                    observed_obstacles=int(obs["obstacle_valid"].sum()))
        if done:
            row = info["episode_metrics"]
            row.update(method="sb3_curriculum_transformer", difficulty=self.difficulty,
                       difficulty_name=DIFFICULTIES[self.difficulty]["name"], frontier=self.frontier,
                       curriculum=self.curriculum, prediction=self.prediction, shaping=self.shaping,
                       reward_mode=self.reward_mode,
                       task_return=self.task_return, shaping_return=self.shaping_return,
                       drone_return=self.drone_return, drone_components=dict(self.drone_components),
                       return_sum=self.return_sum,
                       scenario_family=self.scenario_family, localized_obstacles=self.localized_obstacles,
                       predictor_diagnostics=self.belief.diagnostics())
            # The curriculum adapter owns this log (the parent is constructed
            # without episode_log), so append one immutable row per episode.
            # Keeping every row is required for success/collision/removal
            # rates and rolling TensorBoard summaries.
            if self.episode_log_path:
                path = self.episode_log_path
                encoded = json.dumps(row, ensure_ascii=False)
                with path.open("a", encoding="utf-8") as f:
                    f.write(encoded + "\n")
            if self.difficulty == self.frontier:
                self.recent.append(row)
            if self.episode_log_path:
                pass
        return obs, float(reward), done, truncated, info


class CurriculumCallback(RunCallback):
    def _on_step(self):
        result = super()._on_step()
        if self.n_calls % 256 == 0:
            env = self.training_env.envs[0].unwrapped
            path = self.out / "status.json"
            row = json.loads(path.read_text())
            row.update(frontier=env.frontier, difficulty=env.difficulty, scene=env.scene,
                       transitions=env.transitions, observed_obstacles=self.locals["infos"][0]["observed_obstacles"],
                       predictor_diagnostics=env.belief.diagnostics())
            atomic_json(path, row)
        return result


def predictive_model(algorithm, env, seed=41, device="cpu", smoke=False):
    # Reuse the registered SB3 hyperparameters, replacing only the extractor.
    # Passing the class via build_model is explicit; baseline defaults stay intact.
    return build_model(algorithm, env, seed, device, smoke,
                       extractor_class=PredictiveTransformerExtractor)


def evaluate(model, out, clusters=3, prediction=True, shaping=True, count=36, horizon=300., reward_mode="potential", uniform_only=False, navigation_guidance=0., navigation_slow=False, zero_latency=False):
    env = CurriculumVascularEnv(clusters=clusters, prediction=prediction, shaping=shaping,
                               curriculum=False, horizon=horizon, reward_mode=reward_mode,
                               uniform_only=uniform_only, navigation_guidance=navigation_guidance,
                               navigation_slow=navigation_slow, zero_latency=zero_latency)
    splits = json.loads((ROOT / "configs/evaluation_splits.json").read_text())
    pool = splits["multicluster_benchmark_v1"]["development"]
    rows = []
    try:
        with (out / "evaluation.jsonl").open("w") as f:
            for j in range(count):
                anatomy = env.train_anatomies[(j // 4) % len(env.train_anatomies)]
                level = j % 4
                repeat = j // (len(env.train_anatomies) * 4)
                options = dict(anatomy=anatomy, difficulty=level,
                               scenario_family="uniform" if uniform_only else ["uniform", "route_encounter"][(j // 4 + level) % 2],
                               scene_seed=pool["seed_base"] + splits["anatomy_order"].index(anatomy) * pool["stride_per_anatomy"] + 20 + repeat,
                               flow=DIFFICULTIES[level]["flows"][-1],
                               latency=0 if reward_mode == "navigation" or zero_latency else DIFFICULTIES[level]["latencies"][-1])
                obs, _ = env.reset(options=options)
                while True:
                    obs, reward, done, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
                    if done or truncated:
                        row = dict(info["episode_metrics"], split="multicluster_development", evaluation=True)
                        rows.append(row)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        break
        def aggregate(part):
            keys = ("strict_success", "task_success", "navigation_success", "removal",
                    "wall_contact_s", "lost", "particle_contact_s", "particle_collision_events")
            values = {key: float(np.mean([r[key] for r in part])) for key in keys
                      if part and key in part[0]}
            return dict(episodes=len(part), **values,
                        any_obstacle_collision=float(np.mean([r["particle_collision_events"] > 0 for r in part])))
        summary = dict(overall=aggregate(rows), by_difficulty={str(k): aggregate([r for r in rows if r["difficulty"] == k])
                       for k in range(4) if any(r["difficulty"] == k for r in rows)},
                       split="development_only", deterministic=True)
        atomic_json(out / "evaluation_summary.json", summary)
        return summary
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--timesteps", type=int, default=131072)
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--no-prediction", action="store_true")
    p.add_argument("--no-shaping", action="store_true")
    p.add_argument("--reward-mode", choices=("potential", "drone_nav", "navigation"), default="potential")
    p.add_argument("--uniform-only", action="store_true", help="use stable uniform particle scenes; disable synthetic route encounter placement")
    p.add_argument("--navigation-guidance", type=float, default=0., help="route-follower mix for navigation mode, 0..1")
    p.add_argument("--navigation-slow", action="store_true", help="enable the deployable narrow-lumen center/recovery primitive")
    p.add_argument("--no-latency", action="store_true", help="fix image/command latency to zero while retaining the selected task reward")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-count", type=int, default=36)
    a = p.parse_args()
    if (a.timesteps < 1 or not 1 <= a.eval_count <= 360 or not 0 <= a.seed <= 10000
            or not 0. <= a.navigation_guidance <= 1.):
        p.error("positive timesteps, eval-count 1..360, seed 0..10000 required")
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = dict(vars(a), out=str(a.out), framework="stable-baselines3 2.7.1", gamma=GAMMA,
               environment_steps_requested=512 if a.smoke else a.timesteps,
               difficulties=DIFFICULTIES, train_anatomies=json.loads((ROOT / "configs/evaluation_splits.json").read_text())["anatomy_holdout_v1"]["train"],
               curriculum=not a.no_curriculum, prediction=not a.no_prediction, shaping=not a.no_shaping,
               reward_mode=a.reward_mode,
               uniform_only=a.uniform_only,
               navigation_guidance=a.navigation_guidance,
               navigation_slow=a.navigation_slow,
               zero_latency=a.no_latency,
               navigation_latency=0 if a.reward_mode == "navigation" or a.no_latency else None,
               architecture="shared CNN+GRU; 4-head robot-obstacle/peer cross-attention transformer, 64 width",
               predictor="causal gated detection tracks; constant velocity, 0.2/0.5/1.0 s; heuristic uncertainty; not learned/calibrated",
               initialization="random actor/critic; no action prior, no pretrained model",
               budget_unit="actual joint physical environment step", horizon_s=300.,
               curriculum_rule="70% frontier, 20% uniformly from unlocked stages, 10% all difficulties; mastery or quarter-budget exposure; balanced anatomy bag",
               mastery_rule=("six frontier episodes: navigation success >=0.8, no lost, mean wall contact <=20s, "
                             "mean obstacle collision events <=0.5" if a.reward_mode == "navigation" else
                             "six frontier episodes: mean removal >=0.1/0.3/0.6, no lost, mean particle contact <=0.1s"),
               shaping_rule="gamma*Phi(next)-Phi(current); Phi=-0.2*mean capped remaining map route mm; terminal Phi=0",
               drone_reward_rule="1.5*clipped route progress mm + 50*removed fraction +25 successful terminal -8 particle-contact-s -25 collision-events -1.5 particle-near-s -4 wall-s -12 lost -0.05 action smoothness -.02 time",
               sealed_test_access=False)
    cfg["scenario_families"] = "50% original uniform particles; 50% synthetic route encounters, at most 4 particles per robot localized within 0.3..1.4mm, valid lumen and no initial overlaps; final 36 scenes balanced over families"
    cfg["source_sha256"] = {}
    for folder in ("scripts", "marl", "environments", "configs"):
        for source in sorted((ROOT / folder).glob("*.json" if folder == "configs" else "*.py")):
            name = str(source.relative_to(ROOT))
            cfg["source_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
            dest = a.out / "source_snapshot" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    atomic_json(a.out / "config.json", cfg)
    env = DummyVecEnv([lambda: Monitor(CurriculumVascularEnv(
        seed=a.seed, clusters=a.clusters, budget=cfg["environment_steps_requested"],
        curriculum=cfg["curriculum"], prediction=cfg["prediction"], shaping=cfg["shaping"],
        reward_mode=cfg["reward_mode"], uniform_only=cfg["uniform_only"],
        navigation_guidance=cfg["navigation_guidance"],
        navigation_slow=cfg["navigation_slow"],
        zero_latency=cfg["zero_latency"],
        episode_log=a.out / "episodes.jsonl"))])
    try:
        model = predictive_model(a.algorithm, env, a.seed, a.device, a.smoke)
        cfg["hyperparameters"] = {k: getattr(model, k) for k in
            ("gamma", "learning_rate", "n_steps", "n_epochs", "batch_size", "buffer_size", "learning_starts", "gradient_steps", "tau", "policy_delay") if hasattr(model, k)}
        cfg["train_freq"] = [model.train_freq.frequency, model.train_freq.unit.value] if hasattr(model, "train_freq") else None
        atomic_json(a.out / "config.json", cfg)
        # Keep CSV/JSON audit logs and emit TensorBoard scalars for live
        # monitoring during long single-seed runs.
        model.set_logger(configure(str(a.out), ["csv", "json", "tensorboard"]))
        model.save(a.out / "initial_model")
        initial = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        started = time.monotonic()
        model.learn(total_timesteps=cfg["environment_steps_requested"], callback=CurriculumCallback(a.out, 8192))
        model.save(a.out / "final_model")
        if a.algorithm != "ppo":
            model.save_replay_buffer(a.out / "replay_buffer.pkl")
        change = sum(float((v.detach().cpu() - initial[k]).square().sum()) for k, v in model.policy.state_dict().items()) ** .5
        summary = dict(environment_steps=model.num_timesteps, updates=model._n_updates,
                       parameter_change_l2=change, training_seconds=time.monotonic() - started,
                       checkpoint_resume="model/optimizer/replay; not exact in-flight environment/RNG")
        atomic_json(a.out / "TRAINING_COMPLETE.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        env.close()
    summary = evaluate(model, a.out, a.clusters, cfg["prediction"], cfg["shaping"], a.eval_count,
                       reward_mode=cfg["reward_mode"], uniform_only=cfg["uniform_only"],
                       navigation_guidance=cfg["navigation_guidance"],
                       navigation_slow=cfg["navigation_slow"], zero_latency=cfg["zero_latency"])
    atomic_json(a.out / "RUN_COMPLETE.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
