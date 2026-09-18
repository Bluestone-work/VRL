"""Territory-balanced batching built from vectorized vascular environments."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from environments.vector_env import VectorVascularEnv
from environments.vessel_geometry import resolve_pool


class BalancedVectorVascularEnv:
    """Keep every requested territory represented in every policy rollout.

    Each child batch shares one geometry, while different children use different
    territories. This retains numpy vectorization without allowing a random pool
    to omit rare territories for an entire run.
    """

    def __init__(
        self,
        n_envs: int = 64,
        scenario_pool: str | Sequence[str] = "anatomical",
        tree_resample_interval: int = 900,
        seed: int = 0,
        **env_kwargs,
    ) -> None:
        self.scenarios = tuple(resolve_pool(scenario_pool))
        if n_envs < len(self.scenarios):
            raise ValueError(
                f"n_envs={n_envs} must be at least the {len(self.scenarios)} "
                "territories in the selected pool"
            )
        self.n_envs = int(n_envs)
        self.num_robots = int(env_kwargs.get("num_robots", 3))
        self.num_clots = int(env_kwargs.get("num_clots", 3))
        self.horizon = int(env_kwargs.get("horizon", 300))

        counts = np.full(len(self.scenarios), n_envs // len(self.scenarios), dtype=int)
        remainder = n_envs % len(self.scenarios)
        rotation = seed % len(self.scenarios)
        counts[(np.arange(remainder) + rotation) % len(self.scenarios)] += 1
        self.group_sizes = tuple(int(v) for v in counts)

        self.envs = []
        self.scenario_ids = []
        for group, (scenario, count) in enumerate(zip(self.scenarios, counts)):
            child = VectorVascularEnv(
                n_envs=int(count),
                scenario=scenario,
                scenario_pool=[scenario],
                randomize_scenario=False,
                tree_resample_interval=tree_resample_interval,
                seed=seed + 1009 * group,
                **env_kwargs,
            )
            self.envs.append(child)
            self.scenario_ids.append(np.full(int(count), group, dtype=np.int16))
        self.scenario_ids = np.concatenate(self.scenario_ids)
        self.scenario_names = np.concatenate([
            np.full(count, scenario, dtype=f"U{max(32, len(scenario))}")
            for scenario, count in zip(self.scenarios, self.group_sizes)
        ])

    @property
    def robot_positions(self) -> np.ndarray:
        return np.concatenate([env.robot_positions for env in self.envs], axis=0)

    @property
    def robot_velocities(self) -> np.ndarray:
        return np.concatenate([env.robot_velocities for env in self.envs], axis=0)

    @property
    def steps(self) -> np.ndarray:
        return np.concatenate([env.steps for env in self.envs], axis=0)

    @property
    def geometry_ids(self) -> np.ndarray:
        return np.concatenate([
            np.full(env.n_envs, group * 1_000_000 + env.active_geometry_id,
                    dtype=np.int64)
            for group, env in enumerate(self.envs)
        ])

    @property
    def geometry_features(self) -> np.ndarray:
        features = []
        for group, env in enumerate(self.envs):
            tree = env.tree
            bbox = tree.points.max(axis=0) - tree.points.min(axis=0)
            chord = float(np.linalg.norm(tree.points[-1] - tree.points[0]))
            feature = np.asarray([
                tree.total_length,
                tree.n_stations / 1000.0,
                len(tree.branches) / 20.0,
                tree.radii.min() / env.tube_radius,
                tree.radii.max() / env.tube_radius,
                tree.radii.mean() / env.tube_radius,
                tree.radii.std() / env.tube_radius,
                bbox[0], bbox[1], bbox[2],
                chord / max(tree.total_length, 1e-8),
                group / max(len(self.envs) - 1, 1),
            ], dtype=np.float32)
            features.append(np.repeat(feature[None, :], env.n_envs, axis=0))
        return np.concatenate(features, axis=0)

    @property
    def difficulty(self) -> float:
        difficulties = [env._difficulty for env in self.envs]
        if not np.allclose(difficulties, difficulties[0]):
            raise RuntimeError("territory groups have inconsistent curriculum difficulty")
        return float(difficulties[0])

    def set_difficulty(self, difficulty: float) -> None:
        """Apply one curriculum stage consistently to every territory group."""
        for env in self.envs:
            env.set_difficulty(difficulty)

    @staticmethod
    def _concat_obs(observations: list[dict[str, np.ndarray]]):
        return {
            key: np.concatenate([obs[key] for obs in observations], axis=0)
            for key in observations[0]
        }

    def reset_all(self) -> dict[str, np.ndarray]:
        return self._concat_obs([env.reset_all() for env in self.envs])

    def observe(self) -> dict[str, np.ndarray]:
        return self._concat_obs([env._observe() for env in self.envs])

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        observations = []
        rewards = []
        terminated = []
        truncated = []
        infos = []
        offset = 0
        for env in self.envs:
            result = env.step(action[offset:offset + env.n_envs])
            obs, reward, term, trunc, info = result
            observations.append(obs)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
            offset += env.n_envs

        obs = self._concat_obs(observations)
        term = np.concatenate(terminated)
        trunc = np.concatenate(truncated)
        info: dict[str, Any] = {}
        for key in (
            "agent_rewards", "team_reward", "success", "removal_rate",
            "wall_collisions", "clots_engaged", "first_contact_step",
            "contact_miss",
        ):
            info[key] = np.concatenate([np.asarray(item[key]) for item in infos], axis=0)
        info["scenario"] = self.scenario_names.copy()
        info["scenario_id"] = self.scenario_ids.copy()
        info["geometry_id"] = np.concatenate([
            np.full(env.n_envs, group * 1_000_000 + int(item["geometry_id"]),
                    dtype=np.int64)
            for group, (env, item) in enumerate(zip(self.envs, infos))
        ])
        info["tree_resampled"] = np.concatenate([
            np.full(env.n_envs, bool(item.get("tree_resampled", False)))
            for env, item in zip(self.envs, infos)
        ])

        if any("final_observation" in item for item in infos):
            finals = [
                item.get("final_observation", child_obs)
                for item, child_obs in zip(infos, observations)
            ]
            info["final_observation"] = self._concat_obs(finals)
            final_positions = []
            final_velocities = []
            for env, item in zip(self.envs, infos):
                context = item.get("final_context")
                final_positions.append(
                    env.robot_positions if context is None else context["positions"]
                )
                final_velocities.append(
                    env.robot_velocities if context is None else context["velocities"]
                )
            info["final_context"] = {
                "positions": np.concatenate(final_positions, axis=0),
                "velocities": np.concatenate(final_velocities, axis=0),
            }
        return obs, np.concatenate(rewards), term, trunc, info

    def state_dict(self) -> dict[str, Any]:
        return {"children": [env.state_dict() for env in self.envs]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if len(state["children"]) != len(self.envs):
            raise ValueError("checkpoint territory groups do not match this environment")
        for env, child_state in zip(self.envs, state["children"]):
            env.load_state_dict(child_state)

    def close(self) -> None:
        for env in self.envs:
            env.close()
