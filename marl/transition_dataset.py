"""Auditable, append-only transition shards for world-model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeTransitionWriter:
    """Persist batched transitions while retaining episode and geometry identity."""

    def __init__(
        self,
        root: str | Path,
        n_envs: int,
        run_id: int = 0,
        shard_size: int = 32768,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.n_envs = int(n_envs)
        self.run_id = int(run_id)
        self.shard_size = int(shard_size)
        existing = sorted(self.root.glob("transitions_*.npz"))
        self.shard_index = len(existing)
        self.total_written = sum(
            int(item.get("transitions", 0))
            for item in self._read_manifest()
        )
        self._episode_counter = self.total_written + self.run_id * 1_000_000_000
        self.episode_ids = np.arange(
            self._episode_counter,
            self._episode_counter + self.n_envs,
            dtype=np.int64,
        )
        self._episode_counter += self.n_envs
        self.episode_steps = np.zeros(self.n_envs, dtype=np.int32)
        self._buffer: dict[str, list[np.ndarray]] = {}
        self._buffered = 0

    def _read_manifest(self) -> list[dict[str, Any]]:
        path = self.root / "manifest.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def append(self, *, done: np.ndarray, **arrays: np.ndarray) -> None:
        done = np.asarray(done, dtype=bool)
        if done.shape != (self.n_envs,):
            raise ValueError(f"done must have shape {(self.n_envs,)}, got {done.shape}")
        payload = {
            key: np.asarray(value)
            for key, value in arrays.items()
        }
        payload["done"] = done
        payload["episode_id"] = self.episode_ids.copy()
        payload["episode_step"] = self.episode_steps.copy()
        for key, value in payload.items():
            if value.shape[0] != self.n_envs:
                raise ValueError(f"{key} has leading dimension {value.shape[0]}")
            self._buffer.setdefault(key, []).append(value.copy())
        self._buffered += self.n_envs

        self.episode_steps += 1
        finished = np.flatnonzero(done)
        for env_index in finished:
            self.episode_ids[env_index] = self._episode_counter
            self._episode_counter += 1
            self.episode_steps[env_index] = 0
        if self._buffered >= self.shard_size:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buffered:
            return None
        data = {
            key: np.concatenate(values, axis=0)
            for key, values in self._buffer.items()
        }
        path = self.root / f"transitions_{self.shard_index:05d}.npz"
        np.savez_compressed(path, **data)
        record = {
            "shard": path.name,
            "transitions": int(next(iter(data.values())).shape[0]),
            "fields": {key: list(value.shape) for key, value in data.items()},
        }
        with (self.root / "manifest.jsonl").open("a") as manifest:
            manifest.write(json.dumps(record, sort_keys=True) + "\n")
        self.total_written += record["transitions"]
        self.shard_index += 1
        self._buffer.clear()
        self._buffered = 0
        return path

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode_counter": self._episode_counter,
            "episode_ids": self.episode_ids.copy(),
            "episode_steps": self.episode_steps.copy(),
            "total_written": self.total_written,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._episode_counter = int(state["episode_counter"])
        self.episode_ids = state["episode_ids"].copy()
        self.episode_steps = state["episode_steps"].copy()

    def close(self) -> None:
        self.flush()


def load_transition_shards(root: str | Path) -> dict[str, np.ndarray]:
    paths = sorted(Path(root).glob("transitions_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no transition shards found under {root}")
    chunks: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path) as shard:
            for key in shard.files:
                chunks.setdefault(key, []).append(shard[key])
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def split_by_geometry(
    geometry_id: np.ndarray,
    seed: int = 0,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    """Split whole geometries, never individual correlated transitions."""
    unique = np.unique(geometry_id)
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    n_train = max(1, int(len(unique) * train_fraction))
    n_validation = max(1, int(len(unique) * validation_fraction))
    if n_train + n_validation >= len(unique):
        n_train = max(1, len(unique) - 2)
        n_validation = 1
    assignments = {
        "train": unique[:n_train],
        "validation": unique[n_train:n_train + n_validation],
        "test": unique[n_train + n_validation:],
    }
    return {
        name: np.flatnonzero(np.isin(geometry_id, ids))
        for name, ids in assignments.items()
    }
