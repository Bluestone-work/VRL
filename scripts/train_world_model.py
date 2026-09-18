"""Train and validate the graph dynamics ensemble on recorded real transitions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.transition_dataset import load_transition_shards, split_by_geometry
from marl.world_model import GraphWorldModelEnsemble, WorldModelConfig


MODEL_FIELDS = (
    "obs", "next_obs", "action", "rewards", "state", "next_state",
    "adjacency", "positions", "next_positions", "velocities",
    "next_velocities", "scenario_id", "geometry_features", "done",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", nargs="+", required=True)
    parser.add_argument("--action-field", default="action",
                        help="dataset field used as the simulator action")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-transitions", type=int, default=0)
    parser.add_argument("--dataset-start-fraction", type=float, default=0.0,
                        help="discard this leading fraction from each ordered dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_datasets(paths, max_transitions=0, start_fraction=0.0):
    if start_fraction < 0.0 or start_fraction >= 1.0:
        raise ValueError("dataset start fraction must be in [0, 1)")
    datasets = []
    provenance = []
    for dataset_index, path in enumerate(paths):
        data = load_transition_shards(path)
        raw_transitions = len(data["obs"])
        start = int(raw_transitions * start_fraction)
        data = {key: value[start:] for key, value in data.items()}
        data["geometry_id"] = (
            data["geometry_id"].astype(np.int64)
            + dataset_index * 1_000_000_000
        )
        datasets.append(data)
        provenance.append({
            "path": str(Path(path).resolve()),
            "raw_transitions": raw_transitions,
            "selected_start": start,
            "selected_transitions": len(data["obs"]),
            "selected_geometries": int(np.unique(data["geometry_id"]).size),
        })
    common = set.intersection(*(set(data) for data in datasets))
    merged = {
        key: np.concatenate([data[key] for data in datasets], axis=0)
        for key in common
    }
    if max_transitions and len(merged["obs"]) > max_transitions:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(
            len(merged["obs"]), size=max_transitions, replace=False
        ))
        merged = {key: value[keep] for key, value in merged.items()}
    missing = set(MODEL_FIELDS) - set(merged)
    if missing:
        raise ValueError(f"dataset is missing fields: {sorted(missing)}")
    return merged, provenance


def tensor_batch(data, indices, device):
    return {
        key: torch.as_tensor(data[key][indices], device=device)
        for key in MODEL_FIELDS
    }


@torch.no_grad()
def one_step_metrics(model, data, indices, device, max_samples=20000):
    if len(indices) > max_samples:
        indices = np.random.default_rng(0).choice(indices, max_samples, replace=False)
    totals = {
        "obs_squared_error": 0.0,
        "obs_persistence_squared_error": 0.0,
        "state_squared_error": 0.0,
        "reward_absolute_error": 0.0,
        "reward_zero_absolute_error": 0.0,
        "done_tp": 0,
        "done_fp": 0,
        "done_fn": 0,
        "count": 0,
        "uncertainty": 0.0,
    }
    uncertainty_values = []
    for start in range(0, len(indices), 2048):
        idx = indices[start:start + 2048]
        batch = tensor_batch(data, idx, device)
        prediction = model.predict(batch)
        count = len(idx)
        totals["obs_squared_error"] += float(
            (prediction["obs"] - batch["next_obs"]).square().mean().item() * count
        )
        totals["obs_persistence_squared_error"] += float(
            (batch["obs"] - batch["next_obs"]).square().mean().item() * count
        )
        totals["state_squared_error"] += float(
            (prediction["state"] - batch["next_state"]).square().mean().item() * count
        )
        totals["reward_absolute_error"] += float(
            (prediction["rewards"] - batch["rewards"]).abs().mean().item() * count
        )
        totals["reward_zero_absolute_error"] += float(
            batch["rewards"].abs().mean().item() * count
        )
        predicted_done = prediction["done_probability"] >= 0.5
        actual_done = batch["done"].bool()
        totals["done_tp"] += int((predicted_done & actual_done).sum().item())
        totals["done_fp"] += int((predicted_done & ~actual_done).sum().item())
        totals["done_fn"] += int((~predicted_done & actual_done).sum().item())
        totals["uncertainty"] += float(prediction["uncertainty"].mean().item() * count)
        uncertainty_values.append(prediction["uncertainty"].cpu().numpy())
        totals["count"] += count
    count = max(totals["count"], 1)
    precision = totals["done_tp"] / max(totals["done_tp"] + totals["done_fp"], 1)
    recall = totals["done_tp"] / max(totals["done_tp"] + totals["done_fn"], 1)
    reward_mae = totals["reward_absolute_error"] / count
    reward_zero_mae = totals["reward_zero_absolute_error"] / count
    uncertainty_values = np.concatenate(uncertainty_values)
    return {
        "obs_rmse": float(np.sqrt(totals["obs_squared_error"] / count)),
        "obs_persistence_rmse": float(np.sqrt(
            totals["obs_persistence_squared_error"] / count
        )),
        "state_rmse": float(np.sqrt(totals["state_squared_error"] / count)),
        "reward_mae": reward_mae,
        "reward_zero_mae": reward_zero_mae,
        "reward_relative_to_zero": reward_mae / max(reward_zero_mae, 1e-8),
        "done_f1": 2 * precision * recall / max(precision + recall, 1e-8),
        "mean_uncertainty": totals["uncertainty"] / count,
        "uncertainty_p95": float(np.quantile(uncertainty_values, 0.95)),
        "uncertainty_p99": float(np.quantile(uncertainty_values, 0.99)),
    }


def sequence_starts(data, split_indices, horizon, limit=2048):
    allowed = set(np.asarray(split_indices).tolist())
    lookup = {
        (int(episode), int(step)): index
        for index, (episode, step) in enumerate(zip(
            data["episode_id"], data["episode_step"]
        ))
        if index in allowed
    }
    sequences = []
    seen = 0
    rng = np.random.default_rng(10_000 + horizon)
    for (episode, step), index in lookup.items():
        sequence = [lookup.get((episode, step + offset)) for offset in range(horizon)]
        if all(item is not None for item in sequence):
            seen += 1
            if len(sequences) < limit:
                sequences.append(sequence)
            else:
                replacement = int(rng.integers(seen))
                if replacement < limit:
                    sequences[replacement] = sequence
    return np.asarray(sequences, dtype=np.int64)


@torch.no_grad()
def multistep_metrics(model, data, split_indices, device, horizons=(1, 3, 5)):
    output = {}
    for horizon in horizons:
        sequences = sequence_starts(data, split_indices, horizon)
        if not len(sequences):
            output[str(horizon)] = {"count": 0}
            continue
        first = sequences[:, 0]
        current = tensor_batch(data, first, device)
        initial_obs = current["obs"].clone()
        for offset in range(horizon):
            action_indices = sequences[:, offset]
            current["action"] = torch.as_tensor(
                data["action"][action_indices], device=device
            )
            prediction = model.predict(current)
            current.update({
                "obs": prediction["obs"],
                "state": prediction["state"],
                "positions": prediction["positions"],
                "velocities": prediction["velocities"],
                "adjacency": prediction["adjacency"],
            })
        target_indices = sequences[:, -1]
        target = torch.as_tensor(data["next_obs"][target_indices], device=device)
        rmse = torch.sqrt((current["obs"] - target).square().mean()).item()
        persistence = torch.sqrt((initial_obs - target).square().mean()).item()
        output[str(horizon)] = {
            "count": int(len(sequences)),
            "obs_rmse": float(rmse),
            "persistence_rmse": float(persistence),
            "relative_to_persistence": float(rmse / max(persistence, 1e-8)),
        }
    return output


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    data, provenance = load_datasets(
        args.dataset_dir, args.max_transitions, args.dataset_start_fraction
    )
    if args.action_field not in data:
        raise ValueError(
            f"action field {args.action_field!r} is not present; available fields: "
            f"{sorted(data)}"
        )
    if args.action_field != "action":
        data["action_source"] = data[args.action_field]
        data["action"] = data[args.action_field]
    dataset_record = {
        "sources": provenance,
        "merged_transitions": int(len(data["obs"])),
        "merged_geometries": int(np.unique(data["geometry_id"]).size),
        "fields": {key: list(value.shape) for key, value in data.items()},
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_record, indent=2))
    split = split_by_geometry(data["geometry_id"], seed=args.seed)
    split_record = {
        name: {
            "transitions": int(len(indices)),
            "geometries": int(np.unique(data["geometry_id"][indices]).size),
        }
        for name, indices in split.items()
    }
    (out_dir / "split.json").write_text(json.dumps(split_record, indent=2))

    config = WorldModelConfig(
        obs_dim=data["obs"].shape[-1],
        action_dim=data["action"].shape[-1],
        state_dim=data["state"].shape[-1],
        geometry_dim=data["geometry_features"].shape[-1],
        hidden_dim=args.hidden_dim,
        ensemble_size=args.ensemble_size,
    )
    model = GraphWorldModelEnsemble(config).to(args.device)
    optimizers = [
        torch.optim.AdamW(member.parameters(), lr=args.lr, weight_decay=1e-5)
        for member in model.members
    ]
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    epoch_log = (out_dir / "training_metrics.jsonl").open("w")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        member_totals = np.zeros(args.ensemble_size, dtype=np.float64)
        member_updates = np.zeros(args.ensemble_size, dtype=np.int64)
        order = rng.permutation(split["train"])
        for start in range(0, len(order), args.batch_size):
            base_indices = order[start:start + args.batch_size]
            for member_index, (member, optimizer) in enumerate(
                zip(model.members, optimizers)
            ):
                bootstrap = rng.choice(
                    base_indices, size=len(base_indices), replace=True
                )
                batch = tensor_batch(data, bootstrap, args.device)
                losses = member.loss(batch)
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(member.parameters(), 10.0)
                optimizer.step()
                member_totals[member_index] += losses["total"].item()
                member_updates[member_index] += 1

        model.eval()
        validation_losses = []
        validation_indices = split["validation"]
        if len(validation_indices) > 20000:
            validation_indices = rng.choice(
                validation_indices, 20000, replace=False
            )
        with torch.no_grad():
            for member in model.members:
                total, count = 0.0, 0
                for start in range(0, len(validation_indices), args.batch_size):
                    idx = validation_indices[start:start + args.batch_size]
                    loss = member.loss(tensor_batch(data, idx, args.device))["total"]
                    total += loss.item() * len(idx)
                    count += len(idx)
                validation_losses.append(total / max(count, 1))
        validation_loss = float(np.mean(validation_losses))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(member_totals / np.maximum(member_updates, 1))),
            "validation_loss": validation_loss,
            "member_validation_loss": validation_losses,
            "seconds": time.time() - started,
        }
        epoch_log.write(json.dumps(record) + "\n")
        epoch_log.flush()
        print(json.dumps(record))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    epoch_log.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    one_step = one_step_metrics(model, data, split["test"], args.device)
    multistep = multistep_metrics(model, data, split["test"], args.device)
    five_step = multistep.get("5", {})
    gate_passed = bool(
        five_step.get("count", 0) > 0
        and five_step.get("relative_to_persistence", float("inf")) <= 0.8
        and one_step["reward_relative_to_zero"] <= 0.9
    )
    metrics = {
        "best_validation_loss": best_loss,
        "one_step": one_step,
        "multistep": multistep,
        "gate": {
            "passed": gate_passed,
            "criteria": {
                "five_step_relative_to_persistence_max": 0.8,
                "reward_relative_to_zero_max": 0.9,
            },
        },
        "training_seconds": time.time() - started,
        "split": split_record,
    }
    model.save(out_dir / "best_world_model.pt", extra=metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
