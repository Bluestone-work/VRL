import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_navigation_dataset import aggregate, grouped_aggregates
from scripts.multicluster_protocol import SpacingTracker


def test_repaired_paired_report_reproduces_legacy_counts_and_rejects_mismatch():
    from scripts.summarize_repaired_comparison import OUT, load, strict_pairs
    train = load(OUT / "ippo_s41_train180.json", "train")
    pursuit = load(OUT / "pursuit_train180.json", "train")
    pairs = strict_pairs(train, pursuit, "regression")
    assert len(pairs) == 180
    assert sum(int(a["navigation_success"]) for _, a, _ in pairs) == 153
    assert sum(int(b["navigation_success"]) for _, _, b in pairs) == 152
    broken = [dict(row) for row in pursuit]
    broken[0]["accepted_seed"] += 1
    try:
        strict_pairs(train, broken, "broken")
    except ValueError as exc:
        assert "identity_mismatch" in str(exc)
    else:
        raise AssertionError("identity mismatch was silently accepted")


def test_repaired_regression_counts_and_legacy_intersections():
    from scripts.summarize_repaired_comparison import LEGACY, OUT, load, validate_rows
    expected = {
        ("pursuit", "train", "navigation_success"): (152, 180),
        ("ippo_s41", "train", "navigation_success"): (153, 180),
        ("mappo_s41", "train", "navigation_success"): (151, 180),
        ("pursuit", "holdout", "navigation_success"): (80, 100),
        ("ippo_s41", "holdout", "navigation_success"): (80, 100),
        ("mappo_s41", "holdout", "navigation_success"): (80, 100),
    }
    for (method, split, field), (n, d) in expected.items():
        suffix = "train180" if split == "train" else "holdout100"
        rows = load(OUT / f"{method}_{suffix}.json", split)
        assert (sum(bool(row[field]) for row in rows), len(rows)) == (n, d)
        assert sum(bool(row["strict_success"]) for row in rows) == (142 if method == "ippo_s41" and split == "train" else 140 if split == "train" else 77)
    for split, expected_intersection in (("train180", 4), ("holdout100", 2)):
        p, _ = validate_rows(load(LEGACY / f"pursuit_{split}.json", "train" if split.startswith("train") else "holdout"), "p")
        i, _ = validate_rows(load(LEGACY / f"ippo_s41_{split}_chunked.json", "train" if split.startswith("train") else "holdout"), "i")
        assert len(set(p) & set(i)) == expected_intersection


def test_mappo_v2_critic_can_fit_identity_conditioned_targets():
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.train_sb3_ippo import CentralizedValueExtractor, IPPOAgentVecEnv
    physical = CurriculumVascularEnv(seed=41, clusters=2, curriculum=False,
                                      reward_mode="navigation", zero_latency=True, direct_action=True)
    env = IPPOAgentVecEnv(physical, action_set="semantic_reduced", centralized_critic=True,
                          critic_identity=True)
    try:
        obs = env.reset()
        space = env.observation_space
        extractor = CentralizedValueExtractor(space)
        optimizer = torch.optim.Adam(extractor.parameters(), lr=0.02)
        inputs = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}
        target = torch.tensor([[1.0], [-1.0]])
        head = torch.nn.Linear(128, 1)
        optimizer = torch.optim.Adam(list(extractor.parameters()) + list(head.parameters()), lr=0.02)
        for _ in range(150):
            loss = torch.nn.functional.mse_loss(head(extractor(inputs)), target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        assert torch.max(torch.abs(head(extractor(inputs)) - target)).item() < 0.15
    finally:
        env.close()


def test_local_observation_only_generates_action():
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.train_sb3_ippo import IPPOAgentVecEnv, load_navigation_ppo
    checkpoint = Path(__file__).resolve().parents[1] / "research/COMPARISON_20261125/models/ippo_s41_final_model.zip"
    if not checkpoint.exists():
        return
    physical = CurriculumVascularEnv(seed=41, clusters=3, curriculum=False,
                                      reward_mode="navigation", zero_latency=True, direct_action=True)
    env = IPPOAgentVecEnv(physical, action_set="semantic_reduced", flow_feature=True)
    try:
        obs = env.reset()
        assert "central_state" not in obs
        action, _ = load_navigation_ppo(checkpoint, env=env, device="cpu").predict(obs, deterministic=True)
        assert action.shape == (3,)
    finally:
        env.close()


def test_fixed_three_entity_contact_is_read_once_at_episode_end():
    from scripts.evaluate_fixed_three_lesions import entity_pair_contact_seconds
    class Episode:
        pair = 1.25
    assert entity_pair_contact_seconds(Episode()) == 1.25


def _row(anatomy, flow, success=True):
    return dict(anatomy=anatomy, clusters=3, flow_mode=flow, clot_pattern="historical_sites",
                obstacle_density="low", obstacle_layout="uniform", navigation_success=success,
                strict_success=success, wall_contact_s=0., wall_after_reach_s=0., lost=0,
                particle_collision_events=0., particle_contact_s=0.)


def test_grouped_aggregates_partition_rows():
    rows = [_row("a", "weak"), _row("a", "strong", False), _row("b", "weak")]
    groups = grouped_aggregates(rows)
    assert sum(x["requested"] for x in groups["by_anatomy"].values()) == len(rows)
    assert sum(x["requested"] for x in groups["by_flow_mode"].values()) == len(rows)
    assert sum(x["requested"] for x in groups["by_condition"].values()) == len(rows)
    assert aggregate(rows)["requested"] == len(rows)


def test_spacing_tracker_counts_duration_once():
    tracker = SpacingTracker(np.array([[0., 0., 0.], [2., 0., 0.]]), np.array([True, True]), 2.)
    tracker.begin_step()
    tracker.update(np.array([[0., 0., 0.], [2., 0., 0.]]),
                   np.array([[0., 0., 0.], [1., 0., 0.]]),
                   np.array([True, True]), np.array([1., 1.]), 1.)
    tracker.end_step()
    assert tracker.summary()["spacing_violation_pair_s"] > 0.
    assert tracker.summary()["spacing_violation_steps"] == 1


def test_mappo_actor_ignores_central_state_but_critic_reads_it():
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.train_sb3_ippo import IPPOAgentVecEnv, load_navigation_ppo
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "research/COMPARISON_20261125/models/mappo_s41_final_model.zip"
    if not checkpoint.exists():
        return
    physical = CurriculumVascularEnv(seed=41, clusters=3, curriculum=False,
                                     reward_mode="navigation", zero_latency=True, direct_action=True)
    env = IPPOAgentVecEnv(physical, action_set="semantic_reduced", flow_feature=True,
                          centralized_critic=True)
    try:
        obs = env.reset()
        tensor = {key: torch.as_tensor(value) for key, value in obs.items()}
        policy = load_navigation_ppo(checkpoint, env=env, device="cpu").policy
        logits_a = policy.get_distribution(tensor).distribution.logits.detach()
        changed = dict(tensor, central_state=tensor["central_state"] + 3.0)
        logits_b = policy.get_distribution(changed).distribution.logits.detach()
        assert torch.allclose(logits_a, logits_b)
        value_a = policy.predict_values(tensor)
        value_b = policy.predict_values(changed)
        assert not torch.allclose(value_a, value_b)
    finally:
        env.close()
