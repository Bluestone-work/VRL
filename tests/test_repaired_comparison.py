import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_navigation_dataset import aggregate, grouped_aggregates
from scripts.multicluster_protocol import SpacingTracker


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
