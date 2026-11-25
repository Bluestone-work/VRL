import numpy as np
import pytest
import torch

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import (
    IPPOAgentVecEnv, IPPOExtractor, MaskedMultiInputActorCriticPolicy,
)


ANATOMY = "hcc_portal_multifocal"


def navigation_env(clusters=3, horizon=10., **kwargs):
    physical = CurriculumVascularEnv(clusters=clusters, horizon=horizon,
                                     curriculum=False, reward_mode="navigation", direct_action=True)
    physical.train_anatomies = [ANATOMY]
    return IPPOAgentVecEnv(physical, action_set="semantic_reduced", **kwargs)


def scene(seed, level):
    return dict(anatomy=ANATOMY, scene_seed=seed, scenario_family="uniform",
                difficulty=level, flow=(.025, .05, .10, .15)[level])


def test_frozen_navigation_targets_drive_routes_candidates_commands_and_score():
    env = navigation_env()
    try:
        physical = env.physical
        for seed, level, expected in [(1910000002, 2, [3, 2, 0]), (1910000101, 1, [2, 3, 1])]:
            obs = env.reset(options=scene(seed, level))
            ep = physical.ep
            for _ in range(2):
                np.testing.assert_array_equal(physical.current[1], expected)
                np.testing.assert_array_equal(ep.ctl.goal, expected)
                np.testing.assert_array_equal(obs["ctx"][:, 27], 1.)
                assert (obs["candidate_valid"][:, 1] == 1.).all()
                for path, target in zip(ep.coord.paths, expected):
                    assert np.linalg.norm(path[-1] - ep.env.clot_positions_mm[target]) <= ep.coord.step + 1e-5
                obs, _, done, infos = env.step(np.ones(3, int))
                assert not done.any()
                assert (np.linalg.norm(ep.sent_world, axis=1) > 0.).all()
                np.testing.assert_array_equal(infos[0]["navigation_targets"], expected)
                np.testing.assert_allclose(physical._navigation_distances(),
                                           ep.env._target_distances()[np.arange(3), expected])
    finally:
        env.close()


@pytest.mark.parametrize("clusters", [1, 3])
def test_image_motion_accumulates_and_resets_at_episode_boundary(clusters):
    env = navigation_env(clusters, horizon=2., flow_feature=True)
    try:
        env.reset(options=scene(1910000001, 1))
        for t in range(16):
            obs, _, done, _ = env.step(np.full(clusters, t % 2))
            assert not done.any()
        motion = env._om
        counts = [len(samples) for samples in motion.samples]
        assert min(counts) >= 6
        assert (obs["flow"][:, 5] > 0.).all()
        assert (np.linalg.norm(obs["flow"][:, :3], axis=1) > 0.).all()
        np.testing.assert_array_equal(env._flow_features(), obs["flow"])
        assert [len(samples) for samples in motion.samples] == counts
        while not done.any():
            obs, _, done, infos = env.step(np.full(clusters, 1))
        assert infos[0]["terminal_observation"]["flow"][5] > 0.
        assert env._om is not motion
        assert not any(env._om.samples)
        np.testing.assert_array_equal(obs["flow"][:, 5], 0.)
    finally:
        env.close()


def test_policy_and_execution_respect_masked_zero_command(monkeypatch):
    env = navigation_env()
    try:
        obs = env.reset(options=scene(1910000002, 2))
        obs["candidate_valid"][:] = 0.
        obs["candidate_valid"][:, 1] = 1.
        env._candidate_valid = obs["candidate_valid"]
        policy = MaskedMultiInputActorCriticPolicy(
            env.observation_space, env.action_space, lambda _: .001,
            features_extractor_class=IPPOExtractor, normalize_images=False)
        tensors, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            actions, _, log_probs = policy(tensors)
            _, evaluated, _ = policy.evaluate_actions(tensors, actions)
            probs = policy.get_distribution(tensors).distribution.probs
        assert torch.all(actions == 1) and torch.all(probs[:, 0] == 0.)
        torch.testing.assert_close(log_probs, evaluated)
        expected = env._candidate_commands[:, 1].copy()
        step = env.physical.step

        def check_command(command):
            np.testing.assert_array_equal(command.reshape(3, 3), expected)
            return step(command)

        monkeypatch.setattr(env.physical, "step", check_command)
        env.step(np.zeros(3, int))  # An external unmasked policy selected the unsafe slot.
    finally:
        env.close()


def test_classic_scores_arrival_using_shared_distances_on_terminal_step(monkeypatch):
    from scripts.benchmark_lysis import LysisEpisode, NAV_GOAL_TOLERANCE_MM
    from scripts.evaluate_hcc_navigation_matrix import run_classic

    def distances(ep):
        np.testing.assert_array_equal(ep.navigation_targets, [3, 2, 0])
        assert ep.steps == 1
        return np.full(ep.n, NAV_GOAL_TOLERANCE_MM)

    monkeypatch.setattr(LysisEpisode, "navigation_distances", distances)
    row = run_classic("stpg", 2, 1910000002, horizon=.1)
    assert row["navigation_success"] and row["navigation_reached"] == 3
    assert row["navigation_time_s"] == pytest.approx(.1)
    assert row["task_success"] and row["strict_success"] == row["strict_navigation_success"]
    assert row["collision_free_success"] == row["collision_free_navigation"]
    assert row["termination"] == "navigation_success"
    assert row["navigation_targets"] == [3, 2, 0]
    assert row["accepted_seed"] == 1910000002


def test_team_and_actor_returns_logged_separately(tmp_path):
    import json
    from scripts.live_training_monitor import curves_for

    env = navigation_env(horizon=.4, flow_feature=True, exit_mask=True)
    try:
        env.physical.episode_log_path = tmp_path / "episodes.jsonl"
        env.reset(options=scene(1910000002, 2))
        returns = np.zeros(3)
        for t in range(4):
            _, rewards, done, infos = env.step(np.full(3, t % 2))
            returns += rewards
        assert done.all()
        logged = json.loads(env.physical.episode_log_path.read_text())
        np.testing.assert_allclose(logged["agent_returns"], returns)
        assert logged["navigation_return"] == infos[0]["navigation_return"]
        curves = curves_for(tmp_path)
        assert curves["reward_kind"] == "agent_mean"
        assert curves["episodes"][-1][1] == pytest.approx(returns.mean())
        del logged["mean_agent_return"]
        env.physical.episode_log_path.write_text(json.dumps(logged))
        curves = curves_for(tmp_path)
        assert curves["reward_kind"] == "team"
        assert curves["episodes"][-1][1] == logged["navigation_return"]
    finally:
        env.close()


def test_integration_failure_ends_episode_without_bootstrap(monkeypatch, tmp_path):
    env = navigation_env(flow_feature=True, exit_mask=True, failure_log=tmp_path / "fail.json")
    try:
        env.physical.episode_log_path = tmp_path / "episodes.jsonl"
        obs = env.reset(options=scene(1910004000, 0))
        scene_hash = env.physical.scene["scenario_hash"]

        def fail(_action):
            raise RuntimeError("Physical integration budget exceeded")

        monkeypatch.setattr(env.physical, "step", fail)
        _, rewards, done, infos = env.step(np.ones(3, int))
        assert done.all() and not rewards.any()
        import json
        logged = json.loads(env.physical.episode_log_path.read_text())
        assert logged["integration_failure"] and logged["environment_steps"] == 0
        assert not logged["navigation_success"]
        assert logged["scenario_hash"] == scene_hash
        assert [row["integration_failure"] for row in env.physical.recent] == [True]
        for i, info in enumerate(infos):
            assert info["integration_failure"] and not info["is_success"]
            assert not info["TimeLimit.truncated"]
            for key in obs:
                np.testing.assert_array_equal(info["terminal_observation"][key], obs[key][i])
    finally:
        env.close()


def test_paired_flow_override_survives_training_curriculum():
    env = CurriculumVascularEnv(clusters=1, budget=100, reward_mode='navigation', flow_curriculum=True)
    try:
        env.train_anatomies = [ANATOMY]
        _, info = env.reset(options=scene(1910000001, 3))
        assert info["scene"]["flow"] == pytest.approx(.15)
        _, info = env.reset(options={"difficulty": 3})
        assert info["scene"]["flow"] == pytest.approx(.025)
    finally:
        env.close()


@pytest.mark.parametrize(("parity", "exit_mask"), [(True, False), (False, True)])
def test_continuous_exit_protection_has_its_own_switch(parity, exit_mask, monkeypatch):
    import marl.exit_mask as outlet

    phys = CurriculumVascularEnv(clusters=1, reward_mode="navigation", direct_action=True)
    phys.train_anatomies = [ANATOMY]
    env = IPPOAgentVecEnv(phys, action_set="continuous", exit_mask=exit_mask, safety_parity=parity)
    calls = []
    compensate = outlet.compensate_exit_command

    def record(*args, **kwargs):
        calls.append(1)
        return compensate(*args, **kwargs)

    monkeypatch.setattr(outlet, "compensate_exit_command", record)
    try:
        env.reset(options=scene(1910000002, 2))
        env.step(np.zeros((1, 3)))
        assert len(calls) == int(exit_mask)
    finally:
        env.close()


def test_live_snapshot_reports_physical_map_in_mm(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.train_sb3_ippo import IPPOCallback

    env = navigation_env()
    try:
        env.reset(options=scene(1910000001, 1))
        callback = IPPOCallback(tmp_path, 3000000, 3)
        callback.model = SimpleNamespace(get_env=lambda: env)
        callback.n_calls = 256
        callback.num_timesteps = 768
        assert callback._on_step()
        live = json.loads((tmp_path / "live_trace.json").read_text())
        points = env.physical.ep.env.transport.points
        np.testing.assert_allclose(live["centerline_points_mm"], points[::int(np.ceil(len(points)/1800))], atol=5.1e-6)
        np.testing.assert_array_equal(live["target_indices"], env.physical.nav_targets)
    finally:
        env.close()

def test_replan_preserves_teammates_command_and_route_state(monkeypatch):
    from scripts.benchmark_lysis import WallGuard

    env = navigation_env()
    try:
        env.reset(options=scene(1910000002, 2))
        ep = env.physical.ep
        est, targets, hold, _ = env.physical.current
        guard = WallGuard(ep)
        monkeypatch.setattr(guard.replan, "update", lambda *args: [0])
        monkeypatch.setattr(guard.exec, "act", lambda t, est, ctl, rule, hold: rule)
        commands = np.full((3, 3), .123)
        ep.ctl.nominal[1:] = 17.
        fixed = guard(ep, est, targets, commands, hold)
        assert not np.allclose(fixed[0], commands[0])
        np.testing.assert_array_equal(fixed[1:], commands[1:])
        np.testing.assert_array_equal(ep.ctl.nominal[1:], 17.)
    finally:
        env.close()


def test_spacing_packet_represents_last_world_command_in_current_frame(monkeypatch):
    env = navigation_env()
    try:
        env.reset(options=scene(1910000002, 2))
        ep = env.physical.ep
        est = env.physical.current[0]
        frames = np.tile([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], (3, 1, 1))
        ep.sent_world[:] = [.2, .3, .4]
        ep.prev_local[:] = [.2, .3, .4]
        monkeypatch.setattr(ep.ctl, "frames", lambda est: frames)
        packet = ep.spacing_packet(est)
        np.testing.assert_allclose(packet.navigation[:, :3], np.tile([.3, -.2, .4], (3, 1)))
    finally:
        env.close()


def test_hcc_runner_keeps_incomplete_groups_visible(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import evaluate_hcc_navigation_matrix as matrix

    def evaluate(method, level, seed, horizon, centerline):
        if level == 0:
            raise ValueError("failed scene")
        return dict(method=method, difficulty=level, integration_failure=(level == 1),
                    navigation_success=False, strict_navigation_success=False,
                    navigation_time_s=None, wall_contact_s=0., lost=0, particle_collision_events=0)

    out = tmp_path / "rows.json"
    monkeypatch.setattr(matrix, "run_classic", evaluate)
    monkeypatch.setattr(sys, "argv", ["matrix", "--methods", "pursuit", "--out", str(out)])
    with pytest.raises(SystemExit, match="Incomplete"):
        matrix.main()
    summary = json.loads(out.with_suffix(".summary.json").read_text())
    for level in (0, 1):
        group = summary[f"pursuit|{level}"]
        assert group["errors"] == group["requested"] == 1 and group["episodes"] == 0
        assert not group["complete"] and group["navigation_success_rate"] is None
    assert summary["pursuit|2"]["navigation_success_rate"] == 0.


@pytest.mark.parametrize("mode", ["semantic_reduced", "semantic_minimal"])
def test_semantic_mask_deduplicates_only_retained_commands(mode):
    from scripts.train_sb3_ippo import direct_candidates

    env = navigation_env()
    try:
        env.reset(options=scene(1910004002, 2))
        commands, valid = direct_candidates(env.physical, mode)
        for robot_commands, mask in zip(commands, valid):
            keys = [tuple(command.round(6)) for command in robot_commands]
            expected = [keys.index(key) == slot for slot, key in enumerate(keys)]
            np.testing.assert_array_equal(mask, expected)
    finally:
        env.close()


def test_vec_reset_consumes_shared_options_seeds_and_cancels_old_action():
    env = navigation_env(horizon=.1)
    try:
        options = scene(1910004002, 2)
        assert env.seed(27) == [27, 27, 27]
        env.set_options(options)
        env.reset()
        assert all(i["scene"]["seed"] == options["scene_seed"] for i in env.reset_infos)
        env.step_async(np.ones(3))
        env.reset()
        assert env.physical.scene["seed"] != options["scene_seed"]
        with pytest.raises(RuntimeError, match="step_async"):
            env.step_wait()

        env.seed(27)
        env.reset()
        first = env.physical.scene.copy()
        env.step(np.ones(3))
        assert env.physical.scene["seed"] != first["seed"]
        env.seed(27)
        env.reset()
        assert env.physical.scene["scenario_hash"] == first["scenario_hash"]
        with pytest.raises(ValueError, match="share reset options"):
            env.set_options([{}, {}, options])
    finally:
        env.close()


def test_resume_uses_updated_policy_and_exact_tail(tmp_path, monkeypatch):
    import sys
    from scripts.resume_sb3_3m import BudgetedPPO
    from scripts.train_sb3_ippo import IPPOCallback, load_navigation_ppo

    env = navigation_env(clusters=1, horizon=.2)
    try:
        # Reproduce old zip files written by running the training script as __main__.
        legacy = type("MaskedMultiInputActorCriticPolicy", (MaskedMultiInputActorCriticPolicy,),
                      {"__module__": "__main__"})
        monkeypatch.setattr(sys.modules["__main__"], legacy.__name__, legacy, raising=False)
        base = BudgetedPPO(legacy, env, n_steps=4, batch_size=2, n_epochs=1, device="cpu",
                           policy_kwargs=dict(features_extractor_class=IPPOExtractor, normalize_images=False))
        base.learn(4)
        checkpoint = tmp_path / "legacy.zip"
        base.save(checkpoint)
        restored = load_navigation_ppo(checkpoint, device="cpu", algorithm=BudgetedPPO)
        assert type(restored.policy) is MaskedMultiInputActorCriticPolicy
        for old, new in zip(base.policy.parameters(), restored.policy.parameters()):
            torch.testing.assert_close(old, new)
        assert restored.policy.optimizer.state_dict()["state"]
        restored.set_env(env)
        callback = IPPOCallback(tmp_path, 10, 1, save_every=4)
        callback.last_saved = 4
        restored.learn(6, reset_num_timesteps=False, callback=callback)
        assert restored.num_timesteps == 10
        assert restored.rollout_buffer.buffer_size == 2
        saved = load_navigation_ppo(tmp_path / "policy_agent_steps8.zip", device="cpu")
        assert saved._n_updates == base._n_updates + 1
    finally:
        env.close()


def test_paired_healthy_inlet_boundary_preserves_anatomy_and_targets():
    env = navigation_env()
    try:
        options = scene(2944100001, 3)
        env.reset(options=options)
        physical = env.physical
        layout = physical.scene["scenario_hash"]
        targets = physical.nav_targets.copy()
        original = physical.ep.env.solution["station_inflow_mm3_s"].copy()
        healthy = physical.ep.env.flow_model.solve()["inlet_flow_ml_min"]
        env.reset(options=dict(options, healthy_inlet_ml_min=638.6, active_speed_mm_s=1.2))
        sim = physical.ep.env
        assert physical.scene["scenario_hash"] == layout
        np.testing.assert_array_equal(physical.nav_targets, targets)
        assert sim.config.robot_speed_mm_s == 1.2
        assert sim.flow_model.solve()["inlet_flow_ml_min"] == pytest.approx(638.6)
        np.testing.assert_allclose(sim.solution["station_inflow_mm3_s"], original * (638.6 / healthy))
        assert physical.scene["inlet_ml_min"] < 638.6
    finally:
        env.close()
