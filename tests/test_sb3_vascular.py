import numpy as np
import pytest
import torch
from stable_baselines3.common.env_checker import check_env

from scripts.train_sb3_vascular import VascularSB3Env, build_model


def test_gymnasium_contract_and_deterministic_reset():
    env = VascularSB3Env(clusters=1, dynamic_obstacle_count=2, horizon=1.)
    try:
        check_env(env, warn=False, skip_render_check=True)
        first, _ = env.reset(seed=42)
        scene = dict(env.scene)
        env.step(np.zeros(3, np.float32))
        second, _ = env.reset(seed=42)
        assert env.scene == scene
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
    finally:
        env.close()


def test_deadline_is_terminal_and_only_caller_resets():
    env = VascularSB3Env(clusters=1, dynamic_obstacle_count=2, horizon=.3)
    try:
        env.reset(seed=41)
        episode = env.ep
        obs, reward, done, truncated, info = env.step(np.zeros(3, np.float32))
        assert not done
        np.testing.assert_allclose(env.ep.sent_world, 0., atol=1e-8)
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(np.zeros(3, np.float32))
            if done:
                break
        assert done and not truncated
        assert env.ep is episode
        assert env.observation_space.contains(obs)
        assert info['episode_metrics']['termination'] == 'time_limit'
        assert env.ep.steps == env.total_environment_steps
        with pytest.raises(RuntimeError, match='reset'):
            env.step(np.zeros(3, np.float32))
    finally:
        env.close()


def test_observation_uses_delayed_estimate_not_current_positions_or_mass():
    env = VascularSB3Env(clusters=1, dynamic_obstacle_count=2)
    try:
        env.reset(seed=41)
        est = env.current[0]
        env.ep.observe = lambda: est
        env.ep.env.positions_mm[:] = np.nan
        env.ep.env.masses[:] = np.nan
        obs = env._observe()
        assert all(np.isfinite(value).all() for value in obs.values())
    finally:
        env.close()


def test_policy_joint_actions_and_save_reload(tmp_path):
    torch.set_num_threads(1)
    env = VascularSB3Env(clusters=3, dynamic_obstacle_count=2)
    try:
        obs, _ = env.reset(seed=41)
        for algo in ('ppo', 'sac', 'td3'):
            model = build_model(algo, env, seed=41, smoke=True)
            action, _ = model.predict(obs, deterministic=True)
            assert action.shape == (9,)
            assert np.isfinite(action).all() and np.abs(action).max() <= 1.
            path = tmp_path / algo
            model.save(path)
            restored = type(model).load(path, device='cpu')
            np.testing.assert_allclose(restored.predict(obs, deterministic=True)[0], action, atol=1e-7)
    finally:
        env.close()


def test_evaluation_uses_development_pool_and_complete_episodes(tmp_path, monkeypatch):
    import scripts.train_sb3_vascular as module
    from scripts.multicluster_protocol import reserved_seed

    class ZeroPolicy:
        def predict(self, obs, deterministic):
            assert deterministic
            return np.zeros(9, np.float32), None

    original_env = module.VascularSB3Env
    monkeypatch.setattr(module, 'VascularSB3Env', lambda **kwargs: original_env(horizon=.3, **kwargs))
    summary = module.evaluate(ZeroPolicy(), tmp_path, clusters=3, obstacles=2, count=9)
    import json
    rows = [json.loads(line) for line in (tmp_path / 'evaluation.jsonl').read_text().splitlines()]
    assert summary['episodes'] == len(rows) == 9
    assert len({row['anatomy'] for row in rows}) == 9
    assert all(not reserved_seed(row['seed']) for row in rows)
    assert all(row['termination'] == 'time_limit' for row in rows)


def test_real_ppo_rollout_preserves_episode_and_separate_progress_files(tmp_path):
    import json
    from stable_baselines3.common.logger import configure
    from scripts.train_sb3_vascular import RunCallback

    torch.set_num_threads(1)
    env = VascularSB3Env(clusters=1, dynamic_obstacle_count=2)
    try:
        model = build_model('ppo', env, seed=41, smoke=True)
        model.set_logger(configure(str(tmp_path), ['json']))
        model.learn(total_timesteps=512, callback=RunCallback(tmp_path, 10000))
        status = json.loads((tmp_path / 'status.json').read_text())
        assert status['environment_steps'] == env.total_environment_steps == 512
        assert env.episode_index == 1 and env.ep.steps == 512
        logs = [json.loads(line) for line in (tmp_path / 'progress.json').read_text().splitlines()]
        assert logs[-1]['time/total_timesteps'] == 512
        assert model._n_updates > 0
    finally:
        env.close()
