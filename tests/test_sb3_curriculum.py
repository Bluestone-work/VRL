from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from stable_baselines3.common.env_checker import check_env

from marl.sb3_curriculum import ObstacleBelief, PredictiveTransformerExtractor
from scripts.train_sb3_curriculum import CurriculumVascularEnv, evaluate, predictive_model


def estimate(t, detections):
    return SimpleNamespace(frame_time_s=t, particles=[[(np.array(p), np.zeros(3)) for p in detections]],
                           pos=np.zeros((1, 3)), vel=np.zeros((1, 3)), active=np.ones(1, bool))


def test_causal_tracks_forecast_held_frames_occlusion_and_ablation():
    frame = np.eye(3)[None]
    belief = ObstacleBelief(1)
    belief.observe(estimate(0., [[.5, 0., 0.]]), frame, 0., .1)
    tokens, valid = belief.observe(estimate(.1, [[.48, 0., 0.]]), frame, .1, .1)
    assert valid.sum() == 1
    assert tokens[0, 0, 6] > tokens[0, 0, 12]  # approaching future x
    track_hits = belief.tracks[0][0]['hits']
    belief.observe(estimate(.1, [[.48, 0., 0.]]), frame, .2, .1)
    assert belief.tracks[0][0]['hits'] == track_hits  # held frame is not new evidence
    _, valid = belief.observe(estimate(.3, []), frame, .3, .1)
    assert valid.sum() == 1  # short occlusion memory
    _, valid = belief.observe(estimate(1.5, []), frame, 1.5, .1)
    assert valid.sum() == 0
    off = ObstacleBelief(1, prediction=False)
    tokens, valid = off.observe(estimate(0., [[.5, 0., 0.]]), frame, 0., .1)
    assert valid.sum() == 1 and not tokens[..., 6:21].any()


def test_curriculum_gym_contract_difficulty_and_balanced_anatomies():
    env = CurriculumVascularEnv(clusters=1, horizon=.3, budget=100)
    try:
        check_env(env, warn=False, skip_render_check=True)
        anatomies = set()
        for _ in range(9):
            env.reset()
            anatomies.add(env.scene['anatomy'])
        # Start a fresh full anatomy bag after check_env's partial bag.
        env.anatomy_bag = []
        anatomies = {env.reset()[1]['scene']['anatomy'] for _ in range(9)}
        assert len(anatomies) == 9
        env.total_environment_steps = 80
        obs, info = env.reset(options={'difficulty': 3})
        assert env.frontier == 3 and env.dynamic_obstacle_count == 32
        assert info['scene']['latency'] in (3, 4)
        assert env.observation_space.contains(obs)
        assert env.transitions[-1]['reason'] == 'budget_exposure'
    finally:
        env.close()


def test_navigation_isolates_flow_and_dynamic_obstacles_from_latency():
    env = CurriculumVascularEnv(clusters=1, horizon=.3, reward_mode='navigation')
    try:
        for difficulty in range(4):
            _, info = env.reset(options={'difficulty': difficulty, 'flow': .15, 'latency': 4})
            assert info['scene']['latency'] == 0
            assert info['scene']['flow'] == pytest.approx(.15)
            assert env.dynamic_obstacle_count == (4, 8, 16, 32)[difficulty]
    finally:
        env.close()


def test_new_observations_do_not_read_current_truth_and_zero_action_has_no_prior():
    env = CurriculumVascularEnv(clusters=1, horizon=.3)
    try:
        env.reset(seed=41)
        env.step(np.zeros(3, np.float32))
        np.testing.assert_allclose(env.ep.sent_world, 0., atol=1e-8)
        est = env.current[0]
        env.ep.observe = lambda: est
        env.ep.env.positions_mm[:] = np.nan
        env.ep.env.masses[:] = np.nan
        obs = env._observe()
        assert all(np.isfinite(value).all() for value in obs.values())
    finally:
        env.close()


def test_potential_shaping_telescopes_and_terminal_is_zero():
    env = CurriculumVascularEnv(clusters=1, horizon=.3)
    try:
        env.reset(seed=41)
        initial = env.potential
        shaping = []
        while True:
            _, _, done, truncated, info = env.step(np.zeros(3, np.float32))
            shaping.append(info['shaping_reward'])
            if done:
                assert not truncated and env.potential == 0.
                assert info['episode_metrics']['return_sum'] == pytest.approx(env.return_sum)
                break
        assert sum(.9995 ** i * r for i, r in enumerate(shaping)) == pytest.approx(-initial)
    finally:
        env.close()


def test_drone_navigation_reward_is_dense_and_logged(tmp_path):
    env = CurriculumVascularEnv(clusters=1, horizon=.3, reward_mode='drone_nav',
                                episode_log=tmp_path / 'episodes.jsonl')
    try:
        env.reset(seed=41)
        rewards = []
        while True:
            _, reward, done, truncated, info = env.step(np.zeros(3, np.float32))
            rewards.append(reward)
            if done:
                assert not truncated
                assert info['reward_mode'] == 'drone_nav'
                assert set(('progress', 'goal', 'collision', 'near', 'wall', 'lost', 'smooth', 'time')) <= set(info['drone_components'])
                break
        row = json.loads((tmp_path / 'episodes.jsonl').read_text().splitlines()[-1])
        assert row['reward_mode'] == 'drone_nav'
        assert row['return_sum'] == pytest.approx(sum(rewards))
        assert 'drone_return' in row and 'drone_components' in row
    finally:
        env.close()


@pytest.mark.parametrize('algorithm', ['ppo', 'sac', 'td3'])
def test_all_sb3_algorithms_really_update_and_reload(algorithm, tmp_path):
    torch.set_num_threads(1)
    env = CurriculumVascularEnv(clusters=1, horizon=5., budget=512)
    try:
        model = predictive_model(algorithm, env, smoke=True)
        initial = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        model.learn(total_timesteps=512)
        assert model._n_updates > 0
        assert any(not torch.equal(v, initial[k]) for k, v in model.policy.state_dict().items())
        obs, _ = env.reset(options={'difficulty': 3})
        action = model.predict(obs, deterministic=True)[0]
        assert np.isfinite(action).all()
        model.save(tmp_path / algorithm)
        restored = type(model).load(tmp_path / algorithm, device='cpu')
        np.testing.assert_allclose(restored.predict(obs, deterministic=True)[0], action, atol=1e-7)
    finally:
        env.close()


def test_attention_masks_padding_and_accepts_no_detections():
    torch.set_num_threads(1)
    env = CurriculumVascularEnv(clusters=3)
    try:
        obs, _ = env.reset(seed=41)
        obs['obstacle_valid'][:] = 0.
        obs['peer_valid'][:] = 0.
        tensors = {k: torch.as_tensor(v[None]).float() for k, v in obs.items()}
        net = PredictiveTransformerExtractor(env.observation_space).eval()
        with torch.no_grad():
            before = net(tensors)
            tensors['obstacles'][:] = 9.
            tensors['peers'][:] = -9.
            after = net(tensors)
        assert torch.isfinite(before).all()
        torch.testing.assert_close(before, after)
    finally:
        env.close()


def test_encounter_scenes_are_valid_observable_and_paired_across_ablations():
    reference = None
    for prediction in (True, False):
        env = CurriculumVascularEnv(clusters=3, prediction=prediction)
        try:
            obs, _ = env.reset(seed=41, options=dict(difficulty=3, scenario_family='route_encounter',
                                                     anatomy='mca_m1_lvo', scene_seed=2600000020))
            assert env.localized_obstacles > 0
            ep = env.ep
            positions = ep.env.positions_mm
            radius = ep.env.body_radius
            pair_dist = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
            needed = radius[:, None] + radius[None]
            np.fill_diagonal(pair_dist, np.inf)
            # Original uniform particles may overlap each other; localized
            # particles must start clear of robots and other particles.
            assert np.all(pair_dist[:3, 3:3 + env.localized_obstacles] > needed[:3, 3:3 + env.localized_obstacles])
            _, lumen, radial, _ = ep.env.transport.coordinates(positions, ep.env.edges, ep.env.solution)
            assert np.all(radial + radius <= lumen + 1e-6)
            value = (ep.manifest['scenario_hash'], positions.copy())
            if reference is None:
                reference = value
            else:
                assert value[0] == reference[0]
                np.testing.assert_array_equal(value[1], reference[1])
            assert obs['obstacle_valid'].sum() > 0
        finally:
            env.close()


def test_evaluation_spans_all_anatomies_and_difficulties_with_paired_scenes(tmp_path):
    class ZeroPolicy:
        def predict(self, obs, deterministic):
            return np.zeros(3, np.float32), None
    from scripts.multicluster_protocol import reserved_seed
    summary = evaluate(ZeroPolicy(), tmp_path, clusters=1, count=36, horizon=.1)
    rows = [json.loads(line) for line in (tmp_path / 'evaluation.jsonl').read_text().splitlines()]
    assert summary['overall']['episodes'] == 36
    assert len({(r['anatomy'], r['difficulty']) for r in rows}) == 36
    assert all(not reserved_seed(r['seed']) for r in rows)
    assert set(summary['by_difficulty']) == {'0', '1', '2', '3'}
