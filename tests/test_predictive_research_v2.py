"""Causality, measurement and controller regressions for the isolated protocol."""
import numpy as np
import pytest
import torch
import io
import json
from types import SimpleNamespace

from marl.predictive_research import (SensorHistory, MotionGRU, LocalWorldEnsemble,
    ActionController, analytic_cost, bounded, HISTORY, HORIZONS)
from environments.research_predictive_env import ResearchSingle, ResearchVector, assignments, replan
from marl.geometric_control import direct_local_action
from scripts.research16_20_worker import load_world
from scripts.research16_20_queue import Queue


def make(single=True, **kw):
    cls = ResearchSingle if single else ResearchVector
    options = dict(num_robots=5, num_clots=3, seed=71, particle_seed=81,
        scenario='coronary_rca', randomize_scenario=False,
        obs_mode='geometric_predictive', dynamic_intravascular_particles=True,
        robot_radius=.0011, contact_mode='geodesic', **kw)
    return cls(**options) if single else cls(n_envs=2, **options)


def test_history_is_causal_idempotent_and_reacquisition_clears():
    h = SensorHistory((1, 1))
    robot = np.zeros((1, 1, 3), np.float32)
    particles = np.array([[[.03, 0, 0], [.08, 0, 0]]], np.float32)
    vel = np.zeros_like(particles)
    for t in range(HISTORY):
        h.observe(robot, robot, particles, vel, [t])
    assert h.valid.item() == HISTORY
    before = h.values.copy()
    h.observe(robot, robot, particles, vel, [HISTORY-1])
    np.testing.assert_array_equal(before, h.values)
    particles[0, 0, 0] = .2
    h.observe(robot, robot, particles, vel, [HISTORY])
    assert h.valid.item() == 1 and h.ids.item() == 1
    particles[:] = 2
    h.observe(robot, robot, particles, vel, [HISTORY+1])
    assert h.valid.item() == 0 and h.ids.item() == -1
    h.reset()
    assert not h.values.any()


def test_motion_starts_at_cv_and_backpropagates():
    model = MotionGRU()
    x = torch.randn(8, 6, 6)
    mu, lv = model(x)
    expected = x[:, -1, 3:6][:, None]*torch.tensor(HORIZONS)[None, :, None]
    torch.testing.assert_close(mu, expected)
    (mu.square().mean()+lv.mean()).backward()
    assert model.head.weight.grad.abs().sum() > 0


def test_world_gate_cannot_be_bypassed(tmp_path):
    path = tmp_path/'failed.pt'
    torch.save(dict(model=LocalWorldEnsemble().state_dict(), gate=dict(passed=False)), path)
    with pytest.raises(ValueError, match='gate failed'):
        load_world(path, 'cpu')


def test_untrusted_world_uses_analytic_fallback():
    class Uncertain(torch.nn.Module):
        def forward(self, obs, action):
            pred = torch.zeros(5, len(obs), 57)
            pred[:, :, 52:55] = torch.arange(5)[:, None, None]*10
            return pred
    nodes = np.zeros((5, 52), np.float32)
    nodes[:, 19] = nodes[:, 18] = nodes[:, 24] = 1
    nodes[:, 6] = .2
    proposal = np.full((5, 3), .2, np.float32)
    expected, _ = ActionController().choose(nodes, proposal)
    actual, diag = ActionController(Uncertain()).choose(nodes, proposal)
    np.testing.assert_array_equal(expected, actual)
    assert diag['model_trusted'] == 0
    assert np.max(np.linalg.norm(actual, axis=-1)) <= 1.000001


def test_prediction_detects_crossing_not_only_endpoint():
    n = np.zeros((1,52), np.float32)
    n[:,18:20] = 1
    n[:,32] = 1
    n[:,43] = 1
    n[:,36] = .03/.12
    n[:,44] = -.015/.12
    action = np.zeros((1,1,3), np.float32)
    approaching = analytic_cost(n, action).item()
    n[:,44] = .075/.12
    departing = analytic_cost(n, action).item()
    assert approaching > departing


def test_seeded_single_rollout_repeats_and_path_is_displacement():
    envs = [make(), make()]
    for env in envs:
        env.reset(seed=71)
    np.testing.assert_array_equal(envs[0].particles.positions, envs[1].particles.positions)
    total = np.zeros(5)
    for _ in range(12):
        for i, env in enumerate(envs):
            obs, _ = replan(env)
            before = env.robot_positions.copy()
            _, _, _, _, info = env.step(direct_local_action(np.full((5,3),.2), env))
            if i == 0:
                total += np.linalg.norm(env.robot_positions-before, axis=-1)
                np.testing.assert_allclose(info['robot_path_length'], total)
        np.testing.assert_array_equal(envs[0].robot_positions, envs[1].robot_positions)
        np.testing.assert_array_equal(envs[0].particles.positions, envs[1].particles.positions)
    envs[0].reset(seed=72)
    assert not envs[0].robot_path_length.any()


def test_vector_terminal_length_excludes_reset_jump():
    env = make(False, horizon=2)
    total = np.zeros((2,5))
    for _ in range(2):
        before = env.robot_positions.copy()
        replan(env)
        _, _, term, trunc, info = env.step(direct_local_action(np.full((2,5,3),.2), env))
        done = term | trunc
        after = info['final_context']['positions'] if done.all() else env.robot_positions
        total += np.linalg.norm(after-before, axis=-1)
        np.testing.assert_allclose(info['robot_path_length'], total, rtol=1e-5, atol=1e-8)
    assert done.all()
    assert not env.robot_path_length.any()
    assert env.features.history.valid.max() <= 1


def test_vector_allocator_passes_stations_and_refreshes_observation():
    env = make(False)
    route = env._route
    seen = []
    def checked(station):
        seen.append(station)
        return route(station)
    env._route = checked
    result = assignments(env, 0)
    live = np.flatnonzero(env.clot_masses[0] > 0)
    assert seen == env.clot_stations[0, live].tolist()
    env.set_task_assignments(np.stack([result, assignments(env,1)]))
    refreshed = env._observe()
    for r, target in enumerate(result):
        delta = env.clot_positions[0,target]-env.robot_positions[0,r]
        st = env.robot_stations[0,r]
        basis = np.stack([env.tree.tangents[st],env.tree.normals[st],env.tree.binormals[st]])
        np.testing.assert_allclose(refreshed['nodes'][0,r,25:28],np.clip(basis@delta/.5,-1,1),atol=1e-6)


def test_invisible_obstacles_do_not_leak_into_actor():
    env = make()
    env.reset(seed=71)
    env.particles.positions[:] = 100
    a = env._build_observation()['nodes'].copy()
    env.particles.positions[:] = -100
    env.particles.velocities[:] = 42
    b = env._build_observation()['nodes'].copy()
    np.testing.assert_array_equal(a[:,36:], b[:,36:])


def test_controller_never_mutates_ppo_proposal():
    env = make()
    obs, _ = env.reset(seed=71)
    action = np.random.default_rng(31).normal(size=(5,3)).astype(np.float32)
    original = action.copy()
    result, _ = ActionController().choose(obs['nodes'], action)
    np.testing.assert_array_equal(action, original)
    assert np.isfinite(result).all()


def test_queue_failed_gate_blocks_20_without_force(tmp_path):
    q = Queue.__new__(Queue)
    q.args = SimpleNamespace(seeds=[42], world_budgets=[100,200])
    q.jobs, q.seen, q.events = {}, set(), io.StringIO()
    t = tmp_path/'17'
    t.mkdir()
    checkpoint = tmp_path/'policy.pt'
    checkpoint.write_bytes(b'unit test checkpoint identity only')
    import hashlib
    (t/'checkpoint_1.ready.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
        transitions=1, arm=17, predictor='predictor.pt', world='', seed=42,
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())))
    q.jobs['17_s42'] = dict(job='train', status='running', out=str(t), seed=42, attempt=1)
    for stage in range(2):
        folder = tmp_path/f'world{stage}'
        folder.mkdir()
        (folder/'metrics.json').write_text(json.dumps(dict(gate=dict(passed=False))))
        q.jobs[f'world_fit{stage}_s42'] = dict(job='fit',status='complete',out=str(folder),seed=42)
    q.discover()
    assert q.jobs['20_s42']['status'] == 'blocked'
    assert 'validation_gate_failed' in q.events.getvalue()


def test_queue_reports_paired_identity_mismatch(tmp_path):
    q = Queue.__new__(Queue)
    q.root, q.running = tmp_path, {}
    q.args, q.jobs = SimpleNamespace(seeds=[42,43,44],max_jobs=18), {}
    for arm in (16,17):
        for seed in (42,43,44):
            folder = tmp_path/f'{arm}_{seed}'
            folder.mkdir()
            record = dict(scenario='test', episode_seed=72, initial_hash='a' if arm==16 else 'b',
                          success=True,removal=1,path_length=2,wall_rate=0,pair_rate=0)
            (folder/'episodes.jsonl').write_text(json.dumps(record)+'\n')
            (folder/'summary.json').write_text(json.dumps(dict(arm=arm,seed=seed,
                macro=dict(success=1.,removal=1.,path_length=2.,wall_rate=0.,pair_rate=0.))))
            q.jobs[f'eval{arm}_{seed}'] = dict(job='evaluate',status='complete',out=str(folder),
                options=dict(checkpoint=str(tmp_path/'policy_100.pt')))
    q.report()
    result = json.loads((tmp_path/'results.json').read_text())
    assert len(result['initial_identity_errors'])==3
    assert not next(x for x in result['aggregate'] if x['arm']==17)['validation_85_reached']
