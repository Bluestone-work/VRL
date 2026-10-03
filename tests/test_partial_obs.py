"""The fair observation and fair reactive controllers never read routes, geodesic targets or allocation."""
import numpy as np

from environments.mca_compiled import CompiledMCAPhysicalEnv
from marl.fair_reactive import fair_reactive_action
from marl.partial_obs import OBS_DIM, PartialObsConfig, PartialObserver
from marl.teacher import execute_local
from scripts.collect_teacher_dataset import student_env_config
from scripts.train_mca_compiled import reset_with_valid_particles


def test_fair_observation_has_no_privileged_access():
    import pytest
    env = CompiledMCAPhysicalEnv(student_env_config('mca_m1_lvo', 5))
    reset_with_valid_particles(env, 940000000)
    observer = PartialObserver(env, PartialObsConfig(noise=0.))
    def forbidden(*a, **k):
        raise AssertionError('fair observation touched privileged information')
    env_names = ('_route_directions', '_assigned_targets', '_target_distances', '_solve_assignment',
                 '_observation', '_write_target_block', '_apply_action_prior')
    for _ in range(30):
        # Privileged accessors raise while the observation is built and the action chosen;
        # env.step itself may use them for its own (unused) legacy observation and reward.
        with pytest.MonkeyPatch.context() as mp:
            for name in env_names:
                mp.setattr(env, name, forbidden)
            for name in ('route_to', 'lookahead'):
                mp.setattr(env.tree, name, forbidden, raising=False)
            mp.setattr(env, 'routes', None); mp.setattr(env, '_route_next_hop', None)
            obs = observer.observe()
            act = fair_reactive_action(obs, 'path')
        assert obs.shape == (5, OBS_DIM) and np.isfinite(obs).all()
        observer.record_action(act)
        env.step(execute_local(env, act))


def test_noise_free_observation_is_deterministic():
    a, b = (CompiledMCAPhysicalEnv(student_env_config('renal_artery', 4)) for _ in range(2))
    for e in (a, b):
        reset_with_valid_particles(e, 940000000)
    oa = PartialObserver(a, PartialObsConfig(noise=0.)).observe()
    ob = PartialObserver(b, PartialObsConfig(noise=0.)).observe()
    assert np.array_equal(oa, ob)
