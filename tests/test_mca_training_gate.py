from dataclasses import replace
import pytest
from environments.mca_physical_env import DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.mca_training_gate import task_feasibility, require_all_clear_capacity
from scripts.train_mca_physical import reset_with_valid_particles


def env_at(duration):
    env=CompiledMCAPhysicalEnv(replace(DynamicsConfig.from_json(),episode_duration_s=duration))
    reset_with_valid_particles(env,42)
    return env


def test_current_task_is_rejected_for_success_target():
    env=env_at(1.)
    with pytest.raises(ValueError,match='provably unreachable'):
        require_all_clear_capacity(env)
    r=task_feasibility(env)
    assert r['optimistic_removal_capacity']==pytest.approx(1.8)
    assert r['initial_mass']==4
    assert r['optimistic_minimum_time_s']==pytest.approx(4/1.8)


def test_explicit_diagnostic_override_does_not_hide_failure():
    r=require_all_clear_capacity(env_at(1.),allow_unreachable_baseline=True)
    assert r['all_clear_proven_impossible']
    assert not r['sufficient_for_training_readiness']


def test_longer_time_is_not_a_reachability_claim():
    r=require_all_clear_capacity(env_at(15.))
    assert not r['all_clear_proven_impossible']
    assert not r['sufficient_for_training_readiness']


@pytest.mark.parametrize('module', ['scripts.train_mca_physical', 'scripts.train_mca_compiled',
                                   'scripts.train_mca_parallel'])
def test_training_entrypoint_rejects_before_creating_run(module, tmp_path):
    import importlib
    from types import SimpleNamespace
    trainer=importlib.import_module(module)
    out=tmp_path/'must_not_be_created'
    args=SimpleNamespace(protocol=str(trainer.DEFAULT_PROTOCOL), seed=42,
                         out=str(out), allow_unreachable_baseline=False)
    with pytest.raises(ValueError,match='provably unreachable'):
        trainer.train(args)
    assert not out.exists()


def test_launcher_rejects_before_creating_run(monkeypatch, tmp_path):
    from scripts import launch_mca_distributed as launcher
    out=tmp_path/'must_not_be_created'
    monkeypatch.setattr(launcher, 'OUT', out)
    with pytest.raises(ValueError,match='provably unreachable'):
        launcher.main()
    assert not out.exists()
