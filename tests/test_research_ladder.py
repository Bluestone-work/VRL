"""Orchestration tests use short fake child programs, never training budgets."""
import json
import sys
import threading
from pathlib import Path
import pytest
from scripts import auto_research_ladder as ladder


def test_fingerprint_and_retry_contract():
    assert ladder.fingerprint({'a':1,'b':2}) == ladder.fingerprint({'b':2,'a':1})
    assert ladder.fingerprint({'timesteps':1}) != ladder.fingerprint({'timesteps':2})
    for code in [-11, 139, -4, 132]: assert ladder.retryable(code)
    for code in [-15,143,-2,130,1,2,124]: assert not ladder.retryable(code)


def test_lock_exclusion(tmp_path):
    with ladder.exclusive_lock(tmp_path/'lock'):
        with pytest.raises(RuntimeError, match='active'):
            with ladder.exclusive_lock(tmp_path/'lock'): pass
    with ladder.exclusive_lock(tmp_path/'lock'): pass


def fake_commands(monkeypatch, codes):
    calls = []
    def command(config, run, device, resume=None):
        calls.append((config,run,device,resume))
        code = codes[min(len(calls)-1,len(codes)-1)]
        script = 'import pathlib,json,sys; p=pathlib.Path(sys.argv[1]); '
        if code == 0:
            script += f'(p/"summary.json").write_text(json.dumps({{"real_transitions":{10}}})); '
        script += f'print("retained output"); sys.exit({code})'
        return [sys.executable,'-c',script,str(run)]
    monkeypatch.setattr(ladder,'command_for',command)
    return calls


def test_native_retry_logs_budget_idempotence(monkeypatch,tmp_path):
    calls = fake_commands(monkeypatch,[139,0])
    job={'arm':'test','config':{'seed':42,'timesteps':10}}
    result=ladder.run_job(job,'cpu',tmp_path,2,5,threading.Event())
    assert len(calls)==2
    run=Path(result['run'])
    assert (run/'attempt_1.stderr.log').exists()
    assert 'retained output' in (run/'attempt_1.stdout.log').read_text()
    assert json.loads((run/'attempt_1.json').read_text())['returncode']==139
    assert ladder.run_job(job,'cpu',tmp_path,2,5,threading.Event())['skipped']
    assert len(calls)==2


@pytest.mark.parametrize('code',[2,143,130])
def test_no_retry_for_config_error_or_cancel(monkeypatch,tmp_path,code):
    calls=fake_commands(monkeypatch,[code,0])
    with pytest.raises(RuntimeError,match='stopped'):
        ladder.run_job({'arm':'test','config':{'seed':1,'timesteps':10}},'cpu',tmp_path,2,5,threading.Event())
    assert len(calls)==1
    with pytest.raises(RuntimeError, match='non-retryable'):
        ladder.run_job({'arm':'test','config':{'seed':1,'timesteps':10}},'cpu',tmp_path,2,5,threading.Event())
    assert len(calls)==1


def test_attempt_budget_persists(monkeypatch,tmp_path):
    calls=fake_commands(monkeypatch,[139])
    job={'arm':'test','config':{'seed':1,'timesteps':10}}
    for _ in range(2):
        with pytest.raises(RuntimeError,match='exhausted'):
            ladder.run_job(job,'cpu',tmp_path,2,5,threading.Event())
    assert len(calls)==2


def test_latest_checkpoint_numeric(tmp_path):
    (tmp_path/'checkpoint_9.pt').touch(); (tmp_path/'checkpoint_10.pt').touch()
    assert ladder.latest_checkpoint(tmp_path).name=='checkpoint_10.pt'


def test_paired_interval_is_descriptive_and_requires_exact_keys():
    rows = [{'training_seed': s, 'scenario': t, 'episode_seed': e, 'success': 0.0}
            for s in (42,43,44) for t in ('a','b') for e in (1,2)]
    better = [dict(row, success=1.0) for row in rows]
    report = ladder.paired_interval(rows, better, draws=100)
    assert report['difference'] == 1 and report['ci95'] == [1,1]
    assert 'p_value' not in report
    with pytest.raises(ValueError, match='keys'):
        ladder.paired_interval(rows, better[:-1], draws=100)


def test_affinity_defaults_override_disable_and_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(ladder.os, 'sched_getaffinity', lambda _: set(range(12)))
    assert ladder.resolve_cpu_affinity() == [0,1,2,3,4,5,8,9,10,11]
    assert ladder.resolve_cpu_affinity('2-4,8') == [2,3,4,8]
    assert ladder.resolve_cpu_affinity(exclude_cpus='0-9') == [10,11]
    assert ladder.resolve_cpu_affinity(disabled=True) is None
    for spec in ('12', '5-2', '', '0-1-2'):
        with pytest.raises(ValueError): ladder.resolve_cpu_affinity(spec)
    command = ladder.command_for({'seed':42, 'cpu-affinity':[2,3]}, tmp_path, 'cpu')
    assert command[:3] == ['taskset','-c','2,3']
    assert '--cpu-affinity' not in command
    # Disabling must not evaluate default exclusions in a restricted cpuset.
    monkeypatch.setattr(ladder.os, 'sched_getaffinity', lambda _: {6,7})
    command = ladder.command_for({'seed':42, 'cpu-affinity':None}, tmp_path, 'cpu')
    assert command[0] == sys.executable


def test_research_defaults_and_matrix(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['ladder', '--dry-run', '--timesteps', '1000000'])
    assert ladder.main() == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['jobs']) == 12
    assert {(j['config']['contact-mode'], j['config']['critic-value-mode'], j['config']['seed'])
            for j in plan['jobs']} == {(c, v, s) for c in ('euclidean', 'geodesic')
                                      for v in ('q', 'v') for s in (42, 43, 44)}
    assert plan['jobs_per_device'] == 1
    for job in plan['jobs']:
        config = job['config']
        for key, value in {'n-envs':64, 'n-steps':128, 'batch-size':2048, 'n-epochs':5,
                           'eval-episodes':5, 'final-eval-episodes':20, 'timesteps':1000000}.items():
            assert config[key] == value
        assert config['n-envs'] * config['n-steps'] == 8192
        assert not {6, 7}.intersection(config['cpu-affinity'])


def test_training_cli_overrides_reach_child_command(monkeypatch, capsys, tmp_path):
    options = {'n-envs':8, 'n-steps':32, 'batch-size':256, 'n-epochs':2,
               'eval-episodes':2, 'final-eval-episodes':3, 'timesteps':6000}
    argv = ['ladder', '--dry-run']
    for key, value in options.items(): argv.extend(['--'+key, str(value)])
    monkeypatch.setattr(sys, 'argv', argv)
    assert ladder.main() == 0
    plan = json.loads(capsys.readouterr().out)
    for job in plan['jobs']:
        command = ladder.command_for(job['config'], tmp_path, 'cpu')
        for key, value in options.items():
            assert job['config'][key] == value
            assert command[command.index('--'+key)+1] == str(value)


@pytest.mark.parametrize('option', ['batch-size', 'n-epochs', 'eval-episodes', 'final-eval-episodes'])
def test_training_cli_rejects_zero(monkeypatch, option):
    monkeypatch.setattr(sys, 'argv', ['ladder', '--'+option, '0'])
    with pytest.raises(ValueError, match='positive'):
        ladder.main()


def test_dry_run_has_no_files(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(sys,'argv',['ladder','--output',str(tmp_path/'untouched'),'--timesteps','16','--seeds','42','--devices','cpu','--arms','geodesic_v'])
    assert ladder.main()==0
    plan=json.loads(capsys.readouterr().out)
    assert len(plan['jobs'])==1 and plan['jobs_per_device']==1
    assert not (tmp_path/'untouched').exists()
