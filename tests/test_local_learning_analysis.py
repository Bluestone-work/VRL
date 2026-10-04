"""Scientific comparison gates, including tracked stopping-time semantics."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_zero_successes_do_not_imply_zero_population_uncertainty():
    from scripts.analyze_local_learning import exact_binomial_ci
    lower, upper = exact_binomial_ci(0, 12)
    assert lower == 0.
    assert .264 < upper < .265


def matrix(folder, arm, *, stage='validation', observed=True):
    folder.mkdir()
    manifest = dict(stage=stage, observation_contract='tracked', source_hashes={},
                    requested=1, scenes=[1420000000], arms=[dict(label=arm)])
    row = dict(arm=arm, scene_seed=1420000000, scenario_hash='same_scene',
               clusters=3, sensor_assumptions={'latency_s':.1},
               reset_info={'actual_config':{'num_robots':3,'lysis_mass_per_s':.12},
                           'aggregate_lysis_capacity_mass_per_s':.36,'resource_budget':'fixed_total'},
               source_hashes={}, status='completed', cluster_safe_success=False,
               task_success=False, removal=.5, removal_auc_180=.25,
               wall_contact_s=2., spacing_violation_pair_s=0.,
               sensor_confirmed_all_cleared=False, false_visual_completion=False,
               physical_cluster_safe_success=False, elapsed_s=180., command_squared_s=12.,
               time_to_removal_s={'90': None}, completion_uses_observed_confirmation=observed)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    (folder/'attempts.jsonl').write_text(json.dumps(row)+'\n')
    return folder


def analyze(tmp_path, *inputs):
    out = tmp_path/'analysis'
    cmd = [sys.executable, str(ROOT/'scripts/analyze_local_learning.py'), '--out', str(out)]
    for folder in inputs: cmd += ['--input', str(folder)]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads((out/'analysis.json').read_text())


def test_tracked_t90_failures_keep_horizon_penalty(tmp_path):
    result = analyze(tmp_path, matrix(tmp_path/'a', 'joint_n3'))
    assert result['audit_passed']
    assert result['arms']['joint_n3']['conditional_means']['T90_with_horizon_penalty'] == 180.


def test_oracle_termination_blocks_tracked_ranking(tmp_path):
    result = analyze(tmp_path, matrix(tmp_path/'a', 'joint_n3', observed=False))
    assert not result['audit_passed']
    assert 'Tracked rollout used oracle completion' in result['issues']
    assert result['comparisons'] == []


def test_development_and_confirmation_cannot_be_pooled(tmp_path):
    result = analyze(tmp_path, matrix(tmp_path/'a', 'joint_n3'),
                     matrix(tmp_path/'b', 'joint_n1', stage='confirmation'))
    assert not result['audit_passed']
    assert 'Mixed development and confirmation stages' in result['issues']


def test_same_source_hash_does_not_excuse_changed_sensor_settings(tmp_path):
    a,b=matrix(tmp_path/'a','joint_n3'),matrix(tmp_path/'b','memory_n3')
    path=b/'attempts.jsonl';row=json.loads(path.read_text())
    row['sensor_assumptions']['latency_s']=0.
    path.write_text(json.dumps(row)+'\n')
    result=analyze(tmp_path,a,b)
    assert not result['audit_passed']
    assert 'Unmatched actual sensor assumptions' in result['issues']
