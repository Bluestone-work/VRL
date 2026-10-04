"""Protect scientific estimands used in the paired diagnostic report."""
import pytest

from scripts.analyze_exp0046_paired import metrics, compare, scene_vectors, analyze


def episode(seed=1, control_seed=42, **changes):
    row = dict(seed=seed, control_seed=control_seed,
        task_success=True, safe_success=True, cluster_safe_success=True,
        removal=1., wall_contact_s=0., particle_contact_s=0., robot_pair_contact_s=0.,
        spacing_violation_pair_s=0., lost_clusters=0, path_mm=10., command_squared_s=5.,
        mass_per_cluster_s=.1, active_cluster_s=10., timeout=False, elapsed_s=10.,
        removal_auc_s=5., spacing_compliant=True,
        time_to_removal_s={'50': 5., '90': 9., '100': 10.}, controller={})
    row.update(changes)
    return row


def test_early_clearance_beats_late_clearance_at_equal_final_removal():
    early = episode()
    late = episode(elapsed_s=180., removal_auc_s=90.,
                   time_to_removal_s={'50': 90., '90': 162., '100': 180.})
    # Raw AUC alone reverses the efficiency ranking when the fast run terminates.
    assert early['removal_auc_s'] < late['removal_auc_s']
    assert metrics(early, 180.)['auc_180_fraction'] == pytest.approx(175/180)
    assert metrics(early, 180.)['auc_180_fraction'] > metrics(late, 180.)['auc_180_fraction']


def test_lost_cluster_early_exit_is_not_treated_as_fast_completion():
    row = episode(elapsed_s=5., removal=.25, removal_auc_s=.5, lost_clusters=1,
                  task_success=False, cluster_safe_success=False,
                  time_to_removal_s={'50': None, '90': None, '100': None})
    values = metrics(row, 180.)
    assert values['t100_horizon_penalty_s'] == 180.
    assert values['t100_reached_rate'] == 0.
    assert values['auc_180_fraction'] == pytest.approx((.5+175*.25)/180)


def test_repeated_control_seeds_do_not_increase_independent_scene_count():
    a = [episode(1, removal=.1), episode(2, removal=.9)]
    b = [episode(1, removal=0.), episode(2, removal=0.)]
    repeated_a = [dict(row, control_seed=k) for row in a for k in (42, 43, 44)]
    repeated_b = [dict(row, control_seed=k) for row in b for k in (42, 43, 44)]
    plain = compare('plain', a, b, 180.)['metrics']['removal']
    repeated = compare('repeated', repeated_a, repeated_b, 180.)['metrics']['removal']
    assert repeated['scenes'] == 2
    assert repeated['mean'] == pytest.approx(plain['mean'])
    assert repeated['ci95'] == pytest.approx(plain['ci95'])


def test_within_scene_mean_and_paired_identity_are_explicit():
    rows = [episode(1, 42, removal=0.), episode(1, 43, removal=1.),
            episode(2, 42, removal=.2), episode(2, 43, removal=.4)]
    scenes = scene_vectors(rows, 180.)
    assert scenes[1]['removal'] == .5
    assert scenes[2]['removal'] == pytest.approx(.3)
    same = compare('same', rows, rows, 180.)
    assert same['metrics']['removal']['ci95'] == [0., 0.]
    with pytest.raises(ValueError, match='incomplete pairing'):
        compare('missing scene', rows, rows[:2], 180.)
    with pytest.raises(ValueError, match='unmatched control seeds'):
        compare('missing repeat', rows, rows[::2], 180.)


def test_native_failure_blocks_comparisons_and_stays_in_bounds(monkeypatch, tmp_path):
    arm = 'single_sequential_n1_d2_fixed_total'
    manifest = dict(duration_s=180., scenarios=[1, 2], cells=[dict(
        method='single_sequential', clusters=1, d_min_mm=2., control_seed=42, budget='fixed_total')])
    rows = [episode(1, status='completed', _arm=arm),
            dict(seed=2, status='process_exit', returncode=-11, _arm=arm)]
    monkeypatch.setattr('scripts.analyze_exp0046_paired.audit',
                        lambda directory: (manifest, rows, {'passed': False, 'issues': ['native crash']}))
    result = analyze(tmp_path)
    assert result['comparisons_blocked']
    assert 'comparisons' not in result
    summary = result['descriptive_only'][arm]
    assert summary['requested'] == 2
    assert summary['failed'] == 1
    assert summary['cluster_safe_success_bounds'] == [.5, 1.]
    assert summary['conditional_completed_means']['cluster_safe_success'] == 1.
