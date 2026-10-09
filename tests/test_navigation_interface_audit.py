import json

import numpy as np
import pytest

from scripts.audit_navigation_interface import dataset_audit, paired_audit


def test_dataset_audit_records_ambiguous_but_aligned_labels(tmp_path):
    costs = np.ones((2, 7), np.float32)
    costs[0, 0], costs[0, 6] = 0., .01
    costs[1, 3], costs[1, 6] = 0., .02
    sequence = np.zeros((2, 1, 65), np.float32)
    sequence[:, -1, 25] = .1
    sequence[:, -1, 26] = 1.
    np.savez(tmp_path/'episode.npz', label=np.array([0, 3]), costs=costs, seq=sequence)
    result = dataset_audit(tmp_path)
    assert result['label_argmin_consistent']
    assert result['margin_below_005'] == 1.
    assert result['detected_gap_below_02']['samples'] == 2
    assert result['always_stop_mean_regret'] == pytest.approx(.015)


def episode(method, strict, scenario='same', initial='initial'):
    row = dict(anatomy='test', seed=2600000000, clusters=1, method=method, scenario_hash=scenario,
               initial_state_hash=initial, cluster_safe_success=strict, task_success=True,
               lost=0, obstacle_events_static=0, obstacle_events_dynamic=0, wall_contact_s=0.,
               max_continuous_wall_contact_s=0.)
    return dict(method=method, row=row, error=None)


def test_pairing_validates_states_and_uses_strict_obstacle_safe_key(tmp_path):
    path = tmp_path/'pairs.jsonl'
    records = [episode('base', False), episode('candidate', True)]
    path.write_text('\n'.join(json.dumps(record) for record in records))
    result = paired_audit(path, path, 'base', 'candidate')
    assert result['paired'] == 1
    assert result['metrics']['cluster_safe_success']['delta'] == 1.


@pytest.mark.parametrize('field', ['scenario_hash', 'initial_state_hash'])
def test_pairing_rejects_unmatched_initial_conditions(tmp_path, field):
    path = tmp_path/'pairs.jsonl'
    records = [episode('base', False), episode('candidate', True)]
    records[1]['row'][field] = 'different'
    path.write_text('\n'.join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match='mismatch'):
        paired_audit(path, path, 'base', 'candidate')


def test_pairing_rejects_duplicate_episode_rows(tmp_path):
    path = tmp_path/'pairs.jsonl'
    records = [episode('base', False), episode('base', False), episode('candidate', True)]
    path.write_text('\n'.join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match='Duplicate'):
        paired_audit(path, path, 'base', 'candidate')
