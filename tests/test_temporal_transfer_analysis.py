from copy import deepcopy
from scripts.analyze_temporal_transfer import audit_shared_sources


def manifests():
    a = dict(source_hashes={'sensor': 'a', 'physics': 'b', 'reward': 'c'},
             observation_contract='tracked', stage='validation', scenes=[1520000000])
    b = deepcopy(a)
    b['source_hashes']['temporal_model'] = 'd'
    b['observation_contract'] = 'temporal'
    return a,b


def test_model_extension_is_allowed_but_shared_sensor_changes_are_not():
    a,b = manifests()
    assert not audit_shared_sources(a,b)
    b['source_hashes']['sensor'] = 'changed'
    assert 'Shared tracked source changed between model revisions' in audit_shared_sources(a,b)


def test_transfer_rejects_different_physical_scene_pool_and_confirmation():
    a,b = manifests()
    b['scenes'] = [1530000000]
    b['stage'] = 'confirmation'
    issues = audit_shared_sources(a,b)
    assert 'Unmatched scene pools' in issues
    assert 'Cross-revision development analysis rejects confirmation data' in issues
