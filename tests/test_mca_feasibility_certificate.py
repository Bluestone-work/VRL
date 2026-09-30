import hashlib
import json
import pytest
from scripts.mca_training_gate import require_feasibility_certificate


def evidence(tmp_path):
    physics='configs/task.json';source='environments/mca_physical_env.py'
    for name in (physics,source):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('original')
    hashes={n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in (physics,source)}
    protocol=dict(physics_config=physics,feasibility_certificate='compiled.json',reference_certificate='reference.json')
    for filename,backend in [('compiled.json','compiled'),('reference.json','reference')]:
        rows=[dict(seed=i,success=True,backend=backend,lost_robots=0,max_action_norm=.9,
                   remaining_masses=[0.,0.,0.,0.]) for i in range(20)]
        (tmp_path/filename).write_text(json.dumps(dict(passed=True,config=physics,source_sha256=hashes,episodes=rows)))
    return protocol


def test_bound_evidence_passes_without_claiming_policy_success(tmp_path):
    result=require_feasibility_certificate(evidence(tmp_path),tmp_path)
    assert result['constructive_feasibility_validated']
    assert result['policy_success_rate'] is None


def test_code_or_config_changes_invalidate_previous_witnesses(tmp_path):
    protocol=evidence(tmp_path)
    (tmp_path/protocol['physics_config']).write_text('changed flow or horizon')
    with pytest.raises(ValueError,match='Stale feasibility'):
        require_feasibility_certificate(protocol,tmp_path)


def test_success_label_cannot_hide_incomplete_clearance(tmp_path):
    protocol=evidence(tmp_path);p=tmp_path/'compiled.json';report=json.loads(p.read_text())
    report['episodes'][0]['remaining_masses'][0]=.01;p.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='failed constructive'):
        require_feasibility_certificate(protocol,tmp_path)


def test_missing_reference_evidence_blocks_launch(tmp_path):
    protocol=evidence(tmp_path);protocol.pop('reference_certificate')
    with pytest.raises(ValueError,match='Missing reference'):
        require_feasibility_certificate(protocol,tmp_path)
