"""Fixed validation / diagnostic / sealed-test layout splits and the sealed-test ledger."""
import json
from pathlib import Path

import pytest

from scripts.mca_eval_splits import REGISTRY, check_disjoint, check_protocol, load_registry, split
from scripts import evaluate_mca_sealed_test as sealed

ROOT = Path(__file__).resolve().parents[1]


def test_registry_splits_are_disjoint_and_sized():
    registry = load_registry()
    assert split('test') == (990000000, 500)
    assert split('validation') == (950000000, 200)
    assert split('diagnostic') == (940000000, 100)
    bad = json.loads(REGISTRY.read_text())
    bad['anatomies']['mca_m1_lvo']['validation']['seed_base'] = 990000100
    with pytest.raises(ValueError, match='Overlapping'):
        check_disjoint(bad)
    with pytest.raises(ValueError):
        split('test', 'unknown_anatomy', registry)


def test_no_existing_protocol_touches_the_sealed_test_layouts():
    for path in sorted((ROOT/'configs/experiments').glob('EXP_00*_PURE_RL.json')):
        protocol = json.loads(path.read_text())
        protocol.pop('checkpoint_selection', None)  # older studies used 100 selection layouts
        assert check_protocol(protocol), path.name


def test_formal_study_selects_on_the_registered_validation_split():
    for arm in ('ASSIGNED', 'CONTROL'):
        protocol = json.loads((ROOT/f'configs/experiments/EXP_0039_{arm}_PURE_RL.json').read_text())
        assert check_protocol(protocol)


def test_protocols_using_test_layouts_are_rejected():
    base = dict(validation_seed_base=940000000, training_seed_base=1900000000, seeds=[42])
    with pytest.raises(ValueError, match='sealed test'):
        check_protocol(dict(base, validation_seed_base=990000010))
    with pytest.raises(ValueError, match='registered validation'):
        check_protocol(dict(base, checkpoint_selection=dict(selection_seed_base=950000000, selection_layouts=100)))
    with pytest.raises(ValueError, match='overlaps'):
        check_protocol(dict(base, training_seed_base=990000000-42*10_000_000))


PREFLIGHT = ROOT/'research/runs/EXP0039_ASSIGNED_PREFLIGHT_20261001/latest.pt'
PROTOCOL = ROOT/'configs/experiments/EXP_0039_ASSIGNED_PURE_RL.json'


def test_partial_runs_cannot_write_the_real_ledger():
    with pytest.raises(ValueError, match='tests only'):
        sealed.main(['--protocol', str(PROTOCOL), '--checkpoint', str(PREFLIGHT), '--study', 'X', '--label', 'x',
                     '--declare-candidates', '1', '--count', '1'])


@pytest.mark.skipif(not PREFLIGHT.exists(), reason='local preflight checkpoint not available')
def test_ledger_caches_repeats_and_enforces_the_declared_budget(tmp_path):
    common = ['--protocol', str(PROTOCOL), '--count', '1', '--workers', '1', '--ledger-dir', str(tmp_path)]
    with pytest.raises(ValueError, match='declare'):
        sealed.main(common+['--checkpoint', str(PREFLIGHT), '--study', 'S', '--label', 'a'])
    first = sealed.main(common+['--checkpoint', str(PREFLIGHT), '--study', 'S', '--label', 'a', '--declare-candidates', '1'])
    assert [e['seed'] for e in first['episodes']] == [990000000]
    again = sealed.main(common+['--checkpoint', str(PREFLIGHT), '--study', 'S', '--label', 'b'])
    assert again['success_rate'] == first['success_rate']
    ledger = json.loads((tmp_path/'LEDGER.json').read_text())
    assert ledger['studies']['S']['used'] == 1 and len(ledger['entries']) == 1
    other = tmp_path/'other.pt'
    other.write_bytes(PREFLIGHT.read_bytes()+b'')  # identical bytes: still cached by sha
    sealed.main(common+['--checkpoint', str(other), '--study', 'S', '--label', 'c'])
    assert json.loads((tmp_path/'LEDGER.json').read_text())['studies']['S']['used'] == 1


def test_checkpoint_from_different_code_is_refused():
    payload = dict(meta=dict(source_sha256={'environments/mca_physical_env.py': '0'*64}))
    with pytest.raises(ValueError, match='does not match'):
        sealed._check_frozen_source(payload)
