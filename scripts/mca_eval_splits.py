"""Fixed per-anatomy layout splits (validation / diagnostic / sealed test).

The registry is configs/evaluation_splits.json. Splits are seed ranges for
reset_with_valid_particles; this module only resolves and checks them.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT/'configs/evaluation_splits.json'
SPLITS = ('validation', 'diagnostic', 'test')
DEFAULT_ANATOMY = 'mca_m1_lvo'


def load_registry(path=REGISTRY):
    registry = json.loads(Path(path).read_text())
    check_disjoint(registry)
    return registry


def split(name, anatomy=DEFAULT_ANATOMY, registry=None):
    """Return (seed_base, count) for one split of one anatomy."""
    if name not in SPLITS:
        raise ValueError(f'Unknown split {name}')
    registry = registry or load_registry()
    if anatomy not in registry['anatomies']:
        raise ValueError(f'No fixed splits registered for anatomy {anatomy}')
    entry = registry['anatomies'][anatomy][name]
    return int(entry['seed_base']), int(entry['count'])


def seeds(name, anatomy=DEFAULT_ANATOMY, registry=None):
    base, count = split(name, anatomy, registry)
    return list(range(base, base+count))


def check_disjoint(registry):
    """Every split of every anatomy is disjoint from every other and from reserved ranges."""
    ranges = []
    for anatomy, entry in registry['anatomies'].items():
        if set(entry) != set(SPLITS):
            raise ValueError(f'{anatomy}: expected splits {SPLITS}')
        for name in SPLITS:
            base, count = int(entry[name]['seed_base']), int(entry[name]['count'])
            if count <= 0 or base < 0:
                raise ValueError(f'{anatomy}/{name}: invalid range')
            ranges.append((base, base+count, f'{anatomy}/{name}'))
    for r in registry.get('reserved_ranges', []):
        ranges.append((int(r['start']), int(r['stop']), f'reserved:{r["name"]}'))
    ranges.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(ranges, ranges[1:]):
        if b0 < a1:
            raise ValueError(f'Overlapping layout ranges: {an} and {bn}')


def check_protocol(protocol, registry=None, anatomy=DEFAULT_ANATOMY):
    """A training protocol may never use test layouts for training, development or selection."""
    registry = registry or load_registry()
    test_base, test_count = split('test', anatomy, registry)
    used = [protocol.get('validation_seed_base')]
    selection = protocol.get('checkpoint_selection')
    if selection:
        used.append(selection.get('selection_seed_base'))
        vbase, vcount = split('validation', anatomy, registry)
        if (selection['selection_seed_base'], selection['selection_layouts']) != (vbase, vcount):
            raise ValueError('Checkpoint selection must use the registered validation split')
    for base in used:
        if base is not None and test_base <= int(base) < test_base+test_count:
            raise ValueError('Protocol uses sealed test layouts')
    train = protocol.get('training_seed_base')
    if train is not None:
        for seed in protocol.get('seeds', []):
            start = int(train)+int(seed)*10_000_000
            if start < test_base+test_count and test_base < start+10_000_000:
                raise ValueError('Training stream overlaps sealed test layouts')
    return True
