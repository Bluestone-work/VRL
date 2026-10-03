"""Evaluate one frozen checkpoint on the sealed test split, at most once.

Rules enforced here (see configs/evaluation_splits.json):
  * every access is appended to research/sealed_test/LEDGER.json;
  * a checkpoint (by sha256) is evaluated once; asking again returns the stored
    result without running anything, so repeated looks add no information;
  * a study declares how many checkpoints it will test on its first access and
    can never exceed it;
  * the imported environments/marl code must match the checkpoint's recorded
    source hashes, so run from the arm's source_snapshot (PYTHONPATH=.).
Deterministic actions, the same evaluate() as development validation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT/'research/sealed_test'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_frozen_source(payload):
    import environments
    code_root = Path(environments.__file__).resolve().parents[1]
    recorded = payload['meta'].get('source_sha256') or {}
    names = [n for n in recorded if n.startswith(('environments/', 'marl/'))]
    if not names:
        raise ValueError('Checkpoint has no recorded source hashes')
    bad = [n for n in names if not (code_root/n).exists() or _sha(code_root/n) != recorded[n]]
    if bad:
        raise ValueError(f'Imported code does not match the checkpoint sources: {bad[:5]}')
    return str(code_root)


def _worker(job):
    protocol, checkpoint, base, count, anatomy = job
    import torch
    torch.set_num_threads(1)
    from environments.mca_compiled import CompiledMCAPhysicalEnv
    from environments.mca_physical_env import DynamicsConfig
    from marl.mca_physical_policy import make_physical_agent
    from scripts.train_mca_compiled import evaluate
    cfg = DynamicsConfig.from_json(Path(protocol['physics_config']))
    if anatomy != getattr(cfg, 'anatomy', 'mca_m1_lvo'):
        # Older snapshots have no anatomy field and can only evaluate MCA.
        from dataclasses import replace
        cfg = replace(cfg, anatomy=anatomy)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'],
                                hidden_dim=protocol['hidden_dim'], device='cpu', ppo=protocol.get('ppo'))
    agent.load(checkpoint, load_optimizers=False)
    return evaluate(agent, cfg, dict(protocol, validation_seed_base=base), count)['episodes']


def summarize(records):
    return dict(success_rate=float(np.mean([r['success'] for r in records])),
                collision_free_success_rate=float(np.mean([r['collision_free_success'] for r in records])),
                particle_collision_episode_rate=float(np.mean([r['particle_contact_s'] > 1e-12 for r in records])),
                mean_removal_fraction=float(np.mean([r['removal_fraction'] for r in records])),
                **({'safe_success_rate': float(np.mean([r['safe']['safe_success'] for r in records]))}
                   if all('safe' in r for r in records) else {}))


class Ledger:
    def __init__(self, directory=LEDGER_DIR):
        self.dir = Path(directory); self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir/'LEDGER.json'
        self._lock = open(self.dir/'.lock', 'a+')

    def __enter__(self):
        fcntl.flock(self._lock, fcntl.LOCK_EX)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else dict(studies={}, entries=[])
        return self

    def save(self):
        tmp = self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(self.data, indent=1)+'\n'); tmp.replace(self.path)

    def __exit__(self, *exc):
        fcntl.flock(self._lock, fcntl.LOCK_UN)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--study', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--anatomy', default='mca_m1_lvo')
    p.add_argument('--declare-candidates', type=int, help='Required on a study first access')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--ledger-dir', type=Path, default=LEDGER_DIR)
    p.add_argument('--count', type=int, help='Testing only: evaluate the first N test layouts (not a reportable result)')
    args = p.parse_args(argv)
    # Load the registry helpers from this repository even when run inside a
    # frozen source_snapshot whose scripts/ package predates them.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_mca_eval_splits', ROOT/'scripts/mca_eval_splits.py')
    splits = importlib.util.module_from_spec(spec); spec.loader.exec_module(splits)
    split, check_protocol = splits.split, splits.check_protocol
    import torch
    protocol = json.loads(args.protocol.read_text())
    check_protocol({k: v for k, v in protocol.items() if k != 'checkpoint_selection'}, anatomy=args.anatomy)
    base, count = split('test', args.anatomy)
    partial = args.count is not None and args.count < count
    if partial:
        if args.ledger_dir.resolve() == LEDGER_DIR.resolve():
            raise ValueError('--count is for tests only and needs a separate --ledger-dir')
        count = args.count
    checkpoint = args.checkpoint.resolve(); digest = _sha(checkpoint)
    with Ledger(args.ledger_dir) as ledger:
        for entry in ledger.data['entries']:
            # The same weights under a different protocol (prior, shield) are a different policy.
            if (entry['checkpoint_sha256'] == digest and entry['anatomy'] == args.anatomy and entry['complete']
                    and entry['partial'] == partial and entry['protocol'] == str(args.protocol.resolve())):
                print(json.dumps(dict(cached=True, **entry)), flush=True)
                return json.loads(Path(entry['result']).read_text())
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    code_root = _check_frozen_source(payload)
    with Ledger(args.ledger_dir) as ledger:
        study = ledger.data['studies'].get(args.study)
        if study is None:
            if not args.declare_candidates:
                raise ValueError('First sealed-test access of a study must declare its candidate count')
            study = ledger.data['studies'][args.study] = dict(declared=args.declare_candidates, used=0, created=time.time())
        if study['used'] >= study['declared']:
            raise ValueError(f'Study {args.study} already used all {study["declared"]} declared sealed-test evaluations')
        study['used'] += 1
        entry = dict(study=args.study, label=args.label, anatomy=args.anatomy, checkpoint=str(checkpoint),
                     checkpoint_sha256=digest, protocol=str(args.protocol.resolve()), code_root=code_root,
                     test_seed_base=base, layouts=count, started=time.time(), complete=False, partial=partial)
        ledger.data['entries'].append(entry); ledger.save()
    chunks = np.array_split(np.arange(count), max(1, min(args.workers, count)))
    jobs = [(protocol, str(checkpoint), base+int(c[0]), len(c), args.anatomy) for c in chunks if len(c)]
    with ProcessPoolExecutor(len(jobs)) as pool:
        records = [r for part in pool.map(_worker, jobs) for r in part]
    if [r['seed'] for r in records] != list(range(base, base+count)):
        raise RuntimeError('Incomplete sealed-test evaluation')
    result = dict(summarize(records), episodes=records,
                  provenance=dict(entry, purpose='sealed test', finished=time.time()))
    out = args.ledger_dir/args.study/f'{args.label}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    out.write_text(json.dumps(result)+'\n')
    with Ledger(args.ledger_dir) as ledger:
        for e in ledger.data['entries']:
            # Match this exact job: parallel jobs may share weights (other anatomies/protocols).
            if (e['checkpoint_sha256'] == digest and e['study'] == args.study and e['label'] == args.label
                    and e['anatomy'] == args.anatomy and e['protocol'] == entry['protocol'] and not e['complete']):
                e.update(complete=True, result=str(out), success_rate=result['success_rate'], finished=time.time())
        ledger.save()
    print(json.dumps({k: v for k, v in result.items() if k not in ('episodes', 'provenance')}), flush=True)
    return result


if __name__ == '__main__':
    main()
