"""Evaluate an immutable MCA checkpoint with an explicit, auditable sample count."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.mca_physical_policy import make_physical_agent
from scripts.train_mca_compiled import evaluate
from scripts.train_mca_physical import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    # Running research can continue editing the working tree. Evaluation must
    # still import the exact environment/policy implementation of this arm.
    source_root = args.protocol.resolve().parents[2]
    if source_root.name == 'source_snapshot' and source_root != ROOT:
        frozen_helper = source_root / 'scripts/evaluate_mca_frozen_checkpoint.py'
        if not frozen_helper.is_file():
            raise ValueError('The frozen evaluator helper has not been staged')
        os.chdir(source_root)
        os.execv(sys.executable, [sys.executable, '-m', 'scripts.evaluate_mca_frozen_checkpoint', *sys.argv[1:]])
    if args.count < 1 or args.out.exists():
        raise ValueError('Positive episode count and a new output file are required')
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    for name, digest in payload['meta']['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Evaluation source mismatch: {name}')
    protocol = json.loads(args.protocol.read_text())
    cfg = DynamicsConfig.from_json(ROOT / protocol['physics_config'])
    env = CompiledMCAPhysicalEnv(cfg)
    agent = make_physical_agent(env, seed=payload['meta']['training_seed'],
                               hidden_dim=protocol['hidden_dim'], device=args.device,
                               ppo=protocol.get('ppo'))
    for key in ('observation_schema', 'action_semantics', 'physical_action_semantics'):
        if agent.meta[key] != payload['meta'][key]:
            raise ValueError(f'Evaluation checkpoint mismatch: {key}')
    agent.load(args.checkpoint, load_optimizers=False)
    result = evaluate(agent, cfg, protocol, args.count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    atomic_json(args.out.with_suffix('.provenance.json'), dict(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_steps=payload['training_state']['transitions'],
        episodes=args.count, validation_seed_base=protocol['validation_seed_base'],
        protocol=str(args.protocol.resolve()), evaluation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        world_model=False, scope='Repeated development validation on fixed anatomy; not a sealed final test'))
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}), flush=True)


if __name__ == '__main__':
    main()
