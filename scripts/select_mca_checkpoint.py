"""Evaluate one candidate checkpoint on the disjoint selection layouts.

Selection layouts use protocol['checkpoint_selection']['selection_seed_base'],
never the development validation base, so the development result of the
chosen checkpoint stays an unbiased (non-selected-on) estimate.
Run from inside the arm's source_snapshot so the frozen code is imported.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig
from marl.mca_physical_policy import make_physical_agent
from scripts.train_mca_compiled import evaluate
from scripts.train_mca_physical import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite a selection result')
    torch.set_num_threads(1)
    protocol = json.loads(args.protocol.read_text())
    selection = protocol['checkpoint_selection']
    if selection['selection_seed_base'] == protocol['validation_seed_base']:
        raise ValueError('Selection layouts must differ from development layouts')
    cfg = DynamicsConfig.from_json(Path(protocol['physics_config']))
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    agent = make_physical_agent(CompiledMCAPhysicalEnv(cfg), seed=payload['meta']['training_seed'],
                                hidden_dim=protocol['hidden_dim'], device='cpu', ppo=protocol.get('ppo'))
    agent.load(args.checkpoint, load_optimizers=False)
    result = evaluate(agent, cfg, dict(protocol, validation_seed_base=selection['selection_seed_base']),
                      selection['selection_layouts'])
    result['provenance'] = dict(checkpoint=str(args.checkpoint.resolve()),
                                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                selection_seed_base=selection['selection_seed_base'], purpose='checkpoint selection only')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('episodes', 'provenance')}), flush=True)


if __name__ == '__main__':
    main()
