"""Retrospective read-only action/metric tap for one preserved diagnostic row.

The policy source is unchanged. Observed association is not causal attribution.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from environments.mca_physical_env import MCAPhysicalEnv, DynamicsConfig
from marl.multicluster import MultiClusterConfig, MultiClusterController
from marl.fair_reactive import fair_reactive_action
from scripts.evaluate_multicluster import run_episode, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    original = next(r for r in rows if r['seed'] == args.seed)
    assert original['method'] in ('multi_parallel', 'multi_unshielded')
    assert original['status'] == 'completed'
    assert original['source_hashes'] == source_hashes()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'manifest.json').write_text(json.dumps(dict(
        purpose='retrospective diagnostic tap; association only; no policy change',
        source_row=str(args.input), seed=args.seed, source_hashes=source_hashes(),
        original_final_state_hash=original['final_state_hash']), indent=2)+'\n')
    protocol = MultiClusterConfig(**original['controller_config'])
    names = {f.name for f in fields(DynamicsConfig)}
    config = DynamicsConfig(**{k: v for k, v in original['actual_config'].items() if k in names})
    # paired_environment expects the base (pre-N-division) catalytic rate.
    if original['resource_budget'] == 'fixed_total':
        from dataclasses import replace
        config = replace(config, lysis_mass_per_s=config.lysis_mass_per_s*protocol.clusters)
    act, step = MultiClusterController.act, MCAPhysicalEnv.step
    pending, counts = {}, dict(agent_steps=0, filter_changed_agent_steps=0,
        wall_agent_steps=0, wall_with_filter_changed=0, wall_with_nominal_outward=0,
        wall_with_filter_newly_outward=0, wall_without_filter_change=0)
    with (args.out/'trace.jsonl').open('x') as stream:
        def tapped_act(self, observation):
            nominal = fair_reactive_action(observation.navigation, mode='path')
            local, control = act(self, observation)
            offset = observation.navigation[:, 6:9].astype(float)
            radial = offset/np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), 1e-12)
            pending.clear()
            pending.update(nominal=nominal, actual=local.copy(),
                nominal_outward=np.sum(radial*nominal, axis=1),
                actual_outward=np.sum(radial*local, axis=1),
                clearance=observation.navigation[:, 9].copy(), active=observation.active.copy(),
                visible_peer_count=observation.peer_visible.sum(axis=1))
            return local, control
        def tapped_step(self, action):
            result = step(self, action)
            info = result[-1]
            wall = np.asarray(info['wall_contact_s']) > 0
            changed = np.linalg.norm(pending['actual']-pending['nominal'], axis=1) > 1e-6
            outward = pending['nominal_outward'] > .05
            new_outward = (pending['actual_outward'] > .05) & ~outward
            counts['agent_steps'] += int(pending['active'].sum())
            counts['filter_changed_agent_steps'] += int(changed.sum())
            counts['wall_agent_steps'] += int(wall.sum())
            counts['wall_with_filter_changed'] += int((wall & changed).sum())
            counts['wall_without_filter_change'] += int((wall & ~changed).sum())
            counts['wall_with_nominal_outward'] += int((wall & outward).sum())
            counts['wall_with_filter_newly_outward'] += int((wall & changed & new_outward).sum())
            trace = {k: v.tolist() for k, v in pending.items()}
            trace.update(elapsed_s=float(info['elapsed_s']), wall_contact_s=info['wall_contact_s'].tolist())
            stream.write(json.dumps(trace)+'\n')
            return result
        MultiClusterController.act = tapped_act
        MCAPhysicalEnv.step = tapped_step
        try:
            replay = run_episode(config, protocol, args.seed, budget=original['resource_budget'])
        finally:
            MultiClusterController.act = act
            MCAPhysicalEnv.step = step
    (args.out/'replay.json').write_text(json.dumps(replay, indent=2)+'\n')
    same = replay['final_state_hash'] == original['final_state_hash']
    summary = dict(seed=args.seed, final_state_identical=same, counts=counts,
        wall_contact_s=replay['wall_contact_s'], removal=replay['removal'],
        spacing_violation_pair_s=replay['spacing_violation_pair_s'],
        warning='Wall/action associations do not establish a causal controller failure.')
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    assert same, 'Diagnostic tap changed final state; discard its interpretation'


if __name__ == '__main__':
    main()
