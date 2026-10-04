"""Post-hoc measured-state shadow audit; never replace registered evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from marl.tpg_learning import TPGAgent
from scripts.run_tpg_learning import make_episode, high_state, low_state, high_forward, low_forward
from scripts.tpg_episode import tpg_hashes


def evaluate_choices(distribution, preferred, valid):
    logits = distribution.logits.detach().numpy()
    choices = logits.argmax(-1)
    selected = np.take_along_axis(logits, preferred[..., None], axis=-1)[..., 0]
    others = np.where(valid, logits, -1e9).copy()
    np.put_along_axis(others, preferred[..., None], -1e9, axis=-1)
    multiple = valid.sum(axis=-1) > 1
    margins = selected - others.max(axis=-1)
    probability = distribution.probs.detach().numpy()
    nominal_probability = np.take_along_axis(probability, preferred[..., None], axis=-1)[..., 0]
    return dict(total=int(preferred.size), different=int((choices != preferred).sum()),
                multiple=int(multiple.sum()),
                margins=margins[multiple].tolist(), probabilities=nominal_probability.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    audit = json.loads((args.study / 'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:
        raise ValueError('Registered comparison has not passed admission')
    source = tpg_hashes()
    manifest = json.loads((args.study / 'manifest.json').read_text())
    if source != manifest['source_hashes']:
        raise ValueError('Frozen source does not match')
    p = manifest['protocol']
    torch.set_num_threads(1)
    models = []
    for variant in p['variants']:
        for seed in p['training_seeds']:
            folder = args.study / 'training' / f'train_{variant}_{seed}'
            state = json.loads((folder / 'status.json').read_text())
            checkpoint = Path(state['checkpoint'])
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != state['checkpoint_sha256']:
                raise ValueError('Checkpoint digest mismatch')
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            net = TPGAgent(p['hidden_dim'])
            net.load_state_dict(payload['model']); net.eval()
            models.append((variant, seed, net))
    request = dict(post_hoc=True, comparison='shadow on measured rule-policy trajectories',
                   uses_true_navigation_state=False, candidate_actions_executed=False,
                   registered_results_replaced=False, confirmation_accessed=False,
                   scenes=list(range(p['validation_scene_base'], p['validation_scene_base']+p['validation_scenes'])),
                   source_hashes=source)
    (args.out / 'requested.json').write_text(json.dumps(request, indent=2)+'\n')
    records, replay_checks = [], []
    for scene in request['scenes']:
        ep = make_episode(scene, duration=p['duration_s'])
        highs, high_choices, lows, low_choices = [], [], [], []
        try:
            while not ep.done:
                if ep.needs_decision:
                    highs.append(high_state(ep)); preferred = ep.conventional_priority()
                    high_choices.append(preferred); ep.choose_priority(preferred)
                lows.append(low_state(ep)); preferred = ep.conventional_low()
                low_choices.append(preferred.copy()); ep.step_control(preferred)
            result = ep.result('rule_shadow_diagnostic')
        finally:
            ep.close()
        baseline = json.loads((args.study / 'results' / f'tpg_{scene}.jsonl').read_text())
        check = dict(scene=scene, same_initial=result['actual_initial_snapshot_hash']==baseline['actual_initial_snapshot_hash'],
                     same_final=result['final_state_hash']==baseline['final_state_hash'])
        replay_checks.append(check)
        if not check['same_initial'] or not check['same_final']:
            raise RuntimeError('Shadow instrumentation changed the reference replay')
        for variant, seed, net in models:
            for kind, states, preferred in [('high', highs, high_choices), ('low', lows, low_choices)]:
                if kind != variant and variant != 'both':
                    continue
                totals = dict(total=0, different=0, multiple=0)
                margins, probabilities = [], []
                for start in range(0, len(states), 128):
                    state = {key: np.asarray([s[key] for s in states[start:start+128]]) for key in states[0]}
                    with torch.no_grad():
                        if kind == 'high':
                            dist, _ = net.high(torch.from_numpy(state['nodes']), torch.from_numpy(state['pairs']))
                            valid = dist.logits.detach().numpy() > -1e8
                            chosen = np.asarray(preferred[start:start+128])
                        else:
                            dist, _ = net.low(*(torch.from_numpy(state[key]) for key in ('history','candidates','valid','active')))
                            # Active clusters only; invisible/inactive placeholders are not decisions.
                            active = state['active']
                            dist = torch.distributions.Categorical(logits=dist.logits[torch.from_numpy(active)])
                            valid = state['valid'][active]
                            chosen = np.asarray(preferred[start:start+128])[active]
                    counts = evaluate_choices(dist, chosen, valid)
                    for key in totals: totals[key] += counts[key]
                    margins.extend(counts['margins']); probabilities.extend(counts['probabilities'])
                row = dict(variant=variant, seed=seed, scene=scene, level=kind, **totals,
                           minimum_nominal_margin=float(min(margins)) if margins else None,
                           mean_nominal_probability=float(np.mean(probabilities)) if probabilities else None,
                           trained_score_head_weight_norm=float(getattr(net,kind).score[-1].weight.detach().norm()))
                records.append(row)
        (args.out / f'scene_{scene}.json').write_text(json.dumps([r for r in records if r['scene']==scene],indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(scene=scene, replay=check, model_level_rows=sum(r['scene']==scene for r in records))),flush=True)
    if source != tpg_hashes(): raise RuntimeError('Algorithm source changed')
    result = dict(**request, completed=True, replay_checks=replay_checks, records=records,
                  note='Shadow agreement describes these visited states, not unvisited states or retraining efficacy. No bias/mask/reward/checkpoint was changed.')
    (args.out / 'AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
