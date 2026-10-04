"""EXP0054 guarded training/evaluation, separate from every frozen older run."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

from marl.measured_marl import MeasuredMARL, bids_to_priority, temporal_gae
from scripts.tpg_episode import ROOT, tpg_hashes
from scripts.run_tpg_learning import make_episode
from scripts.run_option_learning import atomic_json

PROTOCOL = ROOT/'configs/experiments/EXP_0054_MEASURED_MARL.json'
STATE_KEYS = ('history', 'nodes', 'pairs', 'candidates', 'valid', 'active')


def source_hashes():
    result = tpg_hashes()
    for name in ('marl/measured_marl.py', 'scripts/run_measured_marl.py',
                 'scripts/run_measured_marl_study.py', 'scripts/analyze_measured_marl.py',
                 'configs/experiments/EXP_0054_MEASURED_MARL.json'):
        result[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def runtime_context():
    from numpy.core import _multiarray_umath
    return dict(executable=sys.executable, numpy=np.__version__, torch=torch.__version__,
        numpy_binary_sha256=hashlib.sha256(Path(_multiarray_umath.__file__).read_bytes()).hexdigest())


def enforce_runtime():
    reference = json.loads((ROOT/'research/validation/EXP0053_RUNTIME_REPAIR_20261004/generic_runtime_context.json').read_text())
    context = runtime_context()
    for key in ('executable', 'numpy', 'torch', 'numpy_binary_sha256'):
        if context[key] != reference[key]:
            raise RuntimeError('Unadmitted runtime: '+key)
    return context


def observation(ep):
    # Only data already derived by the common measured observer/controller.
    return dict(history=ep.low_history.copy(), nodes=ep.nodes.copy(), pairs=ep.pairs.copy(),
                candidates=ep.low_candidates.copy(), valid=ep.low_valid.copy(), active=ep.scene.active.copy())


def tensors(state):
    return {k: torch.from_numpy(np.asarray(state[k]))[None] for k in STATE_KEYS}


def act(net, level, state, greedy=False):
    with torch.no_grad():
        dist, value = net(level, tensors(state))
        action = dist.logits.argmax(-1) if greedy else dist.sample()
        return action[0].numpy(), dist.log_prob(action)[0].numpy(), value[0].numpy()


def priority_action(net, action, state):
    return int(action) if net.joint_high else bids_to_priority(action, state['nodes'], state['pairs'])


def update(net, optimizer, records, level, bootstrap, p):
    if not records:
        return dict(samples=0)
    states = {k: torch.from_numpy(np.asarray([r[k] for r in records])) for k in STATE_KEYS}
    actions = torch.from_numpy(np.asarray([r['action'] for r in records]))
    oldlog = torch.from_numpy(np.asarray([r['logp'] for r in records], np.float32))
    adv, ret = temporal_gae(records, bootstrap, p['gamma_per_control_step'], p['gae_lambda_per_control_step'])
    adv, ret = torch.from_numpy(adv), torch.from_numpy(ret)
    joint = level == 'high' and net.joint_high
    mask = states['active'].any(-1) if joint else states['active']
    if not mask.any():
        return dict(samples=len(records), active_samples=0, skipped=True)
    chosen_adv = adv[mask]
    adv = (adv-chosen_adv.mean())/(chosen_adv.std(unbiased=False)+1e-8)
    aux_enabled = level == 'low' and net.variant == 'vctpg_ac'
    if aux_enabled:
        commands = torch.from_numpy(np.asarray([r['commands'] for r in records], np.float32))
        labels = torch.from_numpy(np.asarray([r['measured_velocity'] for r in records], np.float32))
        fresh = torch.from_numpy(np.asarray([r['measurement_valid'] for r in records], bool))
    losses, kls, auxs, entropies = [], [], [], []
    for epoch in range(p['epochs']):
        epoch_kl = []
        for indices in torch.randperm(len(records)).split(p['minibatch_size']):
            valid = mask[indices]
            if not valid.any():
                continue
            state = {k: v[indices] for k, v in states.items()}
            dist, value = net(level, state)
            logp = dist.log_prob(actions[indices])
            delta = logp-oldlog[indices]
            ratio = delta.exp()
            gain = torch.minimum(ratio*adv[indices],
                ratio.clamp(1-p['clip_coef'], 1+p['clip_coef'])*adv[indices])
            entropy = dist.entropy()[valid].mean()
            loss = -gain[valid].mean() + p['value_coef']*(value-ret[indices]).square()[valid].mean() - p['entropy_coef']*entropy
            auxiliary = torch.zeros(())
            if aux_enabled and fresh[indices].any():
                prediction = net.low_actor.predict(state, commands[indices])
                auxiliary = torch.nn.functional.smooth_l1_loss(prediction[fresh[indices]], labels[indices][fresh[indices]])
                loss = loss + p['auxiliary_weight']*auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite measured-MARL loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters_for(level), .5)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite measured-MARL gradient')
            optimizer.step()
            kl = float(((ratio-1)-delta)[valid].mean().detach())
            losses.append(float(loss.detach())); kls.append(kl); epoch_kl.append(kl)
            auxs.append(float(auxiliary.detach())); entropies.append(float(entropy.detach()))
        if epoch_kl and np.mean(epoch_kl) > p['target_kl']:
            break
    return dict(samples=len(records), active_samples=int(mask.sum()), loss=float(np.mean(losses)),
                kl=float(np.mean(kls)), auxiliary=float(np.mean(auxs)), entropy=float(np.mean(entropies)))


def train(args):
    runtime = enforce_runtime()
    p = json.loads(PROTOCOL.read_text()); source = source_hashes()
    torch.set_num_threads(1); torch.manual_seed(args.seed); np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    net = MeasuredMARL(args.variant, p['hidden_dim'], p['clusters'], p['prior_logit_bias'])
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    optimizers = {k: torch.optim.Adam(net.parameters_for(k), lr=p['learning_rate']) for k in ('high', 'low')}
    atomic_json(args.out/'manifest.json', dict(protocol=p, source_hashes=source, runtime=runtime,
        variant=args.variant, training_seed=args.seed, requested_steps=args.steps, scene_base=args.scene_base,
        started_at=datetime.now().astimezone().isoformat(), confirmation_accessed=False,
        parameters=sum(x.numel() for x in net.parameters())))
    steps = episodes = low_updates = high_updates = high_samples = 0
    high_records = []; pending = ep = None
    checkpoint_files = []; started = time.monotonic()
    with (args.out/'episodes.jsonl').open('x', buffering=1) as episode_log, \
         (args.out/'attempts.jsonl').open('x', buffering=1) as attempts, \
         (args.out/'updates.jsonl').open('x', buffering=1) as updates:
        def reset():
            seed = args.scene_base+episodes
            attempts.write(json.dumps(dict(event='reset_requested', scene_seed=seed, steps=steps))+'\n')
            new = make_episode(seed, duration=p['duration_s'])
            attempts.write(json.dumps(dict(event='reset_completed', scene_seed=seed,
                accepted_seed=new.manifest['accepted_seed'], scenario_hash=new.manifest['scenario_hash']))+'\n')
            return new

        def save():
            if source != source_hashes():
                raise RuntimeError('Source changed during training')
            delta = {name: float(sum((v.detach()-initial[k]).square().sum().item()
                for k, v in net.state_dict().items() if k.startswith(name+'.'))**.5)
                for name in ('high_actor', 'low_actor', 'high_critic', 'low_critic')}
            path = args.out/f'policy_{steps}.pt'
            torch.save(dict(model=net.state_dict(), variant=args.variant, protocol=p,
                source_hashes=source, runtime=runtime, training_seed=args.seed,
                transitions=steps, model_l2_change=delta), path)
            checkpoint_files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), steps=steps))
            return delta

        try:
            ep = reset()
            while steps < args.steps:
                low_records = []
                for _ in range(min(p['rollout_control_steps'], args.steps-steps)):
                    if ep.needs_decision:
                        state = observation(ep)
                        with torch.no_grad():
                            _, vnext = net('high', tensors(state))
                        if pending is not None:
                            pending['next_value'] = vnext[0].numpy()
                            high_records.append(pending); pending = None
                        if len(high_records) >= p['high_update_events']:
                            stat = update(net, optimizers['high'], high_records, 'high', vnext[0].numpy(), p)
                            high_samples += len(high_records); high_updates += 1
                            updates.write(json.dumps(dict(level='high', steps=steps, **stat))+'\n')
                            high_records = []
                        action, logp, value = act(net, 'high', state)
                        pending = dict(**state, action=action, logp=logp, value=value, reward=0., duration=0, done=False)
                        ep.choose_priority(priority_action(net, action, state))
                    state = observation(ep)
                    action, logp, value = act(net, 'low', state)
                    reward, done, aux = ep.step_control(action)
                    low_records.append(dict(**state, action=action, logp=logp, value=value,
                        reward=reward, done=done, duration=1, **aux))
                    if pending is not None:
                        pending['reward'] += p['gamma_per_control_step']**pending['duration']*reward
                        pending['duration'] += 1; pending['done'] = done
                    steps += 1
                    if done:
                        if pending is not None:
                            pending['next_value'] = np.zeros_like(pending['value'])
                            high_records.append(pending); pending = None
                        row = ep.result('sampled_'+args.variant)
                        row.update(training_seed=args.seed, training_steps=steps)
                        episode_log.write(json.dumps(row, allow_nan=False)+'\n')
                        attempts.write(json.dumps(dict(event='episode_completed', scene_seed=ep.scene_seed, steps=steps))+'\n')
                        ep.close(); episodes += 1
                        ep = reset() if steps < args.steps else None
                bootstrap = np.zeros(p['clusters'], np.float32)
                if ep is not None:
                    with torch.no_grad():
                        _, value = net('low', tensors(observation(ep)))
                    bootstrap = value[0].numpy()
                stats = update(net, optimizers['low'], low_records, 'low', bootstrap, p)
                low_updates += 1
                updates.write(json.dumps(dict(level='low', steps=steps, **stats))+'\n')
                status = dict(phase='training', steps=steps, episodes=episodes, high_samples=high_samples,
                    high_updates=high_updates, low_updates=low_updates, pending_high_events=len(high_records),
                    low=stats, wall_s=time.monotonic()-started)
                atomic_json(args.out/'status.json', status)
                print(json.dumps(status), flush=True)
                if steps in p['checkpoints'] and steps != args.steps:
                    save()
            if high_records:
                bootstrap = high_records[-1]['next_value']
                stats = update(net, optimizers['high'], high_records, 'high', bootstrap, p)
                high_samples += len(high_records); high_updates += 1
                updates.write(json.dumps(dict(level='high_final_completed_events', steps=steps, **stats))+'\n')
            if pending is not None:
                attempts.write(json.dumps(dict(event='censored_high_option', duration=pending['duration'], not_trained=True))+'\n')
            if ep is not None:
                attempts.write(json.dumps(dict(event='budget_cutoff', scene_seed=ep.scene_seed, incomplete=True))+'\n')
            delta = save()
            atomic_json(args.out/'status.json', {**status, 'phase':'completed',
                'high_samples':high_samples, 'high_updates':high_updates, 'checkpoints':checkpoint_files,
                'model_l2_change':delta, 'wall_s':time.monotonic()-started})
        except BaseException as exc:
            atomic_json(args.out/'failure.json', dict(error=repr(exc), steps=steps, episodes=episodes))
            raise
        finally:
            if ep is not None:
                ep.close()


def evaluate(args):
    runtime = enforce_runtime()
    p = json.loads(PROTOCOL.read_text()); source = source_hashes()
    torch.set_num_threads(1); torch.manual_seed(42)
    net = MeasuredMARL(args.variant, p['hidden_dim'], p['clusters'], p['prior_logit_bias'])
    payload = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if payload['source_hashes'] != source or payload['runtime'] != runtime or payload['variant'] != args.variant:
            raise ValueError('Checkpoint provenance mismatch')
        net.load_state_dict(payload['model'])
    net.eval(); args.out.parent.mkdir(parents=True, exist_ok=True)
    ep = make_episode(args.scene, duration=args.duration)
    high_calls = low_calls = high_nonrule = low_nonrule = active_choices = 0
    prior_probability = 0.
    started = time.monotonic()
    with args.out.open('x') as stream:
        try:
            while not ep.done:
                if ep.needs_decision:
                    conventional = ep.conventional_priority()
                    if args.rule:
                        priority = conventional
                    else:
                        state = observation(ep)
                        action, _, _ = act(net, 'high', state, True)
                        priority = priority_action(net, action, state)
                        high_calls += 1
                        high_nonrule += priority != conventional
                    ep.choose_priority(priority)
                conventional = ep.conventional_low()
                if args.rule:
                    choices = conventional
                else:
                    state = observation(ep)
                    with torch.no_grad():
                        dist, _ = net('low', tensors(state))
                        choices = dist.logits.argmax(-1)[0].numpy()
                        probabilities = dist.probs[0].numpy()
                    active = state['active']
                    low_nonrule += int(np.sum((choices != conventional) & active))
                    active_choices += int(active.sum())
                    prior_probability += float(probabilities[np.arange(p['clusters']), conventional][active].sum())
                    low_calls += 1
                ep.step_control(choices)
            row = ep.result('tpg' if args.rule else args.variant)
            row.update(experiment=p['experiment'], variant='tpg' if args.rule else args.variant,
                training_seed=payload['training_seed'] if payload else None,
                training_steps=payload['transitions'] if payload else 0,
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if payload else None,
                source_hashes=source, runtime=runtime, high_network_calls=high_calls, low_network_calls=low_calls,
                high_nonrule_choices=high_nonrule, low_nonrule_choices=low_nonrule, active_low_choices=active_choices,
                mean_rule_probability=prior_probability/max(active_choices, 1), wall_clock_s=time.monotonic()-started,
                baseline_scope=p['scope'], confirmation_accessed=False)
            if source != source_hashes():
                raise RuntimeError('Source changed during evaluation')
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps({k: row[k] for k in ('variant','scene_seed','removal','cluster_safe_success','high_nonrule_choices','low_nonrule_choices')}), flush=True)
        finally:
            ep.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    variants = ('r_mappo', 'r_ippo', 'vctpg_ac', 'vctpg_no_ac')
    tr = sub.add_parser('train')
    tr.add_argument('--variant', choices=variants, required=True)
    tr.add_argument('--seed', type=int, required=True)
    tr.add_argument('--steps', type=int, required=True)
    tr.add_argument('--scene-base', type=int, required=True)
    tr.add_argument('--out', type=Path, required=True)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--variant', choices=variants, default='r_mappo')
    ev.add_argument('--scene', type=int, required=True)
    ev.add_argument('--duration', type=float, default=180.)
    ev.add_argument('--checkpoint', type=Path)
    ev.add_argument('--rule', action='store_true')
    ev.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.mode == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
