"""Train/evaluate EXP0051 with duration-correct option PPO and fixed provenance."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from marl.measured_options import OptionActorCritic, smdp_gae
from scripts.option_learning_episode import OptionEpisode, PROTOCOL, ROOT, option_hashes


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def check_development_seed(seed):
    from scripts.multicluster_protocol import reserved_seed
    if reserved_seed(seed):
        raise ValueError('Reserved evaluation scene')
    for path in (ROOT/'configs/experiments').glob('EXP_00*_*.json'):
        data = json.loads(path.read_text())
        start, count = data.get('confirmation_scene_base'), data.get('confirmation_scenes')
        if start is not None and count is not None and start <= seed < start+count:
            raise ValueError('Confirmation pool must remain unopened')


def make_episode(seed, **kwargs):
    from unittest.mock import patch
    from environments.mca_physical_env import MCAPhysicalEnv
    check_development_seed(seed)
    original_reset = MCAPhysicalEnv.reset
    def guarded_reset(env, *, seed=None, options=None):
        if seed is not None:
            check_development_seed(seed)
        return original_reset(env, seed=seed, options=options)
    # This also guards deterministic rejection-resampling seeds BEFORE reset.
    # Each runner is process-isolated, so the scoped guard has no thread race.
    with patch.object(MCAPhysicalEnv, 'reset', guarded_reset):
        episode = OptionEpisode(seed, **kwargs)
    # Rejection sampling may change the accepted seed; audit it as well.
    try:
        check_development_seed(episode.manifest['accepted_seed'])
    except BaseException:
        episode.close()
        raise
    return episode


def train(args):
    protocol = json.loads(PROTOCOL.read_text())
    source = option_hashes()
    torch.set_num_threads(1)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    net = OptionActorCritic(protocol['hidden_dim'], protocol['memory_initial_logit_bias'])
    initial_model = {k: v.detach().clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.Adam(net.parameters(), lr=protocol['learning_rate'])
    manifest = dict(experiment=protocol['experiment'], mode=args.variant,
        training_seed=args.seed, scene_base=args.scene_base, requested_control_steps=args.steps,
        protocol=protocol, source_hashes=source, confirmation_accessed=False,
        started_at=datetime.now().astimezone().isoformat())
    atomic_json(args.out/'manifest.json', manifest)
    steps = episodes = updates = 0
    episode = None
    started = time.monotonic()
    with (args.out/'episode_attempts.jsonl').open('x', buffering=1) as attempt_log, \
            (args.out/'episodes.jsonl').open('x', buffering=1) as ep_log, \
            (args.out/'updates.jsonl').open('x', buffering=1) as update_log:
        def reset():
            seed = args.scene_base+episodes
            attempt_log.write(json.dumps(dict(event='reset_requested', scene_seed=seed, control_steps=steps))+'\n')
            ep = make_episode(seed, option_steps=protocol['option_steps'][args.variant])
            attempt_log.write(json.dumps(dict(event='reset_completed', scene_seed=seed,
                accepted_seed=ep.manifest['accepted_seed'], scenario_hash=ep.manifest['scenario_hash']))+'\n')
            return ep
        try:
            episode = reset()
            while steps < args.steps:
                batch = {k: [] for k in ('features', 'valid', 'action', 'logp', 'value', 'reward', 'done', 'duration')}
                collected = 0
                while collected < protocol['rollout_control_steps'] and steps < args.steps:
                    features, valid = episode.high_features.copy(), episode.high_valid.copy()
                    with torch.no_grad():
                        dist, value = net(torch.from_numpy(features)[None], torch.from_numpy(valid)[None])
                        action = dist.sample()
                        logp = dist.log_prob(action).sum(dim=-1)
                    remaining = min(args.steps-steps, protocol['rollout_control_steps']-collected)
                    reward, done, duration = episode.step_option(action[0].numpy(), maximum_steps=remaining)
                    for key, item in dict(features=features, valid=valid, action=action[0].numpy(),
                        logp=float(logp[0]), value=float(value[0]), reward=reward, done=done, duration=duration).items():
                        batch[key].append(item)
                    steps += duration; collected += duration
                    if done:
                        row = episode.result('learning_sampled_'+args.variant)
                        row.update(training_seed=args.seed, training_steps=steps)
                        ep_log.write(json.dumps(row, allow_nan=False)+'\n')
                        attempt_log.write(json.dumps(dict(event='completed', scene_seed=episode.scene_seed,
                            control_steps=steps, removal=row['removal']))+'\n')
                        episode.close(); episodes += 1
                        episode = reset()
                with torch.no_grad():
                    _, next_value = net(torch.from_numpy(episode.high_features)[None],
                                        torch.from_numpy(episode.high_valid)[None])
                adv, returns = smdp_gae(batch['reward'], batch['value'], batch['done'], batch['duration'],
                    float(next_value[0]), protocol['gamma_per_control_step'], protocol['gae_lambda_per_control_step'])
                features = torch.from_numpy(np.asarray(batch['features']))
                valid = torch.from_numpy(np.asarray(batch['valid']))
                actions = torch.from_numpy(np.asarray(batch['action']))
                oldlog = torch.tensor(batch['logp'], dtype=torch.float32)
                advantages = torch.from_numpy(adv)
                advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
                targets = torch.from_numpy(returns)
                losses, kls, entropies = [], [], []
                for epoch in range(protocol['epochs']):
                    indices = torch.randperm(len(features))
                    epoch_kl = []
                    for offset in range(0, len(indices), protocol['minibatch_size']):
                        ii = indices[offset:offset+protocol['minibatch_size']]
                        dist, value = net(features[ii], valid[ii])
                        logp = dist.log_prob(actions[ii]).sum(dim=-1)
                        ratio = torch.exp(logp-oldlog[ii])
                        gain = torch.minimum(ratio*advantages[ii],
                            ratio.clamp(1-protocol['clip_coef'], 1+protocol['clip_coef'])*advantages[ii])
                        entropy = dist.entropy().sum(dim=-1).mean()
                        value_loss = .5*(value-targets[ii]).square().mean()
                        loss = -gain.mean()+.5*value_loss-protocol['entropy_coef']*entropy
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite option PPO loss')
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), .5); optimizer.step()
                        with torch.no_grad():
                            kl = float(((ratio-1)-(logp-oldlog[ii])).mean())
                        losses.append(float(loss.detach())); kls.append(kl); epoch_kl.append(kl)
                        entropies.append(float(entropy.detach()))
                    if np.mean(epoch_kl) > protocol['target_kl']:
                        break
                updates += 1
                row = dict(control_steps=steps, macro_samples=len(features), episodes=episodes,
                    updates=updates, loss=float(np.mean(losses)), kl=float(np.mean(kls)),
                    entropy=float(np.mean(entropies)), wall_s=time.monotonic()-started,
                    fps=steps/(time.monotonic()-started))
                update_log.write(json.dumps(row, allow_nan=False)+'\n')
                atomic_json(args.out/'status.json', dict(phase='training', **row))
                print(json.dumps(row), flush=True)
            assert source == option_hashes(), 'Frozen training source changed'
            delta = float(sum((v.detach()-initial_model[k]).square().sum().item()
                              for k, v in net.state_dict().items())**.5)
            payload = dict(model=net.state_dict(), optimizer=optimizer.state_dict(), source_hashes=source,
                protocol=protocol, variant=args.variant, training_seed=args.seed,
                transitions=steps, model_l2_change=delta, scene_base=args.scene_base)
            path = args.out/f'policy_{steps}.pt'
            torch.save(payload, path)
            attempt_log.write(json.dumps(dict(event='budget_cutoff', scene_seed=episode.scene_seed,
                control_steps=steps, elapsed_s=episode.env.elapsed_s, not_completed_episode=True))+'\n')
            atomic_json(args.out/'status.json', dict(phase='completed', checkpoint=str(path),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_l2_change=delta, **row))
        except BaseException as exc:
            atomic_json(args.out/'python_failure.json', dict(error=repr(exc), control_steps=steps,
                current_scene=args.scene_base+episodes))
            raise
        finally:
            if episode is not None:
                episode.close()


def evaluate(args):
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL.read_text())
    payload = None
    net = None
    if args.policy in ('learned', 'untrained'):
        net = OptionActorCritic(protocol['hidden_dim'], protocol['memory_initial_logit_bias'])
        if args.policy == 'learned':
            if args.checkpoint is None:
                raise ValueError('Checkpoint required')
            payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            if payload['source_hashes'] != option_hashes():
                raise ValueError('Checkpoint source mismatch')
            if payload['variant'] != args.variant:
                raise ValueError('Wrong option duration for checkpoint')
            net.load_state_dict(payload['model'])
        net.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', buffering=1) as stream:
        for seed in range(args.scene_base, args.scene_base+args.episodes):
            ep = make_episode(seed, clusters=args.clusters,
                option_steps=protocol['option_steps'][args.variant],
                supervisor='legacy' if args.policy == 'legacy_memory' else 'joint',
                coupling=args.coupling, duration=args.duration)
            try:
                while not ep.done:
                    if net is None:
                        choices = ep.conventional_options(args.policy)
                    else:
                        with torch.no_grad():
                            dist, _ = net(torch.from_numpy(ep.high_features)[None],
                                          torch.from_numpy(ep.high_valid)[None])
                            choices = dist.logits.argmax(dim=-1)[0].numpy()
                    ep.step_option(choices)
                row = ep.result(args.policy)
                row.update(variant=args.variant,
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if payload else None,
                    training_seed=payload['training_seed'] if payload else None,
                    training_steps=payload['transitions'] if payload else 0)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                print(json.dumps({k:row[k] for k in ('scene_seed','scheduler','removal','cluster_safe_success',
                    'wall_contact_s','spacing_violation_pair_s','minimum_spacing_mm')}), flush=True)
            finally:
                ep.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    train_p = sub.add_parser('train')
    train_p.add_argument('--seed', type=int, required=True)
    train_p.add_argument('--steps', type=int, default=16384)
    train_p.add_argument('--scene-base', type=int, required=True)
    train_p.add_argument('--variant', choices=('hierarchical','flat'), required=True)
    train_p.add_argument('--out', type=Path, required=True)
    eval_p = sub.add_parser('evaluate')
    eval_p.add_argument('--policy', choices=('memory','balanced','priority','legacy_memory','learned','untrained'), required=True)
    eval_p.add_argument('--variant', choices=('hierarchical','flat'), default='hierarchical')
    eval_p.add_argument('--checkpoint', type=Path)
    eval_p.add_argument('--clusters', type=int, default=3)
    eval_p.add_argument('--scene-base', type=int, required=True)
    eval_p.add_argument('--episodes', type=int, default=1)
    eval_p.add_argument('--duration', type=float, default=180.)
    eval_p.add_argument('--coupling', type=float, default=0.)
    eval_p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    train(args) if args.mode == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
