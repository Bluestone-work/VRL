"""Train/evaluate the registered observation-only EXP0047 PPO selector."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from marl.local_maneuver_learning import LocalManeuverActorCritic
from scripts.local_learning_episode import LearningEpisode, PROTOCOL, ROOT, learning_hashes


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def advantages(rewards, values, dones, last_value, gamma, lam):
    adv = np.zeros_like(rewards)
    carry = np.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        following = last_value if t == len(rewards)-1 else values[t+1]
        alive = 1-dones[t]
        delta = rewards[t]+gamma*following*alive-values[t]
        carry = delta+gamma*lam*alive*carry
        adv[t] = carry
    return adv, adv+values


def train(args):
    protocol = json.loads(PROTOCOL.read_text())
    source = learning_hashes()
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    for file in source:
        dest = out/'source_snapshot'/file
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/file, dest)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = LocalManeuverActorCritic(protocol['hidden_dim'])
    optimizer = torch.optim.Adam(net.parameters(), lr=protocol['learning_rate'])
    completed = transitions = updates = 0
    parent = None
    if args.initialize is not None:
        parent = torch.load(args.initialize, map_location='cpu', weights_only=False)
        if parent['source_hashes'] != source or parent['training_seed'] != args.seed:
            raise ValueError('Continuation checkpoint/source/seed mismatch')
        if args.steps <= parent['transitions']:
            raise ValueError('Continuation target must exceed parent budget')
        net.load_state_dict(parent['model']); optimizer.load_state_dict(parent['optimizer'])
        transitions = parent['transitions']
        torch.set_rng_state(parent['torch_rng']); np.random.set_state(parent['numpy_rng'])
    initial_steps = transitions
    atomic_json(out/'manifest.json', dict(protocol=protocol, training_seed=args.seed,
        scene_base=args.scene_base, target_steps=args.steps, source_hashes=source,
        started=datetime.now().astimezone().isoformat(), device='cpu', inference_no_privileged_state=True,
        continuation=str(args.initialize) if args.initialize is not None else None,
        initial_steps=initial_steps, continuation_starts_fresh_disjoint_scenes=parent is not None))
    started = time.monotonic()
    attempts_path = out/'attempts.jsonl'
    attempts_path.touch(exist_ok=False)
    def attempt_event(event, **extra):
        with attempts_path.open('a') as f:
            f.write(json.dumps(dict(event=event, scene_seed=args.scene_base+completed,
                                   training_steps=transitions, **extra))+'\n')
    def new_episode():
        attempt_event('started')
        return LearningEpisode(args.scene_base+completed, control_seed=args.seed,
                               reward_config=protocol['reward'])
    episode = new_episode()
    next_checkpoint = (transitions//8192+1)*8192
    def checkpoint():
        payload = dict(model=net.state_dict(), optimizer=optimizer.state_dict(),
            transitions=transitions, completed_episodes=completed, training_seed=args.seed,
            source_hashes=source, protocol=protocol, torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(), resume_supported='weights_optimizer_rng_with_fresh_disjoint_scenes')
        path = out/f'policy_{transitions}.pt'
        temp = path.with_suffix('.tmp'); torch.save(payload, temp); temp.replace(path)
        return path
    checkpoint()
    try:
        with (out/'episodes.jsonl').open('x', buffering=1) as episode_log, \
             (out/'updates.jsonl').open('x', buffering=1) as update_log:
            while transitions < args.steps:
                data = {k: [] for k in ('x', 'valid', 'actions', 'logp', 'value', 'reward', 'done', 'active')}
                length = min(protocol['rollout_steps'], args.steps-transitions)
                for _ in range(length):
                    x = torch.from_numpy(episode.features)
                    valid = torch.from_numpy(episode.valid)
                    with torch.no_grad():
                        dist = net.distribution(x, valid)
                        action = dist.sample()
                        logp, value = dist.log_prob(action).numpy(), net.value(x).numpy()
                    before_x, before_valid = episode.features.copy(), episode.valid.copy()
                    reward, done, active, lost = episode.step(action.numpy())
                    row = dict(x=before_x, valid=before_valid, actions=action.numpy(), logp=logp,
                        value=value, reward=reward, done=np.full(len(active), done, dtype=float)+lost.astype(float), active=active)
                    row['done'] = np.minimum(row['done'], 1.)
                    for k, v in row.items(): data[k].append(v)
                    transitions += 1
                    if done:
                        result = episode.result('learning_training_sampled')
                        result.update(training_step=transitions, training_seed=args.seed)
                        episode_log.write(json.dumps(result, allow_nan=False)+'\n')
                        attempt_event('completed', elapsed_s=result['elapsed_s'], removal=result['removal'])
                        episode.close(); completed += 1
                        episode = new_episode()
                with torch.no_grad(): last_value = net.value(torch.from_numpy(episode.features)).numpy()
                arrays = {k: np.asarray(v) for k, v in data.items()}
                adv, returns = advantages(arrays['reward'], arrays['value'], arrays['done'], last_value,
                                          protocol['gamma'], protocol['gae_lambda'])
                keep = arrays['active'].reshape(-1)
                def tensor(key):
                    a = arrays[key]
                    return torch.from_numpy(a.reshape((-1,)+a.shape[2:])[keep])
                x, valid, actions = tensor('x'), tensor('valid'), tensor('actions')
                oldlog = tensor('logp')
                adv = torch.from_numpy(adv.reshape(-1)[keep]).float()
                returns = torch.from_numpy(returns.reshape(-1)[keep]).float()
                adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
                losses, entropies, kls = [], [], []
                for _ in range(protocol['epochs']):
                    order = torch.randperm(len(adv))
                    for offset in range(0, len(order), protocol['minibatch_size']):
                        index = order[offset:offset+protocol['minibatch_size']]
                        dist = net.distribution(x[index], valid[index])
                        logp = dist.log_prob(actions[index])
                        ratio = (logp-oldlog[index]).exp()
                        unclipped = ratio*adv[index]
                        clipped = ratio.clamp(1-protocol['clip_coef'], 1+protocol['clip_coef'])*adv[index]
                        entropy = dist.entropy().mean()
                        value_loss = .5*(net.value(x[index])-returns[index]).square().mean()
                        loss = -torch.minimum(unclipped, clipped).mean()+.5*value_loss-protocol['entropy_coef']*entropy
                        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), .5); optimizer.step()
                        losses.append(float(loss.detach())); entropies.append(float(entropy.detach()))
                        kls.append(float((oldlog[index]-logp.detach()).mean()))
                updates += 1
                record = dict(steps=transitions, episodes=completed, updates=updates,
                    loss=float(np.mean(losses)), entropy=float(np.mean(entropies)),
                    approximate_kl=float(np.mean(kls)), wall_s=time.monotonic()-started,
                    fps=(transitions-initial_steps)/(time.monotonic()-started))
                update_log.write(json.dumps(record, allow_nan=False)+'\n')
                atomic_json(out/'status.json', dict(phase='training', **record))
                print(json.dumps(record), flush=True)
                if transitions >= next_checkpoint or transitions == args.steps:
                    checkpoint(); next_checkpoint = (transitions//8192+1)*8192
            assert learning_hashes() == source, 'Source changed during training'
            attempt_event('budget_cutoff', elapsed_s=episode.env.elapsed_s,
                          removal=episode.previous_removal, not_a_completed_episode=True)
            atomic_json(out/'status.json', dict(phase='completed', **record))
    except BaseException as exc:
        atomic_json(out/'python_failure.json', dict(error=repr(exc), steps=transitions,
                    current_scene=args.scene_base+completed))
        raise
    finally:
        episode.close()


def evaluate(args):
    torch.set_num_threads(1)
    net = None
    checkpoint_hash = None
    if args.policy == 'learned':
        assert args.checkpoint is not None
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if payload['source_hashes'] != learning_hashes(): raise ValueError('Checkpoint/source mismatch')
        net = LocalManeuverActorCritic(payload['protocol']['hidden_dim'])
        net.load_state_dict(payload['model']); net.eval()
        checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', buffering=1) as log:
        for seed in range(args.scene_base, args.scene_base+args.episodes):
            episode = LearningEpisode(seed, control_seed=args.control_seed, clusters=args.clusters)
            try:
                while not episode.done:
                    if net is not None:
                        with torch.no_grad():
                            choice = net.distribution(torch.from_numpy(episode.features),
                                torch.from_numpy(episode.valid)).logits.argmax(dim=-1).numpy()
                    else:
                        choice = episode.library.conventional_choice(episode.packet, episode.valid,
                            episode.details, memory=args.policy == 'memory')
                    episode.step(choice, legacy=args.policy == 'v2_spacing')
                row = episode.result(args.policy)
                row.update(checkpoint_sha256=checkpoint_hash,
                           training_seed=payload['training_seed'] if net is not None else None,
                           training_steps=payload['transitions'] if net is not None else 0)
                log.write(json.dumps(row, allow_nan=False)+'\n')
                print(json.dumps({k: row[k] for k in ('scene_seed', 'method', 'removal',
                    'cluster_safe_success', 'wall_contact_s', 'spacing_violation_pair_s')}), flush=True)
            finally:
                episode.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    train_p = sub.add_parser('train')
    train_p.add_argument('--out', type=Path, required=True)
    train_p.add_argument('--seed', type=int, required=True)
    train_p.add_argument('--scene-base', type=int, required=True)
    train_p.add_argument('--steps', type=int, default=32768)
    train_p.add_argument('--initialize', type=Path)
    eval_p = sub.add_parser('evaluate')
    eval_p.add_argument('--policy', choices=('learned', 'joint', 'memory', 'v2_spacing'), required=True)
    eval_p.add_argument('--checkpoint', type=Path)
    eval_p.add_argument('--out', type=Path, required=True)
    eval_p.add_argument('--scene-base', type=int, required=True)
    eval_p.add_argument('--episodes', type=int, default=1)
    eval_p.add_argument('--control-seed', type=int, default=42)
    eval_p.add_argument('--clusters', type=int, default=3)
    args = parser.parse_args()
    train(args) if args.mode == 'train' else evaluate(args)


if __name__ == '__main__':
    main()
