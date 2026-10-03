"""Behaviour cloning of the Graph Transformer student on teacher-labelled scene graphs.

Losses (robots active in the scene only):
  direction  1 - cos(student direction, teacher direction)  on steps the teacher moves
  stop       BCE(stop logit, teacher stop), positive weight --stop-weight
  subgoal    CE over alive clots vs the teacher's allocation, weight --subgoal-weight (auxiliary)
The teacher never enters inference: the saved student acts from the scene graph alone.
usage: train_student_bc.py --data DIR[,DIR...] --val DIR --out DIR [--epochs 8]
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from marl.graph_transformer_student import GraphTransformerStudent, collate_scenes, save_student


def load_episodes(dirs):
    """Episode list: (static dict, dynamic dict of stacked arrays, labels, meta)."""
    episodes = []
    for d in dirs:
        for f in sorted(Path(d).glob('*.npz')):
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z['meta']))
            static = {k[2:]: z[k] for k in z.files if k.startswith('s_')}
            dynamic = {k[2:]: z[k] for k in z.files if k.startswith('d_')}
            labels = (z['teacher_local'], z['teacher_stop'], z['teacher_subgoal'])
            episodes.append((static, dynamic, labels, meta))
    return episodes


def scene_at(static, dynamic, t, constants):
    scene = dict(static)
    scene.update({k: v[t] for k, v in dynamic.items()})
    scene.update(constants)
    return scene


CONSTANTS = dict(robot_radius=np.float32(.08), particle_radius=np.float32(.02), robot_speed=np.float32(1.))


def batches(episodes, index, batch_size, rng=None):
    order = rng.permutation(len(index)) if rng is not None else np.arange(len(index))
    for start in range(0, len(order), batch_size):
        chunk = [index[i] for i in order[start:start+batch_size]]
        scenes = [scene_at(episodes[e][0], episodes[e][1], t, CONSTANTS) for e, t in chunk]
        labels = [tuple(l[t] for l in episodes[e][2]) for e, t in chunk]
        yield scenes, labels


def label_tensors(labels, R, device):
    B = len(labels)
    direction = np.zeros((B, R, 3), np.float32); stop = np.zeros((B, R), np.float32)
    move = np.zeros((B, R), bool); subgoal = np.full((B, R), -1, np.int64)
    for b, (local, s, g) in enumerate(labels):
        n = len(local)
        norm = np.linalg.norm(local, axis=1)
        direction[b, :n] = local/np.maximum(norm, 1e-9)[:, None]
        stop[b, :n] = s; move[b, :n] = (~s) & (norm > 1e-9); subgoal[b, :n] = g
    return [torch.as_tensor(x, device=device) for x in (direction, stop, move, subgoal)]


def losses(model, scenes, labels, device, stop_weight, subgoal_weight):
    batch = collate_scenes(scenes, device)
    out = model(batch)
    R = out['direction'].shape[1]
    direction, stop, move, subgoal = label_tensors(labels, R, device)
    active = batch['robot_mask']
    cos = nn.functional.cosine_similarity(out['direction'], direction, dim=-1)
    mv = move & active
    l_dir = (1-cos)[mv].mean() if mv.any() else cos.sum()*0
    bce = nn.functional.binary_cross_entropy_with_logits(out['stop_logit'], stop, reduction='none',
                                                         pos_weight=torch.tensor(stop_weight, device=device))
    l_stop = bce[active].mean()
    sg = (subgoal >= 0) & active
    l_sub = nn.functional.cross_entropy(out['subgoal_logits'][sg], subgoal[sg]) if sg.any() else cos.sum()*0
    pred_stop = out['stop_logit'] > 0
    stats = dict(cos=cos[mv].mean().item() if mv.any() else float('nan'),
                 stop_acc=(pred_stop == (stop > .5))[active].float().mean().item(),
                 stop_recall=(pred_stop & (stop > .5))[active].sum().item()/max(((stop > .5) & active).sum().item(), 1),
                 subgoal_acc=(out['subgoal_logits'].argmax(-1) == subgoal)[sg].float().mean().item() if sg.any() else float('nan'))
    return l_dir+l_stop+subgoal_weight*l_sub, dict(l_dir=l_dir.item(), l_stop=l_stop.item(), l_sub=l_sub.item(), **stats)


def evaluate(model, episodes, index, args, device):
    model.eval(); agg = {}; n = 0
    with torch.no_grad():
        for scenes, labels in batches(episodes, index, args.batch_size):
            _, s = losses(model, scenes, labels, device, args.stop_weight, args.subgoal_weight)
            for k, v in s.items():
                agg[k] = agg.get(k, 0)+v*len(scenes)
            n += len(scenes)
    model.train()
    return {k: v/n for k, v in agg.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--val', required=True); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--epochs', type=int, default=8); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dim', type=int, default=128); p.add_argument('--layers', type=int, default=4); p.add_argument('--heads', type=int, default=4)
    p.add_argument('--stop-weight', type=float, default=5.); p.add_argument('--subgoal-weight', type=float, default=.5)
    p.add_argument('--max-steps', type=int, default=0, help='smoke tests only')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    train = load_episodes(args.data.split(',')); val = load_episodes([args.val])
    tindex = [(e, t) for e, ep in enumerate(train) for t in range(len(ep[2][0]))]
    vindex = [(e, t) for e, ep in enumerate(val) for t in range(0, len(ep[2][0]), 4)]
    model = GraphTransformerStudent(args.dim, args.heads, args.layers).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.epochs*((len(tindex)+args.batch_size-1)//args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=max(total, 1), pct_start=.05)
    meta = dict(vars(args), out=str(args.out), train_episodes=len(train), train_scenes=len(tindex), val_episodes=len(val),
                anatomies=sorted({m[3]['anatomy'] for m in train}), parameters=sum(x.numel() for x in model.parameters()))
    (args.out/'config.json').write_text(json.dumps(meta, indent=1)+'\n')
    log = (args.out/'log.jsonl').open('a'); step = 0; t0 = time.time(); best = None
    for epoch in range(args.epochs):
        for scenes, labels in batches(train, tindex, args.batch_size, rng):
            loss, s = losses(model, scenes, labels, args.device, args.stop_weight, args.subgoal_weight)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); sched.step(); step += 1
            if step % 200 == 0:
                log.write(json.dumps(dict(step=step, epoch=epoch, loss=loss.item(), t=time.time()-t0, **s))+'\n'); log.flush()
            if args.max_steps and step >= args.max_steps:
                break
        v = evaluate(model, val, vindex, args, args.device)
        log.write(json.dumps(dict(step=step, epoch=epoch, val=v, t=time.time()-t0))+'\n'); log.flush()
        print(json.dumps(dict(epoch=epoch, step=step, t=round(time.time()-t0), **{k: round(x, 4) for k, x in v.items()})), flush=True)
        save_student(model, args.out/f'student_epoch{epoch}.pt', **meta, epoch=epoch, val_metrics=v)
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
