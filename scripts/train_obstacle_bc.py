"""Collect and train a deployable 8-option obstacle policy by privileged labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from marl.obstacle_control import History, DiscreteTemporalPolicy, save_discrete_checkpoint, token, token_dim
from marl.lookahead_teacher import label, option_local
from scripts.benchmark_obstacles import Episode


def collect_episode(anatomy, seed, clusters=1, horizon=300., obs_vel=False, teacher_horizon=10):
    ep=Episode(clusters, anatomy, seed, horizon=horizon, avoid=True)
    hist=History(clusters, dim=token_dim(obs_vel)); rows=[]; ys=[]
    try:
        while True:
            est,tgt,rule,hold=ep.observe(); T,_=token(ep.env,ep.sensor,est,ep.ctl,rule,hold,tgt,ep.prev_local,obs_vel)
            seq,mask=hist.push(T); y,_=label(ep,est,rule,hold,horizon=teacher_horizon)
            live=est.active & (np.asarray(tgt)>=0) & ~hold
            for i in np.flatnonzero(live): rows.append((seq[i].copy(),mask[i].copy())); ys.append(y)
            local=option_local(ep,est,rule,y); done,_=ep.step(est,local,hold)
            if done: break
    finally: ep.close()
    if not rows: return None
    return np.stack([x[0] for x in rows]),np.stack([x[1] for x in rows]),np.asarray(ys,np.int64)

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('collect'); c.add_argument('--out',type=Path,required=True); c.add_argument('--anatomies',nargs='+',required=True); c.add_argument('--first-seed',type=int,required=True); c.add_argument('--episodes',type=int,required=True); c.add_argument('--clusters',type=int,default=1); c.add_argument('--teacher-horizon',type=int,default=10); c.add_argument('--obs-vel',action='store_true')
    t=sub.add_parser('train'); t.add_argument('--data',type=Path,required=True); t.add_argument('--out',type=Path,required=True); t.add_argument('--arch',choices=('transformer','gru','mlp'),default='transformer'); t.add_argument('--epochs',type=int,default=20); t.add_argument('--batch-size',type=int,default=256); t.add_argument('--lr',type=float,default=3e-4); t.add_argument('--device',default='cpu'); t.add_argument('--obs-vel',action='store_true')
    a=p.parse_args()
    if a.cmd=='collect':
        a.out.mkdir(parents=True,exist_ok=True); meta=[]
        for k in range(a.episodes):
            an=a.anatomies[k%len(a.anatomies)]; seed=a.first_seed+k; r=collect_episode(an,seed,a.clusters,obs_vel=a.obs_vel,teacher_horizon=a.teacher_horizon)
            if r is None: continue
            np.savez_compressed(a.out/f'{an}_{seed}.npz',seq=r[0],mask=r[1],label=r[2],meta=json.dumps(dict(anatomy=an,seed=seed,clusters=a.clusters,obs_vel=a.obs_vel,teacher_horizon=a.teacher_horizon)))
            meta.append(dict(anatomy=an,seed=seed,samples=len(r[2])))
            print(json.dumps(meta[-1]),flush=True)
        (a.out/'manifest.json').write_text(json.dumps(meta,indent=2))
    else:
        files=sorted(a.data.glob('*.npz')); X=[];M=[];Y=[]
        for f in files:
            z=np.load(f,allow_pickle=False); X.append(z['seq']); M.append(z['mask']); Y.append(z['label'])
        if not Y: raise ValueError('no training files')
        X=torch.as_tensor(np.concatenate(X)); M=torch.as_tensor(np.concatenate(M),dtype=torch.bool); Y=torch.as_tensor(np.concatenate(Y),dtype=torch.long)
        net=DiscreteTemporalPolicy(a.arch,dim=token_dim(a.obs_vel)).to(a.device); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.01); dl=DataLoader(TensorDataset(X,M,Y),a.batch_size,shuffle=True)
        for ep in range(a.epochs):
            net.train(); loss=0.; n=0
            for x,m,y in dl:
                x,m,y=x.to(a.device),m.to(a.device),y.to(a.device); l=torch.nn.functional.cross_entropy(net.logits(x,m),y); opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); loss+=float(l)*len(y); n+=len(y)
            print(json.dumps(dict(epoch=ep+1,loss=loss/n,samples=n)),flush=True)
        a.out.parent.mkdir(parents=True,exist_ok=True); save_discrete_checkpoint(a.out,net,dict(arch=a.arch,dim=token_dim(a.obs_vel),actions=8,obs_vel=a.obs_vel,window=16,teacher='lookahead_teacher'))
if __name__=='__main__': main()
