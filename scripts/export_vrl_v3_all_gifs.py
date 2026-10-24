"""Resumable, provenance-preserving export of the V3 paper's entire media matrix.

Training NPZs are original measured truth; evaluation trajectories are fresh
deterministic runs of the paper's local implementations, never metric animations.
The renderer projects the actual 3-D geometry and trajectories with a fixed
perspective camera. Rendering does not enter the controller or physics.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path('/home/wj/桌面/VRL_V3_全部场景_3D_GIF_20261011')
PA = ','.join(f'research/runs/EXP0090_20261011/predictor/predictor_s{i}.pt' for i in range(3))
PB = ','.join(f'research/runs/EXP0091_20261011/predictor_B/predictor_s{i}.pt' for i in (10,11,12))
PC = ','.join(f'research/runs/EXP0091_20261011/predictor_C/predictor_s{i}.pt' for i in (20,21,22))
TURBO = 'research/runs/EXP0093_20261011/turbo_hard_s0/policy.pt'
WEIGHTS = dict(delta=.05, prog=1.5, appr=3., kappa=0)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
COLORS = [(38,132,230),(34,174,116),(239,156,34)]
PANELS = dict(low_delay='低延迟',moderate_delay='中延迟',high_delay='高延迟',strong_flow='强流',variable_response='动力学变化',ood_flow_delay='分布外')

def condition_text(task):
    cell = task.get('cell','')
    section = task.get('section','')
    if '弱流' in section or cell.startswith('v5_'):
        s = task['job'][5]
        return f'弱流扰动强度 {s:g}×（0=基准，1=弱流，1.5=更强扰动）'
    names = {'low_delay':'低延迟：观测延迟 1 步', 'moderate_delay':'中延迟：观测延迟 2 步',
             'high_delay':'高延迟：观测延迟 4 步', 'strong_flow':'强流：入口流速提高',
             'variable_response':'动力学变化：流量/脉动变化', 'ood_flow_delay':'分布外：流量与延迟同时变化',
             'normal':'标准血流', 'pulsatile_response':'脉动血流与响应变化',
             'strong_pulsatile_ood':'更强血流 + 脉动 + 延迟（超出训练范围）'}
    return names.get(cell.split('|')[0], cell.split('|')[0])

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)); temp.replace(path)

def geometry(ep):
    e=ep.env
    return dict(points=e.transport.points.tolist(), ends=np.asarray(e.transport.ends).tolist(),
                radius=np.asarray(e.flow_model.healthy_radius_mm).tolist(), clots=e.clot_positions_mm.tolist(),
                clot_mass0=e.initial_mass.tolist(), body=float(e.config.robot_radius_mm))

def label_method(m):
    if m.startswith('fac='): return m[4:].replace('+',' / ')
    kind,_,arg=m.partition('=')
    if kind=='sel_learned_fb':
        return '本文预测选择 + 回退（种子 '+('B' if 'predictor_B' in arg else 'C' if 'predictor_C' in arg else 'A')+'）'
    if kind in ('nav','nav_off'): return kind+' / '+Path(arg).parent.name
    return kind

def spec_from_row(r, manifests):
    m=r['method']+('='+r['method_arg'] if r.get('method_arg') else '')
    opts=dict(r.get('opts',{}))
    matched=[x['args'] for x in manifests if m in x['args']['methods'].split(',')]
    if matched:
        a=matched[-1]; opts['weights']=json.loads(a.get('weights','{}'))
        for k in ('predictors','fallback_std','turbo','ablation','no_tpg'): opts[k]=a.get(k)
    else: opts['weights']=WEIGHTS.copy()
    return m,(r['anatomy'],int(r['clusters']),int(r['seed']),r['flow_inlet_mm_s'],int(r['latency_steps']),float(r['variation_s'])),opts

def registry(out):
    split=json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']
    held=set(split['held_out']); tasks=[]; keys=set()
    def add(m,job,opts,section,cell,source,reference=None,training=None):
        fingerprint=json.dumps([m,job,opts,section,cell,training],sort_keys=True)
        uid=hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        if uid in keys: return
        keys.add(uid); an,n,seed,f,L,v=job
        method=label_method(m); folder=m.partition('=')[0] if '=' not in m else (m[4:].replace('+','__') if m.startswith('fac=') else method.split(' / ')[-1].replace(' ','_'))
        if m.startswith('sel_learned_fb='): folder='ours_'+('B' if 'predictor_B' in m else 'C' if 'predictor_C' in m else 'A')
        topo='留出解剖' if an in held else '训练解剖'
        name=f'{an}_N{n}_seed{seed}_f{f if f is not None else "legacy"}_L{L}_s{v:g}_{uid}'
        rel=str(Path(section)/cell.split('|')[0]/topo/folder/(name+'.gif'))
        tasks.append(dict(id=uid,method=m,label=method,job=list(job),opts=opts,section=section,cell=cell,
                          topology=topo,source=source,reference=reference,training=training,gif=rel))
    # Exactly the recorded factorial design, including all 23 hard-matrix arms.
    dirs=['EXP0093_20261011/hard','EXP0091_20261011/dev_hard_ours','EXP0091_20261011/dev_hard_online',
          'EXP0090_20261011/hard','EXP0091_20261011/dev_v5_ours','EXP0091_20261011/dev_v5_online',
          'EXP0091_20261011/dev_v5_paper','EXP0090_20261011/v5',
          'EXP0090_20261011/hard_notpg','EXP0090_20261011/hard_no_guard',
          'EXP0090_20261011/hard_no_shield','EXP0090_20261011/hard_no_safety']
    nav_seen=set()
    for d in dirs:
        base=ROOT/'research/validation'/d
        manifests=[json.loads(l) for l in (base/'manifest.jsonl').open()]
        for l in (base/'episodes.jsonl').open():
            r=json.loads(l); m,job,opts=spec_from_row(r,manifests)
            # Early EXP0090 candidate-selector prototypes are not the V3 algorithm.
            if d=='EXP0090_20261011/hard' and r['method'] not in ('fixed_settle','adaptive_settle','switch_settle','nav','nav_off','sel_learned','sel_learned_inv'):
                continue
            if d=='EXP0090_20261011/hard' and r['method']=='sel_learned' and r.get('method_arg'): continue
            if d=='EXP0090_20261011/v5' and r['method'] not in ('fixed_settle','switch_settle','nav','nav_off'): continue
            section='02_测试回放_困难矩阵' if 'hard' in d else '03_测试回放_弱流V5'
            if '/hard_' in d:
                section='05_安全层消融'; r['cell']=d.split('/')[-1]+'__'+r['cell']
            add(m,job,opts,section,r['cell'],str(base/'episodes.jsonl'),r)
    # Complete the V5 factorial set as well, even when its original job is still running.
    from scripts.eval_exp0090 import jobs_for
    from types import SimpleNamespace
    jobs=jobs_for(SimpleNamespace(protocol='v5',strengths='0,1,1.5',count=10))
    fac=['A_TPG+ours','A_TPG+pursuit','A_TPG+pac','A_TPG+mppi','A_TPG+cbf','STPG+ours','OT+ours','OT+pursuit']
    opts=dict(predictors=PA,fallback_std=.0445,weights=WEIGHTS,no_tpg=False,ablation=None,turbo=None)
    for cell,job in jobs:
        for arm in fac: add('fac='+arm,job,opts,'03_测试回放_弱流V5',cell,'EXP0093 V5 complete registered scene matrix')
    # Original trajectories from all three independent predictor training datasets.
    for seed,data in [('A','EXP0090_20261011/data'),('B','EXP0091_20261011/data_B'),('C','EXP0091_20261011/data_C')]:
        for p in sorted((ROOT/'research/runs'/data).glob('train_*.npz')):
            with np.load(p) as z: meta=json.loads(str(z['meta']))
            job=(meta['anatomy'],meta['n'],meta['seed'],meta['flow'],meta['latency'],meta['s'])
            add('training_behavior_'+seed,job,{},'01_真实训练轨迹','predictor_'+seed,str(p),training=str(p))
    # PPO training does not persist its sampled rollouts. Export early/middle/final
    # checkpoints on every training anatomy x N, and label them as new replays.
    policies=[('Turbo','research/runs/EXP0093_20261011/turbo_hard_s0','fac=A_TPG+turbo'),
              ('Nav7101','research/runs/EXP0073_LONG1M_20261009/nav_tf_v3_s7101','nav'),
              ('Nav7102','research/runs/EXP0081_MULTI_SEED_AUX_20261010/nav_tf_v3_s7102','nav'),
              ('Nav7103','research/runs/EXP0081_MULTI_SEED_AUX_20261010/nav_tf_v3_s7103','nav')]
    for name,folder,kind in policies:
        checkpoints=sorted((ROOT/folder).glob('policy_it*.pt'),key=lambda x:int(x.stem[9:]))
        if not checkpoints: continue
        stages=[('early',checkpoints[0]),('middle',checkpoints[len(checkpoints)//2]),('final',ROOT/folder/'policy.pt')]
        for stage,p in stages:
            for k,an in enumerate(split['train']):
                for n in (1,2,3):
                    job=(an,n,2126000000+k*1000+n,.05,2,.625)
                    o=dict(opts); method='nav='+str(p.relative_to(ROOT))
                    if kind.startswith('fac='): method=kind; o['turbo']=str(p.relative_to(ROOT))
                    add(method,job,o,'04_训练阶段模型回放',name+'_'+stage,str(p))
    # Put one scene for every method/anatomy/N before subsequent condition replicates.
    tasks.sort(key=lambda t:(0 if t['section'].startswith('02') and t['cell'].startswith('low_delay') else
                              1 if t['section'].startswith('01') else 2 if t['section'].startswith('02') else 3,
                              t['job'][2],t['method']))
    atomic_json(out/'tasks.json',tasks)
    return tasks

def evaluate(task):
    import torch
    torch.set_num_threads(1)
    from scripts.eval_exp0090 import make_episode,controller
    from marl.deployable_sensing import DeployableConfig
    from marl import factorial as FA
    m,job,opts=task['method'],task['job'],task['opts']
    if m.startswith('fac=') or m=='stpg':
        g,l=m[4:].split('+') if m.startswith('fac=') else ('STPG','pursuit')
        FA.PRED.update(ensemble=(opts.get('predictors') or PA).split(','),fallback_std=opts.get('fallback_std'),weights=opts.get('weights'),turbo=opts.get('turbo'))
        an,n,seed,f,L,v=job; kw=dict(sense_cfg=DeployableConfig(latency_steps=L),variation=v if v>0 else None)
        if f is not None: kw['flow_inlet_mm_s']=f
        ep=FA.make(g,n,an,seed,**kw); ctl=FA.local(l,ep)
    else:
        ep,_=make_episode((m,task['cell'],job,opts)); ctl=controller(m,ep,opts)
        abl=opts.get('ablation')
        if abl in ('no_guard','no_safety'):
            class NoGuard:
                class exec:
                    @staticmethod
                    def act(t,est,ctl,local,hold):return local
                def __call__(self,ep,est,tgt,command,hold):return command
            if hasattr(ctl,'guard'):ctl.guard=NoGuard()
        if abl in ('no_shield','no_safety'):ep.shield=None
    try:
        geo=geometry(ep); e=ep.env
        rec=dict(t=[0.],pos=[e.positions_mm[:ep.n].copy()],mass=[e.masses.copy()],wall=[0.],removal=[0.])
        while True:
            est=ep.observe(); tgt=ep.plan_targets_now(est)
            if getattr(ep,'stpg',None) is not None: ep.stpg.update(e.elapsed_s)
            rule=ep.ctl.act(tgt,est); hold=ep.hold(est)
            done,res=ep.step(est,ctl(ep,est,tgt,rule,hold),hold)
            rec['t'].append(float(e.elapsed_s)); rec['pos'].append(e.positions_mm[:ep.n].copy())
            rec['mass'].append(e.masses.copy()); rec['wall'].append(float(np.sum(res['wall'])))
            rec['removal'].append(1-float(e.masses.sum())/ep.initial)
            if done: break
        extra=ctl.summary() if hasattr(ctl,'summary') else {}
        return {k:np.asarray(v) for k,v in rec.items()},geo,ep.row(m,**extra)
    finally: ep.close()

def training_record(task):
    from marl.deployable_sensing import DeployableConfig
    from scripts.benchmark_lysis import LysisEpisode
    with np.load(task['training']) as z: R=z['rows']; meta=json.loads(str(z['meta']))
    an,n,seed,f,L,v=task['job']
    ep=LysisEpisode(n,an,seed,sense_cfg=DeployableConfig(latency_steps=L),variation=v if v>0 else None,flow_inlet_mm_s=f)
    try: geo=geometry(ep)
    finally: ep.close()
    pos=R[:,45:48].reshape(-1,n,3)
    rec=dict(t=np.arange(len(pos))*.1,pos=pos,mass=np.tile(geo['clot_mass0'],(len(pos),1)),wall=np.zeros(len(pos)),removal=np.full(len(pos),np.nan))
    return rec,geo,dict(training_meta=meta,mass_recorded=False,original_training_positions=True)

class View:
    """Perspective-projected 3D vessel with translucent tube surfaces and depth shading."""
    def __init__(self,geo,W=900,H=650):
        self.W,self.H=W,H; self.geo=geo; P=np.asarray(geo['points'],float)
        self.c=(P.min(0)+P.max(0))*.5; ext=max(float(np.linalg.norm(np.ptp(P,axis=0))),1.)
        yaw=np.radians(35); pitch=np.radians(32)
        self.right=np.array([np.cos(yaw),-np.sin(yaw),0.])
        self.up=np.array([np.sin(yaw)*np.sin(pitch),np.cos(yaw)*np.sin(pitch),np.cos(pitch)])
        self.forward=np.cross(self.right,self.up); self.dist=ext*2.5; self.scale=min(W*.8,(H-145)*.88)/ext*self.dist
        self.center=np.array([W*.5,110+(H-145)*.5]); self.ext=ext
        # Fit the projected geometry, not the diagonal of its world bounding box.
        uv=self.project(P); span=np.ptp(uv,axis=0)
        self.scale*=min(W*.82/max(span[0],1),(H-180)*.82/max(span[1],1))
        uv=self.project(P); self.center+=np.array([W*.5,110+(H-160)*.5])-(uv.min(0)+uv.max(0))*.5
        base=Image.new('RGB',(W,H),(247,249,253)); d=ImageDraw.Draw(base)
        d.rounded_rectangle((14,105,W-14,H-49),18,fill=(255,255,255),outline=(220,228,238))
        ends=np.asarray(geo['ends']); uv=self.project(P); radii=np.asarray(geo['radius'])
        order=sorted(ends,key=lambda ab:np.dot((P[ab[0]]+P[ab[1]])*.5-self.c,self.forward))
        for a,b in order:
            r=float((radii[a]+radii[b])*.5); thick=max(2,int(2*self.radius((P[a]+P[b])*.5,r)))
            line=[tuple(uv[a]),tuple(uv[b])]
            d.line(line,fill=(247,211,217),width=thick+2)
            d.line(line,fill=(251,232,235),width=thick)
            d.line(line,fill=(230,168,180),width=1)
        # The lumen itself uses the project's 3-D mesh renderer. Draw dynamic
        # positions and tracks with the exact same camera projection afterwards.
        from scripts.render_lysis_paper import Scene
        import shutil
        os.environ['LYSIS_DISABLE_EGL']='1'
        scene=Scene(geo,W,H-155,vessel_alpha=.32)
        try:
            scene.camera(P,yaw=35,pitch=-35)
            span=np.ptp(scene.project(P),axis=0)
            zoom=min(1.6,W*.85/max(span[0],1),(H-155)*.82/max(span[1],1))
            projection=np.asarray(scene.proj).reshape(4,4).T.copy()
            projection[0,0]*=zoom;projection[1,1]*=zoom
            scene.proj=projection.T.ravel().tolist()
            scene.M=projection@np.asarray(scene.view).reshape(4,4).T
            lumen=scene.render([],np.zeros(len(geo['clots'])))
            self.camera_M=scene.M.copy()
            V=np.asarray(scene.view).reshape(4,4).T
            self.camera_right=V[0,:3].copy()
            self.right=V[0,:3].copy();self.up=V[1,:3].copy();self.forward=V[2,:3].copy()
            base.paste(Image.fromarray(lumen),(0,105))
        finally:
            scene.close();shutil.rmtree(scene.tmp,ignore_errors=True)
        d=ImageDraw.Draw(base)
        origin=(55,H-95)
        for axis,color,txt in zip(np.eye(3),[(195,61,61),(53,153,83),(60,105,207)],['X','Y','Z']):
            vec=np.array([axis@self.right,-axis@self.up])*26
            tip=tuple(np.array(origin)+vec);d.line([origin,tip],fill=color,width=2);d.text(tip,txt,fill=color)
        d.text((22,H-36),'3D perspective  |  trajectories in mm  |  spheres enlarged for visibility',fill=(95,109,130))
        self.base=base; self.font=ImageFont.truetype(FONT,17);self.small=ImageFont.truetype(FONT,14)
    def project(self,P):
        if hasattr(self,'camera_M'):
            Q=np.asarray(P,float);shape=Q.shape[:-1];Q=Q.reshape(-1,3)
            h=np.c_[Q,np.ones(len(Q))]@self.camera_M.T;ndc=h[:,:3]/h[:,3:4]
            return np.stack([(ndc[:,0]+1)*self.W*.5,(1-ndc[:,1])*(self.H-155)*.5+105],axis=-1).reshape(*shape,2)
        Q=np.asarray(P,float)-self.c; depth=np.maximum(self.dist-Q@self.forward,self.dist*.1)
        return self.center+np.stack([Q@self.right,-Q@self.up],axis=-1)*self.scale/depth[...,None]
    def radius(self,p,r):
        if hasattr(self,'camera_M'):
            uv=self.project([p,np.asarray(p)+self.camera_right*r]);return float(np.linalg.norm(uv[1]-uv[0]))
        return float(r*self.scale/max(self.dist-np.dot(np.asarray(p)-self.c,self.forward),self.dist*.1))
    def frame(self,task,rec,k,metrics):
        im=self.base.copy();d=ImageDraw.Draw(im);an,n,seed,f,L,v=task['job']
        d.text((22,12),task['label'],font=self.font,fill=(20,40,70))
        d.text((22,39),f'{an}   N={n}   {task["topology"]}   {PANELS.get(task["cell"].split("|")[0],task["cell"].split("|")[0])}',font=self.small,fill=(61,79,105))
        d.text((22,63),f'seed={seed}   inlet={f if f is not None else "legacy"} mm/s   latency={L}   dynamics={v:g}',font=self.small,fill=(61,79,105))
        if task['training']: status='真实训练轨迹 · 血栓质量未记录'
        else:
            wall=float(np.sum(rec['wall'][:k+1])); status=f'清除 {rec["removal"][k]*100:.1f}%   壁接触 {wall:.2f} robot·s'
            if task.get('actual_rl_training'):status=task.get('training_algorithm','PPO')+' 真实训练轨迹 · '+status
            if k==len(rec['t'])-1: status+='   '+('Strict 成功' if metrics.get('strict_success') else 'Strict 未成功')
        d.text((22,101),condition_text(task),font=self.small,fill=(135,74,25))
        d.text((22,122),f't={rec["t"][k]:.1f} s   {status}',font=self.small,fill=(44,65,85))
        P=rec['pos'][:k+1]; stride=max(1,len(P)//700)
        for i in range(n):
            uv=self.project(P[::stride,i]); d.line([tuple(x) for x in uv],fill=COLORS[i],width=2)
        objects=[]
        for j,p in enumerate(self.geo['clots']):
            m0=self.geo['clot_mass0'][j];ratio=max(0,float(rec['mass'][k,j]))/max(m0,1e-10)
            if ratio<1e-5: continue
            rad=float(np.asarray(self.geo['radius'])[np.argmin(np.linalg.norm(np.asarray(self.geo['points'])-p,axis=1))])*.65*ratio**(1/3)
            objects.append((np.dot(np.asarray(p)-self.c,self.forward),p,max(3,self.radius(p,rad)),(145,32,57)))
        for i,p in enumerate(P[-1]):objects.append((np.dot(p-self.c,self.forward),p,8,COLORS[i]))
        for _,p,r,col in sorted(objects,key=lambda x:x[0]):
            x,y=self.project(np.asarray(p));d.ellipse((x-r,y-r,x+r,y+r),fill=col,outline=(255,255,255),width=1)
            d.ellipse((x-r*.5,y-r*.6,x+r*.1,y),fill=tuple(min(255,c+65) for c in col))
        for i,p in enumerate(P[-1]):
            x,y=self.project(np.asarray(p)); d.rounded_rectangle((x+7,y-12,x+36,y+10),radius=4,fill=(255,255,255),outline=COLORS[i],width=2)
            d.text((x+11,y-10),f'R{i+1}',font=self.small,fill=COLORS[i])
        if 'targets' in rec:
            for j,p in enumerate(self.geo['clots']):
                x,y=self.project(np.asarray(p))
                d.text((x+7,y+7),f'C{j+1}',font=self.small,fill=(126,48,65))
        for i in range(n):
            yy=112+i*20; d.ellipse((650,yy,661,yy+11),fill=COLORS[i])
            state=f'R{i+1} 机器人'
            if 'targets' in rec:
                target=int(rec['targets'][k,i])
                state+=f' → C{target+1}' if target>=0 else ' · 停车'
            if 'hold' in rec and rec['hold'][k,i]:state+=' · TPG 等待'
            d.text((666,yy-3),state,font=self.small,fill=(44,65,85))
        if 'flow_factor' in rec:
            # Recorded simulator flow multiplier for visualization only. It
            # is never an actor observation or a reconstructed phase signal.
            times=np.asarray(rec['t']);q=np.asarray(rec['flow_factor'])
            x0,y0,x1,y1=self.W-248,self.H-148,self.W-22,self.H-57
            d.rounded_rectangle((x0,y0,x1,y1),6,fill=(255,255,255),outline=(205,218,235))
            d.text((x0+7,y0+3),f'流量系数 q(t) = {q[k]:.2f}',font=self.small,fill=(53,79,114))
            first=max(0,int(np.searchsorted(times,times[k]-8.)))
            tt=times[first:k+1];values=q[first:k+1];maximum=max(1.1,float(q.max())*1.05)
            xx=x0+8+(tt-max(0.,times[k]-8.))/8.*(x1-x0-16)
            yy=y1-9-values/maximum*(y1-y0-34)
            if len(tt)>1:d.line(list(zip(xx,yy)),fill=(63,116,190),width=2)
            d.text((x0+8,y1-17),'最近 8 s',font=self.small,fill=(108,124,146))
        return im

def render(task,rec,geo,metrics,dest,frames):
    view=View(geo); indices=np.unique(np.linspace(0,len(rec['t'])-1,min(frames,len(rec['t']))).astype(int))
    ims=[view.frame(task,rec,int(k),metrics).quantize(colors=128,method=Image.Quantize.FASTOCTREE) for k in indices]
    temp=dest.with_suffix('.tmp.gif'); dest.parent.mkdir(parents=True,exist_ok=True)
    durations=[80]*len(ims);durations[-1]=1500
    ims[0].save(temp,save_all=True,append_images=ims[1:],duration=durations,loop=0,optimize=False,disposal=2)
    with Image.open(temp) as z:
        if z.n_frames<2: raise RuntimeError('Not an animated GIF')
        z.seek(z.n_frames-1);z.load()
    temp.replace(dest)
    return len(ims)

def work(args):
    task,out,frames=args;out=Path(out);dest=out/task['gif']; meta=dest.with_suffix('.json'); start=time.monotonic()
    try:
        if dest.exists() and meta.exists(): return dict(id=task['id'],state='exists',gif=task['gif'])
        trace=dest.with_suffix('.npz')
        if trace.exists():
            with np.load(trace) as z:
                rec={k:z[k] for k in ('t','pos','mass','wall','removal')};geo=json.loads(str(z['geometry']));metrics=json.loads(str(z['metrics']))
                for key in ('flow_factor','targets','hold'):
                    if key in z:rec[key]=z[key]
        else:
            rec,geo,metrics=training_record(task) if task['training'] else evaluate(task)
            trace.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(trace,**rec,geometry=json.dumps(geo),metrics=json.dumps(metrics))
        mismatch={}
        ref=task.get('reference')
        if ref:
            for k in ('removal','elapsed_s','wall_contact_s','strict_success','scenario_hash'):
                a,b=metrics.get(k),ref.get(k)
                equal=(abs(float(a)-float(b))<1e-6) if isinstance(a,(float,int)) and isinstance(b,(float,int)) else a==b
                if not equal:mismatch[k]=dict(replay=a,reference=b)
        count=render(task,rec,geo,metrics,dest,frames)
        result=dict(id=task['id'],state='complete',gif=task['gif'],frames=count,simulation_duration_s=float(rec['t'][-1]),
                    walltime_s=round(time.monotonic()-start,2),metrics=metrics,reference_mismatch=mismatch,
                    provenance=task,renderer='3D perspective of recorded physical geometry and truth trajectories')
        atomic_json(meta,result);return result
    except Exception:return dict(id=task['id'],state='failed',gif=task['gif'],error=traceback.format_exc())

def update_index(out,tasks):
    success=sum((out/t['gif']).with_suffix('.json').exists() for t in tasks)
    states=dict(expected=len(tasks),completed=success,pending=len(tasks)-success,updated=time.strftime('%Y-%m-%d %H:%M:%S'))
    atomic_json(out/'progress.json',states)
    data=[dict(gif=t['gif'],method=t['label'],section=t['section'],anatomy=t['job'][0],N=t['job'][1],panel=t['cell'],topology=t['topology'],ready=(out/t['gif']).with_suffix('.json').exists()) for t in tasks]
    page='''<!doctype html><meta charset="utf-8"><title>VRL V3 全场景 3D GIF</title><style>body{font:16px sans-serif;background:#f5f7fb;color:#17304c;margin:35px}input{padding:12px;width:70%;font-size:16px}table{border-collapse:collapse;width:100%;background:white;margin-top:20px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#1265ab}</style><h1>VRL V3 全场景 3D GIF</h1><p>STATUS</p><p>真实训练轨迹、困难矩阵、弱流 V5、训练阶段模型回放分别保存。测试回放使用论文的本地适配代码；失败过程同样保留。每个 GIF 旁有完整 NPZ 轨迹及 JSON 参数/指标。原论文使用开发评估池，页面“测试回放”不表示封存测试已经执行。</p><input id="q" placeholder="搜索算法、解剖、N、条件…"><p id="count"></p><table><thead><tr><th>阶段</th><th>算法</th><th>解剖</th><th>N</th><th>条件</th><th>文件</th></tr></thead><tbody id="rows"></tbody></table><script>const data=DATA;function show(){let q=document.getElementById('q').value.toLowerCase();let list=data.filter(x=>JSON.stringify(x).toLowerCase().includes(q));document.getElementById('count').textContent='匹配 '+list.length+' 项；显示前 300 项（输入关键词筛选）';document.getElementById('rows').innerHTML=list.slice(0,300).map(x=>`<tr><td>${x.section}</td><td>${x.method}</td><td>${x.anatomy} / ${x.topology}</td><td>${x.N}</td><td>${x.panel}</td><td>${x.ready?`<a href="${encodeURI(x.gif)}" target="_blank">打开 GIF</a>`:'生成中'}</td></tr>`).join('')}document.getElementById('q').addEventListener('input',show);show();</script>'''
    page=page.replace('STATUS',f'已完成 {success} / {len(tasks)}；更新时间 {states["updated"]}').replace('DATA',json.dumps(data,ensure_ascii=False))
    if any(t['method'].startswith('rl_') for t in tasks):page=page.replace('VRL V3 全场景 3D GIF','VRL PPO 局部导航 3D GIF')
    (out/'index.html').write_text(page)
    return states

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=DEFAULT_OUT);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--frames',type=int,default=180);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--prepare',action='store_true');ap.add_argument('--index',action='store_true')
    a=ap.parse_args();os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=True)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
    tasks=json.loads((a.out/'tasks.json').read_text()) if (a.out/'tasks.json').exists() else registry(a.out)
    print(update_index(a.out,tasks),flush=True)
    if a.prepare or a.index:return
    import fcntl
    lock=(a.out/'.export.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An export batch is already running for this output directory.')
    pending=[t for t in tasks if not (a.out/t['gif']).with_suffix('.json').exists()]
    if a.limit:pending=pending[:a.limit]
    retry=[]
    with mp.get_context('spawn').Pool(a.workers,maxtasksperchild=40) as pool:
        with (a.out/'export_log.jsonl').open('a') as log:
            for i,r in enumerate(pool.imap_unordered(work,[(t,str(a.out),a.frames) for t in pending],chunksize=1)):
                log.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');log.flush()
                print(json.dumps({k:r[k] for k in ('id','state','gif')}),flush=True)
                if r['state']=='failed':
                    print(r['error'],flush=True);retry.append(r['id'])
                if i%10==0:update_index(a.out,tasks)
            for attempt in range(2):
                if not retry:break
                todo=[t for t in tasks if t['id'] in set(retry)];retry=[]
                for r in pool.imap_unordered(work,[(t,str(a.out),a.frames) for t in todo],chunksize=1):
                    r['retry_attempt']=attempt+1
                    log.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');log.flush()
                    if r['state']=='failed':retry.append(r['id'])
                update_index(a.out,tasks)
    print(update_index(a.out,tasks),flush=True)
    atomic_json(a.out/'batch_finished.json',dict(finished=time.strftime('%Y-%m-%d %H:%M:%S'),**update_index(a.out,tasks)))

if __name__=='__main__':main()
