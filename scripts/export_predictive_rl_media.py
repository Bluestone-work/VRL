"""Export genuine PPO training traces and all matched RL navigation scenarios."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from scripts.export_vrl_v3_all_gifs import ROOT,atomic_json,update_index,work

OUT=Path('/home/wj/桌面/VRL_RL_局部导航_训练测试_GIF_20261011')

def make_tasks(checkpoint,out,include_v5):
    import numpy as np
    from scripts.eval_exp0090 import jobs_for
    split=json.load(open(ROOT/'configs/evaluation_splits.json'))['anatomy_holdout_v1'];held=set(split['held_out'])
    tasks=[];seed_folder=Path(checkpoint).parent;initial=str(seed_folder/'initial.pt')
    def add(method,label,job,section,cell,source,training_source=None):
        uid=hashlib.sha256(json.dumps([method,job,section,cell],sort_keys=True).encode()).hexdigest()[:16]
        an,n,s,f,L,v=job;topo='留出解剖' if an in held else '训练解剖'
        folder='RL_main' if method=='rl_nav='+checkpoint else 'RL_initial' if method=='rl_nav='+initial else 'parent_no_RL'
        if training_source:folder='actual_PPO_training'
        rel=str(Path(section)/cell.split('|')[0]/topo/folder/f'{an}_N{n}_seed{s}_{uid}.gif')
        task=dict(id=uid,method=method,label=label,job=list(job),opts={},section=section,cell=cell,topology=topo,
                  source=source,reference=None,training=None,gif=rel,actual_rl_training=bool(training_source))
        tasks.append(task)
        if training_source:
            dest=(out/rel).with_suffix('.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(training_source,dest)
    for p in sorted((ROOT/seed_folder/'training_traces').glob('*.npz')):
        with np.load(p) as z:meta=json.loads(str(z['meta']));metrics=json.loads(str(z['metrics']))
        job=(meta['anatomy'],meta['n'],meta['seed'],meta['flow'],meta['latency'],meta['s'])
        stages=metrics.get('policy_iterations',[])
        add('rl_training','PPO 局部导航 · 真实训练回合'+(f'（更新 {min(stages)}–{max(stages)}）' if stages else ''),job,
            '01_真实PPO训练','training',str(p),p)
    args=SimpleNamespace(protocol='hard',panels='low_delay,moderate_delay,high_delay,strong_flow,variable_response,ood_flow_delay',train_only=False)
    for cell,job in jobs_for(args):
        for method,label in [('rl_nav='+checkpoint,'PPO RL 局部导航 · 已训练'),('rl_nav_off='+initial,'原预测选择器 · 关闭 RL'),('rl_nav='+initial,'PPO 导航 · 未训练初始化')]:
            add(method,label,job,'02_全部困难场景',cell,'Original V3 252 paired scene registry')
    if include_v5:
        for cell,job in jobs_for(SimpleNamespace(protocol='v5',strengths='0,1,1.5',count=10)):
            for method,label in [('rl_nav='+checkpoint,'PPO RL 局部导航 · 已训练'),('rl_nav_off='+initial,'原预测选择器 · 关闭 RL')]:
                add(method,label,job,'03_全部弱流V5场景',cell,'Original V5 1260 paired scene registry')
    atomic_json(out/'tasks.json',tasks);return tasks

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--training-only',action='store_true');p.add_argument('--no-v5',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--frames',type=int,default=180);p.add_argument('--limit',type=int,default=0);a=p.parse_args();os.chdir(ROOT)
    if hasattr(os,'sched_getaffinity'):
        allowed=os.sched_getaffinity(0)-{6,7}
        if allowed:os.sched_setaffinity(0,allowed)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    a.out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.out/'.export.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('An export batch is already running for this output directory.')
    if (a.out/'tasks.json').exists():tasks=json.loads((a.out/'tasks.json').read_text())
    else:
        if not a.checkpoint:p.error('--checkpoint required to prepare tasks')
        tasks=make_tasks(a.checkpoint,a.out,not a.no_v5)
    scoped=[t for t in tasks if not a.training_only or t.get('actual_rl_training')]
    pending=[t for t in scoped if not ((a.out/t['gif']).exists() and (a.out/t['gif']).with_suffix('.json').exists())]
    if a.limit:pending=pending[:a.limit]
    print(update_index(a.out,tasks),flush=True)
    with mp.get_context('spawn').Pool(a.workers,maxtasksperchild=30) as pool,(a.out/'export_log.jsonl').open('a') as log:
        todo=pending
        for attempt in range(3):
            retry=set()
            for k,r in enumerate(pool.imap_unordered(work,[(t,str(a.out),a.frames) for t in todo])):
                if attempt:r['retry_attempt']=attempt
                log.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');log.flush()
                print(json.dumps({key:r.get(key) for key in ('id','state','gif','error')},ensure_ascii=False),flush=True)
                if r['state']=='failed':retry.add(r['id'])
                if k%10==0:update_index(a.out,tasks)
            if not retry:break
            todo=[t for t in pending if t['id'] in retry]
    progress=update_index(a.out,tasks);print(progress,flush=True)
    remaining=[t['id'] for t in pending if not ((a.out/t['gif']).exists() and (a.out/t['gif']).with_suffix('.json').exists())]
    if remaining:raise RuntimeError(f'{len(remaining)} GIF exports are incomplete after retries; see export_log.jsonl')
    if not a.training_only and not a.limit and progress['pending']:
        raise RuntimeError('The full GIF matrix is incomplete')

if __name__=='__main__':main()
