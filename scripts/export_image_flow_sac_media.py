"""Render stored physical SAC training/evaluation traces, including failed episodes.

No simulator is invoked by this exporter: the GIF, NPZ and metric JSON all refer
to the same actual recorded episode. Restarting resumes completed artifacts.
"""
import os
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse
import hashlib
import html
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from scripts.export_vrl_v3_all_gifs import ROOT,atomic_json,work

OUT=Path('/home/wj/桌面/VRL_图像血流SAC_训练测试_3D_GIF_20261011')
VALIDATION=ROOT/'research/validation/EXP0095_IMAGE_FLOW_SAC_20261011'
RUN=ROOT/'research/runs/EXP0095_IMAGE_FLOW_SAC_20261011/seed0'


def registry(out,include_wide=False):
    held=set(json.load(open(ROOT/'configs/evaluation_splits.json'))['anatomy_holdout_v1']['held_out'])
    sources=[('01_真实SAC训练',RUN/'training_traces')]
    if include_wide:sources.append(('05_失败的全幅SAC训练',RUN.parent/'seed0_wide_action/training_traces'))
    sources.append(('07_随机策略复现校验',VALIDATION/'replay_verification/traces_True'))
    sources.extend([(section,VALIDATION/folder/'traces') for section,folder in
                    [('02_全部困难场景','full_hard'),('03_强流脉动全场景','stress'),('06_全部弱流V5场景','v5'),
                     ('07_已知故障诊断','diagnostics'),
                     ('04_图像消融','ablation_no_images'),('04_脉动估计消融','ablation_no_pulse'),
                     ('04_原位控制消融','ablation_no_station'),('04_横向控制消融','ablation_speed_only'),
                     ('04_速度适配幅度消融','ablation_speed_gain'),
                     ('04_近目标控制幅度消融','ablation_contact_range')]])
    tasks=[]
    for section,folder in sources:
        for source in sorted(folder.glob('*.npz')):
            try:
                with np.load(source) as z:
                    meta=json.loads(str(z['meta']));metrics=json.loads(str(z['metrics']))
                    if len(z['t'])<2:raise ValueError(f'Incomplete trace: {source}')
            except (OSError,EOFError,ValueError,zipfile.BadZipFile):
                # A running evaluator can still be writing the newest trace.
                # It remains pending in the full scope, never counted complete.
                continue
            training='真实' in section or '失败' in section
            method=metrics['method'];cell=metrics.get('cell','training')
            if training:
                label='图像/历史 SAC · 实际训练回合'
                iterations=metrics.get('policy_iterations',[])
                if iterations:label+=f'（更新 {min(iterations)}–{max(iterations)}）'
                training_config=source.parent.parent/'config.json'
                if training_config.exists() and iterations:
                    warmup=json.load(open(training_config)).get('critic_warmup_iterations')
                    if warmup is not None and max(iterations)<=warmup:
                        label+=' · 仅价值网络预训练'
                    elif warmup is not None and min(iterations)<=warmup:
                        label+=' · 含价值预训练及策略更新'
                arm='actual_SAC_training'
            elif method=='parent':label='原最高成功率方法 · 无 RL';arm='parent'
            elif method=='old_ppo':label='上一版类别 PPO';arm='old_PPO'
            else:
                p=Path(method.partition('=')[2]);arm=p.stem
                label='图像/历史 SAC · '+('未训练初始化' if arm=='initial' else '已训练 '+arm)
            an=meta['anatomy'];n=meta['n'];seed=meta['seed'];topology='留出解剖' if an in held else '训练解剖'
            job=[an,n,seed,meta['flow'],meta['latency'],meta['s']]
            uid=hashlib.sha256((section+'|'+str(source.relative_to(ROOT))).encode()).hexdigest()[:16]
            gif=str(Path(section)/cell.split('|')[0]/topology/arm/f'{an}_N{n}_seed{seed}_{uid}.gif')
            task=dict(id=uid,method=method,label=label,job=job,section=section,cell=cell,topology=topology,
                      opts={},source=str(source),reference=metrics,training=None,gif=gif,
                      actual_rl_training=training,training_algorithm='SAC')
            dest=(out/gif).with_suffix('.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(source,dest)
            tasks.append(task)
    atomic_json(out/'tasks.json',tasks);return tasks


def planned_count(include_wide):
    training=len(list((RUN/'training_traces').glob('*.npz')))
    if include_wide:training+=len(list((RUN.parent/'seed0_wide_action/training_traces').glob('*.npz')))
    diagnostics=len(list((VALIDATION/'diagnostics/traces').glob('*.npz')))+len(list((VALIDATION/'replay_verification/traces_True').glob('*.npz')))
    return training+252*4+168*4+168*3+1260*2+diagnostics


def index(out,tasks,expected=None):
    ready=lambda t:(out/t['gif']).exists() and (out/t['gif']).with_suffix('.json').exists()
    count=sum(ready(t) for t in tasks)
    expected=len(tasks) if expected is None else expected
    progress=dict(expected=expected,completed=count,pending=expected-count,recorded_episodes=len(tasks),
                  recorded_pending=len(tasks)-count,updated=time.strftime('%F %T'))
    atomic_json(out/'progress.json',progress)
    records=[dict(section=t['section'],method=t['label'],anatomy=t['job'][0],N=t['job'][1],
                  cell=t['cell'],topology=t['topology'],gif=t['gif'],ready=ready(t)) for t in tasks]
    page='''<!doctype html><meta charset="utf-8"><title>VRL 图像/血流 SAC 3D GIF</title>
<style>body{font:16px sans-serif;background:#f5f7fb;color:#17304c;margin:32px}input{padding:12px;width:75%}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#1265ab}</style>
<h1>图像与运动历史 SAC：真实训练及全场景回放</h1><p>STATUS</p>
<p>每份动画对应一个实际记录的完整三维回合，成功和失败均保留；相邻 NPZ 保存轨迹、血栓质量和执行指令，JSON 保存参数、指标和来源。
训练回合中的策略随更新变化，不能把它称作最终模型测试。测试回放来自开发评估池，未访问封存测试池。模型仅模拟接触溶栓。</p>
<input id="q" placeholder="搜索训练/测试、算法、解剖、N 或强流/脉动条件"><p id="count"></p><table><thead><tr><th>阶段</th><th>方法</th><th>解剖</th><th>N</th><th>条件</th><th>GIF</th></tr></thead><tbody id="rows"></tbody></table>
<script>const records=DATA;function show(){const q=document.getElementById('q').value.toLowerCase();const rs=records.filter(r=>JSON.stringify(r).toLowerCase().includes(q));document.getElementById('count').textContent='匹配 '+rs.length+' 项，显示前 300 项';document.getElementById('rows').innerHTML=rs.slice(0,300).map(r=>`<tr><td>${r.section}</td><td>${r.method}</td><td>${r.anatomy} / ${r.topology}</td><td>${r.N}</td><td>${r.cell}</td><td>${r.ready?`<a href="${encodeURI(r.gif)}" target="_blank">打开</a>`:'生成中'}</td></tr>`).join('')}document.getElementById('q').addEventListener('input',show);show();</script>'''
    page=page.replace('STATUS',f'全范围计划 {expected} 份，已生成 {count} 份；已记录轨迹 {len(tasks)} 份，其他场景正在评估或等待导出。更新 {progress["updated"]}').replace('DATA',json.dumps(records,ensure_ascii=False))
    (out/'index.html').write_text(page);return progress


def main():
    global RUN,VALIDATION
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--frames',type=int,default=180);p.add_argument('--include-wide',action='store_true')
    p.add_argument('--prepare',action='store_true');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--run',type=Path,default=RUN);p.add_argument('--validation',type=Path,default=VALIDATION)
    p.add_argument('--recorded-only',action='store_true')
    a=p.parse_args();os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=True)
    RUN=a.run.resolve();VALIDATION=a.validation.resolve()
    import fcntl
    lock=(a.out/'.export.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=registry(a.out,a.include_wide)
    expected=len(tasks) if a.recorded_only else planned_count(a.include_wide)
    if not tasks:raise RuntimeError('No actual recorded episodes available')
    print(index(a.out,tasks,expected),flush=True)
    if a.prepare:return
    pending=[t for t in tasks if not ((a.out/t['gif']).exists() and (a.out/t['gif']).with_suffix('.json').exists())]
    if a.limit:pending=pending[:a.limit]
    with mp.get_context('spawn').Pool(a.workers,maxtasksperchild=30) as pool,(a.out/'export_log.jsonl').open('a') as log:
        todo=pending
        for attempt in range(3):
            retry=set()
            for k,result in enumerate(pool.imap_unordered(work,[(t,str(a.out),a.frames) for t in todo])):
                result['attempt']=attempt;log.write(json.dumps(result,ensure_ascii=False,default=str)+'\n');log.flush()
                if result['state']=='failed':retry.add(result['id']);print(result['error'],flush=True)
                if k%10==0:print(index(a.out,tasks,expected),flush=True)
            if not retry:break
            todo=[t for t in pending if t['id'] in retry]
    progress=index(a.out,tasks,expected);print(progress,flush=True)
    if progress['recorded_pending'] and not a.limit:raise RuntimeError('Incomplete recorded media; inspect export_log.jsonl')
    atomic_json(a.out/'LAST_EXPORT_BATCH.json',progress)
    if not progress['pending']:atomic_json(a.out/'COMPLETE.json',progress)


if __name__=='__main__':main()
