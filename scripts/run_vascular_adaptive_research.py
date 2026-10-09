"""Registered, evidence-driven rounds after EXP0060; negatives trigger another round.

All rounds are development research, never automatic claims of publication or
clinical success. A 09:00 snapshot is generated without ending the round queue.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/validation/EXP0061_ADAPTIVE_WORLD_RL_20261006'
RUNS=ROOT/'research/runs/EXP0061_ADAPTIVE_WORLD_RL_20261006'
OLD=ROOT/'research/validation/EXP0060_VASCULAR_OPTION_RL_20261006'
OLD_RUNS=ROOT/'research/runs/EXP0060_VASCULAR_OPTION_RL_20261006'
PY='/home/wj/.cache/vascular-research/cpu312-clean-20261004/bin/python'
SCREEN='mca_m1_lvo,ica_terminus_t,renal_artery'
FIELDS=['cluster_safe_success','removal','removal_auc','wall_contact_s','particle_events']


def records(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def paired(candidate,baseline):
    key=lambda r:(r['anatomy'],r['seed'],r['clusters'],r['sensing'])
    b={key(r):r for r in baseline};c={key(r):r for r in candidate}
    if len(b)!=len(baseline) or len(c)!=len(candidate) or set(b)!=set(c):
        raise ValueError('Incomplete or duplicated paired scenes')
    keys=sorted(b)
    if any((b[k]['scenario_hash'],b[k]['initial_state_hash'])!=(c[k]['scenario_hash'],c[k]['initial_state_hash']) for k in keys):
        raise ValueError('Actual scene hashes differ')
    result=dict(n=len(keys),scene_hashes_match=True)
    rng=np.random.default_rng(61)
    for f in FIELDS:
        d=np.array([float(c[k][f])-float(b[k][f]) for k in keys])
        draws=d[rng.integers(0,len(d),(2000,len(d)))].mean(1)
        result[f]=dict(candidate=float(np.mean([c[k][f] for k in keys])),baseline=float(np.mean([b[k][f] for k in keys])),
                       delta=float(d.mean()),ci95=np.percentile(draws,[2.5,97.5]).tolist())
    result['spacing_violation_pair_s']=float(np.mean([c[k]['spacing']['spacing_violation_pair_s'] for k in keys]))
    result['baseline_spacing_violation_pair_s']=float(np.mean([b[k]['spacing']['spacing_violation_pair_s'] for k in keys]))
    # An explicitly exploratory advancement gate; these are not publication claims.
    safe=result['cluster_safe_success']['delta'];rem=result['removal']['delta'];auc=result['removal_auc']['delta']
    wall=result['wall_contact_s']['delta'];particle=result['particle_events']['delta']
    no_regression=(rem>=-.005 and wall<=.1 and particle<=.1 and
                   result['spacing_violation_pair_s']<=result['baseline_spacing_violation_pair_s']+1e-9)
    result['promising']=bool(no_regression and ((safe>0 and auc>=-.01) or
                            (safe>=0 and auc>=0 and wall<-.1) or (safe>=0 and auc>.005)))
    result['label']='development_promising_not_confirmed' if result['promising'] else 'not_accepted_continue_research'
    return result


def next_recipe(recipe,result,last_model,round_id):
    r=dict(recipe);reasons=[]
    if result['wall_contact_s']['delta']>.1:
        r['wall_weight']=min(32.,r['wall_weight']*1.5);reasons.append('wall contact regression: increase safety cost equally in both arms')
    if result['particle_events']['delta']>0:
        r['particle_weight']=min(24.,r['particle_weight']+4);reasons.append('particle events: strengthen contact cost in both arms')
    if result['removal']['delta']<-.005:
        r['gamma']=min(.9995,r['gamma']+.002);reasons.append('incomplete clearance: longer reward horizon')
    if last_model and last_model.get('state_mse',0)>last_model.get('persistence_mse',float('inf')):
        r['world_weight']=max(.02,r['world_weight']*.5);r['model_warmup']=min(32,r['model_warmup']+8)
        reasons.append('predictive MSE worse than persistence: reduce auxiliary interference and increase warmup')
    if not reasons:
        r['entropy']=min(.05,r['entropy']+.005);reasons.append('no robust gain: broaden policy exploration')
    if round_id>=1:
        r['updates']=min(240,120+40*((round_id+1)//2))
    # Alternate after failed repair attempts: a measured pursuit-initialized
    # policy is another registered learning start, not a fabricated trained model.
    if not result['promising'] and round_id%3==1:
        r['parent_kind']='initial' if r['parent_kind']=='final' else 'final'
        reasons.append('compare another initialization after repeated failure; both arms share it')
    if r==recipe:
        r['lr']=max(1e-5,r['lr']*.5)
        reasons.append('reduce update size after bounded safety coefficients stop changing')
        if r==recipe:reasons.append('bounded-parameter replication on a new registered training stream; not a new mechanism')
    return r,reasons


def seed_scene_interval(candidates,baseline):
    """Resample training seeds and paired scene units; never pool 3x repeated scenes."""
    key=lambda r:(r['anatomy'],r['seed'],r['clusters'],r['sensing'])
    b={key(r):r for r in baseline};keys=sorted(b);diff=[]
    for rows in candidates:
        paired(rows,baseline)
        c={key(r):r for r in rows}
        diff.append([float(c[k]['cluster_safe_success'])-float(b[k]['cluster_safe_success']) for k in keys])
    diff=np.array(diff);rng=np.random.default_rng(6101);si=rng.integers(0,len(diff),(2000,len(diff)))
    strata={name:[j for j,k in enumerate(keys) if k[0]==name] for name in sorted({k[0] for k in keys})}
    ci=np.concatenate([np.asarray(ix)[rng.integers(0,len(ix),(2000,len(ix)))] for ix in strata.values()],axis=1)
    samples=diff[si[:,:,None],ci[:,None,:]].mean((1,2))
    return dict(training_seeds=len(diff),scenes_per_seed=len(keys),safe_delta=float(diff.mean()),
                hierarchical_ci95=np.percentile(samples,[2.5,97.5]).tolist(),development_only=True)


def write_report(state):
    OUT.mkdir(parents=True,exist_ok=True)
    lines=['# EXP0061 连续科研进展',f'更新时间：{datetime.now().isoformat(timespec="seconds")}',
        '本队列持续执行：实际训练 → 同场景评估 → 失败分型 → 登记下一轮。负结果不会被改写为成功。所有比较均为开发集探索，原密封测试不访问。',
        f'当前阶段：{state.get("stage")}；已登记轮次上限 {state.get("max_rounds",24)}。09:00 生成阶段快照，不因单个实验结束而停止队列。',
        '## 已完成的上一轮诊断',
        'EXP0060 seed0 的 480 更新模型对传统追踪：Safe 同为 5/12；清除 97.92% vs 100%；壁接触 22.54 s vs .502 s。该模型未被接受为优于传统方法。',
        '## 实际学习方法',
        'GRU 历史编码 + 三成员动作条件观测/风险预测器 + PPO。预测器学习下一次测量变化和训练回报/接触事件；actor 只见测量历史和网络预测，不读取候选动作的真实仿真后果。',
        '每轮两个训练臂共享父检查点、追加预算、场景流和奖励。continued-PPO 是相同预算的 DRL 对照；world-PPO 加预测辅助和预测门。不是 DreamerV3 复现，也不是临床已验证的世界模型。',
        '初始配置同时增加两臂的安全训练代价，因此相对旧模型提升不能全部归因于预测模型；只能用两臂差异检验其增量贡献。',
        '## 每轮结果',
        '| 轮次 | 方法 | 场景 | Safe % | 清除 % | AUC % | 壁接触 s | 结论 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for rr in state.get('rounds',[]):
        for method,stats in rr.get('comparisons',{}).items():
            lines.append(f'| {rr["round"]} | {method} | {stats["n"]} | {100*stats["cluster_safe_success"]["candidate"]:.1f} | '
                        f'{100*stats["removal"]["candidate"]:.1f} | {100*stats["removal_auc"]["candidate"]:.2f} | '
                        f'{stats["wall_contact_s"]["candidate"]:.3f} | {stats["label"]} |')
        if rr.get('next_reasons'):lines+=['',f'轮 {rr["round"]} 下一轮依据：'+ '; '.join(rr['next_reasons']),'']
    lines+=['','## 解释边界',
        'promising 只表示值得扩展验证，不能称稳定超过启发式。需要强固定选项对照、多种子与额外场景复核。单次 seed0 筛查、重复使用开发集和多轮选择存在选择偏差。',
        '额外场景也标注为开发复核，不冒充密封测试。移除预测门的评估属于依赖性干预，不能代替同预算重训练消融。',
        '真实磁场尚未校准，2 mm 仅为间距代理。体外能破栓与学习导航可迁移是两个需要分别验证的结论。',
        '失败/预算未完成的作业留在 jobs 与 STATUS.json；三个连续运行基础设施失败会停在明确失败状态，避免无意义重试。',
        '',f'状态与全部证据：{OUT}']
    formatted=[]
    for l in lines:formatted.extend([l] if l.startswith('|') else ['',l,''])
    (OUT/'REPORT.md').write_text('\n'.join(formatted)+'\n')
    if datetime.now().hour>=9 and not (OUT/'MORNING_0900.json').exists():
        (OUT/'MORNING_0900.json').write_text(json.dumps(state,indent=2))
        shutil.copyfile(OUT/'REPORT.md',OUT/'MORNING_0900.md')


def main():
    os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
    OUT.mkdir(parents=True,exist_ok=True);RUNS.mkdir(parents=True,exist_ok=True)
    snap=OUT/'source_snapshot'
    if snap.exists():raise RuntimeError('Registered adaptive queue already exists; preserve it')
    snap.mkdir()
    for folder in ['marl','scripts','environments','configs']:
        shutil.copytree(ROOT/folder,snap/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    hashes={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file()}
    (OUT/'SOURCE_SHA256.json').write_text(json.dumps(hashes,indent=2))
    env=dict(os.environ,PYTHONPATH=str(snap),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    lock=threading.RLock()
    state=dict(pid=os.getpid(),stage='initializing',max_rounds=24,rounds=[],jobs=[],failures=[])
    def update(stage=None):
        with lock:
            if stage:state['stage']=stage
            state['time']=datetime.now().isoformat()
            tmp=OUT/'STATUS.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'STATUS.json')
            write_report(state)
    def run(name,args,timeout=3000,python=PY):
        logdir=OUT/'jobs';logdir.mkdir(exist_ok=True)
        cmd=['nice','-n','10','taskset','-c','0-5',python,'-u','-m',*map(str,args)]
        with (logdir/f'{name}.log').open('w') as log:
            log.write(json.dumps(dict(cmd=cmd,cwd=str(snap)))+'\n');log.flush()
            pr=subprocess.Popen(cmd,cwd=snap,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            start=time.time();job=dict(name=name,pid=pr.pid,status='running')
            with lock:state['jobs'].append(job)
            update()
            while True:
                try:rc=pr.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    update()
                    if time.time()-start>timeout:
                        os.killpg(pr.pid,signal.SIGTERM)
                        try:pr.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(pr.pid,signal.SIGKILL);pr.wait()
                        rc=pr.returncode
                        with lock:job['timeout']=True
                        break
            with lock:job.update(status='complete' if rc==0 else 'failed',returncode=rc,seconds=time.time()-start)
            if rc!=0:
                with lock:state['failures'].append(dict(job))
            update();return rc==0
    def evaluate(name,checkpoint,path,scenes=4,anatomies=SCREEN,scene_start=0,fixed=0,disable=False,capture=False):
        args=['scripts.vascular_predictive_study','eval' if checkpoint else 'baseline','--arm',name,'--out',path,
              '--scenes',scenes,'--anatomies',anatomies,'--scene-start',scene_start,'--fixed-option',fixed]
        if checkpoint:args+=['--checkpoint',checkpoint]
        if disable:args+=['--disable-world']
        if capture:args+=['--capture']
        return run(name,args,timeout=3600)
    def best_fixed():
        scored=[]
        for option in range(9):
            rows=[]
            for p in (OLD/'eval').glob(f'fixed_{option}_N3_noise_*.jsonl'):rows+=records(p)
            if len(rows)!=54:return 0
            scored.append((np.mean([r['cluster_safe_success'] for r in rows]),np.mean([r['removal_auc'] for r in rows]),option))
        return int(max(scored)[-1])
    recipe=dict(lr=1e-4,gamma=.995,entropy=.01,world_weight=.1,model_warmup=8,wall_weight=12.,
                particle_weight=12.,updates=120,parent_kind='final')
    (OUT/'ADAPTIVE_PROTOCOL.json').write_text(json.dumps(dict(initial_recipe=recipe,max_rounds=24,
        screen_anatomies=SCREEN,screen_scenes=4,workers=2,rollout=128,minutes_per_run=45,
        advancement='safe/efficiency improvement with no removal/wall/particle/spacing regression; development only',
        morning='09:00 stage snapshot, not a stop',formal_EXP0060_unchanged=True),indent=2))
    update('running');consecutive_failures=0
    for round_id in range(24):
        if (OUT/'STOP_AFTER_ROUND').exists():update('stopped_by_local_control');return
        rd=OUT/f'round_{round_id:02d}';rd.mkdir(exist_ok=False)
        parent=OLD_RUNS/'memory_mappo_s0'/f'{recipe["parent_kind"]}.pt'
        if not parent.exists():raise FileNotFoundError(parent)
        rr=dict(round=round_id,recipe=dict(recipe),parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),comparisons={})
        state['rounds'].append(rr);(rd/'PREREGISTRATION.json').write_text(json.dumps(rr,indent=2));update(f'round_{round_id}_training')
        outputs={}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={}
            for arm in ['continued_ppo','world_ppo']:
                name=f'r{round_id:02d}_{arm}';out=RUNS/name;outputs[arm]=out
                args=['scripts.vascular_predictive_study','train','--arm',arm,'--out',out,'--parent',parent,
                      '--round',round_id,'--workers',2,'--rollout',128,'--minutes',45]
                for key,value in recipe.items():
                    if key!='parent_kind':args+=['--'+key.replace('_','-'),value]
                futures[pool.submit(run,name,args,3000)]=arm
            ok=all([f.result() for f in futures])
        ok=ok and all((o/'DONE.json').exists() and json.loads((o/'DONE.json').read_text())['budget_completed'] for o in outputs.values())
        if not ok:
            consecutive_failures+=1;rr['status']='training_or_budget_failure';update()
            if consecutive_failures>=3:update('infrastructure_blocked_three_rounds');return
            continue
        fixed=best_fixed();rr['fixed_reference']=fixed
        baseline=OUT/f'fixed_{fixed}_screen.jsonl'
        if not baseline.exists():
            if not evaluate(f'fixed_{fixed}_screen',None,baseline,fixed=fixed):raise RuntimeError('Baseline evaluation failed')
        if len(records(baseline))!=12:raise RuntimeError('Incomplete baseline; do not compare')
        update(f'round_{round_id}_paired_evaluation')
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs={}
            for arm,out in outputs.items():
                fs[pool.submit(evaluate,f'r{round_id:02d}_{arm}_screen',out/'final.pt',rd/f'{arm}.jsonl')]=arm
            ok=all([f.result() for f in fs])
        if not ok:
            consecutive_failures+=1;rr['status']='evaluation_failed';update()
            if consecutive_failures>=3:update('infrastructure_blocked_three_rounds');return
            continue
        consecutive_failures=0
        for arm in outputs:rr['comparisons'][arm]=paired(records(rd/f'{arm}.jsonl'),records(baseline))
        rr['world_vs_continued_ppo']=paired(records(rd/'world_ppo.jsonl'),records(rd/'continued_ppo.jsonl'))
        rr['status']='screen_complete';(rd/'COMPARISON.json').write_text(json.dumps(rr,indent=2));update()
        winner=max(rr['comparisons'],key=lambda arm:(rr['comparisons'][arm]['cluster_safe_success']['candidate'],
                        rr['comparisons'][arm]['removal']['candidate'],-rr['comparisons'][arm]['wall_contact_s']['candidate']))
        if rr['comparisons'][winner]['promising']:
            update(f'round_{round_id}_expanded_development')
            anatomies=','.join(json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']['train'])
            broader_baseline=OUT/f'fixed_{fixed}_expanded.jsonl'
            if not broader_baseline.exists():
                evaluate(f'fixed_{fixed}_expanded',None,broader_baseline,6,anatomies,100,fixed=fixed)
            larger=rd/'expanded_winner.jsonl'
            if evaluate(f'r{round_id:02d}_{winner}_expanded',outputs[winner]/'final.pt',larger,6,anatomies,100):
                rr['expanded']=paired(records(larger),records(broader_baseline))
            if winner=='world_ppo':
                ab=rd/'world_gate_disabled.jsonl'
                if evaluate(f'r{round_id:02d}_gate_disabled',outputs[winner]/'final.pt',ab,disable=True):
                    rr['gate_intervention']=paired(records(ab),records(rd/'world_ppo.jsonl'))
            if rr.get('expanded',{}).get('promising'):
                # Replicate both arms, not just the winning model. Parent model
                # provenance distinguishes new finetune seeds from independent
                # full training seeds when EXP0060 seeds are not yet ready.
                rr['replications']=[];update(f'round_{round_id}_seed_replication')
                for seed in [1,2]:
                    seed_parent=OLD_RUNS/f'memory_mappo_s{seed}'/f'{recipe["parent_kind"]}.pt'
                    independent=seed_parent.exists()
                    if not independent:seed_parent=parent
                    seed_outputs={};okays=[]
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        fs=[]
                        for arm in outputs:
                            name=f'r{round_id:02d}_{arm}_s{seed}';out=RUNS/name;seed_outputs[arm]=out
                            args=['scripts.vascular_predictive_study','train','--arm',arm,'--out',out,'--parent',seed_parent,
                                  '--round',round_id,'--seed',seed,'--workers',2,'--rollout',128,'--minutes',45]
                            for key,value in recipe.items():
                                if key!='parent_kind':args+=['--'+key.replace('_','-'),value]
                            fs.append(pool.submit(run,name,args,3000))
                        okays=[f.result() for f in fs]
                    rep=dict(seed=seed,parent=str(seed_parent),independent_parent_seed=independent,comparisons={})
                    if all(okays) and all((o/'DONE.json').exists() and json.loads((o/'DONE.json').read_text())['budget_completed'] for o in seed_outputs.values()):
                        for arm,out in seed_outputs.items():
                            path=rd/f'{arm}_s{seed}_expanded.jsonl'
                            if evaluate(f'r{round_id:02d}_{arm}_s{seed}_expanded',out/'final.pt',path,6,anatomies,100):
                                rep['comparisons'][arm]=paired(records(path),records(broader_baseline))
                    rr['replications'].append(rep);update()
                seed_paths=[larger]+[rd/f'{winner}_s{seed}_expanded.jsonl' for seed in [1,2]]
                if all(p.exists() and len(records(p))==54 for p in seed_paths):
                    rr['seed_scene_analysis']=seed_scene_interval([records(p) for p in seed_paths],records(broader_baseline))
                    rr['seed_scene_analysis']['all_parent_seeds_independent']=all(r['independent_parent_seed'] for r in rr['replications'])
            # Save the actual candidate rollout, even if the broader comparison regresses.
            tracefile=rd/'replay.jsonl'
            if evaluate(f'r{round_id:02d}_replay',outputs[winner]/'final.pt',tracefile,scenes=1,anatomies='mca_m1_lvo',capture=True):
                run(f'r{round_id:02d}_gui',['scripts.view_vascular_option_rl','--trace',tracefile.with_suffix('.trace.json'),
                    '--headless','--out',ROOT/f'research/figures/EXP0061_20261006/round_{round_id:02d}'],120,
                    python='/home/wj/miniconda3/envs/v/bin/python')
        model_logs=records(outputs['world_ppo']/'training.jsonl')
        basis=rr.get('expanded',rr['comparisons']['world_ppo'])
        recipe,reasons=next_recipe(recipe,basis,model_logs[-1] if model_logs else None,round_id)
        rr['next_reasons']=reasons;rr['next_recipe']=recipe
        rr['decision']='continue_next_registered_DRL_round'
        (rd/'COMPARISON.json').write_text(json.dumps(rr,indent=2));update(f'round_{round_id}_finished_next_pending')
    update('registered_24_round_budget_completed')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        OUT.mkdir(parents=True,exist_ok=True)
        fatal=dict(time=datetime.now().isoformat(),error=repr(e),traceback=traceback.format_exc())
        (OUT/'FATAL.json').write_text(json.dumps(fatal,indent=2))
        if (OUT/'STATUS.json').exists():
            status=json.loads((OUT/'STATUS.json').read_text());status['stage']='infrastructure_failure';status['fatal']=fatal
            (OUT/'STATUS.json').write_text(json.dumps(status,indent=2));write_report(status)
        raise
