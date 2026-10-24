"""Freeze training-only selection, validate paired SAC, save reports and all media."""
import os
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'research/validation/EXP0095_IMAGE_FLOW_SAC_20261011'
R=ROOT/'research/runs/EXP0095_IMAGE_FLOW_SAC_20261011/seed0'
MEDIA=Path('/home/wj/桌面/VRL_图像血流SAC_训练测试_3D_GIF_20261011')


def rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.open() if line.strip()]


def status(stage,**extra):
    record=dict(stage=stage,updated=time.strftime('%F %T'),**extra)
    (V/'CURRENT_STATUS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record,ensure_ascii=False),flush=True)


def run(module,args,log):
    with log.open('a') as f:
        subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def check_pool(folder,expected):
    rs=rows(folder/'episodes.jsonl')
    if len(rs)!=expected:raise RuntimeError(f'{folder}: {len(rs)} / {expected} complete episodes')
    failed=rows(folder/'failures.jsonl')
    if failed:raise RuntimeError(f'{folder}: {len(failed)} failures; preserve and inspect before claiming completion')
    traces=list((folder/'traces').glob('*.npz'))
    if len(traces)!=expected:raise RuntimeError(f'{folder}: {len(traces)} / {expected} physical traces')


def select():
    from scripts.report_image_flow_sac import report
    while len(rows(V/'tune/episodes.jsonl'))<540:
        if rows(V/'tune/failures.jsonl'):raise RuntimeError('Tuning failures recorded')
        status('waiting_for_complete_training_anatomy_tuning',complete=len(rows(V/'tune/episodes.jsonl')),expected=540)
        time.sleep(30)
    summary=report([V/'tune/episodes.jsonl'],V/'tune_report')
    candidates={name:r for name,r in summary['results'].items() if name.startswith('seed0/policy_it')}
    if any(r['n']!=108 for r in summary['results'].values()):raise RuntimeError('Unequal tuning coverage')
    best=max(candidates,key=lambda k:(candidates[k]['strict_success'],-candidates[k]['wall_contact_s'],candidates[k]['removal'],-candidates[k]['T90_300']))
    checkpoint=str((R/Path(best).name).relative_to(ROOT))
    import torch
    p=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
    if not p['trained_with_rl'] or p['gradient_updates']<=0:raise RuntimeError('Selected actor was not trained with RL')
    for source,digest in p['config']['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Training source changed: {source}')
    record=dict(checkpoint=checkpoint,sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
        selected_at=time.strftime('%F %T'),selection_pool='108 independent paired cases on nine training anatomies',
        criterion='maximize Strict, then minimize wall contact, then maximize clearance, then minimize T90_300',
        tuning_scores=summary['results'],iterations=p['iterations'],gradient_updates=p['gradient_updates'],agent_steps=p['agent_steps'],
        sealed_test_accessed=False)
    (V/'SELECTED.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    snapshot=V/'frozen_sources';snapshot.mkdir(exist_ok=True)
    for source in (*p['config']['source_sha256'],'scripts/eval_image_flow_sac.py','scripts/report_image_flow_sac.py'):
        target=snapshot/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,target)
    status('checkpoint_frozen',checkpoint=checkpoint,sha256=record['sha256'])
    return checkpoint


def compare_original(checkpoint):
    from scripts.report_image_flow_sac import report
    rs=rows(ROOT/'research/validation/EXP0093_20261011/hard/episodes.jsonl')
    for r in rs:r['method']='V3:'+r['method_arg']
    target=V/'original_paper_hard_controls.jsonl'
    target.write_text(''.join(json.dumps(r)+'\n' for r in rs))
    report([V/'full_hard/episodes.jsonl',target],V/'paper_comparison_report')


def compare_ablations(checkpoint):
    from scripts.report_image_flow_sac import report
    files=[V/'stress/episodes.jsonl']
    for folder,kind in [('ablation_no_images','sac_no_images'),('ablation_no_pulse','sac_no_pulse'),('ablation_no_station','sac_no_station')]:
        rs=rows(V/folder/'episodes.jsonl')
        for r in rs:r['method']=kind+'='+checkpoint
        target=V/(folder+'_labeled.jsonl');target.write_text(''.join(json.dumps(r)+'\n' for r in rs));files.append(target)
    report(files,V/'combined_ablation_report')


def document(checkpoint):
    from scripts.report_image_flow_sac import label
    from scripts.md_to_docx import convert
    complete=json.load(open(R/'TRAINING_COMPLETE.json'));selected=json.load(open(V/'SELECTED.json'))
    audit=json.load(open(V/'authority/summary.json'))
    main=label('sac='+checkpoint)
    text=['# VRL：图像与血流历史强化学习局部控制修订及完整实验记录','',
        '本记录对应 EXP0095。V3 的局部方法是监督运动预测与固定代价选择；新版保留最高成功率组合的测地分配与固定 TPG，使用经过真实奖励训练的连续 SAC 产生最终局部补偿。旧论文数据仍作为原方法对照，不能当作新版 RL 的结果。','',
        '## 高层是否强行终止行为','',
        '旧控制链在 TPG 等待时清零指令并跳过类别 PPO，存在对局部动作的覆盖。30 个独立诊断场景中，73,041 个有效智能体控制步里，等待 77 步（0.1054%）；策略后的壁护改写约 2.99%，间距改写为 0。这批诊断不能支持“主要因为高层终止所以 PPO 没有效果”。回合结束由环境完成、丢失或时限决定，高层没有直接调用终止回合。','',
        '新版仍遵守 TPG 通行顺序，但等待期间允许按图像运动估计抵消漂移并保持原位，轴向动作受保持位置约束，横向补偿受壁护和间距层限制。实际等待、壁护、间距改写与运动均保存。','',
        '## 实际 RL 结构与输入','',
        'actor 接收当前和前一帧的双平面 32×32 裁剪（4 通道）、24×11 检测/已发指令历史和 48 维地图/预测上下文。CNN 提取图像信息，GRU 编码时间历史；延迟补偿和从观测运动拟合的漂移、响应增益、脉动频率信念作为可部署输入。真实心动相位、流速、位置与剩余质量不进入 actor。','',
        '原冻结预测选择器提供参考指令和预测特征，SAC 输出连续三维局部修正：轴向 ±0.2、两个横向各 ±0.06，随后满足执行器范数上限。策略后没有固定代价 argmin 再选动作，也没有专家动作模仿项。学习使用经验回放、双 Q、目标网络、重参数化 actor 更新与熵项。等待投影、壁护、间距层属于受限执行链，单独测量其覆盖，不算作 RL 学习收益。','',
        '奖励结合路径进展、真实清除质量、血栓附近接触、壁接触、机器人丢失、间距及任务完成。仿真真值只用于奖励与评价。物理模型表示接触溶栓，没有独立的机械碎栓指标；下面“清栓/清除”应按该模型理解。','',
        '## 训练、冻结与失败尝试','',
        f"正式 seed0 训练完成 {complete['iterations']} 轮、{complete['gradient_updates']:,} 次 SAC 梯度更新及 {complete['agent_steps']:,} 次真实智能体转移。只使用九种训练解剖，N=1–3，入口流速 0.025/0.05/0.1 mm/s，延迟 1–3 步，扰动强度不超过 1.25。检查点只按训练解剖的 108 个独立场景选择并冻结；所选为 {checkpoint}（更新 {selected['iterations']}）。五种留出解剖未参与训练和检查点选择，未访问封存测试池。当前只有一个正式 RL 训练种子。",'',
        '激进流量补偿/强制末端控制、全幅连续 SAC、近目标伺服和 Q 值门控试验全部保留。激进方法和后期 SAC 检查点发生退化，不能删除失败或把它们写成成功改进。正式方法取消默认强制漂移/末端替换，只由 RL 学习有界补偿。','',
        '## 完整配对验证','']
    for folder,title in [('full_hard_report','原论文全部 252 个困难场景'),('stress_report','14 种解剖 × N=1–3 × 四类强流/脉动条件（168 场景）')]:
        s=json.load(open(V/folder/'summary.json'))
        text.extend(['### '+title,'','| 方法 | 场景 | Strict % | 清除 % | T90_300 s | 壁 robot·s | 路线偏离 RMSE mm | 近目标误差 mm |',
                     '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for name,m in s['results'].items():
            text.append(f"| {name} | {m['n']} | {100*m['strict_success']:.2f} | {100*m['removal']:.2f} | {m['T90_300']:.2f} | {m['wall_contact_s']:.3f} | {m['route_cross_track_rmse_mm']:.4f} | {m['near_target_mean_error_mm']:.4f} |")
        text.append('')
        for ref in ['parent','seed0/initial.pt','old_ppo']:
            pair=s['pairs'][main+' vs '+ref];ci=pair['CI95_pp']
            text.append(f"相对 {ref}：Strict {pair['strict_delta_pp']:+.2f} pp（95% 场景自举区间 {ci[0]:+.2f} 至 {ci[1]:+.2f}）；平均清除 {pair['removal_delta_pp']:+.2f} pp；壁接触 {pair['wall_delta_robot_s']:+.3f} robot·s；配对场景哈希不一致 {pair['scenario_hash_mismatches']}。")
        m=s['results'][main]
        if 'rl_executed_mean_delta' in m:
            text.append(f"同一观测/护盾状态下，相对零 RL 补偿的最终执行指令平均差 {m['rl_executed_mean_delta']:.5f}；超过 0.001 的控制步占 {100*m['rl_executed_nontrivial_fraction']:.2f}%；修正经过执行链的幅度保留比 {100*m['rl_delta_survival_ratio']:.2f}%。该对照只是同状态动作归因，不能替代完整未训练对照回合。")
        text.extend(['','不同流速、延迟、扰动及训练/留出解剖的全部分组记录见本组 by_condition.csv、by_topology.csv 和 all_scenes.csv。近目标误差只在真实距离目标 1 mm 内统计，应同时看接触比例、清除率及 T90。',''])
        if folder=='stress_report':
            text.extend(['| 条件 | 方法 | Strict % | 清除 % | 壁 robot·s | 路线偏离 mm | 近目标误差 mm |','|---|---|---:|---:|---:|---:|---:|'])
            for r in s['by_condition']:
                if r['method'] in ('parent',main):
                    text.append(f"| {r['condition']} | {r['method']} | {100*r['strict_success']:.2f} | {100*r['removal']:.2f} | {r['wall_contact_s']:.3f} | {r['route_cross_track_rmse_mm']:.4f} | {r['near_target_mean_error_mm']:.4f} |")
            text.append('')
    text+=['## 图像、脉动估计与原位控制消融','',
        '分别关闭图像裁剪、从观测拟合脉动信念、原位控制；其余模型和场景相同。关闭图像仍保留从图像检测得到的位置/历史，不能解读为取消全部成像。关闭脉动仍保留运动历史和漂移估计。消融只使用已经冻结的模型，不再选择检查点。','',
        '| 条件 | Strict % | 清除 % | 壁 robot·s | 路线偏离 mm |','|---|---:|---:|---:|---:|']
    for folder,title in [('stress_report','完整 SAC'),('ablation_no_images_report','关闭图像裁剪'),('ablation_no_pulse_report','关闭脉动估计'),('ablation_no_station_report','关闭原位控制')]:
        p=V/folder/'summary.json'
        if not p.exists():continue
        m=json.load(open(p))['results'][main]
        text.append(f"| {title} | {100*m['strict_success']:.2f} | {100*m['removal']:.2f} | {m['wall_contact_s']:.3f} | {m['route_cross_track_rmse_mm']:.4f} |")
    text+=['','CNN/脉动模块被接入与它们带来可测收益是两个不同结论；应按上述消融的实际方向解释。壁护与原位控制的非 RL 变化不能全部归因于 actor 更新。','',
        '## 全场景三维 GIF 与文件','',
        f'媒体目录：{MEDIA}。真实训练、全部困难场景、强流/脉动、消融、弱流 V5 分目录保存；每份 GIF 有原始 NPZ 和 JSON 指标。旧论文的外部方法、基线及安全层消融动画保存在 VRL_V3_全部场景_3D_GIF_20261011，上一版 PPO 动画保存在 VRL_RL_局部导航_训练测试_GIF_20261011。',
        '','GIF 直接从实际记录的完整回合渲染，包含失败过程；不会用指标拼接成功动画。训练 GIF 对应在线训练中策略不断更新的真实回合，不能称作冻结模型的测试。批量导出进度以各目录 progress.json 和 index.html 为准。','',
        f'完整训练日志：{R}/training.jsonl；检查点冻结：{V}/SELECTED.json；原论文 23 组组合对比：{V}/paper_comparison_report/REPORT.md。']
    md=ROOT/'VRL_图像血流SAC_方法修订与完整实验记录_20261011.md'
    md.write_text('\n'.join(text)+'\n')
    target=md.with_suffix('.docx');convert(md.read_text(),target,src_dir=ROOT)
    return target


def resume_media(cpu_set=None):
    affinity=set(range(24))-{6,7} if cpu_set is None else set(cpu_set)-{6,7}
    if not affinity:raise ValueError('Media affinity must include at least one healthy CPU')
    paused=json.load(open(V/'paused_media.json'));restored=[]
    for group in paused['groups']:
        pid=group['pid'];proc=Path('/proc')/str(pid)
        try:
            cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
            if cmd.strip()!=group['args'].strip() or os.getpgid(pid)!=group['pgid']:continue
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    if os.getpgid(int(p.name))==group['pgid']:
                        for task in (p/'task').iterdir():
                            try:os.sched_setaffinity(int(task.name),affinity)
                            except ProcessLookupError:pass
                except (ProcessLookupError,PermissionError):pass
            os.killpg(group['pgid'],signal.SIGCONT);restored.append(group)
        except (ProcessLookupError,FileNotFoundError):pass
    (V/'media_restored.json').write_text(json.dumps(dict(time=time.strftime('%F %T'),groups=restored,cpus=sorted(affinity)),ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=22);p.add_argument('--skip-v5',action='store_true');a=p.parse_args()
    os.chdir(ROOT);V.mkdir(exist_ok=True,parents=True)
    import fcntl
    lock=(V/'.finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=json.load(open(V/'SELECTED.json'))['checkpoint'] if (V/'SELECTED.json').exists() else select()
    selection=json.load(open(V/'SELECTED.json'))
    if hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest()!=selection['sha256']:raise RuntimeError('Frozen checkpoint changed')
    initial=str((R/'initial.pt').relative_to(ROOT));methods=f'parent,old_ppo,sac={initial},sac={checkpoint}'
    from scripts.report_image_flow_sac import report
    for pool,folder,expected in [('hard','full_hard',1008),('stress','stress',672)]:
        status('paired_evaluation',pool=pool,expected=expected,checkpoint=checkpoint)
        run('scripts.eval_image_flow_sac',['--out',str(V/folder),'--pool',pool,'--methods',methods,'--workers',str(a.workers),'--traces','--attribution'],V/(folder+'.log'))
        check_pool(V/folder,expected);report([V/folder/'episodes.jsonl'],V/(folder+'_report'))
    compare_original(checkpoint)
    for folder,cfg in [('ablation_no_images',{'use_images':False}),('ablation_no_pulse',{'pulse':False}),('ablation_no_station',{'station_keeping':False})]:
        status('frozen_model_ablation',ablation=folder)
        config=V/(folder+'.json');config.write_text(json.dumps(cfg))
        run('scripts.eval_image_flow_sac',['--out',str(V/folder),'--pool','stress','--methods','sac='+checkpoint,
            '--controller-json',str(config),'--workers',str(a.workers),'--traces','--attribution'],V/(folder+'.log'))
        check_pool(V/folder,168);report([V/folder/'episodes.jsonl'],V/(folder+'_report'))
    compare_ablations(checkpoint)
    doc=document(checkpoint)
    status('algorithm_validation_complete',checkpoint=checkpoint,document=str(doc))
    run('scripts.export_image_flow_sac_media',['--workers',str(a.workers),'--include-wide'],V/'media.log')
    if not a.skip_v5:
        status('all_v5_scenes_evaluation',expected=2520)
        run('scripts.eval_image_flow_sac',['--out',str(V/'v5'),'--pool','v5','--methods','parent,sac='+checkpoint,
            '--workers',str(a.workers),'--traces','--attribution'],V/'v5.log')
        check_pool(V/'v5',2520);report([V/'v5/episodes.jsonl'],V/'v5_report')
        run('scripts.export_image_flow_sac_media',['--workers',str(a.workers),'--include-wide'],V/'media_v5.log')
    document(checkpoint)
    progress=json.load(open(MEDIA/'progress.json'))
    if progress['pending']:raise RuntimeError('SAC media matrix incomplete')
    resume_media()
    status('new_SAC_complete_legacy_comparison_media_resumed',checkpoint=checkpoint,document=str(doc),media=progress)
    (V/'COMPLETE.json').write_text(json.dumps(dict(checkpoint=checkpoint,document=str(doc),media=progress),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
