"""Select on fresh training anatomies; freeze, evaluate and export scalar speed SAC."""
import os
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.finish_image_flow_sac import ROOT,rows,run,check_pool,resume_media

V=ROOT/'research/validation/EXP0098_ENVELOPED_IMAGE_SAC_20261011'
R=ROOT/'research/runs/EXP0098_ENVELOPED_IMAGE_SAC_20261011/seed0'
MEDIA=Path('/home/wj/桌面/VRL_分阶段图像速度SAC_训练测试_3D_GIF_20261011')


def status(stage,**extra):
    record=dict(stage=stage,updated=time.strftime('%F %T'),**extra)
    (V/'CURRENT_STATUS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record,ensure_ascii=False),flush=True)


def select(workers):
    from scripts.report_image_flow_sac import report
    previous=ROOT/'research/validation/EXP0097_SPEED_IMAGE_SAC_20261011/tune/episodes.jsonl'
    projection=ROOT/'research/validation/EXP0096_SAFE_IMAGE_SAC_20261011/tune_speed_projection/episodes.jsonl'
    while not (R/'TRAINING_COMPLETE.json').exists() or len(rows(previous))<540 or len(rows(projection))<864:
        status('waiting_for_training_and_previous_tuning',training_complete=(R/'TRAINING_COMPLETE.json').exists(),
            previous_tuning=len(rows(previous)),previous_expected=540,projection=len(rows(projection)),projection_expected=864)
        time.sleep(30)
    methods='parent,'+','.join('sac_enveloped='+str((R/f'policy_it{k}.pt').relative_to(ROOT)) for k in [20,80,160,240])
    status('complete_training_anatomy_tuning',expected=540)
    run('scripts.eval_image_flow_sac',['--out',str(V/'tune'),'--pool','tune','--scene-seed-offset','39000000',
        '--methods',methods,'--workers',str(workers)],V/'tune.log')
    if len(rows(V/'tune/episodes.jsonl'))!=540 or rows(V/'tune/failures.jsonl'):raise RuntimeError('Tuning matrix incomplete')
    summary=report([V/'tune/episodes.jsonl'],V/'tune_report');parent=summary['results']['parent']
    candidates={k:r for k,r in summary['results'].items() if k.startswith('sac_enveloped:seed0/policy_it')}
    if any(r['n']!=108 for r in summary['results'].values()):raise RuntimeError('Unequal tuning coverage')
    admissible={k:r for k,r in candidates.items() if r['strict_success']>=parent['strict_success']-1e-12
        and r['removal']>=parent['removal']-1e-12 and r['wall_contact_s']<=parent['wall_contact_s']+1e-12
        and r['route_cross_track_rmse_mm']<=parent['route_cross_track_rmse_mm']+1e-12}
    jointly_admissible=bool(admissible)
    if not jointly_admissible:
        (V/'NO_JOINT_CANDIDATE.json').write_text(json.dumps(summary['results'],ensure_ascii=False,indent=2))
        status('no_jointly_admissible_trained_checkpoint',results=summary['results'])
    ranking=admissible if jointly_admissible else candidates
    best=max(ranking,key=lambda k:(ranking[k]['strict_success'],-ranking[k]['wall_contact_s'],ranking[k]['removal']))
    checkpoint=str((R/Path(best).name).relative_to(ROOT))
    import torch
    payload=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
    assert payload['trained_with_rl'] and payload['gradient_updates']>0
    for source,digest in payload['config']['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Training source changed: '+source)
    record=dict(checkpoint=checkpoint,sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
        selected_at=time.strftime('%F %T'),selection_pool='108 fresh cases on nine training anatomies, seed block 2189000000',
        criterion='Strict and clearance cannot decrease; wall and route error cannot increase. Then maximize Strict and minimize wall.' if jointly_admissible else
            'Diagnostic only: maximize training-anatomy Strict, then minimize wall and maximize clearance. Not an admissible improvement.',
        admissible=list(admissible),tuning_scores=summary['results'],iterations=payload['iterations'],
        gradient_updates=payload['gradient_updates'],agent_steps=payload['agent_steps'],sealed_test_accessed=False,
        jointly_admissible=jointly_admissible,method_kind='sac_enveloped',controller_config=payload['config']['controller'])
    record_name='SELECTED.json' if jointly_admissible else 'DIAGNOSTIC_SELECTED.json'
    (V/record_name).write_text(json.dumps(record,ensure_ascii=False,indent=2))
    for source in (*payload['config']['source_sha256'],'scripts/eval_image_flow_sac.py','scripts/report_image_flow_sac.py'):
        target=V/'frozen_sources'/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/source,target)
    return checkpoint


def combined_reports(checkpoint):
    from scripts.report_image_flow_sac import report
    original=rows(ROOT/'research/validation/EXP0093_20261011/hard/episodes.jsonl')
    for r in original:r['method']='V3:'+r['method_arg']
    target=V/'original_paper_hard_controls.jsonl';target.write_text(''.join(json.dumps(r)+'\n' for r in original))
    report([V/'full_hard/episodes.jsonl',target],V/'paper_comparison_report')
    files=[V/'stress/episodes.jsonl']
    for folder,kind in [('ablation_no_images','sac_no_images'),('ablation_no_pulse','sac_no_pulse'),('ablation_contact_range','sac_contact_range')]:
        records=rows(V/folder/'episodes.jsonl')
        for r in records:r['method']=kind+'='+checkpoint
        target=V/(folder+'_labeled.jsonl');target.write_text(''.join(json.dumps(r)+'\n' for r in records));files.append(target)
    report(files,V/'combined_ablation_report')


def document(checkpoint):
    from scripts.md_to_docx import convert
    from scripts.report_image_flow_sac import label
    complete=json.load(open(R/'TRAINING_COMPLETE.json'));chosen=selection_record()
    main=label('sac_enveloped='+checkpoint)
    lines=['# VRL：图像/历史分阶段 SAC 速度适配与完整实验记录','',
        'V3 的主方法是监督预测与固定代价选择，没有 RL。本版保留最高成功率组合的测地分配、固定 TPG、冻结预测器和安全执行层，以真实环境奖励训练的连续 SAC 适配沿可靠参考方向的局部速度、减速和驻留。监督预测器与 RL actor 分开记录，不能把旧成绩当成本版成绩。','',
        '## 高层覆盖与失败试验','',
        '旧 TPG 等待会清零命令并跳过类别 PPO；完整回合由环境完成、丢失或时限结束，高层没有直接终止整个回合。30 个诊断场景中等待约占 0.1054%，策略后壁护改写约 2.99%。平均比例很小不能排除个别场景的长时间等待或失效。严重颈内动脉场景实测 2939 个等待步，但退休/终点区前置任务导致的等待均为 0；长等待是在前置导航尚未完成时触发，不能写成已完成机器人堵塞高层。','',
        '旧类别 PPO 252 场景 Strict 与原方法相同，RL 参数更新造成的实际方向/速度改变很少。初版图像 SAC（EXP0095）252 场景 Strict 从 63.1% 到 65.1%，但平均壁接触从 5.37 增到 6.19 robot·s，整体路线偏离变大，未达到共同改善目标。等待/停车闭环及 0.96 参考限幅改变了未训练控制器；颈内动脉 N=3 的严重失败和所有尝试完整保留。EXP0096 修正参考、等待和停车执行后，四个三维 SAC 候选均未同时改善成功、清除、壁接触与路线偏离，因此继续独立训练一维策略。EXP0097 早期一维策略在逆流导航中减速过大而不能到达血栓；本次单独限制远端动作范围，并把进展奖励权重从 0.4 增为 4、gamma 从 0.995 增为 0.9995，以覆盖更长的导航时域。训练、奖励与执行设计变化均明确记录，不把它们全部归因于 CNN 或策略参数更新。','',
        '修正版在零 RL 补偿时精确保留原方法。等待和已完成任务沿用原 TPG、壁护、停车/恢复命令，RL 只改变有效导航；不以解除安全等待换取控制自由。actor 和双 Q 的动作均为一维。速度修正范围由补偿后的延迟成像位置与术前血栓距离确定：0.35 mm 外的对数赔率增益为 0.08，0.12 mm 内为 4，中间连续插值；近目标仍可从满速刹车，输出幅度限制为 0–1。此范围约束属于非 RL 执行设计，RL 学习收益以完整零策略对照归因；参考方向沿用原方法，不独立学习路线几何。原参考为零时保持原零命令，防止把既定驻留变成任意启动/退让。策略后只有执行器限幅、原壁护对新增外向分量的过滤和原间距约束，没有固定代价 argmin 重新挑选方向。','',
        '## 图像、历史与实际强化学习','',
        '输入为当前/上一帧延迟双平面 32×32 裁剪（4 通道）、24×11 成像检测和已发命令历史、48 维地图/参考预测/观测运动上下文。CNN、GRU、双 Q、目标网络、经验回放、重参数化 actor 更新和熵项进行 SAC 训练。脉动信念从带时间戳的运动残差拟合，真实血流、心相位、位置及质量不进入 actor；真值只计算奖励和评价。','',
        '奖励包括进展、实际接触清除量、血栓附近驻留、碰壁、丢失、间距及终端完成。当前物理模型是接触溶栓，没有单独的机械碎栓指标。CNN 被使用与 CNN 带来性能增益须由消融分别验证，不能混为一谈。','',
        f"正式重训完成 {complete['iterations']} 轮、{complete['gradient_updates']:,} 次梯度更新、{complete['agent_steps']:,} 次真实智能体转移。仅用九种训练解剖，N=1–3、入口 0.025/0.05/0.1 mm/s、延迟 1–3 步、扰动 0–1.25；独立训练场景块 2181000000。新调参块 2189000000 的 108 场景只包含训练解剖；候选不能降低成功和清除，不能增加壁接触和路线偏离，再按成功/壁接触排序。所选 {checkpoint}，更新 {chosen['iterations']}。当前是一个正式 RL 训练种子，未访问封存测试池。",'',
        '## 冻结检查点的完整配对验证','']
    if not chosen['jointly_admissible']:
        lines[2:2]=['本版在训练解剖选择池上没有同时满足成功/清除不下降、壁接触/路线偏离不增加的候选。以下冻结检查点只用于完整诊断及保存全部场景，不作为共同改善方法；确认矩阵的结果不会反过来用于重新选择。','']
    for folder,title in [('full_hard_report','原论文完整 252 场景（已知开发基准复测）'),('stress_report','新种子块 2193000000 的 168 个标准/强流/脉动/组合场景'),('v5_report','原弱流 V5 全部 1260 场景')]:
        path=V/folder/'summary.json'
        if not path.exists():continue
        s=json.load(open(path))
        lines+=['### '+title,'','| 方法 | n | Strict % | 清除 % | T90_300 s | 壁 robot·s | 路线偏离 mm | 近目标误差 mm |','|---|---:|---:|---:|---:|---:|---:|---:|']
        fmt=lambda x:'—' if x is None else f'{x:.4f}'
        for name,r in s['results'].items():
            lines.append(f"| {name} | {r['n']} | {100*r['strict_success']:.2f} | {100*r['removal']:.2f} | {r['T90_300']:.2f} | {r['wall_contact_s']:.3f} | {fmt(r['route_cross_track_rmse_mm'])} | {fmt(r['near_target_mean_error_mm'])} |")
        lines.append('')
        for ref in ['parent',label('sac_enveloped='+str((R/'initial.pt').relative_to(ROOT))),'old_ppo']:
            pair=s['pairs'].get(main+' vs '+ref)
            if pair:
                lo,hi=pair['CI95_pp'];lines.append(f"相对 {ref}：Strict {pair['strict_delta_pp']:+.2f} pp，95% 场景自举区间 [{lo:+.2f}, {hi:+.2f}]；清除 {pair['removal_delta_pp']:+.2f} pp；壁接触 {pair['wall_delta_robot_s']:+.3f} robot·s；场景哈希不一致 {pair['scenario_hash_mismatches']}。")
        r=s['results'][main]
        if 'rl_executed_mean_delta' in r:
            lines.append(f"相对同状态零补偿：最终执行指令平均差 {r['rl_executed_mean_delta']:.5f}，超过 0.001 的控制步占 {100*r['rl_executed_nontrivial_fraction']:.2f}%，执行链后的幅度保留比 {100*r['rl_delta_survival_ratio']:.2f}%。这不是等回合性能对照；性能归因使用完整未训练/原方法回合。")
        lines+=['','不同流速、延迟及训练/留出解剖的全部数据见 by_condition.csv、by_topology.csv 和 all_scenes.csv；失败完整计入。近目标误差只统计真实距离目标 1 mm 内，应结合清除、T90 和接触比例解读。','']
        if folder=='stress_report':
            lines+=['| 条件 | 方法 | Strict % | 清除 % | 壁 robot·s | 路线偏离 mm |','|---|---|---:|---:|---:|---:|']
            for r in s['by_condition']:
                if r['method'] in ('parent',main):lines.append(f"| {r['condition']} | {r['method']} | {100*r['strict_success']:.2f} | {100*r['removal']:.2f} | {r['wall_contact_s']:.3f} | {fmt(r['route_cross_track_rmse_mm'])} |")
            lines.append('')
    lines+=['## 冻结模型消融','',
        '分别关闭图像裁剪、观测脉动拟合、近目标大动作范围（contact_logit_gain 从 4 降到 0.08）。关闭裁剪仍保留从图像检测得到的位置/历史；关闭脉动仍保留漂移和历史；小范围版本仍执行真实 RL。各项均使用冻结参数和同一确认场景，不重新选择检查点。','',
        '| 版本 | Strict % | 清除 % | 壁 robot·s | 路线偏离 mm |','|---|---:|---:|---:|---:|']
    for folder,title in [('stress_report','完整 SAC'),('ablation_no_images_report','关闭图像裁剪'),('ablation_no_pulse_report','关闭脉动拟合'),('ablation_contact_range_report','关闭近目标大动作范围')]:
        p=V/folder/'summary.json'
        if p.exists():
            r=json.load(open(p))['results'][main];lines.append(f"| {title} | {100*r['strict_success']:.2f} | {100*r['removal']:.2f} | {r['wall_contact_s']:.3f} | {r['route_cross_track_rmse_mm']:.4f} |")
    lines+=['','## 动画与可复核文件','',f'本版媒体：{MEDIA}。实际训练、困难矩阵、强流/脉动、三项消融和弱流 V5 分目录保存。每份 GIF 对应一个真实完整回合，旁有原始 NPZ 和 JSON 来源/指标；测试动画显示记录的瞬时流量系数。进度由 index.html / progress.json 显示。',
        '',f'冻结记录 {V}/{"SELECTED.json" if chosen["jointly_admissible"] else "DIAGNOSTIC_SELECTED.json"}；完整训练日志 {R}/training.jsonl；全部原论文 23 个组合对比 {V}/paper_comparison_report/REPORT.md。旧 V3/外部方法、旧 PPO 和失败的初版 SAC 的目录保持保存，不能作为本版 RL 的成绩。']
    md=ROOT/'VRL_分阶段图像速度SAC_方法修订与完整实验记录_20261011.md';md.write_text('\n'.join(lines)+'\n')
    convert(md.read_text(),md.with_suffix('.docx'),src_dir=ROOT);return md.with_suffix('.docx')


def selection_record():
    path=V/'SELECTED.json'
    if not path.exists():path=V/'DIAGNOSTIC_SELECTED.json'
    return json.load(open(path))


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=18)
    p.add_argument('--archive-unqualified',action='store_true')
    a=p.parse_args();os.chdir(ROOT)
    import fcntl
    lock=(V/'.finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=selection_record()['checkpoint'] if (V/'SELECTED.json').exists() or (V/'DIAGNOSTIC_SELECTED.json').exists() else select(a.workers)
    chosen=selection_record()
    if hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest()!=chosen['sha256']:raise RuntimeError('Frozen checkpoint changed')
    if a.archive_unqualified and not chosen['jointly_admissible']:
        status('interim_unqualified_training_and_tuning_archival',tuning_episodes=540,checkpoint=checkpoint)
        run('scripts.export_image_flow_sac_media',['--out',str(MEDIA),'--run',str(R),'--validation',str(V),
            '--workers',str(min(a.workers,4)),'--recorded-only'],V/'media_archive.log')
        progress=json.load(open(MEDIA/'progress.json'))
        if progress['pending']:raise RuntimeError('Interim training media incomplete')
        (V/'ARCHIVED.json').write_text(json.dumps(dict(jointly_admissible=False,tuning_episodes=540,
            media=progress,later_experiment='EXP0099_FINITE_IMAGE_SAC_20261011',
            full_matrix_executed=False),ensure_ascii=False,indent=2))
        status('interim_unqualified_archived',media=progress,full_matrix_executed=False)
        run('scripts.update_rl_overview',[],V/'overview.log')
        return
    initial=str((R/'initial.pt').relative_to(ROOT));methods=f'parent,old_ppo,sac_enveloped={initial},sac_enveloped={checkpoint}'
    from scripts.report_image_flow_sac import report
    for pool,folder,expected,offset in [('hard','full_hard',1008,0),('stress','stress',672,40000000)]:
        status('paired_frozen_evaluation',pool=pool,expected=expected,checkpoint=checkpoint)
        run('scripts.eval_image_flow_sac',['--out',str(V/folder),'--pool',pool,'--scene-seed-offset',str(offset),
            '--methods',methods,'--workers',str(a.workers),'--traces','--attribution'],V/(folder+'.log'))
        check_pool(V/folder,expected);report([V/folder/'episodes.jsonl'],V/(folder+'_report'))
    for folder,cfg in [('ablation_no_images',{'use_images':False}),('ablation_no_pulse',{'pulse':False}),('ablation_contact_range',{'contact_logit_gain':.08})]:
        status('frozen_model_ablation',ablation=folder)
        config=V/(folder+'.json');config.write_text(json.dumps(cfg))
        run('scripts.eval_image_flow_sac',['--out',str(V/folder),'--pool','stress','--scene-seed-offset','40000000',
            '--methods','sac_enveloped='+checkpoint,'--controller-json',str(config),'--workers',str(a.workers),'--traces','--attribution'],V/(folder+'.log'))
        check_pool(V/folder,168);report([V/folder/'episodes.jsonl'],V/(folder+'_report'))
    combined_reports(checkpoint);doc=document(checkpoint)
    status('scalar_algorithm_validation_complete',checkpoint=checkpoint,document=str(doc),jointly_admissible=chosen['jointly_admissible'])
    export=['--out',str(MEDIA),'--run',str(R),'--validation',str(V),'--workers',str(a.workers)]
    run('scripts.export_image_flow_sac_media',export,V/'media.log')
    status('all_weak_flow_v5_scenes',expected=2520)
    run('scripts.eval_image_flow_sac',['--out',str(V/'v5'),'--pool','v5','--methods','parent,sac_enveloped='+checkpoint,
        '--workers',str(a.workers),'--traces','--attribution'],V/'v5.log')
    check_pool(V/'v5',2520);report([V/'v5/episodes.jsonl'],V/'v5_report');doc=document(checkpoint)
    run('scripts.export_image_flow_sac_media',export,V/'media_v5.log')
    progress=json.load(open(MEDIA/'progress.json'))
    if progress['pending']:raise RuntimeError('Corrected SAC media matrix incomplete')
    resume_media()
    status('corrected_SAC_complete_legacy_media_resumed',checkpoint=checkpoint,document=str(doc),media=progress)
    (V/'COMPLETE.json').write_text(json.dumps(dict(checkpoint=checkpoint,document=str(doc),media=progress,
        jointly_admissible=chosen['jointly_admissible'],purpose='joint improvement' if chosen['jointly_admissible'] else 'complete diagnostic evaluation'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
