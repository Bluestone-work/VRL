"""Select on fresh training anatomies; freeze, evaluate and export corrected SAC."""
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

V=ROOT/'research/validation/EXP0096_SAFE_IMAGE_SAC_20261011'
R=ROOT/'research/runs/EXP0096_SAFE_IMAGE_SAC_20261011/seed0'
MEDIA=Path('/home/wj/桌面/VRL_安全图像SAC_训练测试_3D_GIF_20261011')


def status(stage,**extra):
    record=dict(stage=stage,updated=time.strftime('%F %T'),**extra)
    (V/'CURRENT_STATUS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record,ensure_ascii=False),flush=True)


def select(workers):
    from scripts.report_image_flow_sac import report
    while not (R/'TRAINING_COMPLETE.json').exists() or len(rows(V/'tune/episodes.jsonl'))<216:
        if rows(V/'tune/failures.jsonl'):raise RuntimeError('Training-anatomy evaluation failures recorded')
        status('waiting_for_training_and_first_tuning',tuning_complete=len(rows(V/'tune/episodes.jsonl')),expected_first_tuning=216)
        time.sleep(30)
    shutil.copyfile(V/'tune/manifest.json',V/'tune/phase1_manifest.json')
    methods=','.join('sac_safe='+str((R/f'policy_it{k}.pt').relative_to(ROOT)) for k in [80,160,240])
    status('complete_training_anatomy_tuning',expected=540)
    run('scripts.eval_image_flow_sac',['--out',str(V/'tune'),'--pool','tune','--scene-seed-offset','19000000',
        '--methods',methods,'--workers',str(workers)],V/'tune_phase2.log')
    if len(rows(V/'tune/episodes.jsonl'))!=540 or rows(V/'tune/failures.jsonl'):raise RuntimeError('Tuning matrix incomplete')
    summary=report([V/'tune/episodes.jsonl'],V/'tune_report');parent=summary['results']['parent']
    candidates={k:r for k,r in summary['results'].items() if k.startswith('sac_safe:seed0/policy_it')}
    if any(r['n']!=108 for r in summary['results'].values()):raise RuntimeError('Unequal tuning coverage')
    admissible={k:r for k,r in candidates.items() if r['strict_success']>=parent['strict_success']-1e-12
        and r['removal']>=parent['removal']-1e-12 and r['wall_contact_s']<=parent['wall_contact_s']+1e-12
        and r['route_cross_track_rmse_mm']<=parent['route_cross_track_rmse_mm']+1e-12}
    if not admissible:
        (V/'NO_JOINT_CANDIDATE.json').write_text(json.dumps(summary['results'],ensure_ascii=False,indent=2))
        status('no_jointly_admissible_trained_checkpoint',results=summary['results'])
        raise RuntimeError('No trained candidate improves the joint training-anatomy criteria; retain data and investigate rather than claim success')
    best=max(admissible,key=lambda k:(admissible[k]['strict_success'],-admissible[k]['wall_contact_s'],admissible[k]['removal']))
    checkpoint=str((R/Path(best).name).relative_to(ROOT))
    import torch
    payload=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
    assert payload['trained_with_rl'] and payload['gradient_updates']>0
    for source,digest in payload['config']['source_sha256'].items():
        if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:raise RuntimeError('Training source changed: '+source)
    record=dict(checkpoint=checkpoint,sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
        selected_at=time.strftime('%F %T'),selection_pool='108 fresh cases on nine training anatomies, seed block 2169000000',
        criterion='Strict and clearance cannot decrease; wall and route error cannot increase. Then maximize Strict and minimize wall.',
        admissible=list(admissible),tuning_scores=summary['results'],iterations=payload['iterations'],
        gradient_updates=payload['gradient_updates'],agent_steps=payload['agent_steps'],sealed_test_accessed=False)
    (V/'SELECTED.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
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
    for folder,kind in [('ablation_no_images','sac_no_images'),('ablation_no_pulse','sac_no_pulse'),('ablation_speed_only','sac_speed_only')]:
        records=rows(V/folder/'episodes.jsonl')
        for r in records:r['method']=kind+'='+checkpoint
        target=V/(folder+'_labeled.jsonl');target.write_text(''.join(json.dumps(r)+'\n' for r in records));files.append(target)
    report(files,V/'combined_ablation_report')


def document(checkpoint):
    from scripts.md_to_docx import convert
    from scripts.report_image_flow_sac import label
    complete=json.load(open(R/'TRAINING_COMPLETE.json'));chosen=json.load(open(V/'SELECTED.json'))
    main=label('sac_safe='+checkpoint)
    lines=['# VRL：安全执行约束下的图像/历史 SAC 局部导航与完整实验记录','',
        'V3 的主方法是监督预测与固定代价选择，没有 RL。本版保留最高成功率组合的测地分配、固定 TPG、冻结预测器和安全执行层，以真实环境奖励训练的连续 SAC 适配局部速度、驻留和小幅横向控制。监督预测器与 RL actor 分开记录，不能把旧成绩当成本版成绩。','',
        '## 高层覆盖与失败试验','',
        '旧 TPG 等待会清零命令并跳过类别 PPO；完整回合由环境完成、丢失或时限结束，高层没有直接终止整个回合。30 个诊断场景中等待约占 0.1054%，策略后壁护改写约 2.99%。平均比例很小不能排除个别场景的长时间等待或失效。','',
        '旧类别 PPO 252 场景 Strict 与原方法相同，RL 参数更新造成的实际方向/速度改变很少。初版图像 SAC（EXP0095）252 场景 Strict 从 63.1% 到 65.1%，但平均壁接触从 5.37 增到 6.19 robot·s，整体路线偏离变大，未达到共同改善目标。等待/停车闭环及 0.96 参考限幅改变了未训练控制器；颈内动脉 N=3 的严重失败和所有尝试完整保留。','',
        '修正版在零 RL 补偿时精确保留原方法。等待和已完成任务沿用原 TPG、壁护、停车/恢复命令，RL 只改变有效导航；不以解除安全等待换取控制自由。速度动作沿原参考指令方向作 ±0.2 修正，两横向动作各限制为 ±0.02。零参考时使用地图路线切向，允许 RL 启动或退让。策略后只有执行器限幅、原壁护对新增外向分量的过滤和原间距约束，没有固定代价 argmin 重新挑选方向。','',
        '## 图像、历史与实际强化学习','',
        '输入为当前/上一帧延迟双平面 32×32 裁剪（4 通道）、24×11 成像检测和已发命令历史、48 维地图/参考预测/观测运动上下文。CNN、GRU、双 Q、目标网络、经验回放、重参数化 actor 更新和熵项进行 SAC 训练。脉动信念从带时间戳的运动残差拟合，真实血流、心相位、位置及质量不进入 actor；真值只计算奖励和评价。','',
        '奖励包括进展、实际接触清除量、血栓附近驻留、碰壁、丢失、间距及终端完成。当前物理模型是接触溶栓，没有单独的机械碎栓指标。CNN 被使用与 CNN 带来性能增益须由消融分别验证，不能混为一谈。','',
        f"正式重训完成 {complete['iterations']} 轮、{complete['gradient_updates']:,} 次梯度更新、{complete['agent_steps']:,} 次真实智能体转移。仅用九种训练解剖，N=1–3、入口 0.025/0.05/0.1 mm/s、延迟 1–3 步、扰动 0–1.25；独立训练场景块 2161000000。新调参块 2169000000 的 108 场景只包含训练解剖；候选不能降低成功和清除，不能增加壁接触和路线偏离，再按成功/壁接触排序。所选 {checkpoint}，更新 {chosen['iterations']}。当前是一个正式 RL 训练种子，未访问封存测试池。",'',
        '## 冻结检查点的完整配对验证','']
    for folder,title in [('full_hard_report','原论文完整 252 场景（已知开发基准复测）'),('stress_report','新种子块 2173000000 的 168 个标准/强流/脉动/组合场景'),('v5_report','原弱流 V5 全部 1260 场景')]:
        path=V/folder/'summary.json'
        if not path.exists():continue
        s=json.load(open(path))
        lines+=['### '+title,'','| 方法 | n | Strict % | 清除 % | T90_300 s | 壁 robot·s | 路线偏离 mm | 近目标误差 mm |','|---|---:|---:|---:|---:|---:|---:|---:|']
        fmt=lambda x:'—' if x is None else f'{x:.4f}'
        for name,r in s['results'].items():
            lines.append(f"| {name} | {r['n']} | {100*r['strict_success']:.2f} | {100*r['removal']:.2f} | {r['T90_300']:.2f} | {r['wall_contact_s']:.3f} | {fmt(r['route_cross_track_rmse_mm'])} | {fmt(r['near_target_mean_error_mm'])} |")
        lines.append('')
        for ref in ['parent',label('sac_safe='+str((R/'initial.pt').relative_to(ROOT))),'old_ppo']:
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
        '分别关闭图像裁剪、观测脉动拟合、横向 RL 补偿。关闭裁剪仍保留从图像检测得到的位置/历史；关闭脉动仍保留漂移和历史；仅速度版本仍是实际 RL。各项均使用冻结参数和同一确认场景，不重新选择检查点。','',
        '| 版本 | Strict % | 清除 % | 壁 robot·s | 路线偏离 mm |','|---|---:|---:|---:|---:|']
    for folder,title in [('stress_report','完整 SAC'),('ablation_no_images_report','关闭图像裁剪'),('ablation_no_pulse_report','关闭脉动拟合'),('ablation_speed_only_report','仅速度 RL')]:
        p=V/folder/'summary.json'
        if p.exists():
            r=json.load(open(p))['results'][main];lines.append(f"| {title} | {100*r['strict_success']:.2f} | {100*r['removal']:.2f} | {r['wall_contact_s']:.3f} | {r['route_cross_track_rmse_mm']:.4f} |")
    lines+=['','## 动画与可复核文件','',f'本版媒体：{MEDIA}。实际训练、困难矩阵、强流/脉动、三项消融和弱流 V5 分目录保存。每份 GIF 对应一个真实完整回合，旁有原始 NPZ 和 JSON 来源/指标；测试动画显示记录的瞬时流量系数。进度由 index.html / progress.json 显示。',
        '',f'冻结记录 {V}/SELECTED.json；完整训练日志 {R}/training.jsonl；全部原论文 23 个组合对比 {V}/paper_comparison_report/REPORT.md。旧 V3/外部方法、旧 PPO 和失败的初版 SAC 的目录保持保存，不能作为本版 RL 的成绩。']
    md=ROOT/'VRL_安全图像SAC_方法修订与完整实验记录_20261011.md';md.write_text('\n'.join(lines)+'\n')
    convert(md.read_text(),md.with_suffix('.docx'),src_dir=ROOT);return md.with_suffix('.docx')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=18);a=p.parse_args();os.chdir(ROOT)
    import fcntl
    lock=(V/'.finish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checkpoint=json.load(open(V/'SELECTED.json'))['checkpoint'] if (V/'SELECTED.json').exists() else select(a.workers)
    chosen=json.load(open(V/'SELECTED.json'))
    if hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest()!=chosen['sha256']:raise RuntimeError('Frozen checkpoint changed')
    initial=str((R/'initial.pt').relative_to(ROOT));methods=f'parent,old_ppo,sac_safe={initial},sac_safe={checkpoint}'
    from scripts.report_image_flow_sac import report
    for pool,folder,expected,offset in [('hard','full_hard',1008,0),('stress','stress',672,20000000)]:
        status('paired_frozen_evaluation',pool=pool,expected=expected,checkpoint=checkpoint)
        run('scripts.eval_image_flow_sac',['--out',str(V/folder),'--pool',pool,'--scene-seed-offset',str(offset),
            '--methods',methods,'--workers',str(a.workers),'--traces','--attribution'],V/(folder+'.log'))
        check_pool(V/folder,expected);report([V/folder/'episodes.jsonl'],V/(folder+'_report'))
    for folder,cfg in [('ablation_no_images',{'use_images':False}),('ablation_no_pulse',{'pulse':False}),('ablation_speed_only',{'action_scales':[.2,0.,0.]})]:
        status('frozen_model_ablation',ablation=folder)
        config=V/(folder+'.json');config.write_text(json.dumps(cfg))
        run('scripts.eval_image_flow_sac',['--out',str(V/folder),'--pool','stress','--scene-seed-offset','20000000',
            '--methods','sac_safe='+checkpoint,'--controller-json',str(config),'--workers',str(a.workers),'--traces','--attribution'],V/(folder+'.log'))
        check_pool(V/folder,168);report([V/folder/'episodes.jsonl'],V/(folder+'_report'))
    combined_reports(checkpoint);doc=document(checkpoint)
    status('corrected_algorithm_validation_complete',checkpoint=checkpoint,document=str(doc))
    export=['--out',str(MEDIA),'--run',str(R),'--validation',str(V),'--workers',str(a.workers)]
    run('scripts.export_image_flow_sac_media',export,V/'media.log')
    status('all_weak_flow_v5_scenes',expected=2520)
    run('scripts.eval_image_flow_sac',['--out',str(V/'v5'),'--pool','v5','--methods','parent,sac_safe='+checkpoint,
        '--workers',str(a.workers),'--traces','--attribution'],V/'v5.log')
    check_pool(V/'v5',2520);report([V/'v5/episodes.jsonl'],V/'v5_report');doc=document(checkpoint)
    run('scripts.export_image_flow_sac_media',export,V/'media_v5.log')
    progress=json.load(open(MEDIA/'progress.json'))
    if progress['pending']:raise RuntimeError('Corrected SAC media matrix incomplete')
    resume_media()
    status('corrected_SAC_complete_legacy_media_resumed',checkpoint=checkpoint,document=str(doc),media=progress)
    (V/'COMPLETE.json').write_text(json.dumps(dict(checkpoint=checkpoint,document=str(doc),media=progress),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
