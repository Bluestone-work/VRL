"""Regenerate an honest interim/final EXP0060 report from completed raw episodes."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/validation/EXP0060_VASCULAR_OPTION_RL_20261006'
RUNS=ROOT/'research/runs/EXP0060_VASCULAR_OPTION_RL_20261006'
FIG=ROOT/'research/figures/EXP0060_20261006'


def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    import fcntl
    report_lock=(OUT/'.report.lock').open('a')
    fcntl.flock(report_lock,fcntl.LOCK_EX)
    rows=[]
    for p in sorted((OUT/'eval').glob('*.jsonl')):
        for line in p.read_text().splitlines():
            if line.strip():rows.append(json.loads(line))
    groups=defaultdict(list); hashes=defaultdict(set); duplicates=[]; seen=set()
    for r in rows:
        key=(r['method'],r['clusters'],r['sensing']); groups[key].append(r)
        scene=(r['anatomy'],r['seed'],r['clusters'],r['sensing'])
        hashes[scene].add((r['scenario_hash'],r['initial_state_hash']))
        sk=key+scene
        if sk in seen:duplicates.append(sk)
        seen.add(sk)
    bad=[k for k,v in hashes.items() if len(v)!=1]
    train_hashes=set()
    for p in RUNS.glob('*/episodes.jsonl'):
        for l in p.read_text().splitlines():
            if l.strip():train_hashes.add(json.loads(l).get('scenario_hash'))
    overlap=sorted({r['scenario_hash'] for r in rows}&train_hashes)
    pairing=dict(episodes=len(rows),unique_scenes=len(hashes),mismatched_scenes=bad,duplicate_rows=duplicates,
                 train_development_scene_overlap=overlap)
    (OUT/'PAIRING_AUDIT.json').write_text(json.dumps(pairing,indent=2))
    lines=['# EXP0060：血管记忆型多智能体 DRL 研究报告',
           f'生成时间：{datetime.now().isoformat(timespec="seconds")}。仅开发集结果，非临床/密封测试结论。',
           '## Claude 后续更新核查',
           '以 0bdc01e（2026-10-06 01:04）的初稿和代码为基础。Claude 修复了分叉并集几何，加入可部署位置测量与双视角图像感知、TPG 全路线时序协调，并生成六幅论文图。其主方法仍为传统追踪；纯共享 PPO 的失败不能证明 DRL 不适用，旧 HRL 在新物理上未重训也不能作为充分算法对比。',
           '本轮新增可训练的 V-MORL 原型：每 0.5 s 由 GRU actor 选择局部反馈选项，高层不频繁重分配。MAPPO critic 只汇总同等级观测。候选名字不等于已证明原创性。',
           '这是传统高层协调与学习低层控制的组合，尚不是两层都由 RL 学习的完整 HRL。本文 MAPPO/IPPO 对照共享相同的 9 类反馈选项，不等同于无导航先验的直接连续动作 MAPPO。',
           '网络总参数：memory-MAPPO 46602，FF-MAPPO 21642，memory-IPPO 38410。无记忆消融同时减少了 GRU 参数，因此若观察到优势，还需要等参数容量 MLP 对照才能进一步归因于记忆；不能仅凭该消融宣称记忆机制已被独立证明。',
           '## 信息与指标修正',
           '所有本轮主对照共享含噪测量、术前图、动力学、动作选项及安全过滤；初始位置来自首帧定位，TPG 驻留固定 15 s。相对旧稿有协议变化，不能把本轮百分比直接减旧稿 80.2%。图像策略转移另行报告，当前训练不是端到端视觉 RL。',
           '二值血栓清除/目标注册/完整健康地图仍属理想化可测假设，未验证真实成像或配准误差。2 mm 只是未校准的欧式间距代理，不能宣称零磁场干扰。',
           'AUC 已补齐提前结束回合至统一 300 s 时限。主指标 SafeSuccess 需清除完成、壁接触总计 <1 robot-s、零粒子/集群接触、无流失、无子步间距违规。',
           f'场景哈希核查：{len(rows)} 回合，{len(hashes)} 个场景×N×感知组合；不一致 {len(bad)}，重复 {len(duplicates)}，与已完成训练回合重叠 {len(overlap)}。',
           '## 完成的配对评估',
           '| 方法 | N | 感知 | 回合 | Safe % | 清除 % | 壁接触 s | 粒子事件 | 间距违规 pair-s | AUC % |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    summaries=[]
    for (method,n,sensing),rs in sorted(groups.items()):
        avg=lambda k:float(np.mean([r[k] for r in rs]))
        safe=100*avg('cluster_safe_success'); removal=100*avg('removal'); wall=avg('wall_contact_s')
        spacing=float(np.mean([r['spacing']['spacing_violation_pair_s'] for r in rs]))
        lines.append(f'| {method} | {n} | {sensing} | {len(rs)} | {safe:.1f} | {removal:.1f} | {wall:.3f} | {avg("particle_events"):.2f} | {spacing:.4f} | {100*avg("removal_auc"):.1f} |')
        summaries.append(dict(method=method,n=n,sensing=sensing,episodes=len(rs),safe=safe))
    if not rows:lines.append('暂无正式评估完成；不能判断超过基线。')
    lines+=['## 对同场景传统基线的差异',
            '以每个训练种子单独配对报告，避免把重复评估同一场景当成独立样本。开发集配对 bootstrap 95% CI，仅作方向筛查，不作多重比较后的显著性结论。',
            '| 方法 | 配对场景 | Safe 差 pp [95% CI] | AUC 差 pp [95% CI] |',
            '|---|---:|---:|---:|']
    comparisons=[]
    baseline={(r['anatomy'],r['seed']):r for r in groups.get(('fixed_0',3,'noise'),[])}
    for (method,n,sensing),rs in sorted(groups.items()):
        if n!=3 or sensing!='noise' or method=='fixed_0':continue
        paired=[(r,baseline[(r['anatomy'],r['seed'])]) for r in rs if (r['anatomy'],r['seed']) in baseline]
        if not paired:continue
        cis=[]
        for metric in ['cluster_safe_success','removal_auc']:
            d=np.array([float(r[metric])-float(b[metric]) for r,b in paired])*100
            # Stratify by anatomy; resample whole scene units within each stratum.
            rng=np.random.default_rng(60); names=np.array([r['anatomy'] for r,b in paired]); samples=np.zeros(2000)
            for name in sorted(set(names)):
                z=d[names==name]; samples+=z[rng.integers(0,len(z),(2000,len(z)))].sum(1)/len(d)
            ci=np.percentile(samples,[2.5,97.5]); cis.append((float(d.mean()),ci.tolist()))
        lines.append(f'| {method} | {len(paired)} | {cis[0][0]:+.1f} [{cis[0][1][0]:+.1f}, {cis[0][1][1]:+.1f}] | {cis[1][0]:+.2f} [{cis[1][1][0]:+.2f}, {cis[1][1][1]:+.2f}] |')
        comparisons.append(dict(method=method,n=len(paired),safe_delta=cis[0],auc_delta=cis[1]))
    pilot_files=[OUT/'pilot_baseline.jsonl',OUT/'pilot_rl.jsonl']
    if all(p.exists() for p in pilot_files):
        pilots=[[json.loads(l) for l in p.read_text().splitlines() if l.strip()] for p in pilot_files]
        if all(len(rs)==12 for rs in pilots):
            b={r['initial_state_hash']:r for r in pilots[0]}
            assert all(r['initial_state_hash'] in b for r in pilots[1])
            lines+=['## 已完成的短程先导实验（独立于正式三种子比较）',
                    '40 次 PPO 更新、268450 agent control steps；3 解剖各 4 个固定开发场景，逐场初始状态哈希匹配。']
            for name,rs in zip(['同信息传统追踪','记忆 MAPPO 先导'],pilots):
                lines.append(f'{name}：Safe {sum(r["cluster_safe_success"] for r in rs)}/12；平均清除 {100*np.mean([r["removal"] for r in rs]):.1f}%；平均 AUC {100*np.mean([r["removal_auc"] for r in rs]):.2f}%。')
            lines.append('先导用于验证训练与比较链路，不作算法优越性证据，不能用随机探索回合成功率代替确定性配对评估。')
    complete=[]; interrupted=[]
    budgets={}
    for p in sorted(RUNS.glob('*/config.json')):
        d=p.parent; status='已完成' if (d/'DONE.json').exists() else '尚未完成/中断，查日志'
        logs=[json.loads(l) for l in (d/'training.jsonl').read_text().splitlines()] if (d/'training.jsonl').exists() else []
        if d.name.startswith(('memory_mappo_s','ff_mappo_s','memory_ippo_s')):
            budgets[d.name]=dict(updates=logs[-1]['iteration'] if logs else 0,expected_updates=480,
                                complete=(d/'DONE.json').exists() and bool(logs) and logs[-1]['iteration']==480)
        (complete if (d/'DONE.json').exists() else interrupted).append(d.name)
        if logs:lines.append(f'\n训练 {d.name}：{status}；更新 {logs[-1]["iteration"]}，agent control steps {logs[-1]["agent_control_steps"]:,}。')
    lines+=['## 当前判断与验收边界',
            '只有完整三种子、同预算、同场景的学习/传统对照与消融齐全后，才能判断学习收益。原始探索先验、固定选项选择、时序记忆、中心化训练和 TPG 的贡献需要分开解释。部分数据或单种子提高不能称稳定超过基线。',
            f'正式预算完成：{sum(b["complete"] for b in budgets.values())}/9；每臂每种子 480 updates × 6 workers × 128 决策 × 5 控制步（终止回合最后一个决策可能不足 5 步）。时间上限导致提前结束的模型不视为相同预算完成。',
            '未成功和中断的试验均保留；最初未限核的 admission 出现 SIGILL/SIGSEGV，之后按仓库已有约定排除 CPU 6/7，不能把换核通过当成硬件根因诊断。',
            '## 网络与实际 GUI',
            f'![实际实现的网络结构]({FIG}/network_architecture.png)',
            '网络图对应已运行代码：66→64 编码，8 帧 GRU，9 类动作，观测池化 critic。GUI 读取真实物理执行轨迹，显示真值仅供检查，绝不作为 actor 输入。',
            '## 与两篇 Nature Machine Intelligence 论文的关系',
            'Medany 2025（10.1038/s42256-025-01054-2）采用 DreamerV3、PPO 对照与真实显微视觉控制；Turbo（10.1038/s42256-026-01252-6）采用时序网络 RL、观测噪声/域随机化并验证实物导航。共同关键是学习机制证据、合适对照和真实闭环，而不是仅在结构图中放一个 RL 模块。',
            '体外破栓实验可以支持破栓能力；若没有运行同一学习控制器的闭环导航实验，不能由此声称 DRL 导航 sim-to-real 已验证。',
            '下一步优先级：完成已登记比较；若主方法未胜出，按失败分型再登记新试验；增加参数扰动与成像误差；最后独立密封验证。']
    adaptive=ROOT/'research/validation/EXP0061_ADAPTIVE_WORLD_RL_20261006'
    if (adaptive/'STATUS.json').exists():
        state=json.loads((adaptive/'STATUS.json').read_text())
        lines+=['## 用户追加的连续科研任务',
                f'EXP0061 已启动：{state["stage"]}。每轮训练 DRL/预测辅助 DRL、配对评估并据失败调整下一轮；09:00 是阶段快照，不因单轮结束而停止。',
                f'连续迭代验收报告：{adaptive}/验收报告_连续DRL与世界模型.docx',
                '新队列与 EXP0060 冻结的三种子比较独立，不能将开发筛查模型的改善当成已经完成的正式多种子结论。']
    gui=FIG/'gui'/'gui_10s.png'
    if not gui.exists():gui=FIG/'gui_pilot'/'gui_10s.png'
    if gui.exists():lines.insert(lines.index('## 与两篇 Nature Machine Intelligence 论文的关系'),f'![真实仿真 GUI 回放（检查图中策略标签）]({gui})')
    (OUT/'SUMMARY.json').write_text(json.dumps(dict(pairing=pairing,summaries=summaries,comparisons=comparisons,complete=complete,unfinished=interrupted,budgets=budgets),indent=2))
    formatted=[]
    for line in lines:
        formatted.extend([line] if line.startswith('|') else ['',line,''])
    report=OUT/'REPORT.md'; report.write_text('\n'.join(formatted)+'\n')
    # Curves are raw training statistics, not evaluation success claims.
    fig,ax=plt.subplots(figsize=(8,4))
    for p in sorted(RUNS.glob('*/training.jsonl')):
        if 'pilot' in str(p) or 'admission' in str(p):continue
        rs=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rs=[r for r in rs if r['safe'] is not None]
        if rs:ax.plot([r['agent_control_steps'] for r in rs],[r['safe'] for r in rs],alpha=.6,label=p.parent.name)
    ax.set(xlabel='Agent control steps',ylabel='Training safe fraction / completed episodes',ylim=(-.02,1.02))
    if ax.lines:ax.legend(fontsize=6,ncol=2)
    fig.tight_layout(); fig.savefig(FIG/'training_curves.png',dpi=180); plt.close(fig)
    if summaries:
        s=[z for z in summaries if z['n']==3 and z['sensing']=='noise']
        fig,ax=plt.subplots(figsize=(10,max(3,len(s)*.28)))
        ax.barh([z['method'] for z in s],[z['safe'] for z in s],color=['#98a7b4' if z['method'].startswith('fixed') else '#277f99' for z in s])
        ax.set(xlabel='Development SafeSuccess (%)',xlim=(0,100)); fig.tight_layout(); fig.savefig(FIG/'development_results.png',dpi=180); plt.close(fig)
    import subprocess,sys
    tempdoc=OUT/'研究进展与DRL方法_20261006.tmp.docx'
    subprocess.run([sys.executable,'-m','scripts.md_to_docx',str(report),str(tempdoc)],cwd=ROOT,check=True)
    tempdoc.replace(OUT/'研究进展与DRL方法_20261006.docx')
    print(json.dumps(dict(episodes=len(rows),paired_mismatches=len(bad),report=str(report)),ensure_ascii=False))


if __name__=='__main__':main()
