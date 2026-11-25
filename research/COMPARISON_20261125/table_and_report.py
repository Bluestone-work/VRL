import json, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).parent

def overall(path, method=None):
 d=json.load(open(path)); x=d.get('overall',d)
 if method and method in x: x=x[method]
 return x
rows=[]
for method, paths in [('pursuit',[('train',R/'pursuit_train180.json'),('holdout',R/'pursuit_holdout100.json')]),('IPPO',[('train180_s41',R/'ippo_s41_train180_chunked.json'),('holdout100_s41',R/'ippo_s41_holdout100_chunked.json'),('train12_s42',R/'ippo_s42_train12.json'),('train6_s43',R/'ippo_s43_train6.json')]),('MAPPO',[('train12',R/'mappo_s41_train12.json'),('train12',R/'mappo_s42_train12.json'),('train12',R/'mappo_s43_train12.json')]),('CSN-adapted',[('train12',R/'domain_csn_train12.json')])]:
 for split,p in paths:
  x=overall(p,'pursuit' if method=='pursuit' else 'csn' if method=='CSN-adapted' else None)
  rows.append(dict(method=method,split=split,episodes=x.get('episodes'),success=x.get('navigation_success_rate'),strict=x.get('strict_success_rate'),lost=x.get('lost_rate'),wall=x.get('wall_contact_s'),collision=x.get('collision_episode_rate'),complete=x.get('complete')))
with open(R/'local_control_table.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
fixed=json.load(open(R/'fixed_three_pursuit.json'))
with open(R/'fixed_three_lesion_table.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['clusters','episodes','all_three_rate','synchronized_rate','last_arrival_s','arrival_span_s','wall_contact_s','lost_rate']); w.writeheader()
 for n,x in sorted(fixed['by_clusters'].items(),key=lambda z:int(z[0])): w.writerow(dict(clusters=n,**x))
lines=['# 多集群血管导航对比实验报告','', '## 实验状态','', '本报告来自独立 worktree `exp/vascular-comparison-20261125`，基线提交 `ec9a097b46e4d865b2b20b079f1bb44dbb1fbf91`。主协议使用 `navigation_dataset_v1`，N=3、13 个语义动作、合法动作掩码、在线流速特征、共同安全层和零延迟。IPPO 与 MAPPO 为本轮新训练，种子 41、42、43，各 50,000 个物理环境步。历史 EXP0141 checkpoint 不在当前 checkout，未将本轮训练冒充历史结果。','', '## 局部控制器实测表','', '| 方法 | 数据 | 回合 | 导航成功 | 严格成功 | 丢失率 | 壁接触 robot·s | 碰撞回合率 |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 lines.append(f"| {r['method']} | {r['split']} | {r['episodes']} | {100*r['success']:.1f}% | {100*r['strict']:.1f}% | {100*r['lost']:.1f}% | {r['wall']:.3f} | {100*r['collision']:.1f}% |")
lines += ['', '## 固定三个病灶 pursuit 实测', '', '| 集群数 | 回合 | 三病灶全覆盖 | 全覆盖且到达时间差≤30 s | 最后到达 s | 到达时间差 s | 壁接触 robot·s | 丢失率 |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for n,x in sorted(fixed['by_clusters'].items(),key=lambda z:int(z[0])):
 lines.append(f"| {n} | {x['episodes']} | {100*x['all_three_rate']:.1f}% | {100*x['synchronized_rate']:.1f}% | {x['last_arrival_s']:.2f} | {x['arrival_span_s']:.2f} | {x['wall_contact_s']:.2f} | {100*x['lost_rate']:.1f}% |")
lines += ['', '## 当前算法与可包装的方法故事','', '高层固定病灶分配、血管路线和 TPG 通行顺序。局部控制器在每一步从 13 个语义候选动作中选择，候选动作经过合法动作掩码、近目标减速、WallGuard、间距安全层和执行限幅后才成为实际指令。pursuit 是规则基线。IPPO 使用参数共享的局部 actor、CNN 图像编码、GRU 历史编码、障碍与邻居池化、流速残差分支和 PPO clipped objective。MAPPO 与 IPPO 共享 actor 和部署观测，训练时额外使用集中式 joint-state critic。CSN-adapted 将 concentric-sector 搜索思想映射到三维 Frenet 框架。','', '## 文献方法适配','', 'CSN 仅按 Su et al. DOI 10.1002/aisy.70466 的 concentric-sector 动态避障思想实现三维 Frenet 扩展，未使用作者代码，属于 CSN 适配版。12 个 N=3 训练行上成功率 50.0%，严格成功率 8.3%，丢失率 16.7%，壁接触 88.13 robot·s。Yang、Bevan、Li 的层次 DRL 文献 DOI 10.1002/aisy.202200168 已核对为高层短程子目标加低层 3D DRL。仓库没有同规格可接入的连续 3D 策略，短程航点移植冒烟 2 回合为 0/2 成功且壁接触 798.30 robot·s，因此没有把它列入主比较表。','', '## 失败和缺口','', 'IPPO 种子 41 的 180 行训练集和 100 行留出集已通过分块隔离评估，0 技术失败。MAPPO 仍只有每种子 12 行配对评估，长进程全量运行曾触发 exit code 139，未把失败运行计入统计。QMIX、连续 PPO/SAC、CCBS、完整 Yang N=1/N=3 迁移尚未形成可复核结果。','', '## 结论边界','', '在 N=3 全量分块评估中，IPPO 种子 41 训练集导航成功率 97.2%、留出集 99.0%，pursuit 对应为 84.4% 和 80.0%。IPPO 严格成功率和壁接触仍需与 pursuit 一起解释，不能只用导航成功率宣称整体优越。MAPPO 目前只有 12 行配对结果，不能外推全量。固定三个病灶的 6 回合 pursuit 诊断中，N=3 的平均最后到达时间 61.18 s，低于 N=1 的 103.28 s 和 N=2 的 98.97 s，但三病灶全覆盖率为 66.7%，同步窗口成功率仅 16.7%，所以“并行缩短最后到达时间”得到初步支持，“并行提高同步递药效率”尚未得到支持。']
(R/'yang_n1_train2.json').exists() and lines.insert(lines.index('## 失败和缺口') if '## 失败和缺口' in lines else len(lines), 'Yang 适配诊断（每组 2 回合，非主结果）：N=1 成功率 0/2，壁接触 293.998 robot·s；N=3 成功率 0/2，壁接触 798.303 robot·s，其中到达后壁接触 142.200 robot·s。该实现只复现短程航点层与 VRL 局部执行器映射，连续转向控制器和原文训练细节没有声称复现。')
(R/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
# plots
fig,ax=plt.subplots(figsize=(8,4)); labels=[]; vals=[]; colors=[]
for r in rows:
 labels.append(f"{r['method']}\n{r['split']}"); vals.append(100*r['success']); colors.append({'pursuit':'#2a78d6','IPPO':'#eb6834','MAPPO':'#1baf7a','CSN-adapted':'#9467bd'}[r['method']])
ax.bar(labels,vals,color=colors); ax.set_ylim(0,100); ax.set_ylabel('Navigation success (%)'); ax.set_title('N=3 local-controller comparison'); ax.tick_params(axis='x',rotation=30); fig.tight_layout(); fig.savefig(R/'local_control_comparison.png',dpi=180); plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4)); ns=sorted(fixed['by_clusters'],key=int); ax.plot(ns,[100*fixed['by_clusters'][n]['all_three_rate'] for n in ns],'o-',label='all three reached'); ax.plot(ns,[100*fixed['by_clusters'][n]['synchronized_rate'] for n in ns],'o-',label='synchronized τ≤30 s'); ax.set_xlabel('Clusters'); ax.set_ylabel('Episodes (%)'); ax.set_ylim(0,100); ax.legend(); ax.set_title('Fixed three-lesion pursuit'); fig.tight_layout(); fig.savefig(R/'fixed_three_lesion_scaling.png',dpi=180); plt.close(fig)
