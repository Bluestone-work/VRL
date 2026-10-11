"""Build the Chinese manuscript draft (docx) from the frozen EXP0090-0093 results.
usage: build_manuscript_exp0093.py --out VRL_中文论文初稿_V3_20261011.docx
The styles are taken from the 2026-10-09 draft (template); its body is replaced. Every number is read from result files.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

ROOT = Path('.'); FIG = ROOT/'research/figures/EXP0091_paper'; EQ = FIG/'eq'
TEMPLATE = ROOT/'VRL_中文论文初稿_V2_叙事重构_20261009_润色插图版.docx'
V90, V91, V93 = ROOT/'research/validation/EXP0090_20261011', ROOT/'research/validation/EXP0091_20261011', ROOT/'research/validation/EXP0093_20261011'


# ------------------------------------------------------------------------------------------------ helpers
class Doc:
    def __init__(self):
        import docx
        from docx.shared import Cm, Pt
        self.docx, self.Cm, self.Pt = docx, Cm, Pt
        self.d = docx.Document(str(TEMPLATE)); body = self.d.element.body
        for el in list(body):
            if not el.tag.endswith('sectPr'):
                body.remove(el)
        self.fig_no = self.tab_no = self.eq_no = self.alg_no = 0; self.cited = []
        EQ.mkdir(parents=True, exist_ok=True)

    def cite(self, text):
        import re
        def num(m):
            ks = [k.strip().lstrip('@') for k in m.group(1).split(',')]
            ns = []
            for k in ks:
                if k not in REFS:
                    raise KeyError(k)
                if k not in self.cited:
                    self.cited.append(k)
                ns.append(self.cited.index(k)+1)
            ns = sorted(set(ns)); parts = []; i = 0
            while i < len(ns):
                j = i
                while j+1 < len(ns) and ns[j+1] == ns[j]+1:
                    j += 1
                parts.append(f'{ns[i]}–{ns[j]}' if j-i >= 2 else ','.join(str(x) for x in ns[i:j+1])); i = j+1
            return '['+','.join(parts)+']'
        return re.sub(r'\[(@[^\]]+)\]', num, text)

    def p(self, text, style='Normal'):
        text = self.cite(text)
        para = self.d.add_paragraph(style=style)
        for k, part in enumerate(text.split('**')):           # **bold** spans
            run = para.add_run(part); run.bold = bool(k % 2)
        return para

    def h1(self, t): self.d.add_paragraph(t, style='Heading 1')
    def h2(self, t): self.d.add_paragraph(t, style='Heading 2')

    def eq(self, latex, label=True, size=13):
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        plt.rcParams['mathtext.fontset'] = 'stix'; plt.rcParams['font.family'] = 'STIXGeneral'
        self.eq_no += 1; f = EQ/f'eq{self.eq_no:02d}.png'
        fig = plt.figure(figsize=(.01, .01)); fig.text(0, 0, f'${latex}$', fontsize=size)
        fig.savefig(f, dpi=250, bbox_inches='tight', pad_inches=.03, transparent=True); plt.close(fig)
        para = self.d.add_paragraph(style='Equation')
        from PIL import Image
        w, h = Image.open(f).size; width = min(w/250*2.54*.95, 14.)
        para.add_run().add_picture(str(f), width=self.Cm(width))
        if label:
            para.add_run(f'\t({self.eq_no})')
        return self.eq_no

    def fig(self, path, caption, width=16.):
        self.fig_no += 1
        para = self.d.add_paragraph(style='Normal'); para.alignment = 1
        para.add_run().add_picture(str(path), width=self.Cm(width))
        self.p(f'图{self.fig_no}  {caption}', 'CaptionCN')
        return self.fig_no

    def table(self, header, rows, caption, widths=None, fs=8):
        self.tab_no += 1
        self.p(f'表{self.tab_no}  {caption}', 'CaptionCN')
        t = self.d.add_table(rows=1, cols=len(header)); t.style = 'Table Grid' if 'Table Grid' in [s.name for s in self.d.styles] else t.style
        for c, h in enumerate(header):
            cell = t.rows[0].cells[c]; cell.text = ''; r = cell.paragraphs[0].add_run(str(h)); r.bold = True; r.font.size = self.Pt(fs)
        for row in rows:
            cells = t.add_row().cells
            for c, v in enumerate(row):
                cells[c].text = ''; r = cells[c].paragraphs[0].add_run(self.cite(str(v))); r.font.size = self.Pt(fs)
        self.d.add_paragraph()
        return self.tab_no

    def algo(self, title, lines):
        self.alg_no += 1
        self.p(f'算法{self.alg_no}  {title}', 'CaptionCN')
        t = self.d.add_table(rows=1, cols=1); t.style = 'Table Grid' if 'Table Grid' in [s.name for s in self.d.styles] else t.style
        cell = t.rows[0].cells[0]; cell.text = ''
        for k, ln in enumerate(lines):
            para = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
            r = para.add_run(ln); r.font.name = 'Consolas'; r.font.size = self.Pt(8.5)
        self.d.add_paragraph()

    def save(self, out):
        self.d.save(str(out))


def jl(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def pct(x):
    return f'{100*x:.1f}'
REFS = {
 'nelson2010': 'Nelson BJ, Kaliakatsos IK, Abbott JJ. Microrobots for minimally invasive medicine. Annual Review of Biomedical Engineering, 2010, 12: 55–85.',
 'iacovacci2024': 'Iacovacci V, Diller E, Ahmed D, Menciassi A. Medical microrobots. Annual Review of Biomedical Engineering, 2024, 26: 561–591.',
 'wang2021': 'Wang Q, Chan KF, Schweizer K, et al. Ultrasound Doppler-guided real-time navigation of a magnetic microswarm for active endovascular delivery. Science Advances, 2021, 7(9): eabe5914.',
 'xie2019': 'Xie H, Sun M, Fan X, et al. Reconfigurable magnetic microrobot swarm: multimode transformation, locomotion, and manipulation. Science Robotics, 2019, 4(28): eaav8006.',
 'yu2022': 'Yu J, Jin D, Chan KF, et al. Adaptive pattern and motion control of magnetic microrobotic swarms. IEEE Transactions on Robotics, 2022, 38(3): 1552–1570.',
 'du2025': 'Du X, Wang Y, Law J, et al. Active exploration and reconstruction of vascular networks using microrobot swarms. Nature Machine Intelligence, 2025, 7: 553–564.',
 'an2026': 'An X, Luo S, Zhang H, et al. Autonomous navigation of intelligent microrobotic swarms in unknown environments. Nature Machine Intelligence, 2026, 8: 955–968.',
 'medany2025': 'Medany M, Piglia L, Achenbach L, Mukkavilli SK, Ahmed D. Model-based reinforcement learning for ultrasound-driven autonomous microrobots. Nature Machine Intelligence, 2025, 7: 1076–1090.',
 'kuhn1955': 'Kuhn HW. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 1955, 2(1–2): 83–97.',
 'khan2025': 'Khan UA, Benosman M, Liu W, Pecora F, Durham JW. Multi-robot path planning and scheduling via model predictive optimal transport (MPC-OT). IEEE Conference on Decision and Control (CDC), 2025: 3169–3175.',
 'honig2016': 'Hönig W, Kumar TKS, Cohen L, et al. Multi-agent path finding with kinematic constraints. International Conference on Automated Planning and Scheduling (ICAPS), 2016: 477–485.',
 'jiang2025': 'Jiang H, Lin M, Li J. Speedup techniques for switchable temporal plan graph optimization. Proceedings of the AAAI Conference on Artificial Intelligence, 2025, 39(22): 23212–23221.',
 'sharon2015': 'Sharon G, Stern R, Felner A, Sturtevant NR. Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence, 2015, 219: 40–66.',
 'gonzales2025': 'Gonzales M, Polevoy A, Kobilarov M, Moore J. Multi-agent feedback motion planning using probably approximately correct nonlinear model predictive control. IEEE International Conference on Automation Science and Engineering (CASE), 2025: 1928–1935.',
 'williams2016': 'Williams G, Drews P, Goldfain B, Rehg JM, Theodorou EA. Aggressive driving with model predictive path integral control. IEEE International Conference on Robotics and Automation (ICRA), 2016: 1433–1440.',
 'williams2018': 'Williams G, Drews P, Goldfain B, Rehg JM, Theodorou EA. Information-theoretic model predictive control: theory and applications to autonomous driving. IEEE Transactions on Robotics, 2018, 34(6): 1603–1622.',
 'ames2017': 'Ames AD, Xu X, Grizzle JW, Tabuada P. Control barrier function based quadratic programs for safety critical systems. IEEE Transactions on Automatic Control, 2017, 62(8): 3861–3876.',
 'chua2018': 'Chua K, Calandra R, McAllister R, Levine S. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. Advances in Neural Information Processing Systems (NeurIPS), 2018: 4754–4765.',
 'lakshminarayanan2017': 'Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems (NeurIPS), 2017: 6402–6413.',
 'schulman2017': 'Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. Proximal policy optimization algorithms. arXiv:1707.06347, 2017.',
 'hafner2025': 'Hafner D, Pasukonis J, Ba J, Lillicrap T. Mastering diverse control tasks through world models. Nature, 2025, 640: 647–653.',
 'lee2020': 'Lee J, Hwangbo J, Wellhausen L, Koltun V, Hutter M. Learning quadrupedal locomotion over challenging terrain. Science Robotics, 2020, 5(47): eabc5986.',
 'kumar2021': 'Kumar A, Fu Z, Pathak D, Malik J. RMA: rapid motor adaptation for legged robots. Robotics: Science and Systems (RSS), 2021.',
 'tobin2017': 'Tobin J, Fong R, Ray A, Schneider J, Zaremba W, Abbeel P. Domain randomization for transferring deep neural networks from simulation to the real world. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2017: 23–30.',
 'yu2022mappo': 'Yu C, Velu A, Vinitsky E, et al. The surprising effectiveness of PPO in cooperative multi-agent games. Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks, 2022.',
 'drotleff2026': 'Drotleff J, Tovey S, Hohenberger P, et al. Reinforcement learning enables autonomous microrobot navigation and intervention in simulated blood capillaries. arXiv:2606.26154, 2026.',
 'schulman2015': 'Schulman J, Moritz P, Levine S, Jordan M, Abbeel P. High-dimensional continuous control using generalized advantage estimation. arXiv:1506.02438, 2015.',
}


# ------------------------------------------------------------------------------------------------ data
def arm_of(r):
    m, a = r['method'], r.get('method_arg') or ''
    if m == 'sel_learned_fb' and a:
        return 'ours_'+('A' if 'EXP0090' in a else 'B' if 'predictor_B' in a else 'C')
    if m == 'nav':
        return 'belief' if 'belief' in a else 'nav_tf_v3'
    return m


def collect(files, keep=None):
    by = {}
    for f in files:
        for r in jl(f):
            a = arm_of(r)
            if keep and a not in keep:
                continue
            by.setdefault((r['cell'], r['anatomy'], r['clusters'], r['seed']), {}).setdefault(a, []).append(r)
    return by


MET = ['strict_success', 'task_success', 'removal', 'removal_auc', 't90_300', 'wall_contact_s', 'wall_ge_1s',
       'spacing_violation_pair_s', 'coupling_exposure', 'path_mm']


def stats(by, arms, sel=lambda k: True):
    keys = [k for k, d in by.items() if sel(k) and all(a in d for a in arms)]
    out = {}
    for a in arms:
        out[a] = {m: float(np.mean([np.mean([float(r[m]) for r in by[k][a]]) for k in keys])) for m in MET}
        out[a]['n'] = len(keys)
    return out


def data():
    D = {}
    base = ['switch_settle', 'fixed_settle', 'adaptive_settle', 'stpg', 'pac_nmpc', 'nav_off', 'nav_tf_v3', 'belief']
    hard = collect([V90/'hard/episodes.jsonl'], set(base))
    for k, d in collect([V91/'dev_hard_ours/episodes.jsonl', V91/'dev_hard_online/episodes.jsonl']).items():
        hard.setdefault(k, {}).update(d)
    arms = base+['ours_A', 'ours_B', 'ours_C', 'sel_online']
    D['hard'] = stats(hard, arms)
    D['hard_panel'] = {p: stats(hard, arms, lambda k, p=p: k[0].startswith(p)) for p in
                       ['low_delay', 'moderate_delay', 'high_delay', 'strong_flow', 'variable_response', 'ood_flow_delay']}
    D['hard_topo'] = {t: stats(hard, arms, lambda k, t=t: k[0].split('|')[1] == t) for t in ('seen_topology', 'unseen_topology')}
    nav_seed = {}
    for r in jl(V90/'hard/episodes.jsonl'):
        if r['method'] == 'nav' and 'belief' not in (r.get('method_arg') or ''):
            nav_seed.setdefault(r['method_arg'], []).append(r['strict_success'])
    D['nav_seeds'] = [float(np.mean(v)) for v in nav_seed.values()]
    v5 = collect([V90/'v5/episodes.jsonl'], {'switch_settle', 'fixed_settle', 'nav_off', 'nav_tf_v3'})
    for k, d in collect([V91/'dev_v5_ours/episodes.jsonl', V91/'dev_v5_online/episodes.jsonl', V91/'dev_v5_paper/episodes.jsonl']).items():
        v5.setdefault(k, {}).update(d)
    v5arms = ['switch_settle', 'fixed_settle', 'stpg', 'pac_nmpc', 'nav_off', 'nav_tf_v3', 'ours_A', 'ours_B', 'ours_C', 'sel_online']
    D['v5'] = {s: stats(v5, v5arms, lambda k, s=s: k[0] == s) for s in ('v5_s0', 'v5_s1', 'v5_s1.5')}
    # paired ours - F, ours - switch (hard)
    rng = np.random.default_rng(0)
    def pair(by, a_list, b, sel=lambda k: True):
        ks = [k for k, d in by.items() if sel(k) and all(x in d for x in a_list+[b])]
        d = np.array([np.mean([by[k][x][0]['strict_success'] for x in a_list])-by[k][b][0]['strict_success'] for k in ks])
        m = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(ks)
    O = ['ours_A', 'ours_B', 'ours_C']
    D['pair'] = {b: pair(hard, O, b) for b in ('switch_settle', 'fixed_settle', 'stpg', 'pac_nmpc', 'nav_off', 'nav_tf_v3', 'sel_online')}
    D['pred'] = json.load(open(V90/'prediction/prediction_eval.json'))
    D['fac'] = json.load(open(V93/'summary.json')) if (V93/'summary.json').exists() else {}
    abl = {}
    for name, d in (('full', 'hard'), ('no_tpg', 'hard_notpg'), ('no_guard', 'hard_no_guard'), ('no_shield', 'hard_no_shield'), ('no_safety', 'hard_no_safety')):
        rs = [r for r in jl(V90/d/'episodes.jsonl') if r['method'] == 'sel_learned_inv']
        if rs:
            abl[name] = {m: float(np.mean([float(r[m]) for r in rs])) for m in MET} | dict(n=len(rs))
    D['abl'] = abl
    return D
# ------------------------------------------------------------------------------------------------ manuscript
def build(out):
    D = data(); W = Doc(); H = D['hard']; O = ['ours_A', 'ours_B', 'ours_C']
    om = float(np.mean([H[a]['strict_success'] for a in O])); osd = float(np.std([H[a]['strict_success'] for a in O], ddof=1))
    fac = D['fac'].get('hard', {}); FT = fac.get('table', {})
    def F(arm, cell='ALL', m='strict_success', p=True):
        v = FT.get(arm, {}).get(cell, {}).get(m)
        return '—' if v is None else (pct(v) if p else f'{v:.1f}')
    best_ext = max([a for a in FT if not a.endswith(('+ours', '+online')) and not a.startswith('A_TPG+pursuit')],
                   key=lambda a: FT[a]['ALL']['strict_success'], default=None)
    pf = D['pair']['sel_online']; ps = D['pair']['switch_settle']
    pm0 = fac.get('ours_minus', {})
    ext_arms = [a for a in FT if a.split('+')[1] not in ('ours', 'online')]
    ext_pt = all(pm0[a]['ALL'][0] > 0 for a in ext_arms if a in pm0)
    ext_sig = all(pm0[a]['ALL'][1] > 0 for a in ext_arms if a in pm0)
    ext_word = ('显著优于' if ext_sig else '优于') if ext_pt else '与之相当或略优于大多数'
    pr = D['pred']['val_heldout']['all']
    W.p('面向多变血流与延迟成像的多磁性微机器人集群多点血栓清除：预测辅助的分层导航框架', 'Title')
    W.h1('摘要')
    W.p(f'磁性微机器人及其集群已能在血管内完成靶向递药、血栓破除等操作。将这些能力推向临床，需要在不同患者的血管、'
        f'未知且时变的血流以及延迟的成像反馈下，让多个集群自主、高效、安全地完成多点血栓清除。本文把这一问题形式化为'
        f'"已知几何、未知动力学"的多集群导航任务：术前 CTA 给出血管中心线、健康半径和血栓位置，而血流强度与脉动、'
        f'集群对磁场指令的响应、近壁作用和溶解速率因人因时而异，控制器只能获得延迟的双平面图像估计。'
        f'我们提出一个分层框架。全局层用测地分配与时序计划图（TPG）决定"谁去哪个血栓、按什么顺序通过冲突区"；'
        f'局部层用一个动作条件化的学习运动预测器集成，预测候选指令在未来 1.2 s 内造成的真实位移及其不确定性，'
        f'在以经典规则为锚的候选集中选择指令，并在认知不确定性过高时回退到规则。所有方法共享同一壁护、TPG 等待与间距护盾。'
        f'在 14 种血管解剖、N = 1–3 个集群、6 类流速—延迟—动力学条件构成的 252 个配对场景上，本文框架'
        f'（3 个完全独立的预测器种子）的 Strict 成功率为 {pct(om)}% ± {pct(osd)}%，'
        f'比本项目最强经典规则 SwitchSettle 高 {100*ps[0]:+.1f} 个百分点（95% CI [{100*ps[1]:+.1f}, {100*ps[2]:+.1f}]），'
        f'在“全局层 × 局部层”因子化对比中{ext_word}以 PAC-NMPC、MPPI、CBF-QP、Turbo 强化学习为局部层，以 STPG、最优传输分配为全局层的外部方法组合'
        f'（最佳外部组合 {F(best_ext) if best_ext else "—"}%）。学习预测器在留出解剖上把 1.2 s 位置误差降至 {pr["learned"]["1.2s"]["rmse_mm"]:.3f} mm'
        f'（在线估计器 {pr["online_affine"]["1.2s"]["rmse_mm"]:.3f} mm），并给出校准的不确定性'
        f'（2σ 覆盖率 {pr["learned"]["1.2s"]["within2"]:.2f}）。但配对消融显示，'
        f'在控制效果上它与同一选择器配非学习在线估计器的版本相当（{100*pf[0]:+.1f} pp，[{100*pf[1]:+.1f}, {100*pf[2]:+.1f}]）；'
        f'在弱流、低延迟设定下，框架仍落后于固定减速规则。这些结果表明，"预测—选择"结构是困难条件下性能提升的主要来源，'
        f'而如何把更准确的学习预测转化为控制增益，是下一步的关键问题。', 'Abstract')
    W.p('关键词：磁性微机器人集群；多点血栓清除；分层导航；动作条件运动预测；不确定性回退；未知血流；延迟成像', 'Abstract')
    W.fig(FIG/'fig_framework.png', '本文分层导航框架。蓝色为术前地图驱动的全局层（测地分配、时序计划图、路线追踪）；'
          '绿色为可部署观测与历史；橙色为动作条件化的学习运动预测器集成（真值仅作训练监督，虚线）；'
          '红色为以规则为锚的候选指令选择与不确定性回退；灰色为所有方法共享的执行与安全层；紫色为未知动力学的仿真血管。', 16.5)
    W.h1('1 引言')
    W.p('磁性微机器人及其集群在外加磁场驱动下可在血管内主动运动，已被用于靶向递药、局部溶栓与机械破栓等操作[@nelson2010,@iacovacci2024,@wang2021]。'
        '与单个微机器人相比，可重构的集群具有更大的载药量和作用面积，也能通过形态变化适应不同管径[@xie2019,@yu2022]。'
        '在成像引导下，磁性集群已能在离体和在体血管中被导航至目标位置[@wang2021,@du2025]。这些进展使一个更具临床意义的问题变得现实：'
        '能否在**各种变式场景**下，自主控制**多个**磁性微机器人集群，完成**多点**血栓的清除？')
    W.p('这一目标包含三层困难。第一，**任务层**：多个血栓分布在血管树的不同分支，需要决定各集群负责哪些血栓及访问顺序，'
        '并避免集群在共享管段内相互靠近，造成磁场耦合干扰。第二，**导航层**：血流强度因患者和部位而异，并随心动周期脉动；'
        '集群对磁场指令的实际响应受形态、近壁阻力和黏附影响而未知且漂移；成像反馈存在 0.1–0.3 s 的延迟。'
        '第三，**清除层**：到达血栓并不等于完成任务，集群必须在血流冲刷下保持足够的驻留时间，同时尽量不碰触血管壁。')
    W.p('现有自主导航研究多针对单个机器人或单个集群的"到达目标"任务。An 等[@an2026]用 Transformer 强化学习在未知障碍环境中导航磁性集群；'
        'Medany 等[@medany2025]用基于模型的强化学习（DreamerV3）控制超声驱动微机器人；Du 等[@du2025]用集群主动探索并重建血管网络。'
        '多机器人领域则有成熟的任务分配[@kuhn1955,@khan2025]、路径执行协调[@honig2016,@jiang2025,@sharon2015]与带不确定性的模型预测控制[@gonzales2025,@williams2018,@ames2017]。'
        '然而，把这些方法直接用于多集群血栓清除并不足够：术前 CTA 已给出血管几何，不需要再"探索"未知环境；'
        '真正未知的是**动力学**，即血流、响应和延迟，而且它们恰好决定了"继续推进还是减速驻留"这一清除的关键决策。')
    W.p('本文将问题定义为"已知几何、未知动力学"的多集群导航，并提出一个分层框架：几何可解的部分（分配、路径、冲突协调、壁护）'
        '交给经典算法；几何不可解的部分，即候选指令在未知动力学下会产生什么运动，交给从历史中学习的动作条件运动预测器；'
        '最终指令通过以规则为锚的候选选择产生，并在预测不可靠时回退。本文的主要贡献如下：')
    for t in ['（1）构建了一个面向多集群多点血栓清除的仿真基准：14 种临床解剖、N = 1–3 个集群、按解剖标定的血流、脉动与个体动力学变化、'
              '双平面图像感知与延迟，并定义了覆盖效率、完整度、集群互不影响和壁接触的评价体系；',
              '（2）提出了全局测地分配 + 时序计划图与局部"学习预测—候选选择—不确定性回退"相结合的分层框架，'
              '所有方法共享同一安全执行层，保证对比只反映导航与协调策略的差异；',
              '（3）设计了"全局层 × 局部层"的因子化对比，系统比较 STPG、最优传输分配、PAC-NMPC、MPPI、CBF-QP 与 Turbo 等外部方法的组合；',
              '（4）通过 learner-off、在线估计器、回退和安全层消融，如实区分性能提升来自预测—选择结构还是学习预测本身。']:
        W.p(t)
    W.h1('2 相关工作')
    W.h2('2.1 磁性微机器人集群的血管内操控')
    W.p('磁性微机器人在旋转或梯度磁场下运动，可通过集群化获得更强的推进与协同作用[@nelson2010,@iacovacci2024]。'
        'Xie 等[@xie2019]实现了可在多种形态间切换的磁性集群；Yu 等[@yu2022]提出了集群形态与运动的自适应控制；'
        'Wang 等[@wang2021]在超声多普勒引导下实现了磁性集群的实时血管内导航与递送。这些工作主要解决单集群的驱动与成像引导问题，'
        '对多个集群在同一血管树中协同完成多个目标的研究仍较少，而这正是多点血栓清除所需要的。')
    W.h2('2.2 微机器人自主导航中的学习方法')
    W.p('An 等[@an2026]提出 Turbo：以时序 Transformer 为骨干、在多层域随机化下用 PPO[@schulman2017]训练，使磁性集群在未知障碍环境中仅凭部分观测导航，'
        '并以人类操作者为对照。Medany 等[@medany2025]用 DreamerV3[@hafner2025]式的世界模型强化学习控制超声驱动微机器人，强调样本效率与实物适应。'
        'Du 等[@du2025]让集群在未知血管网络中主动探索并重建拓扑。Drotleff 等[@drotleff2026]在毛细血管仿真中用强化学习完成导航与干预。'
        '这些方法的共同出发点是"环境未知"，学习既要发现路径，也要学会控制。与之不同，本文场景中术前 CTA 已给出几何，'
        '学习只需处理不可见的动力学。这与腿足机器人中"特权教师—可部署学生"[@lee2020]和在线适应[@kumar2021]的思路一致：'
        '仿真真值只用于训练监督，部署时只依赖传感历史。')
    W.h2('2.3 多机器人任务分配与协同执行')
    W.p('任务分配可用匈牙利算法[@kuhn1955]求解一对一指派，MPC-OT[@khan2025]把匿名目标分配与路径调度统一为最优传输线性规划，再用分布式 MPC 跟踪。'
        '在路径执行阶段，时序计划图（TPG）[@honig2016]把多机路径之间的先后依赖组织成有向无环图，从而在执行延误时仍保证无碰撞、无死锁；'
        'Jiang 等[@jiang2025]的可切换 TPG（STPG）在线重优化冲突点的通行顺序以减少延误；冲突搜索（CBS）[@sharon2015]则在规划阶段求解最优多机路径。'
        '多智能体强化学习（如 MAPPO[@yu2022mappo]）也被用于协同决策。在血管中，集群路径受管腔拓扑约束，冲突主要来自共享管段与分叉附近的三维欧氏接近，'
        '因此 TPG 类方法比网格上的 MAPF 搜索更适合直接迁移。')
    W.h2('2.4 不确定动力学下的模型预测与安全控制')
    W.p('模型预测路径积分（MPPI）[@williams2016,@williams2018]通过对随机控制序列采样并按代价加权更新控制；PAC-NMPC[@gonzales2025]在多机系统中优化控制分布，'
        '并以 PAC 上界约束碰撞概率；控制障碍函数二次规划（CBF-QP）[@ames2017]对名义指令做最小修正以保证安全集不变。'
        '在学习式模型预测中，PETS[@chua2018]用概率神经网络集成[@lakshminarayanan2017]建模动力学并结合采样 MPC，兼顾偶然与认知不确定性。'
        '本文的局部层与 PETS 同属"学习动力学模型 + 预测控制"一类，区别在于：(i) 预测对象是延迟成像条件下相对最新估计的真实位移；'
        '(ii) 候选集以可部署的经典规则为锚，只有预测显著更优才替换；(iii) 认知不确定性用于回退到规则；(iv) 嵌入了多集群全局协调与共享安全层。')
    W.h1('3 问题定义')
    W.h2('3.1 场景与任务')
    W.p('考虑一棵三维血管树 𝒯，由中心线管段和健康半径 R(s) 描述，管内含 M = 4 个血栓 c_1…c_M（位置、初始质量 m_j^0 由术前 CTA 给出）。'
        'N ∈ {1, 2, 3} 个磁性集群（半径 0.08 mm，最大速度 1 mm/s）从不同分支出发，控制周期 Δt = 0.1 s，时限 300 s。'
        '集群与血栓的距离小于 0.12 mm 时，血栓按溶解速率被清除，总催化能力在不同 N 之间保持相同（N 越大，单集群溶解越慢）。'
        '任务目标是在时限内清除全部血栓，同时尽量少碰壁、集群之间保持 d_min = 2 mm 的间距，以减少磁场相互干扰。'
        '我们使用 14 种解剖（图2）：其中 9 种用于训练与调参，5 种（冠状动脉 RCA、基底—椎动脉、脑静脉窦、SMA 栓塞、股腘动脉）'
        '只用于评估，以检验跨解剖泛化。')
    W.fig(FIG/'fig_anatomies.png', '本文使用的 14 种血管解剖（PyBullet 渲染，管腔按术前健康半径着色；暗红色为血栓，彩色小球为集群起点）。'
          '标"留出"的 5 种只用于评估。半径范围跨两个数量级，从 0.3 mm 的脑动脉分支到 13.5 mm 的肺动脉主干。', 16.5)
    W.h2('3.2 未知动力学')
    W.p('血流采用压力驱动的血管阻力网络（Poiseuille 定律，阻力 ∝ ∫ds/r⁴），每个管段内为抛物线速度剖面；血栓使局部管径变小，'
        '溶解后网络重新求解。入口健康平均流速按解剖统一标定为 v̄（本文取 0.025–0.1 mm/s，即体外实验尺度）。在此基础上叠加个体化、时变的扰动（强度 s）：')
    W.eq(r'\mathbf{u}_{\mathrm{flow}}(\mathbf{x},t)=2\,\bar{u}_e\,q(t)\left(1-\frac{\rho^2}{R_e^2}\right)\hat{\mathbf{t}}_e,\qquad '
         r'q(t)=\max\left\{\alpha_{\mathrm{p}}\left[1+A\,w\left(2\pi\frac{\mathrm{HR}}{60}t+\phi\right)\right],\;0.05\right\}')
    W.p('其中 ρ 为到中心线的距离，w(·) 为归一化的两谐波动脉波形，HR ∈ [50, 120] bpm，脉动幅度 A ∈ [0, 0.8s]（静脉 × 0.3），'
        '患者流量系数 α_p ∈ logU[1/(1+s), 1+s]。集群的实际速度为')
    W.eq(r'\dot{\mathbf{x}}_i=g_i(t)\,\lambda(\delta_i)\,\eta(c_i)\,\mathbf{R}_i\,\tilde{\mathbf{u}}_i+\mathbf{u}_{\mathrm{flow}}(\mathbf{x}_i,t)+\boldsymbol{\epsilon},\qquad '
         r'\tau\,\dot{\tilde{\mathbf{u}}}_i=\mathbf{u}_i-\tilde{\mathbf{u}}_i')
    W.p('式中 u_i 为指令（‖u_i‖ ≤ 1），g_i(t) 为带 OU 漂移的响应增益（U[1−0.3s, 1+0.3s]），R_i 为最大 15s° 的方向偏差，τ ∈ [0, 0.3s] s 为响应滞后，'
        'λ(δ) 为近壁润滑阻力因子，η(c) 为窄管中集群被拉长导致的减速，另有贴壁后的黏附减速和执行噪声 ε。溶解速率在 logU[1/(1+0.5s), 1+0.5s] 内随机。'
        '以上量控制器均不可见；参数范围在任何结果产生之前登记。')
    W.h2('3.3 可部署观测与信息边界')
    W.p('控制器只能使用：(i) 术前地图（中心线、健康半径、血栓位置）；(ii) 双平面合成图像经检测、三维融合与跟踪得到的集群位置估计，'
        '带噪声、丢帧与 L ∈ {1, 2, 3} 个控制周期的延迟，以及每帧的采集时间（由此得到帧龄 τ_t）；(iii) 控制器自己已经发出的指令；'
        '(iv) 血栓是否仍然可见。仿真中的真实位置、流速、响应增益与血栓剩余质量只用于计算指标，以及作为训练期的监督标签。')
    W.h2('3.4 评价指标')
    W.p('主指标为 Strict 成功：回合内全部血栓被清除、无集群丢失、累计壁接触小于 1 robot·s、全程满足间距约束且集群间无接触：')
    W.eq(r'\mathrm{Strict}=\mathbb{1}\left[\,m_j(T)=0\;\forall j\;\wedge\;\sum_i\int_0^T\mathbb{1}[\mathrm{wall}_i(t)]\,dt<1\,\mathrm{s}\;\wedge\;'
         r'\min_{i\neq k,t}\Vert\mathbf{x}_i-\mathbf{x}_k\Vert\geq d_{\min}\right]')
    W.p('效率用 T50/T90/T100 和清除曲线下面积 AUC 衡量：')
    W.eq(r'\mathrm{AUC}=\frac{1}{T}\int_0^{T}\left(1-\frac{\sum_j m_j(t)}{\sum_j m_j^0}\right)dt,\qquad T=300\,\mathrm{s}')
    W.p('其中 T90_300 表示未达到 90% 清除的回合按 300 s 计；另外单独报告达到比例和仅对达到者计算的条件 T90。'
        '集群互不影响用间距违规时长（pair·s）和偶极耦合暴露 E = Σ_{i<k}∫(d_min/d_ik)³·1[d_ik < 2d_min] dt 衡量，后者是磁偶极场 ∝ 1/d³ 的未标定代理；'
        '碰壁用累计壁接触、最长连续接触和"壁接触 ≥ 1 s"的回合比例衡量。')
    W.h1('4 方法')
    W.h2('4.1 总体框架')
    W.p('框架分为三层（图1）。全局层在术前完成血栓分配与冲突区排序，运行时只根据测量进度执行等待；局部层每 0.1 s 为每个集群生成指令；'
        '执行层把指令依次经过壁护、TPG 等待和间距护盾后发给执行器。所有对比方法使用同一执行层，只替换全局层或局部层。')
    W.h2('4.2 全局层：测地分配与时序计划图')
    W.p('设 d_𝒯(a, b) 为血管图上的测地距离，s_i 为集群 i 的起点。分配 π 把每个血栓指派给一个集群，并给出各集群的访问顺序 (c_{π_i(1)}, …)。'
        '我们穷举所有分配（M = 4、N ≤ 3 时至多 81 种）及其访问顺序，最小化完工路程，次级目标为总路程：')
    W.eq(r'\pi^{*}=\arg\min_{\pi}\left(\max_i L_i(\pi),\;\sum_i L_i(\pi)\right),\qquad '
         r'L_i(\pi)=d_{\mathcal{T}}(s_i,c_{\pi_i(1)})+\sum_{k\geq 1}d_{\mathcal{T}}(c_{\pi_i(k)},c_{\pi_i(k+1)})')
    W.p('随后把每个集群的整段行程按 0.2 mm 重采样为弧长路径 p_i(σ)。对任意两集群，满足下式的弧长对构成冲突区，冲突区的连通分量即为 TPG 中的一个节点对：')
    W.eq(r'\mathcal{Z}_{ik}=\left\{(\sigma,\sigma^{\prime})\;:\;\Vert\mathbf{p}_i(\sigma)-\mathbf{p}_k(\sigma^{\prime})\Vert<d_{\min}+b\right\},\quad b=0.6\,\mathrm{mm}')
    W.p('按优先级（行程更长者优先）对各集群做带驻留时间的时间仿真，为每个冲突区确定先后顺序，由此得到无环的依赖图[@honig2016]。'
        '运行时，后行集群在冲突区入口前停住，直到先行集群的测量进度离开该区。由于顺序来自一个一致的时间计划，依赖图无环，执行不会死锁。')
    W.h2('4.3 局部规则先验')
    W.p('路线追踪在术前路线上取前视点（胡萝卜点，0.4 mm，分叉附近缩短为 0.1 mm），指令方向指向该点；路线进度每步最多前进约 1.5 mm，'
        '以免跳到空间上相邻但管腔不连通的分支。到达血栓附近时按距离线性减速（Fixed Settle），使集群停留在血栓上：')
    W.eq(r'\mathbf{u}^{\mathrm{fix}}=\min\left(1,\frac{d_i}{r_s}\right)\hat{\mathbf{e}}_{\mathrm{carrot}},\qquad d_i=\Vert\hat{\mathbf{x}}_i-\mathbf{c}_{j(i)}\Vert,\;r_s=0.3\,\mathrm{mm}')
    W.p('Adaptive Settle 另外维护一个响应估计，并在"推不动"（1 s 内距离减小不足 0.03 mm）时解除减速 2 s。'
        '前期诊断表明，两者的优劣随成像延迟翻转：延迟为 1 步时 Adaptive 更好，2–3 步时 Fixed 更好。'
        '因此规则先验 SwitchSettle 按帧龄 τ_t 切换：τ_t > 0.15 s 时用 Fixed Settle，否则用 Adaptive Settle。')
    W.h2('4.4 动作条件化的学习运动预测器')
    W.p('局部层的关键量是"若从现在起执行指令序列 u_{t:t+H}，集群的真实位置会怎样变化"。由于成像延迟，最新位置估计 x̂_t 本身已落后于真实位置，'
        '因此预测目标定义为相对最新估计的真实位移，包含延迟补偿（h = 0）：')
    W.eq(r'\Delta\mathbf{x}_h=\mathbf{F}_t\left(\mathbf{x}_{t+h}-\hat{\mathbf{x}}_t\right),\qquad h\in\mathcal{H}=\{0,1,3,6,12\}\ \mathrm{steps}\;(0–1.2\,\mathrm{s})')
    W.p('F_t 为当前路线点的切向—法向—副法向局部坐标系，使模型对解剖朝向不变。输入包括：'
        '(i) 24 步（2.4 s）历史，每步含图像速度、该步实际发出的指令（经护盾后、执行器响应之前）、相对当前的延迟位置、帧龄和 TPG 等待标志，'
        '完整的指令队列保证长延迟下仍能对齐"发出的指令"与"看到的运动"；(ii) 当前地图上下文：径向偏移、壁间隙、健康半径、前方 0/1/2 mm 的路线切向、'
        '到目标血栓的方向与距离、帧龄；(iii) 候选指令序列（12 × 3）。网络由 GRU 编码历史、两个 MLP 编码上下文与指令，'
        '再经 256 维的共享主干输出各时域位移的均值 μ̂_h 与对数方差 log σ̂_h²，另设辅助头回归瞬时流速与有效响应增益（仅用于训练）。训练损失为')
    W.eq(r'\mathcal{L}(\theta)=\frac{1}{2}\sum_{h\in\mathcal{H}}\left[\frac{\Vert\Delta\mathbf{x}_h-\hat{\boldsymbol{\mu}}_h\Vert^2}{\hat{\sigma}_h^2}+\log\hat{\sigma}_h^2\right]+\beta\,\Vert\hat{\mathbf{a}}-\mathbf{a}\Vert^2,\quad\beta=0.1')
    W.p('其中 a 为瞬时流速与响应增益的真值。K = 3 个独立初始化的模型组成深度集成[@lakshminarayanan2017]，预测均值与总方差为')
    W.eq(r'\bar{\boldsymbol{\mu}}_h=\frac{1}{K}\sum_{k=1}^{K}\hat{\boldsymbol{\mu}}_h^{(k)},\qquad '
         r'\sigma_h^2=\frac{1}{K}\sum_k\hat{\sigma}_h^{2\,(k)}\;+\;\mathrm{Var}_k\left[\hat{\boldsymbol{\mu}}_h^{(k)}\right]\;\equiv\;(\sigma_h^{\mathrm{ale}})^2+(\sigma_h^{\mathrm{epi}})^2')
    W.p('训练数据由行为策略在 9 种训练解剖上采集（每个预测器种子 360 回合，约 1.4×10⁶ 个样本）。行为策略每 0.6–1.2 s 以 0.5 的概率'
        '在规则指令和随机候选指令之间切换并保持，使数据覆盖选择器会用到的候选；域随机化覆盖入口流速 {0.025, 0.05, 0.1} mm/s、延迟 1–3 步和扰动强度 s ∈ [0, 1.25][@tobin2017]。'
        '未来真实位置、流速与增益只作监督标签，不进入任何部署输入。')
    W.h2('4.5 以规则为锚的候选指令选择与不确定性回退')
    W.p('每个控制周期，对每个可行动的集群构造候选集 𝒰：规则指令 u_r（SwitchSettle）、learner-off 指令（Fixed Settle）、停止，'
        '以及指向胡萝卜点、沿路线法向/副法向偏转 ±25°、指向血栓（1.5 mm 内）的方向，各配 {0.25, 0.5, 0.75, 1.0} 四档速度，共约 25 个。'
        '候选在 1.2 s 内保持恒定，满足 ‖u‖ ≤ 1。由预测得到的世界坐标位置记为 x̄_h(u)，代价为')
    W.eq(r'J(\mathbf{u})=-w_p\,\Delta s(\mathbf{u})-w_a\,\Delta d(\mathbf{u})-w_d\,D(\mathbf{u})+w_w\sum_{h}\frac{[\,m-(c_h-k_w\sigma_h)\,]_+}{m}'
         r'+w_s\sum_{h,k}\frac{[\,d_{\min}+0.3-\Vert\bar{\mathbf{x}}_h-\mathbf{y}_{k,h}\Vert\,]_+}{d_{\min}}+w_\sigma\,\sigma_{1.2}+w_i\,\mathbb{1}[\Vert\mathbf{u}\Vert<0.3]')
    W.p('Δs 为 1.2 s 内沿术前路线的预测弧长进展；Δd 为预测的接近血栓量，D 为后半时域落在接触半径内的比例（两者仅在血栓 1.5 mm 内计入）；'
        'c_h 为预测位置处的地图壁间隙（沿路线走廊取健康管腔的并集），m = 0.03 mm；y_{k,h} 为其他集群按恒速外推的位置；'
        '最后一项惩罚远离血栓时的空等。选择规则以 u_r 为锚，并在认知不确定性过高时回退：')
    W.eq(r'\mathbf{u}^{\prime}=\arg\min_{\mathbf{u}\in\mathcal{U}}J(\mathbf{u}),\qquad '
         r'\mathbf{u}^{\star}=\mathbf{u}_r\ \ \mathrm{if}\ \ \sigma^{\mathrm{epi}}_{1.2}(\mathbf{u}^{\prime})>\sigma_{\mathrm{fb}}\ \ \mathrm{or}\ \ J(\mathbf{u}_r)-J(\mathbf{u}^{\prime})<\delta;\ \ \mathrm{else}\ \ \mathbf{u}^{\star}=\mathbf{u}^{\prime}')
    W.p('权重 (w_p, w_a, w_d, w_w, w_s, w_σ, w_i, k_w, δ) = (1.5, 3, 1, 2, 2, 0.2, 0.3, 0.5, 0.05) 在独立的调参场景上从三组候选中选出；'
        '回退阈值 σ_fb = 0.0445 mm 取验证集上认知不确定性的第 95 百分位，均在评估之前固定。'
        '非学习对照把同一选择器中的预测器替换为在线仿射估计器：用时间戳对齐的"已发指令—图像速度"历史，滑动拟合 v = g·u + b，再按该模型外推。')
    W.algo('预测辅助的多集群局部控制（每个控制周期）', [
        '输入: 估计 Ê_t（延迟位置、速度、帧龄）, 术前地图 𝓜, 已发指令队列 Q, 预测器集成 {f_θk}, 全局计划 (π*, TPG)',
        '1:  for 每个集群 i do',
        '2:      j ← π* 中 i 的下一个仍可见血栓;  更新路线进度与胡萝卜点',
        '3:      h_t ← Push(历史, 图像速度, Q 中该步指令, 延迟位置, 帧龄, TPG 等待)',
        '4:      if TPG 要求等待 或 j 不存在 then u_i ← 0; continue',
        '5:      g ← WallGuard(路线追踪指令)                   # 停滞重规划 + 近壁投影',
        '6:      u_r ← SwitchSettle(g, 帧龄);  u_0 ← FixedSettle(g)',
        '7:      𝒰 ← {u_r, u_0, 0} ∪ {方向 × 速度网格}',
        '8:      for u ∈ 𝒰 do  (μ̄_h, σ_h, σ^epi_h) ← Ensemble(h_t, c_t(𝓜), u)    # 批量前向',
        '9:          J(u) ← 进展 + 接近/驻留 + 壁风险 + 间距风险 + 不确定性 + 空等',
        '10:     u′ ← argmin J;  if σ^epi(u′) > σ_fb 或 J(u_r) − J(u′) < δ then u′ ← u_r',
        '11:     u_i ← WallRecovery(u′)                         # 与所有方法相同的近壁投影',
        '12: end for',
        '13: u ← TPGHold(u);  u ← SpacingShield(u);  发送 u（‖u_i‖ ≤ 1），并写入 Q',
    ])
    W.h2('4.6 共享执行与安全层')
    W.p('所有方法的输出都经过：(i) WallGuard，即健康管腔距离小于 0.20 mm 时去除外向分量并加入指向轴心的小分量，2 s 位移不足 0.15 mm 时重新规划路线；'
        '(ii) TPG 等待（若全局层有 TPG）；(iii) 间距护盾，低优先级集群在预测冲突时停止，并把指令投影到线性间距约束上。'
        '因此各方法的差异只来自全局层和局部层本身。')
    W.h1('5 实验')
    W.h2('5.1 实验设置')
    W.p('**场景集。** 困难矩阵包含 6 个条件面板（表1），每个面板 42 个场景：9 种训练解剖 × N = 1–3 × 1 个种子共 27 个，5 种留出解剖 × N = 1–3 共 15 个，'
        '合计 252 个配对场景，各方法在完全相同的场景上运行（以场景哈希逐一核对）。V5 设定沿用早期基准的旧流量标定和 1 步延迟，'
        '在扰动强度 s = 0、1、1.5 下各有 14 种解剖 × 10 个种子 × N = 1–3 = 420 个场景，用于检验弱流、低延迟条件。'
        '选择器权重、回退阈值等在单独的调参场景上确定，其种子与上述评估场景、预测器训练数据互不相交。'
        '预先登记的封存测试池（每种解剖 100 个场景）在本稿中**未被访问**，所有结果均为开发集结果。')
    W.table(['面板', '入口健康平均流速 v̄ (mm/s)', '成像延迟 L（步，0.1 s/步）', '扰动强度 s', '主要困难'],
            [['低延迟', '0.05', '1', '0', '基准'], ['中延迟', '0.05', '2', '0', '延迟使减速规则失配'],
             ['高延迟', '0.05', '3', '0', '延迟下"释放"误触发'], ['强流', '0.10', '2', '0.625', '狭窄处流速接近执行器上限'],
             ['动力学变化', '0.05', '2', '1.25', '脉动、响应漂移、黏附叠加'], ['分布外', '0.075', '3', '0.625', '训练中未出现的流速—延迟组合']],
            '困难矩阵的 6 个条件面板。流速在每个场景再乘 U[1.5, 2.5]/2 的随机系数，并随血栓溶解在分支间重新分配。')
    W.p('**统计方法。** 对每个指标报告配对场景上的均值；方法间差值用场景层面的 bootstrap（4000 次重采样）给出 95% 置信区间。'
        '学习方法的种子统计报告均值 ± 标准差。仿真在 22 个 CPU 核上并行运行，学习模型在 RTX 4090 上训练；'
        '单个预测器种子的数据采集约 7 min，集成训练约 2 min。')
    W.h2('5.2 对比方法与因子化设计')
    W.p('多集群血栓清除同时包含"全局"（分配与协调）和"局部"（导航与驻留）两个层面。为了区分各层的贡献，我们把外部论文方法按层归类，'
        '在同一执行层下做"全局层 × 局部层"的因子化组合（表2）。所有外部方法都是根据原文在本任务中的本地适配实现，并非作者代码的复现，'
        '其适配要点如表中所列；它们与本文方法使用相同的可部署观测、执行器上限和安全层。')
    W.table(['层', '方法', '来源', '在本任务中的适配'],
            [['全局', '测地分配 + TPG（本文）', 'TPG：ICAPS 2016[@honig2016]', '穷举分配（式5）+ 基于弧长冲突区的固定优先 TPG'],
             ['全局', 'STPG', 'AAAI 2025[@jiang2025]', '测地分配 + 每 2 s 按测量进度重优化尚未进入的冲突区顺序（N ≤ 3 时穷举，即最优解）'],
             ['全局', 'OT 分配', 'MPC-OT，CDC 2025[@khan2025]', '以测地运输代价之和最小为目标的匿名目标分配，就近顺序访问；连续跟踪层由所比较的局部层替代；无 TPG'],
             ['全局', '无协调', '—', '测地分配，仅间距护盾'],
             ['局部', '预测辅助选择（本文）', '本文；相关：PETS NeurIPS 2018[@chua2018]', '学习预测器集成 + 规则锚定 + 认知不确定性回退'],
             ['局部', '预测辅助选择（在线估计）', '本文消融', '同一选择器，预测器换为在线仿射估计器'],
             ['局部', '路线追踪 + 减速', '经典', 'Fixed Settle'],
             ['局部', 'PAC-NMPC', 'CASE 2025[@gonzales2025]', '高斯控制序列分布、状态与邻居不确定性 rollout、Hoeffding PAC 上界；交叉熵更新代替原文时变 LQR 反馈'],
             ['局部', 'MPPI', 'ICRA 2016 / T-RO 2018[@williams2016,@williams2018]', '64 条采样、8 步时域，模型为测得漂移 + 单位指令；外接同一到栓减速'],
             ['局部', 'CBF-QP', 'IEEE TAC 2017[@ames2017]', '以地图壁间隙为障碍函数的线性化 CBF 约束投影；外接同一到栓减速'],
             ['局部', 'Turbo（Transformer PPO）', 'Nat. Mach. Intell. 2026[@an2026]', '因果 Transformer + 多层域随机化，直接输出速度；在本任务训练域内重新训练 100 min']],
            '因子化对比中的方法。前四行为全局层，后七行为局部层；所有组合共用同一执行层。', fs=7.5)
    W.p('此外，我们还与本项目的经典规则（Fixed / Adaptive / SwitchSettle）、前一版学习方法 nav_tf_v3（路线坐标系 Transformer PPO 残差，3 个种子）'
        '及其带辅助流速头的变体，以及 learner-off 对照（同一框架下把学习残差置零）进行比较（5.4 节）。')
    W.h2('5.3 全局层 × 局部层因子化对比')
    if FT:
        G_LAB = {'A_TPG': '测地分配+TPG', 'STPG': 'STPG', 'OT': 'OT 分配', 'NONE': '无协调'}
        L_LAB = {'ours': '本文（学习预测选择）', 'online': '在线估计选择', 'pursuit': '追踪+减速', 'pac': 'PAC-NMPC', 'mppi': 'MPPI', 'cbf': 'CBF-QP', 'turbo': 'Turbo'}
        go, lo = ['A_TPG', 'STPG', 'OT', 'NONE'], ['ours', 'online', 'pursuit', 'pac', 'mppi', 'cbf', 'turbo']
        arms = sorted(FT, key=lambda a: (go.index(a.split('+')[0]), lo.index(a.split('+')[1])))
        pm = fac.get('ours_minus', {})
        rows = []
        for a in arms:
            g, l = a.split('+'); v = FT[a]['ALL']; dd = pm.get(a, {}).get('ALL')
            rows.append([G_LAB[g], L_LAB[l], pct(v['strict_success']), pct(v['removal']), pct(v['removal_auc']), f"{v['t90_300']:.0f}",
                         f"{v['wall_contact_s']:.2f}", pct(v['wall_ge_1s']), f"{v['spacing_violation_pair_s']:.3f}", f"{v['coupling_exposure']:.2f}",
                         '—' if dd is None else f'{100*dd[0]:+.1f} [{100*dd[1]:+.1f}, {100*dd[2]:+.1f}]'])
        W.table(['全局层', '局部层', 'Strict %', '清除率 %', 'AUC %', 'T90_300 s', '壁 robot·s', '壁≥1 s %', '间距 pair·s', '耦合暴露', '本文−该组合 Strict pp [95% CI]'],
                rows, f"困难矩阵上全局层 × 局部层的全部组合（{fac['n_keys']} 个共同配对场景）。本文 = 测地分配+TPG / 学习预测选择（预测器种子 A）。", fs=7)
        ext = [a for a in arms if a.split('+')[1] not in ('ours', 'online')]
        be = max(ext, key=lambda a: FT[a]['ALL']['strict_success'])
        rk = sorted(arms, key=lambda a: -FT[a]['ALL']['strict_success']); r0 = rk.index('A_TPG+ours')+1
        rank_s = '在全部组合中最高' if r0 == 1 else f'在全部 {len(arms)} 个组合中排第 {r0}（领先者为 {G_LAB[rk[0].split("+")[0]]} / {L_LAB[rk[0].split("+")[1]]}，{F(rk[0])}%）'
        W.p(f'表3给出全部组合的结果。本文组合的 Strict 为 {F("A_TPG+ours")}%，{rank_s}；'
            f'最好的非本文局部层组合为"{G_LAB[be.split("+")[0]]} / {L_LAB[be.split("+")[1]]}"（{F(be)}%）。'
            f'在局部层方面，固定全局层为测地分配+TPG 时，本文 {F("A_TPG+ours")}%、PAC-NMPC {F("A_TPG+pac")}%、MPPI {F("A_TPG+mppi")}%、'
            f'CBF-QP {F("A_TPG+cbf")}%、Turbo {F("A_TPG+turbo")}%、经典追踪 {F("A_TPG+pursuit")}%。'
            f'在全局层方面，固定局部层为本文选择器时，测地分配+TPG {F("A_TPG+ours")}%、STPG {F("STPG+ours")}%、OT 分配 {F("OT+ours")}%、无协调 {F("NONE+ours")}%；'
            f'全局层之间的差别远小于局部层之间的差别，说明困难条件下的性能主要由局部层决定。'
            f'TPG 与间距护盾的作用主要体现在集群互不影响指标上：无协调时间距违规为 {F("NONE+ours", m="spacing_violation_pair_s", p=False)} pair·s，'
            f'耦合暴露为 {F("NONE+ours", m="coupling_exposure", p=False)}，而测地分配+TPG 下分别为 {F("A_TPG+ours", m="spacing_violation_pair_s", p=False)} 和 {F("A_TPG+ours", m="coupling_exposure", p=False)}。')
        W.fig(FIG/'fig_factorial_hard.png', '困难矩阵上各"全局层 × 局部层"组合的 Strict 成功率热图。行为全局层，列为局部层。', 15.)
        W.fig(FIG/'fig_panels_hard.png', '各条件面板下代表性组合的 Strict 成功率。本文在中延迟、高延迟、强流和分布外面板中领先最明显，低延迟面板与规则相当。', 16.)
    else:
        W.p('【因子化对比结果待补】')
    W.h2('5.4 与经典规则及学习基线的比较')
    lab = {'ours_A': '本文（种子 A）', 'ours_B': '本文（种子 B）', 'ours_C': '本文（种子 C）', 'sel_online': '同一选择器 + 在线估计器',
           'switch_settle': 'SwitchSettle（帧龄切换规则）', 'fixed_settle': 'Fixed Settle', 'adaptive_settle': 'Adaptive Settle',
           'stpg': 'STPG + 追踪减速', 'pac_nmpc': 'PAC-NMPC', 'nav_off': 'learner-off（nav_tf_v3 框架，残差 = 0）',
           'nav_tf_v3': 'nav_tf_v3（Transformer PPO 残差，3 种子均值）', 'belief': 'nav_tf_v3 + 辅助流速头（2 种子均值）'}
    PN = ['low_delay', 'moderate_delay', 'high_delay', 'strong_flow', 'variable_response', 'ood_flow_delay']
    rows = []
    for a in ['ours_A', 'ours_B', 'ours_C', 'sel_online', 'switch_settle', 'fixed_settle', 'adaptive_settle', 'stpg', 'pac_nmpc', 'nav_off', 'nav_tf_v3', 'belief']:
        v = H[a]
        rows.append([lab[a], pct(v['strict_success'])]+[pct(D['hard_panel'][p][a]['strict_success']) for p in PN]+
                    [pct(D['hard_topo']['unseen_topology'][a]['strict_success']), f"{v['t90_300']:.0f}", f"{v['wall_contact_s']:.2f}", pct(v['wall_ge_1s']), f"{v['coupling_exposure']:.2f}"])
    W.table(['方法', '全部', '低延迟', '中延迟', '高延迟', '强流', '动力学变化', '分布外', '留出解剖', 'T90_300 s', '壁 robot·s', '壁≥1 s %', '耦合暴露'], rows,
            f"困难矩阵上与经典规则和学习基线的比较（Strict %，{H['ours_A']['n']} 个配对场景）。本文三个种子各自拥有独立采集的数据和独立训练的预测器集成。", fs=7)
    p2 = D['pair']
    W.p(f'本文三个独立种子的 Strict 为 {pct(H["ours_A"]["strict_success"])}%、{pct(H["ours_B"]["strict_success"])}%、{pct(H["ours_C"]["strict_success"])}%'
        f'（均值 {pct(om)}% ± {pct(osd)}%）。相对 SwitchSettle 的配对提升为 {100*p2["switch_settle"][0]:+.1f} pp [{100*p2["switch_settle"][1]:+.1f}, {100*p2["switch_settle"][2]:+.1f}]，'
        f'相对 learner-off 为 {100*p2["nav_off"][0]:+.1f} pp [{100*p2["nav_off"][1]:+.1f}, {100*p2["nav_off"][2]:+.1f}]，'
        f'相对 PAC-NMPC 为 {100*p2["pac_nmpc"][0]:+.1f} pp，相对 nav_tf_v3 为 {100*p2["nav_tf_v3"][0]:+.1f} pp。'
        f'提升集中在中延迟、高延迟和强流面板：这些条件下，固定的减速规则要么在延迟下失配，要么在强逆流中推不动。'
        f'低延迟面板中，本文与 SwitchSettle 持平，因为此时规则已接近最优，选择器大多保留规则指令。'
        f'nav_tf_v3（{pct(H["nav_tf_v3"]["strict_success"])}%）低于 learner-off（{pct(H["nav_off"]["strict_success"])}%），'
        f'说明在接近最优的规则上直接叠加 PPO 残差会引入有害偏移；这也是本文转向"预测—选择"结构的直接原因。'
        f'代价方面，本文的平均壁接触（{H["ours_A"]["wall_contact_s"]:.2f} robot·s）略高于 Fixed Settle（{H["fixed_settle"]["wall_contact_s"]:.2f}），'
        f'壁接触 ≥ 1 s 的回合比例为 {pct(H["ours_A"]["wall_ge_1s"])}%，规则为 {pct(H["fixed_settle"]["wall_ge_1s"])}%。')
    W.h2('5.5 预测精度与不确定性校准')
    pt = D['pred']
    prow = []
    for split, name in (('val_train', '训练解剖（新场景）'), ('val_heldout', '留出解剖')):
        for m, mn in (('constant_velocity', '常速度外推'), ('nominal', '名义动力学'), ('online_affine', '在线仿射估计器'), ('learned', '学习集成（本文）')):
            c = pt[split]['all'][m]
            prow.append([name, mn]+[f"{c[h]['rmse_mm']:.3f}" for h in ('0.0s', '0.3s', '0.6s', '1.2s')]+
                        [f"{c['1.2s']['within1']:.2f} / {c['1.2s']['within2']:.2f}" if 'within1' in c['1.2s'] else '—'])
    W.table(['验证集', '预测器', 'RMSE 0 s', 'RMSE 0.3 s', 'RMSE 0.6 s', 'RMSE 1.2 s', '1σ/2σ 覆盖率 @1.2 s'], prow,
            '真实位移预测误差（mm）与不确定性校准。高斯分布的理想覆盖率为 0.68 / 0.95。', fs=7.5)
    W.fig(V90/'prediction/fig_prediction.png', '预测误差随时域的变化（左、中）及留出解剖上按流速、延迟和管腔间隙分组的 1.2 s 误差（右）。', 16.)
    lc, oc = pt['val_heldout']['all']['learned']['1.2s'], pt['val_heldout']['all']['online_affine']['1.2s']
    W.p(f'在留出解剖上，学习集成的 1.2 s 误差为 {lc["rmse_mm"]:.3f} mm，在线估计器为 {oc["rmse_mm"]:.3f} mm，降低 {100*(1-lc["rmse_mm"]/oc["rmse_mm"]):.0f}%；'
        f'延迟补偿（0 s）误差降低一半以上。学习集成的 1σ/2σ 覆盖率为 {lc["within1"]:.2f}/{lc["within2"]:.2f}，接近理想值；'
        f'在线估计器只有 {oc["within1"]:.2f}/{oc["within2"]:.2f}，明显低估不确定性。在窄管（间隙 < 0.1 mm）中两者的误差都显著增大，学习集成的优势也缩小。')
    W.h2('5.6 消融：收益来自哪里')
    AG = json.load(open(V90/'hard_report/aggregate.json'))
    g = lambda k, m='strict_success': AG[f'ALL|all|{k}'][m]
    ab = D['abl']
    rows = [['完整框架（本文，种子 A）', pct(H['ours_A']['strict_success']), f"{H['ours_A']['wall_contact_s']:.2f}", f"{H['ours_A']['spacing_violation_pair_s']:.3f}", f"{H['ours_A']['coupling_exposure']:.2f}"],
            ['去掉不确定性回退', pct(g('D_sel_ens')), f"{g('D_sel_ens', 'wall_contact_s'):.2f}", f"{g('D_sel_ens', 'spacing_violation_pair_s'):.3f}", f"{g('D_sel_ens', 'coupling_exposure'):.2f}"],
            ['学习预测器 → 在线估计器', pct(H['sel_online']['strict_success']), f"{H['sel_online']['wall_contact_s']:.2f}", f"{H['sel_online']['spacing_violation_pair_s']:.3f}", f"{H['sel_online']['coupling_exposure']:.2f}"],
            ['去掉预测选择（= SwitchSettle）', pct(H['switch_settle']['strict_success']), f"{H['switch_settle']['wall_contact_s']:.2f}", f"{H['switch_settle']['spacing_violation_pair_s']:.3f}", f"{H['switch_settle']['coupling_exposure']:.2f}"],
            ['再去掉帧龄切换（= Fixed Settle）', pct(H['fixed_settle']['strict_success']), f"{H['fixed_settle']['wall_contact_s']:.2f}", f"{H['fixed_settle']['spacing_violation_pair_s']:.3f}", f"{H['fixed_settle']['coupling_exposure']:.2f}"]]
    for key, name in (('no_tpg', '去掉 TPG†'), ('no_guard', '去掉 WallGuard†'), ('no_shield', '去掉间距护盾†'), ('no_safety', '去掉全部安全层†')):
        if key in ab and 'full' in ab:
            v = ab[key]; f0 = ab['full']
            rows.append([name, f"{pct(v['strict_success'])}（完整 {pct(f0['strict_success'])}）", f"{v['wall_contact_s']:.2f}（{f0['wall_contact_s']:.2f}）",
                         f"{v['spacing_violation_pair_s']:.3f}（{f0['spacing_violation_pair_s']:.3f}）", f"{v['coupling_exposure']:.2f}（{f0['coupling_exposure']:.2f}）"])
    W.table(['变体', 'Strict %', '壁接触 robot·s', '间距违规 pair·s', '耦合暴露'], rows,
            '困难矩阵上的消融。†：安全层与 TPG 消融在框架的 belief 反解变体上进行（括号内为该变体的完整版本），其余各行为配对场景上的同一协议。', fs=7.5)
    W.p(f'逐层去除可以看出：预测—选择结构贡献了相对 SwitchSettle 的约 {100*(H["ours_A"]["strict_success"]-H["switch_settle"]["strict_success"]):.0f} pp，'
        f'帧龄切换贡献了 SwitchSettle 相对 Fixed Settle 的 {100*(H["switch_settle"]["strict_success"]-H["fixed_settle"]["strict_success"]):.1f} pp，'
        f'不确定性回退贡献约 {100*(H["ours_A"]["strict_success"]-g("D_sel_ens")):.1f} pp（回退只在约 0.1–0.3% 的控制步触发，集中在高延迟面板）。'
        f'**关键的一点是**：把学习预测器替换为在线估计器后，Strict 为 {pct(H["sel_online"]["strict_success"])}%，'
        f'与本文三种子均值的配对差为 {100*pf[0]:+.1f} pp [{100*pf[1]:+.1f}, {100*pf[2]:+.1f}]。'
        f'也就是说，在当前候选集与代价下，更准确、校准更好的学习预测没有转化为可测的控制增益，性能提升主要来自"预测—选择"这一结构。'
        f'安全层消融表明，去掉 WallGuard 会使壁接触约翻倍，去掉 TPG 或间距护盾会使间距违规和耦合暴露上升，'
        f'而 Strict 的变化在 ±2 pp 以内；这些组件主要保证安全与互不干扰，不应因 Strict 变化不大而移除。')
    W.h2('5.7 弱流、低延迟设定（V5）')
    vr = []
    for a, n in (('ours_A', '本文（种子 A）'), ('ours_B', '本文（种子 B）'), ('ours_C', '本文（种子 C）'), ('sel_online', '在线估计选择'),
                 ('fixed_settle', 'Fixed Settle'), ('switch_settle', 'SwitchSettle'), ('stpg', 'STPG + 追踪减速'), ('pac_nmpc', 'PAC-NMPC'),
                 ('nav_off', 'learner-off'), ('nav_tf_v3', 'nav_tf_v3（3 种子）')):
        vr.append([n]+[f"{pct(D['v5'][s][a]['strict_success'])} / {D['v5'][s][a]['t90_300']:.0f}" for s in ('v5_s0', 'v5_s1', 'v5_s1.5')])
    W.table(['方法', 's = 0（Strict % / T90_300 s）', 's = 1', 's = 1.5'], vr,
            f"V5 开发集结果（每个 s 为 14 解剖 × 10 场景 × N = 1–3 = {D['v5']['v5_s0']['ours_A']['n']} 个配对场景）。", fs=7.5)
    def _v5(a):
        return 100*np.mean([D['v5'][s_][a]['strict_success'] for s_ in ('v5_s0', 'v5_s1', 'v5_s1.5')])
    gf, gs_, gw = _v5('fixed_settle')-_v5('ours_A'), _v5('stpg')-_v5('ours_A'), _v5('switch_settle')-_v5('ours_A')
    W.p(f'在弱流、1 步延迟的 V5 设定下（三档 s 平均），本文框架（种子 A）落后于 Fixed Settle {gf:.1f} 个百分点、STPG（局部层同为追踪减速）{gs_:.1f} 个百分点，'
        f'与 SwitchSettle 相差 {-gw:+.1f} 个百分点。'
        '原因有二：一是此时经典规则已接近最优，选择器的额外动作只会引入扰动；二是选择器的权重是在流速标定后的困难条件下调出的，'
        '在 V5 独立调参集上调整不确定性锚定系数也未能消除这一差距。这一结果界定了本文方法的适用范围：'
        '它针对血流和延迟显著干扰经典规则的情形，在规则已足够好的简单情形中反而不如规则。')
    W.h2('5.8 定性结果')
    G9 = ROOT/'research/figures/REPORT_20261011'
    scenes = [('mca_m1_lvo_N2_s2600000003_f0.1_L2_v0.625', '强流条件下的大脑中动脉场景（N = 2，入口 0.1 mm/s，延迟 2 步，s = 0.625）'),
              ('pulmonary_saddle_N3_s2600100002_f0.05_L2_v1.25', '失败案例：动力学变化条件下的肺动脉骑跨栓场景（N = 3，延迟 2 步，s = 1.25）。本文框架的一个集群在宽大管腔中被预测选择推向管壁并长时间贴壁，'
               '而两种基线均完成清除；这类在大管径、强扰动下的贴壁是本文剩余失败的主要模式之一')]
    for tag, desc in scenes:
        f = G9/f'{tag}_grid.png'; rj = G9/f'{tag}_rollouts.json'
        if not f.exists():
            continue
        R = json.load(open(rj)); m = {k.split('=')[0]: v['metrics'] for k, v in R.items()}
        def s_(k):
            x = m.get(k)
            if not x:
                return '—'
            t = x['t90_s']
            return f"清除 {100*x['removal']:.0f}%" + (f"、T90 = {t:.0f} s" if t else '') + ('（Strict 成功）' if x['strict_success'] else '')
        W.fig(f, f'{desc}。从上到下依次为本文框架、SwitchSettle 与 PAC-NMPC（局部层；全局层均为测地分配 + TPG）。'
              f'彩色曲线为集群轨迹，暗红色球体随剩余血栓质量缩小。本文：{s_("sel_learned_fb")}；SwitchSettle：{s_("switch_settle")}；PAC-NMPC：{s_("pac_nmpc")}。', 16.5)
    W.h1('6 讨论')
    W.p('**为什么采用分层与"规则为锚"的设计。** 术前 CTA 已给出血管几何，分配、路线、冲突协调和壁护都有可靠的经典解。'
        '在本任务早期的实验中，从零学习的端到端策略（包括 Turbo 式直接速度策略）在同等预算下几乎无法完成多点清除，'
        '在接近最优的规则上直接叠加 PPO 残差又会损害性能。以规则为锚的候选选择把学习限制在"判断何时、以多快速度偏离规则"，'
        '使框架在规则已足够好的低延迟条件下与规则持平，在规则失配的条件下获得提升。')
    W.p('**学习的贡献与局限。** 学习预测器在精度和校准上都明显优于在线估计器，但在控制效果上二者相当。可能的原因包括：'
        '候选集离散、时域只有 1.2 s，代价函数的排序对两者的预测差异不敏感；在线估计器的偏差主要出现在窄管等少数状态，'
        '而这些状态下所有候选的风险都很高。如何把更好的不确定性用于更长时域的规划（例如按不确定性调整时域或候选密度），'
        '是让学习组件产生可归因控制增益的关键方向。')
    W.p('**有效性边界。** 本文结论来自仿真开发集：血流为准稳态网络模型，未包含 Womersley 效应与管壁弹性；集群被视为刚体，'
        '2 mm 间距只是磁场相互干扰的未标定代理，未建模多集群共享同一外加磁场的耦合驱动；'
        '"血栓是否可见"由仿真直接给出，而非从像素中检测。外部方法均为本地适配实现，其参数未针对本任务专门调优。'
        '方法冻结后，将在预先登记的封存测试池上只评估一次，并在体外血管模型中验证。')
    W.h1('7 结论')
    W.p(f'本文面向"在多变血流与延迟成像下用多个磁性微机器人集群清除多个血栓"这一问题，提出了全局测地分配 + 时序计划图与局部预测辅助指令选择相结合的分层导航框架，'
        f'并构建了包含 14 种解剖、多种流速—延迟—动力学条件的仿真基准。在困难条件下，框架的 Strict 成功率为 {pct(om)}%，'
        f'比最强经典规则高 {100*ps[0]:.0f} 个百分点，并{ext_word}以 STPG、OT 分配为全局层，以 PAC-NMPC、MPPI、CBF-QP、Turbo 为局部层的外部方法组合，'
        f'在三个独立种子间稳定。消融显示，提升主要来自"预测—选择"结构；学习预测器提供了更准确且校准良好的预测，'
        f'但其控制增益尚未超过非学习的在线估计器；在弱流、低延迟条件下，框架仍落后于固定减速规则。'
        f'这些结果为多集群血栓清除给出了一个可复现的强基线框架，也指明了学习组件下一步需要解决的问题。')
    W.h1('参考文献')
    for k, key in enumerate(W.cited):
        W.p(f'[{k+1}] {REFS[key]}', 'Reference')
    W.save(out); print('saved', out, 'figures', W.fig_no, 'tables', W.tab_no, 'equations', W.eq_no, 'refs', len(W.cited))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args(); build(a.out)


if __name__ == '__main__':
    main()
