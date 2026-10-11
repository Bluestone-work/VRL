"""Paper figures for the EXP0091 manuscript: method framework schematic (and result figures when data are present).
usage: paper_figs_exp0091.py --out research/figures/EXP0091_paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from matplotlib import font_manager
for _f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Serif CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def box(ax, x, y, w, h, text, fc, ec='#333333', fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.02', fc=fc, ec=ec, lw=1.0))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, weight='bold' if bold else 'normal', wrap=True)


def arrow(ax, p, q, color='#333333', style='-|>', ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=1.0, linestyle=ls))


def framework(out):
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    C = dict(pre='#E8F1FA', sense='#EAF6EA', learn='#FDEBD3', sel='#FBE3E3', safe='#EEEEEE', env='#F3ECF8')
    # row 1: pre-operative
    box(ax, .02, .80, .17, .14, '术前 CTA 地图\n中心线·健康半径·血栓位置', C['pre'])
    box(ax, .23, .80, .17, .14, '测地分配 A\n（最小化完工路程）', C['pre'])
    box(ax, .44, .80, .17, .14, '时序计划图 TPG\n冲突区通行顺序', C['pre'])
    box(ax, .65, .80, .15, .14, '路线追踪\n胡萝卜点 + 规则减速', C['pre'])
    arrow(ax, (.19, .87), (.23, .87)); arrow(ax, (.40, .87), (.44, .87)); arrow(ax, (.61, .87), (.65, .87))
    # row 2: sensing + history
    box(ax, .02, .52, .17, .17, '双平面成像\n延迟位置估计 $\\hat{x}_t$\n帧龄 $\\tau_t$', C['sense'])
    box(ax, .23, .52, .19, .17, '可部署历史 $h_t$（2.4 s）\n图像速度·已发指令队列\n·延迟位置·帧龄·TPG 等待', C['sense'])
    arrow(ax, (.19, .605), (.23, .605))
    # learned predictor
    box(ax, .46, .50, .22, .21, '学习运动预测器集成 $f_{\\theta_1..\\theta_3}$\n输入 $h_t$, 地图上下文 $c_t$, 候选指令序列 $u$\n输出 $\\Delta\\hat{x}(t+h\\,|\\,u)$ 与 $\\sigma(h)$\nh ∈ {0, 0.1, 0.3, 0.6, 1.2} s', C['learn'], bold=False)
    arrow(ax, (.42, .605), (.46, .605))
    box(ax, .46, .28, .22, .14, '训练期监督（仅训练）\n未来真实位移·瞬时流速·响应增益\n高斯 NLL + 辅助回归', '#FFF7E6', ec='#C07000', fs=8)
    arrow(ax, (.57, .42), (.57, .50), color='#C07000', ls='--')
    # candidate selection
    box(ax, .72, .52, .26, .19, '候选生成与打分\nU = {规则 u_rule, learner-off, 停止,\n方向×速度网格, belief 反解}\nJ(u) = −进展 − 接近 − 驻留 + 壁风险\n+ 间距风险 + σ 惩罚 + 空等', C['sel'])
    arrow(ax, (.68, .605), (.72, .605))
    arrow(ax, (.80, .80), (.80, .71))
    box(ax, .72, .28, .26, .16, '不确定性感知锚定\n若 J(u_rule) − J(u*) < δ + κ·σ_pair → 用 u_rule\n认知不确定性 > σ_fb → 回退到 SwitchSettle', C['sel'], fs=8)
    arrow(ax, (.85, .52), (.85, .44))
    # safety + plant
    box(ax, .72, .06, .26, .14, '共享安全层（所有方法相同）\nWallGuard 近壁投影/停滞重规划 → TPG 等待\n→ 间距护盾 → 执行器 |u| ≤ 1', C['safe'], fs=8)
    arrow(ax, (.85, .28), (.85, .20))
    box(ax, .40, .06, .26, .14, '仿真血管（未知动力学）\n脉动流·个体流量·响应增益/漂移/延迟\n·窄管形变·近壁阻力·黏附·溶解速率', C['env'], fs=8)
    arrow(ax, (.72, .13), (.66, .13))
    arrow(ax, (.40, .13), (.10, .52), ls=':')
    ax.text(.20, .30, '图像观测（延迟、噪声、丢帧）', fontsize=8, rotation=52, color='#555555')
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(out/f'fig_framework.{e}', dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); framework(a.out); print('saved', a.out)


if __name__ == '__main__':
    main()
