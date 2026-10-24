"""Render paired HCC retraining metrics and differences relative to the retrained paper method."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); data=json.loads(a.input.read_text()); rows=[r for r in data['rows'] if 'error' not in r]
    by=defaultdict(list)
    for r in rows: by[(r['method'],int(r['difficulty']))].append(r)
    methods=sorted({r['method'] for r in rows}); diffs=range(4)
    def mean(m,d,k,default=0.):
        v=[float(r.get(k,default)) for r in by[(m,d)]]
        return float(np.mean(v)) if v else None
    def rate(m,d,k):
        v=[bool(r.get(k,False)) for r in by[(m,d)]]
        return float(np.mean(v)) if v else None
    lines=['# HCC 新构造重新训练后导航复测','',f"输入：`{a.input}`。所有学习方法均使用 HCC 专用训练结果；经典方法沿用原实现。",'',
           '每个难度使用相同 paired scene seeds，难度 0/1/2/3 的动态障碍为 4/8/16/32。', '']
    for d in diffs:
        lines += [f'## 难度 {d}', '', '| 算法 | 成功率 | Strict成功率 | 壁接触 robot·s | 丢失率 | 碰撞回合率 | 完成时间 s |', '|---|---:|---:|---:|---:|---:|---:|']
        for m in methods:
            t=[r for r in by[(m,d)] if r.get('navigation_time_s') is not None]
            tm=float(np.mean([r['navigation_time_s'] for r in t])) if t else None
            lines.append(f"| {m} | {rate(m,d,'navigation_success')*100:.1f}% | {rate(m,d,'strict_navigation_success')*100:.1f}% | {mean(m,d,'wall_contact_s'):.2f} | {rate(m,d,'lost')*100:.1f}% | {rate(m,d,'particle_collision_events')*100:.1f}% | {'—' if tm is None else f'{tm:.1f}'} |")
        lines.append('')
    if 'ours_hcc' in methods:
        lines += ['## 相对重新训练后的本文方法差值','', '成功率/Strict 以百分点计；壁接触、丢失、碰撞、完成时间为“本文方法减对比方法”，正值表示本文方法更优。','', '| 难度 | 对比算法 | 成功率差 pp | Strict差 pp | 壁接触减少 | 丢失率差 pp | 碰撞率差 pp | 完成时间减少 s |','|---:|---|---:|---:|---:|---:|---:|---:|']
        for d in diffs:
            for m in methods:
                if m=='ours_hcc': continue
                vals=[rate('ours_hcc',d,'navigation_success'),rate(m,d,'navigation_success'),rate('ours_hcc',d,'strict_navigation_success'),rate(m,d,'strict_navigation_success'),mean('ours_hcc',d,'wall_contact_s'),mean(m,d,'wall_contact_s'),rate('ours_hcc',d,'lost'),rate(m,d,'lost'),rate('ours_hcc',d,'particle_collision_events'),rate(m,d,'particle_collision_events')]
                if any(v is None for v in vals): continue
                ta=[r['navigation_time_s'] for r in by[('ours_hcc',d)] if r.get('navigation_time_s') is not None]; tb=[r['navigation_time_s'] for r in by[(m,d)] if r.get('navigation_time_s') is not None]
                dt=(float(np.mean(ta))-float(np.mean(tb))) if ta and tb else None
                lines.append(f'| {d} | {m} | {(vals[0]-vals[1])*100:+.1f} | {(vals[2]-vals[3])*100:+.1f} | {vals[4]-vals[5]:+.2f} | {(vals[6]-vals[7])*100:+.1f} | {(vals[8]-vals[9])*100:+.1f} | {"—" if dt is None else f"{dt:+.1f}"} |')
    lines += ['', '## 训练与解释边界', '', '- IPPO：HCC 专用、semantic-reduced、zero latency、500,000 physical steps，随机采样四个难度。', '- 本文方法：HCC 专用 PPO selector，四难度轮换训练，冻结预测器和安全层保持不变。', '- 这些结果是 procedural nominal HCC construct 的导航仿真，不是患者特异性临床验证。']
    a.out.write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
