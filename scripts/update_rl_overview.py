"""Write a factual Chinese handoff and desktop index from saved experiment data."""
import html
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path('/home/wj/桌面')
EXPERIMENTS = [
    ('EXP0099_FINITE_IMAGE_SAC_20261011', '有限时域图像 SAC：critic 预训练与真实策略更新', 'VRL_有限时域图像SAC_训练测试_3D_GIF_20261011'),
    ('EXP0098_ENVELOPED_IMAGE_SAC_20261011', '分阶段图像/历史 SAC：小幅导航适配与近血栓控制', 'VRL_分阶段图像速度SAC_训练测试_3D_GIF_20261011'),
    ('EXP0097_SPEED_IMAGE_SAC_20261011', '单维图像/历史 SAC', 'VRL_单维图像速度SAC_训练测试_3D_GIF_20261011'),
    ('EXP0096_SAFE_IMAGE_SAC_20261011', '三维图像 SAC：恢复原参考与等待/停车', 'VRL_安全图像SAC_训练测试_3D_GIF_20261011'),
    ('EXP0095_IMAGE_FLOW_SAC_20261011', '初版图像/血流 SAC：未达到共同改善', 'VRL_图像血流SAC_训练测试_3D_GIF_20261011'),
    ('EXP0094_PPO_NAV_20261011', '类别 PPO：完整困难矩阵与原方法同分', 'VRL_RL_局部导航_训练测试_GIF_20261011'),
]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def episode_count(path):
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def main():
    from scripts.md_to_docx import convert
    stamp = time.strftime('%F %T')
    lines = [
        '# VRL：高层动作覆盖、图像强化学习与实际实验记录', '',
        f'根据 V3 初稿继续实现与验证，更新于 {stamp}。原 V3 DOCX 保留。本文件只报告实际已有记录；尚未完成的矩阵和媒体不算完成。', '',
        '## 高层是否强行终止行为', '',
        'TPG 在通行顺序不满足时把局部命令清零，旧类别 PPO 在等待步被跳过。因此局部行为确实可能被高层覆盖。高层没有直接结束整个回合；回合结束来自任务完成、机器人丢失或时间上限。', '',
        '30 个诊断场景的 73041 个有效机器人控制步中，TPG 等待 77 步（0.1054%），策略后壁护改写约 2.99%，间距改写为 0。该平均值不能排除个别场景的长等待。对严重颈内动脉场景继续审计得到 2939 个等待步，退休/终点区前置任务导致的等待均为 0；前置导航尚未完成触发等待，不能写成已完成机器人把高层卡死。', '',
        '## 真实 RL 的实现和控制范围', '',
        '原方法的最高成功率组合使用监督预测器和固定代价选择，不能称为 RL。类别 PPO 已进行真实环境训练，但经固定候选筛选与安全执行后，实际动作变化很少，原 252 个困难场景的 Strict 成功率仍为 63.1%。', '',
        '图像版本使用当前/上一帧的延迟双平面裁剪（4×32×32）、24×11 运动/已发命令历史和 48 维地图/预测/运动上下文；CNN、GRU、双 Q、目标网络、经验回放和重参数化 actor 更新构成 SAC。连续真实血流、心动相位、机器人位置与剩余质量不直接作为 actor 特征；状态标志的仿真假设如下。当前是单训练种子，全部结果来自开发验证池，未访问封存测试。', '',
        '观测审计补充：上述“质量不进入 actor”指连续剩余质量。现有 ImageSensor 直接以 env.masses > 0 生成血栓存在标志，并由 env.active 提供机器人有效状态；这些标志参与任务分配和 actor 上下文，属于共同的仿真观测假设，没有经过真实图像分割验证。初始定位模拟操作者标注，后续运动估计来自图像检测。裁剪主要显示机器人运动，不是血栓分割图。性能比较须在这个假设下解释，不能声称已实现完全从图像恢复全部任务状态。', '',
        '一维版本保留可靠参考方向和原 TPG/停车逻辑，actor 与双 Q 的动作均为一维。RL 通过参考速度的对数赔率偏移学习加速、刹车和驻留；零输出精确恢复原方法，满速仍可减速。它学习沿参考方向的速度适配，不独立学习路线几何。未把非 RL 安全修正归因于策略学习。', '',
        '当前仿真表示接触溶栓，没有独立机械碎栓模型。因此清栓效果按清除质量、接触比例和 T90 解释，不能声称已经验证机械破栓。N 是机器人集群数，不是血栓数，单机器人场景也可能含多个血栓。', '',
        '## 各次训练和完整结果', '',
    ]
    cards = []
    for code, title, media_name in EXPERIMENTS:
        validation = ROOT/'research/validation'/code
        run = ROOT/'research/runs'/code/'seed0'
        complete = read(run/'TRAINING_COMPLETE.json')
        state = read(validation/'CURRENT_STATUS.json')
        media = DESKTOP/media_name
        progress = read(media/'progress.json')
        lines += ['### '+title, '']
        if complete:
            updates=f"{complete.get('gradient_updates', '见日志')} 次梯度更新"
            if 'actor_gradient_updates' in complete:
                updates=f"{complete['gradient_updates']} 次价值网络更新、{complete['actor_gradient_updates']} 次策略更新"
            lines += [f"实际训练：{complete.get('iterations', '见日志')} 轮，{updates}，{complete.get('agent_steps', '见日志')} 次智能体转移。", '']
        else:
            lines += ['训练完成状态以该 run 的完成文件和实际日志为准。', '']
        lines += [f'结果目录：{validation}', '']
        for folder, caption in [
            ('tune_report', '训练解剖配对选择'),
            ('tune_speed_projection_report', '冻结三维 actor 的速度投影；不是一维训练'),
            ('full_hard_report', '原论文完整困难矩阵（已知开发基准）'),
            ('full_report', '类别 PPO 原论文完整困难矩阵'),
            ('stress_report', '标准/强流/脉动/组合场景'),
            ('v5_report', '弱流 V5 全场景'),
        ]:
            summary = read(validation/folder/'summary.json')
            if not summary:
                continue
            lines += [caption, '', '| 方法 | n | Strict % | 清除 % | 壁 robot·s | 路线 RMSE mm |', '|---|---:|---:|---:|---:|---:|']
            for name, row in summary['results'].items():
                route = row.get('route_cross_track_rmse_mm')
                route = '—' if route is None else f'{route:.4f}'
                lines.append(f"| {name} | {row['n']} | {100*row['strict_success']:.2f} | {100*row['removal']:.2f} | {row['wall_contact_s']:.3f} | {route} |")
            lines += ['', '按血流条件的真实分组见该报告 by_condition.csv；所有失败回合保留。', '']
        if (validation/'NO_JOINT_CANDIDATE.json').exists():
            lines += ['训练解剖上没有同时满足成功/清除不下降、壁接触/路线偏离不增加的候选；此版本不作为共同改善的方法。', '']
        if code.startswith('EXP0095'):
            lines += ['该中间版本完整困难矩阵 Strict 提升，但整体壁接触与路线误差退化；未实现共同改善。已记录的全部训练/困难/强流脉动回合动画归档。', '']
        state_text = state.get('stage', '见实际日志和目录进度')
        if code.startswith('EXP0099'):
            tune_count = episode_count(validation/'tune/episodes.jsonl')
            stages = [('tune', 972, '训练解剖模型选择'), ('full_hard', 1008, '困难矩阵配对复测'),
                      ('stress', 672, '强流与脉动配对复测'),
                      ('ablation_no_images', 168, '图像裁剪消融'),
                      ('ablation_no_pulse', 168, '脉动估计消融'),
                      ('ablation_contact_range', 168, '近目标动作范围消融'),
                      ('v5', 2520, '弱流 V5 配对复测')]
            coverage = [f'{caption}：{episode_count(validation/folder/"episodes.jsonl")} / {expected}'
                        for folder, expected, caption in stages]
            lines += ['当前完整回合覆盖：'+'；'.join(coverage)+'。', '']
            if tune_count < 972:
                state_text = f'实际 RL 训练完成；训练解剖模型选择 {tune_count} / 972，尚未冻结最终部署模式。'
            regression = read(validation/'replay_verification/VERIFIED.json')
            if regression:
                r = regression['results'][0]
                lines += [f"已知 MCA 回归案例：{r['clusters']} 个机器人、{r['clots']} 个血栓，"
                          f"清除 {100*r['removal']:.1f}%、壁接触 {r['wall_contact_s']:.3f} robot·s、"
                          f"完成 {r['elapsed_s']:.1f} s。两次独立回放及控制归因开关的所有记录数组逐项相等；"
                          '这只证明该已知案例可复现，不作为模型选择依据或全场景改善证据。', '']
            delivery = read(validation/'DELIVERY_VERIFIED.json')
            multi_execution = read(validation/'execution_verification_N3/VERIFIED.json')
            if multi_execution.get('attribution_does_not_change_physics'):
                lines += ['冻结策略在固定三机器人 MCA 场景上，开启/关闭动作归因的所有物理与控制数组逐项相等；'
                          '该检查包含间距过滤层。它验证测量不改变控制过程，不作为模型选择依据。', '']
            if delivery.get('complete'):
                lines += [f"最终来源与覆盖审计通过：{delivery['media_verified']} 份动画逐一验证；"
                          '场景、轨迹、GIF 来源与冻结模型均有核验记录。', '']
        if state.get('full_matrix_executed') is False:
            state_text='未共同达标：已记录训练/诊断动画归档，未执行本版完整测试矩阵。'
        if progress:
            counts = f"已生成 {progress['completed']} / {progress['expected']} 份；剩余 {progress['pending']} 份"
            lines += [f'动画：{media}。{counts}。', '']
        else:
            counts = '尚无媒体完成记录'
        if (media/'index.html').exists():
            target = media_name+'/index.html'
            link = f'<a href="{html.escape(target)}">{html.escape(title)}</a>'
        else:
            link = html.escape(title)
        extra_link = ''
        if code.startswith('EXP0099'):
            tasks = read(media/'tasks.json') or []
            replay = next((t for t in tasks if t['section']=='07_随机策略复现校验'
                           and (media/t['gif']).exists() and (media/t['gif']).with_suffix('.json').exists()), None)
            if replay:
                extra_link = '<br><a href="'+html.escape(media_name+'/'+replay['gif'])+'">已验证的单案例 3D 回放（不是整体结论）</a>'
        cards.append(f'<li>{link}<br>{html.escape(counts)}<br>{html.escape(state_text)}{extra_link}</li>')
    old = DESKTOP/'VRL_V3_全部场景_3D_GIF_20261011'
    progress = read(old/'progress.json')
    counts = f"已生成 {progress.get('completed', 0)} / {progress.get('expected', 0)} 份，剩余 {progress.get('pending', 0)} 份"
    cards.append('<li><a href="VRL_V3_全部场景_3D_GIF_20261011/index.html">原 V3、其他论文方法、经典基线与安全层消融</a><br>'+counts+'<br>经典控制方法没有 RL 训练回合；未记录原始 RL 训练的项目标为训练阶段模型回放。</li>')
    lines += ['## 动画来源与完整性', '',
              '每份动画对应一个完整实际仿真回合，相邻 NPZ、JSON 保存来源、轨迹和指标。训练 GIF 对应在线策略持续更新的回合；冻结检查点回放不冒充原始训练。完整状态以各目录 progress.json / COMPLETE.json 为准。', '',
              '原 V3 与其他论文方法目录：'+str(old)+'。'+counts+'。', '',
              '桌面总入口：'+str(DESKTOP/'VRL_全部算法_训练测试_3D总索引_20261011.html')+'。']
    doc = ROOT/'VRL_RL_高层审计与各版真实实验记录_20261011.md'
    doc.write_text('\n'.join(lines)+'\n')
    convert(doc.read_text(), doc.with_suffix('.docx'), src_dir=ROOT)
    report_rel = 'vascular_marl_local.tar./'+doc.with_suffix('.docx').name
    page = '<!doctype html><meta charset="utf-8"><title>VRL 所有算法训练与测试 3D GIF</title><style>body{font:17px sans-serif;max-width:1100px;margin:45px auto;background:#f6f8fc;color:#17304c}a{color:#125fa7}li{margin:22px 0;line-height:1.7}</style><h1>VRL 所有算法：实际训练及全场景 3D GIF</h1>'
    page += '<p>更新 '+stamp+'。成功、失败与尚未完成的批次均按实际记录标明。打开各目录查看场景索引和最新进度。</p><ul>'+''.join(cards)+'</ul>'
    page += '<p><a href="'+html.escape(report_rel)+'">中文高层审计与各版真实实验记录 DOCX</a></p><p>原 V3 初稿保留；当前物理模型为接触溶栓。所有开发复测均不称为封存测试。</p>'
    (DESKTOP/'VRL_全部算法_训练测试_3D总索引_20261011.html').write_text(page)
    print(doc.with_suffix('.docx'))


if __name__ == '__main__':
    main()
