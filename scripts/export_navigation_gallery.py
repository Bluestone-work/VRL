"""Frozen IPPO navigation replays: 14 anatomies x three flows, visualization only.

Independent illustration seeds never access the registered test manifest.
True positions are recorded after execution for rendering, never fed to PPO.
Every predeclared scene (including failures) is retained with a raw trace.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT = Path('/home/wj/桌面/VRL_导航3D_GIF_14血管_20261018')
FLOW_NAMES = {'weak': '弱流', 'strong': '强流', 'pulsatile': '脉动流'}
DELIVERY_REFERENCE = Path('/home/wj/桌面/新的构造.jpg')


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temp.replace(path)


def catalogue():
    cfg = json.loads((ROOT / 'configs/navigation_dataset_v1.json').read_text())
    anatomies = cfg['train_anatomies'] + cfg['holdout_test_anatomies']
    return [dict(index=3*i+j, anatomy=an, flow_mode=mode, **flow,
                 seed=3908000000+i*1000, clusters=3, obstacles=16,
                 stem=f'{an}_{mode}', illustration_only=True,
                 anatomy_group=('registered_holdout' if an in cfg['holdout_test_anatomies'] else 'registered_train'))
            for i, an in enumerate(anatomies) for j, (mode, flow) in enumerate(cfg['flow_modes'].items())]


def replay(task, model_path, out):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from scripts.train_sb3_ippo import IPPOAgentVecEnv
    from scripts.train_sb3_curriculum import CurriculumVascularEnv
    from scripts.export_vrl_v3_all_gifs import geometry

    torch.set_num_threads(1)
    env = IPPOAgentVecEnv(CurriculumVascularEnv(
        seed=41, clusters=3, curriculum=False, reward_mode='navigation',
        zero_latency=True, direct_action=True), action_set='semantic')
    env.physical.train_anatomies = [task['anatomy']]
    model = PPO.load(model_path, device='cpu')
    options = dict(anatomy=task['anatomy'], scene_seed=task['seed'],
                   flow=task['nominal_flow_inlet_mm_s'], latency=0, difficulty=2,
                   dynamic_obstacle_count=16, clot_initialization='historical_sites',
                   scenario_family='route_encounter',
                   variation=dict(s=task['variation_strength'], pulsatile=task['pulsatile']))
    rec = {k: [] for k in ('t', 'pos', 'particles', 'active', 'particle_active',
                           'reached', 'command', 'wall', 'collisions', 'actions')}
    wall = collisions = 0.

    def capture(actions):
        physical = env.physical
        e = physical.ep.env
        values = dict(t=float(e.elapsed_s), pos=e.positions_mm[:3].copy(),
                      particles=e.positions_mm[3:].copy(), active=e.active[:3].copy(),
                      particle_active=e.active[3:].copy(), reached=physical.nav_reached.copy(),
                      command=physical.ep.sent_world.copy(), wall=wall,
                      collisions=collisions, actions=actions)
        for key, value in values.items():
            rec[key].append(value)

    try:
        obs = env.reset(options=options)
        geo = geometry(env.physical.ep)
        scene = dict(env.physical.scene)
        targets = env.physical.nav_targets.tolist()
        capture(np.full(3, -1, dtype=int))
        for step in range(3001):
            actions, _ = model.predict(obs, deterministic=True)
            # Same candidate-to-command mapping as IPPOAgentVecEnv.step_wait.
            # Invalid duplicates have identical commands to their canonical slot.
            joint = env._candidate_commands[np.arange(3), actions].reshape(-1)
            raw, _, terminated, truncated, info = env.physical.step(joint)
            wall += float(np.sum(info.get('agent_wall_contact_s', 0.)))
            collisions += float(np.sum(info.get('agent_particle_collision_events', 0.)))
            capture(actions.copy())
            if (step+1) % 500 == 0:
                print(f'{task["stem"]}: simulated {rec["t"][-1]:.1f}s / 300s', flush=True)
            if terminated or truncated:
                break
            obs = env._with_candidates(raw)
        else:
            raise RuntimeError('Episode exceeded original 300 s horizon')
        metrics = info['episode_metrics']
        metadata = dict(task=task, task_mode='targeted_drug_delivery',
                        reference_image=str(DELIVERY_REFERENCE),
                        dynamics_unchanged=True, reset_scene=scene, targets=targets, metrics=metrics,
                        model=str(model_path), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        policy='IPPO_s41 final, K18 semantic, deterministic; not a new trained method',
                        training_coverage='Observed training episodes cover mca_m1_lvo only.',
                        scope='Independent illustration; not a sealed test or a statistical comparison.',
                        truth_usage='True positions only for post-execution visualization, not policy input.',
                        pairing='Same requested seed across flows; actual accepted seeds and geometry may differ.',
                        strict_definition='Navigation success and cumulative wall contact < 1 robot.s; not collision-free success.')
        arrays = {k: np.asarray(v) for k, v in rec.items()}
        np.savez_compressed(out / (task['stem']+'.npz'), **arrays,
                            geometry=json.dumps(geo), metadata=json.dumps(metadata))
        write_json(out / (task['stem']+'.json'), metadata)
        return arrays, geo, metadata
    finally:
        env.close()


def render(task, rec, geo, metadata, out, frame_count):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from scripts.export_vrl_v3_all_gifs import COLORS, FONT
    from scripts.render_lysis_paper import Scene
    from types import SimpleNamespace
    import shutil

    # Fit the complete lumen envelope, not just its centreline span. Centre
    # the projected bounds as well, otherwise long vessels clip at the edge.
    scene = Scene(geo, 900, 540, vessel_alpha=.32)
    try:
        points = np.asarray(geo['points'])
        radius = np.asarray(geo['radius'])[:, None]
        envelope = np.concatenate([points+radius*sign for sign in
                                  np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])])
        scene.camera(points, yaw=35, pitch=-35)
        h = np.c_[envelope,np.ones(len(envelope))] @ scene.M.T
        ndc = h[:,:2]/h[:,3:4]
        center = (ndc.min(0)+ndc.max(0))/2
        scale = min(1.8/max(np.ptp(ndc[:,0]),1e-6), 1.7/max(np.ptp(ndc[:,1]),1e-6))
        projection = np.asarray(scene.proj).reshape(4,4).T.copy()
        for axis in (0,1):
            projection[axis] = scale*projection[axis]-scale*center[axis]*projection[3]
        scene.proj = projection.T.ravel().tolist()
        scene.M = projection @ np.asarray(scene.view).reshape(4,4).T
        matrix = scene.M.copy()
        base = Image.new('RGB',(900,700),'#e3f5f6')
        base.paste(Image.fromarray(scene.render([],np.zeros(len(geo['clots'])))),(0,105))
        axes_draw = ImageDraw.Draw(base)
        rotation = np.asarray(scene.view).reshape(4,4).T[:3,:3]
        for axis, color, label in zip(np.eye(3), ('#c33d3d','#359953','#3c69cf'), ('X','Y','Z')):
            tip = np.array([55,600])+28*np.array([rotation[0]@axis,-rotation[1]@axis])
            axes_draw.line([(55,600),tuple(tip)],fill=color,width=2)
            axes_draw.text(tuple(tip),label,fill=color)
        def project(p):
            p = np.asarray(p)
            shape = p.shape[:-1]
            flat = p.reshape(-1,3)
            h = np.c_[flat,np.ones(len(flat))] @ matrix.T
            q = h[:,:2]/h[:,3:4]
            return np.stack(((q[:,0]+1)*450,(1-q[:,1])*270+105),axis=-1).reshape(*shape,2)
        view = SimpleNamespace(base=base,project=project)
    finally:
        scene.close()
        shutil.rmtree(scene.tmp)
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 15)
    context_thumb = None
    if DELIVERY_REFERENCE.exists():
        with Image.open(DELIVERY_REFERENCE) as ref:
            context_thumb = ImageOps.contain(ref.convert('RGB'), (112, 72))
    indices = np.unique(np.linspace(0, len(rec['t'])-1, min(frame_count, len(rec['t']))).astype(int))
    frames = []
    for k in indices:
        im = view.base.copy()
        draw = ImageDraw.Draw(im)
        draw.rectangle((0, 0, 900, 103), fill='#c8ebee')
        draw.text((18, 8), f"IPPO 多集群 3D 递药导航 | {task['anatomy']} | {FLOW_NAMES[task['flow_mode']]}", font=font, fill='#123b55')
        draw.text((18, 35), f"N=3 · 16个动态障碍 · 0延迟 · 名义入口流速 {task['nominal_flow_inlet_mm_s']} mm/s · HCC靶向递药示意", font=small, fill='#28566b')
        status = '运行中'
        if k == len(rec['t'])-1:
            status = '导航成功' if metadata['metrics']['navigation_success'] else '导航未成功'
        reached = int(rec['reached'][k].sum())
        draw.text((18, 59), f"t={rec['t'][k]:.1f}s / 300s · 已抵达靶点 {reached}/3 · 已释放药物 {reached}/3 · {status} · 丢失 {int((~rec['active'][k]).sum())}", font=small, fill='#28566b')
        draw.text((18, 81), '多集群在三维血管内导航至靶点并释放药物；独立可视化场景，失败回放保留', font=small, fill='#476c78')
        if context_thumb is not None:
            draw.rounded_rectangle((774, 6, 892, 82), radius=5, fill='#f4ffff', outline='#73b8bd', width=1)
            im.paste(context_thumb, (777, 8))
        # Physical trajectories and moving obstacles share the 3-D vessel camera.
        for i in range(3):
            uv = view.project(rec['pos'][:k+1:max(1, (k+1)//500), i])
            if len(uv) > 1:
                draw.line([tuple(p) for p in uv], fill=COLORS[i], width=2)
        for p, active in zip(rec['particles'][k], rec['particle_active'][k]):
            if active:
                x, y = view.project(p)
                draw.ellipse((x-4,y-4,x+4,y+4), fill='#7b8495', outline='white')
        target_released = np.zeros(len(geo['clots']), dtype=bool)
        for i, target in enumerate(metadata['targets']):
            if 0 <= int(target) < len(target_released) and rec['reached'][k, i]:
                target_released[int(target)] = True
        for j, p in enumerate(geo['clots']):
            x, y = view.project(np.asarray(p))
            released = bool(target_released[j])
            color = '#1eaa8a' if released else '#d64b63'
            draw.ellipse((x-7,y-7,x+7,y+7), fill=color, outline='white', width=2)
            draw.ellipse((x-13,y-13,x+13,y+13), outline='#44d7b5' if released else '#f19aab', width=2)
            draw.text((x+9,y+4), f'T{j+1}' + (' 已递药' if released else ' 靶点'), font=small, fill='#16756a' if released else '#a72c54')
            if released:
                for dx, dy in ((-10, -7), (11, -5), (-7, 10), (8, 9)):
                    draw.ellipse((x+dx-2, y+dy-2, x+dx+2, y+dy+2), fill='#f4bd3f', outline='white')
        for i, p in enumerate(rec['pos'][k]):
            x, y = view.project(p)
            c = COLORS[i] if rec['active'][k, i] else '#333333'
            draw.ellipse((x-8,y-8,x+8,y+8), fill=c, outline='white', width=2)
            if rec['reached'][k, i]:
                draw.ellipse((x-11,y-11,x+11,y+11), outline=COLORS[i], width=2)
            draw.text((x+10,y-15), f'R{i+1}', font=small, fill=c)
        draw.rectangle((0, 645, 900, 700), fill='#c8ebee')
        targets = '   '.join(f'R{i+1}→T{t+1}' for i,t in enumerate(metadata['targets']))
        draw.text((18, 648), f'蓝/绿/橙：集群与轨迹 · 金色：药物释放 · 红/青：递药靶点/已完成 | {targets}', font=small, fill='#28566b')
        speed = rec['t'][-1] / max(.1, .08*(len(indices)-1))
        draw.text((18, 674), f'真实3D几何透视投影 · 标记放大 · 约{speed:.1f}倍速 · 碰撞事件 {int(rec["collisions"][k])} · seed={task["seed"]}', font=small, fill='#28566b')
        if k == indices[-1]:
            im.save(out / (task['stem']+'.png'))
        frames.append(im.quantize(colors=128, method=Image.Quantize.FASTOCTREE))
    destination = out / (task['stem']+'.gif')
    tmp = destination.with_suffix('.tmp.gif')
    frames[0].save(tmp, save_all=True, append_images=frames[1:],
                   duration=[80]*(len(frames)-1)+[1800], loop=0, disposal=2, optimize=False)
    with Image.open(tmp) as check:
        assert check.n_frames >= 2
        check.seek(check.n_frames-1)
        check.load()
    tmp.replace(destination)


def one(task, args):
    import numpy as np
    directory = args.out / task['anatomy']
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / (task['stem']+'.npz')
    if trace.exists():
        with np.load(trace) as z:
            rec = {k:z[k] for k in z.files if k not in ('geometry', 'metadata')}
            geo, metadata = json.loads(str(z['geometry'])), json.loads(str(z['metadata']))
        if metadata['model_sha256'] != hashlib.sha256(args.model.read_bytes()).hexdigest():
            raise RuntimeError('Cached trace belongs to a different checkpoint')
    else:
        rec, geo, metadata = replay(task, args.model, directory)
    render(task, rec, geo, metadata, directory, args.frames)
    print(json.dumps(dict(scene=task['stem'], navigation_success=metadata['metrics']['navigation_success']), ensure_ascii=False), flush=True)


def gallery(tasks, args):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
    montage = Image.new('RGB', (1080, 90+280*14), 'white')
    draw = ImageDraw.Draw(montage)
    draw.text((15, 8), 'IPPO多集群3D递药导航：14种血管 × 弱流 / 强流 / 脉动流', font=font, fill='#123b55')
    draw.text((15, 38), '目标：导航至分布式靶点并释放药物；完整动态GIF见 index.html；成功与失败均保留', font=font, fill='#28566b')
    sections = []
    records = []
    for task in tasks:
        relative = f"{task['anatomy']}/{task['stem']}"
        gif = args.out / (relative+'.gif')
        if not gif.exists():
            continue
        meta = json.loads((args.out/(relative+'.json')).read_text())
        m = meta['metrics']
        records.append(dict(**task, navigation_success=m['navigation_success'], strict_success=m['strict_success'],
                            wall_contact_s=m['wall_contact_s'], gif=relative+'.gif'))
        img = Image.open(args.out/(relative+'.png')).convert('RGB').resize((360,280))
        montage.paste(img, ((task['index']%3)*360,90+(task['index']//3)*280))
        sections.append(f'<article><h3>{html.escape(task["anatomy"])} · {FLOW_NAMES[task["flow_mode"]]}</h3>'
                        f'<a href="{relative}.gif"><img loading="lazy" src="{relative}.gif"></a>'
                        f'<p>导航：{"成功" if m["navigation_success"] else "未成功"} · 壁接触 {m["wall_contact_s"]:.2f} robot·s '
                        f'· <a href="{relative}.json">场景与指标</a> · <a href="{relative}.npz">原始轨迹</a></p></article>')
    montage.save(args.out/'overview_14x3.jpg', quality=92)
    write_json(args.out/'results.json', records)
    page = '<!doctype html><meta charset="utf-8"><title>VRL 3D导航回放</title><style>body{font:16px sans-serif;background:#f4f7fb;color:#172c49;padding:24px}main{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:12px}article{background:white;padding:12px;border-radius:12px}img{width:100%}h3{font-size:15px}p{line-height:1.6}@media(max-width:1000px){main{grid-template-columns:1fr}}</style>'
    page += '<h1>14种血管 × 三种血流：IPPO 多集群 3D 递药导航</h1><p>固定IPPO_s41最终模型；N=3；16个动态障碍；0延迟。多集群沿三维血管导航至分布式靶点，并在到达后释放药物；红色为待递药靶点，青绿色为已完成递药，金色粒子表示释放药物。使用独立可视化种子，全部预定场景保留，无成功筛选。模型实际仅训练过mca_m1_lvo；这些示意不能作为封闭测试或多种子性能结论。真实位置仅用于渲染。<br>三列依次为弱流0.025、强流0.1、脉动流名义0.05 mm/s。<a href="新的构造.jpg">递药背景参考图</a> · <a href="overview_14x3.jpg">静态总览</a></p><main>'
    (args.out/'index.html').write_text(page+''.join(sections)+'</main>')
    if DELIVERY_REFERENCE.exists():
        import shutil
        shutil.copy2(DELIVERY_REFERENCE, args.out/'新的构造.jpg')
        (args.out/'DELIVERY_CONTEXT.json').write_text(json.dumps({
            'reference_image': str(DELIVERY_REFERENCE),
            'task': 'multi-cluster 3D vascular navigation and targeted drug delivery',
            'render_only_change': True,
            'dynamics_unchanged': True,
        }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--model', type=Path, default=ROOT/'research/runs/NAVIGATION_V1/IPPO_s41/final_model.zip')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--frames', type=int, default=180)
    parser.add_argument('--job', type=int)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--render-only', action='store_true', help='Regenerate GIFs from cached traces; no rollout')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = catalogue()
    if args.job is not None:
        one(tasks[args.job], args)
        return
    write_json(args.out/'manifest.json', dict(tasks=tasks, seed_namespace=3908000000,
               purpose='illustration_only_not_sealed_test', model=str(args.model),
               model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
               fixed_conditions='N=3, historical_sites, 16 route_encounter obstacles, zero latency'))
    chosen = tasks[:args.limit] if args.limit else tasks
    if args.render_only:
        for task in chosen:
            if not (args.out/task['anatomy']/(task['stem']+'.npz')).exists():
                raise SystemExit('render-only requires all selected raw traces')
    def run(task):
        gif = args.out/task['anatomy']/(task['stem']+'.gif')
        if gif.exists() and not args.render_only:
            return task['index'], 0
        cmd = [sys.executable, str(Path(__file__).resolve()), '--job', str(task['index']),
               '--out', str(args.out), '--model', str(args.model), '--frames', str(args.frames)]
        with (args.out/(task['stem']+'.log')).open('a') as log:
            for attempt in range(2):
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode == 0:
                    break
        return task['index'], result.returncode
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, task) for task in chosen]
        for num, future in enumerate(as_completed(futures), 1):
            idx, code = future.result()
            print(f'{num}/{len(chosen)} {tasks[idx]["stem"]} exit={code}', flush=True)
            if code:
                failures.append(idx)
    gallery(tasks, args)
    write_json(args.out/'completion.json', dict(requested=len(chosen), failed=failures))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
