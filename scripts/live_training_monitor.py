#!/usr/bin/env python3
"""Serve a live, browser-based monitor for a vascular RL run.

The monitor is intentionally independent from the training process.  It reads
status.json, episodes.jsonl and progress.csv, so it can be run in a second
NoMachine terminal while training continues.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vascular RL live monitor</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#121d30;--line:#263754;--text:#e7eef9;--muted:#91a4bf;--blue:#57b7ff;--green:#50d890;--red:#ff6b7a;--amber:#f4c95d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1500px;margin:0 auto;padding:20px}h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 10px}.sub{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:10px;margin:18px 0}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px}.card{padding:12px}.label{color:var(--muted);font-size:12px}.value{font-size:22px;font-variant-numeric:tabular-nums;margin-top:5px}.good{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--amber)}.layout{display:grid;grid-template-columns:1.35fr 1fr;gap:12px}.panel{padding:14px;min-height:250px}.wide{grid-column:1/-1}canvas{width:100%;height:260px;display:block;background:#0d1728;border-radius:7px}.scene{display:grid;grid-template-columns:1fr 1fr;gap:7px}.scene div{padding:8px;background:#0d1728;border-radius:6px}.scene b{display:block;color:var(--muted);font-size:11px;font-weight:500}.scene span{display:block;margin-top:3px;overflow-wrap:anywhere}.tablewrap{max-height:340px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;padding:7px 5px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left}th{position:sticky;top:0;background:var(--panel);color:var(--muted);font-weight:500}.pill{padding:2px 6px;border-radius:99px;font-size:11px}.pill.good{background:#123c2a}.pill.bad{background:#4a1e2a}.pill.neutral{background:#26344b}.footer{margin-top:12px;color:var(--muted);font-size:12px}@media(max-width:900px){.grid{grid-template-columns:repeat(3,1fr)}.layout{grid-template-columns:1fr}}@media(max-width:560px){.grid{grid-template-columns:repeat(2,1fr)}main{padding:12px}}
</style>
</head>
<body><main>
<h1>血管强化学习实时监控</h1><div class="sub" id="subtitle">正在读取训练目录…</div>
<section class="grid" aria-label="训练摘要">
<div class="card"><div class="label">环境步数</div><div class="value" id="steps">—</div></div>
<div class="card"><div class="label">训练进度</div><div class="value" id="progress">—</div></div>
<div class="card"><div class="label">最近成功率</div><div class="value" id="success">—</div></div>
<div class="card"><div class="label">最近平均回报</div><div class="value" id="ret">—</div></div>
<div class="card"><div class="label">最近丢失率</div><div class="value" id="lost">—</div></div>
<div class="card"><div class="label">运行状态</div><div class="value" id="runstate">—</div></div>
</section>
<section class="layout">
<div class="panel"><h2>成功率、回报与丢失</h2><canvas id="metrics" width="900" height="300" aria-label="最近训练回合的成功率、回报和丢失曲线"></canvas></div>
<div class="panel"><h2>当前场景 / 物理条件</h2><div class="scene" id="scene"></div></div>
<div class="panel wide"><h2>实时血管轨迹（x-y 投影）</h2><canvas id="trajectory" width="1400" height="360" aria-label="当前血管中心线、机器人位置、目标点和血流方向"></canvas></div>
<div class="panel wide"><h2>最近回合</h2><div class="tablewrap"><table><thead><tr><th>回合</th><th>解剖</th><th>难度</th><th>流速</th><th>回报</th><th>成功</th><th>丢失</th><th>碰壁(s)</th><th>时长(s)</th></tr></thead><tbody id="episodes"></tbody></table></div></div>
<div class="panel wide"><h2>训练指标</h2><canvas id="training" width="1400" height="280" aria-label="PPO 训练损失、近似 KL 和成功率"></canvas></div>
</section><div class="footer" id="updated">—</div>
</main>
<script>
const $=id=>document.getElementById(id); const fmt=(x,d=2)=>x==null||Number.isNaN(Number(x))?'—':Number(x).toFixed(d); const esc=s=>String(s??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function setText(id,v,cls=''){const e=$(id);e.textContent=v;e.className='value '+cls}
function draw(id,series,colors,labels){const c=$(id),ctx=c.getContext('2d'),w=c.width=c.clientWidth*devicePixelRatio,h=c.height=c.clientHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);const W=c.clientWidth,H=c.clientHeight;ctx.fillStyle='#0d1728';ctx.fillRect(0,0,W,H);const pad={l:42,r:14,t:18,b:28};const all=series.flatMap(s=>s.v.filter(Number.isFinite));if(!all.length){ctx.fillStyle='#91a4bf';ctx.fillText('等待训练数据…',pad.l,40);return}let lo=Math.min(...all),hi=Math.max(...all);if(lo===hi){lo-=1;hi+=1}const x=i=>pad.l+(W-pad.l-pad.r)*(i/Math.max(1,series[0].v.length-1));const y=v=>pad.t+(H-pad.t-pad.b)*(1-(v-lo)/(hi-lo));ctx.strokeStyle='#263754';ctx.lineWidth=1;for(let j=0;j<5;j++){let yy=pad.t+(H-pad.t-pad.b)*j/4;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();ctx.fillStyle='#91a4bf';ctx.font='10px system-ui';ctx.fillText(fmt(hi-(hi-lo)*j/4,1),3,yy+3)}series.forEach((s,k)=>{ctx.strokeStyle=colors[k];ctx.lineWidth=2;ctx.beginPath();s.v.forEach((v,i)=>{if(!Number.isFinite(v))return;const xx=x(i),yy=y(v);i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)});ctx.stroke();});labels.forEach((l,k)=>{ctx.fillStyle=colors[k];ctx.fillRect(pad.l+k*145,4,10,10);ctx.fillStyle='#c8d5e8';ctx.font='11px system-ui';ctx.fillText(l,pad.l+k*145+15,13)})}
function drawTrajectory(t){const c=$('trajectory'),ctx=c.getContext('2d'),W=c.clientWidth,H=c.clientHeight;c.width=W*devicePixelRatio;c.height=H*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.fillStyle='#0d1728';ctx.fillRect(0,0,W,H);const pts=(t&&t.centerline_points_mm)||[];const robots=(t&&t.robot_positions_mm)||[];const targets=(t&&t.target_positions_mm)||[];const active=(t&&t.robot_active)||[];if(!pts.length&&!robots.length){ctx.fillStyle='#91a4bf';ctx.fillText('等待训练回调写入 live_trace.json…',24,32);return}const all=pts.concat(robots,targets).filter(p=>p&&p.length>=2),xs=all.map(p=>+p[0]),ys=all.map(p=>+p[1]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const dx=Math.max(1,xmax-xmin),dy=Math.max(1,ymax-ymin),pad=28,scale=Math.min((W-2*pad)/dx,(H-2*pad)/dy),ox=pad+(W-2*pad-scale*dx)/2,oy=pad+(H-2*pad-scale*dy)/2;const P=p=>[ox+(p[0]-xmin)*scale,oy+(ymax-p[1])*scale];ctx.strokeStyle='#60738f';ctx.lineWidth=2;ctx.beginPath();pts.forEach((p,i)=>{const q=P(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.stroke();targets.forEach((p,i)=>{const q=P(p);ctx.fillStyle='#ff6b7a';ctx.beginPath();ctx.arc(q[0],q[1],5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#ffb1ba';ctx.font='10px system-ui';ctx.fillText('T'+i,q[0]+7,q[1]-5)});robots.forEach((p,i)=>{const q=P(p);ctx.fillStyle=active[i]===false?'#76849b':['#50d890','#57b7ff','#f4c95d','#c792ea'][i%4];ctx.beginPath();ctx.arc(q[0],q[1],7,0,Math.PI*2);ctx.fill();ctx.fillStyle='#e7eef9';ctx.font='11px system-ui';ctx.fillText('R'+i,q[0]+9,q[1]+4);const v=(t.flow_velocity_mm_s||[])[i];if(v){const q2=P([p[0]+v[0]*2,p[1]+v[1]*2]);ctx.strokeStyle='#f4c95d';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(q2[0],q2[1]);ctx.stroke()}});ctx.fillStyle='#91a4bf';ctx.font='11px system-ui';ctx.fillText('灰线=血管中心线 · 圆点=机器人 · 红点=目标 · 黄色箭头=瞬时血流（xy）',18,H-12)}
function render(d){const st=d.status||{},eps=d.episodes||[],n=eps.length,last=eps.slice(-30);const avg=a=>a.length?a.reduce((x,y)=>x+Number(y||0),0)/a.length:null;const sr=avg(last.map(x=>x.navigation_success?1:0)),lr=avg(last.map(x=>x.lost>0?1:0)),rr=avg(last.map(x=>x.return_sum));setText('steps',st.environment_steps??'—');const target=Number(st.target_environment_steps||0),done=Number(st.environment_steps||0);setText('progress',target?Math.min(100,100*done/target).toFixed(1)+'%':'—');setText('success',sr==null?'—':(100*sr).toFixed(1)+'%',sr!=null?(sr>=.5?'good':'bad'):'');setText('ret',rr==null?'—':fmt(rr),rr!=null&&rr>0?'good':'');setText('lost',lr==null?'—':(100*lr).toFixed(1)+'%',lr!=null&&lr>.2?'bad':'');const complete=d.complete||d.run_complete;setText('runstate',complete?'完成':(st.agent_steps?'运行中':'等待'),complete?'good':'warn');const scene=st.scene||{};drawTrajectory(d.live_trace||{});$('subtitle').textContent=(d.run_dir||'')+' · 每 2 秒刷新 · '+(st.minutes?fmt(st.minutes,1)+' 分钟':'');const fields=[['解剖',scene.anatomy],['难度',scene.difficulty_name??scene.difficulty],['流模式',scene.flow_mode??'—'],['流速',scene.flow==null?'—':fmt(scene.flow,3)],['障碍物',scene.dynamic_obstacle_count??scene.dynamic_obstacles],['场景族',scene.scenario_family],['局部障碍',scene.localized_obstacles],['frontier',scene.frontier]];$('scene').innerHTML=fields.map(x=>`<div><b>${esc(x[0])}</b><span>${esc(x[1])}</span></div>`).join('');$('episodes').innerHTML=last.slice().reverse().map((e,i)=>`<tr><td>#${n-i}</td><td>${esc(e.anatomy)}</td><td>${esc(e.difficulty_name??e.difficulty)}</td><td>${fmt(e.flow,3)}</td><td>${fmt(e.return_sum)}</td><td><span class="pill ${e.navigation_success?'good':'bad'}">${e.navigation_success?'成功':'失败'}</span></td><td>${esc(e.lost)}</td><td>${fmt(e.wall_contact_s,2)}</td><td>${fmt(e.elapsed_s,1)}</td></tr>`).join('');const vals=last.map(e=>Number(e.return_sum));draw('metrics',[{v:last.map(e=>Number(e.navigation_success?1:0)*100)},{v:last.map(e=>Number(e.lost||0))},{v:vals.map(v=>Number.isFinite(v)?v:0)}],['#50d890','#ff6b7a','#57b7ff'],['成功率 %','丢失数','回报']);const p=d.progress||[];draw('training',[{v:p.map(x=>Number(x['train/value_loss']))},{v:p.map(x=>Number(x['train/approx_kl']))},{v:p.map(x=>Number(x['rollout/success_rate'])*100)}],['#f4c95d','#c792ea','#50d890'],['value loss','approx KL','成功率 %']);$('updated').textContent='最后更新：'+(d.updated||'—')+(d.error?' · '+d.error:'')}
async function tick(){try{const r=await fetch('/api/state?ts='+Date.now(),{cache:'no-store'});render(await r.json())}catch(e){$('updated').textContent='监控读取失败：'+e}}tick();setInterval(tick,2000);
</script></body></html>'''


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _read_jsonl(path: Path, limit: int = 120) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _read_progress(path: Path, limit: int = 300) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))[-limit:]
    except (FileNotFoundError, OSError, csv.Error):
        return []
    out = []
    for row in rows:
        clean: dict[str, Any] = {}
        for k, v in row.items():
            try:
                clean[k] = float(v) if v not in (None, "") else None
            except (TypeError, ValueError):
                clean[k] = v
        out.append(clean)
    return out


def state_for(run_dir: Path) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json")
    episodes = _read_jsonl(run_dir / "episodes.jsonl")
    progress = _read_progress(run_dir / "progress.csv")
    live_trace = _read_json(run_dir / "live_trace.json")
    complete = (run_dir / "RUN_COMPLETE.json").exists() or (run_dir / "TRAINING_COMPLETE.json").exists()
    evaluation = _read_json(run_dir / "evaluation.json")
    return {"run_dir": str(run_dir), "status": status, "episodes": episodes,
            "progress": progress, "live_trace": live_trace, "complete": complete, "run_complete": bool(evaluation),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


def curves_for(run_dir: Path) -> dict[str, Any]:
    status = _read_json(run_dir / "status.json")
    config = _read_json(run_dir / "config.json")
    agents = config.get("clusters")
    reward_window = deque(maxlen=20)
    success_window = deque(maxlen=100)
    episodes = []
    physical_steps = int(config.get("resumed_environment_steps", 0))
    reward_kind = None
    episodes_path = run_dir / "episodes.jsonl"
    if episodes_path.exists():
        with episodes_path.open(encoding="utf-8") as log:
            for line in log:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # The current append may not yet be complete.
                if "environment_steps" not in row:
                    continue
                physical_steps += row["environment_steps"]
                kind = reward_kind or ("agent_mean" if "mean_agent_return" in row else "team")
                reward = (row.get("mean_agent_return") if kind == "agent_mean" else
                          row.get("navigation_return", row.get("return_sum")))
                success = row.get("navigation_success")
                if reward is None or not math.isfinite(reward) or not isinstance(success, bool):
                    continue
                reward_kind = kind
                reward_window.append(reward)
                success_window.append(success)
                episodes.append([physical_steps, reward, sum(reward_window) / len(reward_window),
                                 100 * sum(success_window) / len(success_window)])

    entropy = []
    progress_path = run_dir / "progress.csv"
    if agents and progress_path.exists():
        with progress_path.open(encoding="utf-8", newline="") as log:
            previous_steps = None
            for row in csv.DictReader(log):
                try:
                    steps = int(row["time/total_timesteps"]) // agents
                except (KeyError, TypeError, ValueError):
                    continue
                # SB3 dumps the preceding rollout's optimizer metrics before training this rollout.
                if previous_steps is not None and row.get("train/entropy_loss"):
                    value = -float(row["train/entropy_loss"])
                    if math.isfinite(value):
                        entropy.append([previous_steps, value])
                previous_steps = steps
    return dict(run=run_dir.name, steps=status.get("environment_steps"),
                target=status.get("target_environment_steps"), flow=status.get("scene", {}).get("flow"),
                episodes=episodes, entropy=entropy, reward_kind=reward_kind,
                logged_at=progress_path.stat().st_mtime if progress_path.exists() else None)


class Handler(BaseHTTPRequestHandler):
    run_dir: Path = Path(".")
    def log_message(self, fmt: str, *args: Any) -> None:
        return
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/curves":
            body = Path(__file__).with_name("live_training_curves.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
        elif path == "/api/curves":
            body = json.dumps(curves_for(self.run_dir), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
        elif path == "/api/state":
            body = json.dumps(state_for(self.run_dir), ensure_ascii=False).encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store")
        elif path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Cache-Control", "no-store")
        else:
            body = b"Not found"; self.send_response(404); self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="training run directory")
    ap.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 only for trusted networks")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    run_dir = a.run.expanduser().resolve()
    if not run_dir.is_dir():
        ap.error(f"run directory does not exist: {run_dir}")
    Handler.run_dir = run_dir
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Live monitor: http://{a.host}:{a.port}/")
    print(f"Run directory: {run_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
