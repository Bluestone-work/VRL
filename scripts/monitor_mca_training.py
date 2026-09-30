"""Read-only local dashboard for EXP23. Never loads or mutates training state."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'research/runs/EXP_0023_MCA_PURE_RL_20260929a'
DASH = RUN/'dashboard'


def tail_records(path, limit=2000):
    with path.open('rb') as f:
        f.seek(0, 2); size=f.tell(); start=max(0,size-2_000_000); f.seek(start)
        if start: f.readline()
        lines=f.read().splitlines()
    records=[]
    for line in lines[-limit:]:
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass  # concurrent incomplete last line
    return records


def data():
    now=time.time(); seeds=[]
    pointer=RUN/"active_run.json"
    active_run=Path(json.loads(pointer.read_text())["path"]) if pointer.exists() else RUN
    correction_path=active_run/'correction_status.json'
    correction=json.loads(correction_path.read_text()) if correction_path.exists() else None
    protocol_path=active_run/'protocol.json'
    protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    for seed in (42,43,44):
        d=active_run/f'seed_{seed}'
        state=json.loads((d/'status.json').read_text())
        steps=state['transitions']; fps=state['fps']
        updates=tail_records(d/'updates.jsonl') if (d/'updates.jsonl').exists() else []
        episodes=tail_records(d/'episodes.jsonl') if (d/'episodes.jsonl').exists() else []
        recent=episodes[-100:]
        alive=(Path('/proc')/str(state['pid'])).exists()
        running=alive and state['phase'] in ('running','evaluating') and now-state['timestamp']<60
        remaining=max(0,state['target']-steps)/fps if running and fps>0 else None
        state.pop('traceback',None)
        if 'error' in state: state['error']=state['error'].splitlines()[0][:240]
        state.update(seed=seed,progress=steps/state['target']*100,eta_seconds=remaining,
                     eta_date=datetime.fromtimestamp(now+remaining,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M') if remaining is not None else None,
                     heartbeat_age_s=now-state['timestamp'],process_exists=alive,
                     success_last100=sum(bool(e['success']) for e in recent)/max(1,len(recent)),
                     collision_free_success_last100=sum(bool(e.get('collision_free_success',
                         e['success'] and e.get('particle_contact_s',0.)<=1e-12 and e.get('lost_robots',0)==0))
                         for e in recent)/max(1,len(recent)),
                     removal_last100=sum(e['removed_mass']/4 for e in recent)/max(1,len(recent)),
                     updates_trace=updates,episodes_trace=episodes)
        evaluations=sorted(d.glob('evaluation_*.json'),key=lambda p:int(p.stem.split('_')[-1]))
        if evaluations:
            latest=json.loads(evaluations[-1].read_text())
            state['independent_evaluation']=dict(steps=int(evaluations[-1].stem.split('_')[-1]),
                episodes=len(latest['episodes']),success_rate=latest['success_rate'],
                collision_free_success_rate=latest.get('collision_free_success_rate',
                    sum(e['success'] and e.get('particle_contact_s',0.)<=1e-12 and e.get('lost_robots',0)==0
                        for e in latest['episodes'])/max(1,len(latest['episodes']))),
                particle_collision_episode_rate=latest.get('particle_collision_episode_rate',
                    sum(e.get('particle_contact_s',0.)>1e-12 for e in latest['episodes'])/max(1,len(latest['episodes']))),
                mean_removal_fraction=latest['mean_removal_fraction'])
        else:state['independent_evaluation']=None
        seeds.append(state)
    fresh=all(s['process_exists'] and s['phase'] in ('running','evaluating') and s['heartbeat_age_s']<60 for s in seeds)
    max_eta=max(s['eta_seconds'] for s in seeds) if fresh and all(s['eta_seconds'] is not None for s in seeds) else None
    milestones=[]
    for n in protocol.get('milestones',(100000,1000000,2000000,3000000)):
        remain=max(max(0,n-s['transitions'])/s['fps'] for s in seeds) if max_eta is not None else None
        milestones.append(dict(steps=n,remaining_hours=remain/3600 if remain is not None else None,
                               estimated_at=datetime.fromtimestamp(now+remain,ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M') if remain is not None else None))
    return dict(updated_at=datetime.fromtimestamp(now,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
                timestamp=now,seeds=seeds,all_running=fresh,remaining_days=max_eta/86400 if max_eta is not None else None,active_run=str(active_run),correction=correction,
                estimated_finish=datetime.fromtimestamp(now+max_eta,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M') if max_eta is not None else None,
                milestones=milestones,eta_basis='各进程累计实测吞吐；三种子并行，以最慢进程为准；未计未来停机或吞吐变化',
                trace_scope='最近最多2000条回合/更新记录；曲线仅为训练采样，非独立验证',
                task_limit=protocol.get('task_limit','历史1秒任务完整清除不可达。'))


def snapshot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    report=data(); DASH.mkdir(exist_ok=True)
    (DASH/'snapshot.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False))
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    colors=['#2563eb','#10b981','#f97316']
    for s,color in zip(report['seeds'],colors):
        eps=s['episodes_trace']; ups=[u for u in s['updates_trace'] if 'actor_loss' in u]
        for ax,key,scale in [(axes[0,1],'reward',1),(axes[1,0],'removed_mass',25)]:
            y=np.array([e[key]*scale for e in eps]); width=min(50,len(y))
            if width:
                ax.plot([e['transitions'] for e in eps][width-1:],np.convolve(y,np.ones(width)/width,mode='valid'),color=color,label=f"Seed {s['seed']}")
        axes[1,1].plot([u['transitions'] for u in ups],[u['critic_loss'] for u in ups],color=color,label=f"Seed {s['seed']}")
    axes[0,0].barh([str(s['seed']) for s in report['seeds']],[s['transitions'] for s in report['seeds']],color=colors)
    axes[0,0].set(title='Completed environment steps (budget: 3M each)',xlabel='Environment steps',ylabel='Training seed')
    axes[0,1].set(title='Training episode reward (50-episode mean)',xlabel='Environment steps')
    axes[1,0].set(title='Clot mass removed (%) — 50-episode mean',xlabel='Environment steps',ylabel='% of initial total mass')
    axes[1,1].set(title='Critic loss',xlabel='Environment steps',yscale='symlog',ylabel='Loss')
    for ax in axes.flat: ax.grid(alpha=.2)
    axes[0,1].legend()
    eta=f"{report['remaining_days']:.1f} days" if report['remaining_days'] is not None else 'unavailable (workers not running)'
    fig.suptitle(f"MCA | Training snapshot | ETA {eta}\n{report['updated_at']} Asia/Shanghai | Training metrics, not validation success",fontsize=13)
    fig.savefig(DASH/'training_snapshot.png',dpi=160);plt.close(fig)
    return report


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0]=='/api':
            payload=json.dumps(data(),ensure_ascii=False,allow_nan=False).encode(); kind='application/json; charset=utf-8'
        elif self.path.split('?')[0]=='/scene.png':
            scene=RUN/'vtk_view/scene.png'
            if not scene.exists(): self.send_error(404);return
            payload=scene.read_bytes();kind='image/png'
        elif self.path.split('?')[0]=='/playhead.json':
            playhead=RUN/'vtk_view/playhead.json'
            if not playhead.exists():self.send_error(404);return
            payload=playhead.read_bytes();kind='application/json; charset=utf-8'
        elif self.path.split('?')[0]=='/feasibility.png':
            pointer=RUN/'active_run.json'
            active_run=Path(json.loads(pointer.read_text())['path']) if pointer.exists() else RUN
            protocol_path=active_run/'protocol.json'
            protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
            audit=ROOT/protocol.get('feasibility_image','research/validation/EXP0026_FEASIBILITY_FINAL_20260930/feasibility.png')
            if not audit.exists(): self.send_error(404);return
            payload=audit.read_bytes();kind='image/png'
        elif self.path.split('?')[0] in ('/','/index.html'):
            payload=(DASH/'index.html').read_bytes();kind='text/html; charset=utf-8'
        else:
            self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def log_message(self,*args): pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--serve',action='store_true');p.add_argument('--port',type=int,default=8763);args=p.parse_args()
    if args.serve:
        print(f'http://127.0.0.1:{args.port}',flush=True)
        ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
    else:
        r=snapshot();print(json.dumps({k:v for k,v in r.items() if k!='seeds'},ensure_ascii=False,indent=2))
