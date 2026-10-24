"""Restore the task's suspended V5 evaluation after the new RL media export.

Restart from saved episode records: resuming the stopped evaluator directly
would count the deliberate pause against its per-episode wall-clock timeout.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'research/validation/EXP0094_PPO_NAV_20261011'
MEDIA=Path('/home/wj/桌面/VRL_RL_局部导航_训练测试_GIF_20261011')


def argv(pid):
    try:return [s.decode() for s in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if s]
    except (FileNotFoundError,ProcessLookupError):return []


def is_previous_v5(args):
    return (bool(args) and Path(args[0]).name.startswith('python')
            and 'scripts.eval_exp0090' in args and '--out' in args
            and args[args.index('--out')+1]=='research/validation/EXP0093_20261011/v5')


def main():
    os.chdir(ROOT)
    import fcntl
    lock=(V/'.resume_v5.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    paused=json.loads((V/'paused_previous_v5_eval.json').read_text())
    oldpid=paused['pids'][0];args=argv(oldpid)
    if not is_previous_v5(args):return
    print('Waiting for the new RL GIF batch before restoring the previous V5 evaluation',flush=True)
    while True:
        progress=json.loads((MEDIA/'progress.json').read_text())
        if progress['pending']==0:break
        if not is_previous_v5(argv(oldpid)):return
        time.sleep(30)
    if not is_previous_v5(argv(oldpid)):return
    stopped=[]
    for pid in paused['pids']:
        if not is_previous_v5(argv(pid)):continue
        try:
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT);stopped.append(pid)
        except ProcessLookupError:pass
    time.sleep(1)
    args[args.index('--workers')+1]='8'
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]='1'
    target=ROOT/'research/validation/EXP0093_20261011/v5_resume_after_rl.log'
    with target.open('a') as log:
        child=subprocess.Popen(['taskset','-c','0-5,8-23',sys.executable,*args[1:]],cwd=ROOT,env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,terminated_paused_pids=stopped,argv=args,log=str(target),
                time=time.strftime('%F %T'),reason='Resume from saved records without turning the intentional pause into episode timeouts')
    (V/'previous_v5_restored.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
