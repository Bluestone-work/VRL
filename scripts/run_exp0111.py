"""Fresh matched 1M-environment-step SAC development with exact-state snapshots.

Resume interrupted jobs with an explicit --resume path and a NEW --segment.
Never overwrites prior logs. Evaluation uses fixed development scenes only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts import run_exp0110 as charts

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'research/runs/EXP0111_SAC_1M_20261013'
RESULTS=ROOT/'research/validation/EXP0111_SAC_1M_20261013'
MILESTONES=(100000,250000,500000,1000000)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['auto','fixed','both'],default='both')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--segment',default='seed0')
    args=parser.parse_args()
    if args.resume and args.arm=='both':parser.error('Resume must specify exactly one arm')
    RESULTS.mkdir(parents=True,exist_ok=True)
    commands=[]
    for arm in (('auto','fixed') if args.arm=='both' else (args.arm,)):
        folder=RUNS/arm/args.segment
        if folder.exists():raise FileExistsError(f'Refusing to overwrite {folder}; use an explicit new segment for resume')
        previous_steps=0
        if args.resume:
            import torch
            previous_steps=torch.load(args.resume,map_location='cpu',weights_only=False)['environment_steps']
        train=[sys.executable,'-m','scripts.train_finite_image_flow_sac','--out',str(folder),
            '--seed','0','--workers','2','--env-steps','1000000','--rollout','96',
            '--gradient-steps','16','--batch','256','--capacity','30000','--device','cpu',
            '--critic-warmup-iterations','16','--action-space','vector','--dynamic-obstacles',
            '--dynamic-obstacle-count','16','--obstacle-features','--team-reward','--resume-every','100']
        if arm=='auto':train+=['--auto-alpha']
        if args.resume:train+=['--resume',str(args.resume.resolve())]
        commands.append(train)
        command_file=RESULTS/f'commands_{arm}_{args.segment}.json'
        command_file.write_text(json.dumps(commands,indent=2))
        with (RESULTS/f'{arm}_{args.segment}_training.log').open('x') as log:
            proc=subprocess.Popen(train,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:
                checkpoints=([0] if previous_steps==0 else [])+[m for m in MILESTONES if m>previous_steps]
                for steps in checkpoints:
                    checkpoint=folder/('initial.pt' if steps==0 else f'policy_steps{steps}.pt')
                    while not checkpoint.exists():
                        if proc.poll() is not None:raise RuntimeError(f'Trainer exited before {checkpoint}, code={proc.returncode}')
                        time.sleep(10)
                    methods=f'sac={checkpoint}'
                    if arm=='auto' and steps==0:methods='parent,adaptive_settle,'+methods
                    ev=[sys.executable,'-m','scripts.eval_image_flow_sac','--out',str(RESULTS/f'{arm}_{steps}'),
                        '--pool','screen','--workers','4','--methods',methods,
                        '--dynamic-obstacles','--dynamic-obstacle-count','16','--attribution']
                    commands.append(ev);command_file.write_text(json.dumps(commands,indent=2))
                    subprocess.run(ev,cwd=ROOT,check=True)
                    errors=RESULTS/f'{arm}_{steps}'/'failures.jsonl'
                    if errors.exists() and errors.stat().st_size:raise RuntimeError(f'Evaluation failures: {errors}')
                    charts.RUNS=RUNS;charts.RESULTS=RESULTS
                    charts.report(xlabel='cumulative physical environment steps (milestone)')
                if proc.wait()!=0:raise RuntimeError(f'{arm} training failed')
            finally:
                if proc.poll() is None:proc.terminate();proc.wait()
        (RESULTS/f'COMPLETE_{arm}_{args.segment}.json').write_text(json.dumps({'complete':True}))


if __name__=='__main__':main()
