"""Wait for the fixed main matrix, then run the preregistered conditional repair."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.run_safety_marl import ROOT, PROTOCOL, hashes
from scripts.run_measured_marl import enforce_runtime
from scripts.run_option_learning import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--followup-out',type=Path,required=True)
    args=parser.parse_args();args.out=args.out.resolve();args.followup_out=args.followup_out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    source=hashes();p=json.loads(PROTOCOL.read_text());parent=ROOT/p['parent_study']
    atomic_json(args.out/'requested.json',dict(parent=str(parent),followup=str(args.followup_out),
        source_hashes=source,controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=enforce_runtime(),started_at=datetime.now().astimezone().isoformat(),confirmation_accessed=False))
    start=time.monotonic()
    while True:
        if source!=hashes():raise RuntimeError('Conditional study sources changed while waiting')
        status=json.loads((parent/'status.json').read_text())
        atomic_json(args.out/'status.json',dict(phase='waiting_for_parent',parent_phase=status['phase'],
            completed_parent_attempts=status.get('completed_attempts'),wait_s=time.monotonic()-start))
        if status['phase'].startswith('failed'):
            atomic_json(args.out/'status.json',dict(phase='parent_failed_gate',followup_started=False));raise SystemExit(2)
        if (parent/'AUDIT.json').exists():
            audit=json.loads((parent/'AUDIT.json').read_text())
            if not audit['valid_comparison_gate']:
                atomic_json(args.out/'status.json',dict(phase='parent_failed_audit',followup_started=False));raise SystemExit(2)
            break
        if time.monotonic()-start>7200:
            atomic_json(args.out/'status.json',dict(phase='parent_wait_timeout',followup_started=False));raise SystemExit(2)
        time.sleep(5)
    environment=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    for name,command in (
        ('parent_plot',[sys.executable,'-m','scripts.plot_measured_marl',str(parent)]),
        ('conditional_followup',[sys.executable,'-m','scripts.run_safety_marl_study','--workers','8','--out',str(args.followup_out)])):
        atomic_json(args.out/'status.json',dict(phase=name,command=command))
        with (args.out/(name+'.log')).open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        atomic_json(args.out/(name+'_completion.json'),dict(command=command,returncode=result.returncode))
        if result.returncode:
            atomic_json(args.out/'status.json',dict(phase=name+'_failed',returncode=result.returncode));raise SystemExit(result.returncode)
    atomic_json(args.out/'status.json',dict(phase='completed',parent_audit_passed=True,
        followup_started=args.followup_out.exists(),confirmation_accessed=False))


if __name__=='__main__':main()
