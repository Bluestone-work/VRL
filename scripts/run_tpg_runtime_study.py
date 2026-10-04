"""Explicit, process-local NumPy runtime variant; never edit the global environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('preflight','pilot'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    environment=dict(os.environ,NPY_DISABLE_CPU_FEATURES='AVX2,FMA3',OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1',PYTHONPATH='.')
    # Query package/runtime provenance inside the same child environment.
    code='import json,sys,numpy,torch;print(json.dumps(dict(numpy=numpy.__version__,torch=torch.__version__,python=sys.version)))'
    runtime=json.loads(subprocess.check_output([sys.executable,'-c',code],env=environment,text=True))
    runtime.update(runtime_variant='numpy_disable_avx2_fma3',NPY_DISABLE_CPU_FEATURES='AVX2,FMA3',
        applies_to_all_arms=True,native_fault_fixed=False,prior_failures_replaced=False)
    if args.phase=='pilot':
        if args.preflight is None:raise ValueError('Matching runtime preflight required')
        p=args.preflight
        if json.loads((p/'status.json').read_text())['phase']!='completed' or not json.loads((p/'untrained_identity.json').read_text())['passed']:
            raise ValueError('Preflight gate failed; diagnostic repeats cannot replace failures')
        prior=json.loads((p/'runtime_context.json').read_text())
        if any(prior.get(k)!=runtime[k] for k in ('NPY_DISABLE_CPU_FEATURES','numpy','torch','python')):
            raise ValueError('Runtime differs from preflight')
        from scripts.tpg_episode import tpg_hashes
        if json.loads((p/'manifest.json').read_text())['source_hashes']!=tpg_hashes():
            raise ValueError('Source differs from preflight')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise FileExistsError(args.out)
    child=subprocess.Popen([sys.executable,'-m','scripts.run_tpg_study','--phase',args.phase,
        '--workers',str(args.workers),'--out',str(args.out)],env=environment)
    while not args.out.exists() and child.poll() is None:time.sleep(.05)
    if args.out.exists():
        source=Path(__file__).read_bytes()
        runtime['launcher_sha256']=hashlib.sha256(source).hexdigest()
        runtime['preflight']=str(args.preflight) if args.preflight else None
        (args.out/'runtime_context.json').write_text(json.dumps(runtime,indent=2)+'\n')
        (args.out/'runtime_launcher_snapshot.py').write_bytes(source)
    code=child.wait()
    if code:raise SystemExit(code)
    if json.loads((args.out/'status.json').read_text())['phase']!='completed':raise SystemExit(2)


if __name__=='__main__':main()
