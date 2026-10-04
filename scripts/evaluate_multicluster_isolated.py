"""Crash-audited EXP0046 runner: preserve every requested episode and denominator."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.safe_metrics import aggregate
from scripts.multicluster_protocol import REVISION


def summarize_attempts(attempts):
    rows=[r for r in attempts if r.get('status')=='completed']
    failed=[r for r in attempts if r.get('status')!='completed']
    requested=len(attempts)
    successes=sum(bool(r['cluster_safe_success']) for r in rows)
    return dict(revision=REVISION,requested_episodes=requested,completed_episodes=len(rows),
                failed_episodes=len(failed),failed_attempts=failed,
                conditional_completed_aggregate=aggregate(rows),
                cluster_safe_success_completed=successes/len(rows) if rows else None,
                cluster_safe_success_missing_bounds=[successes/requested,(successes+len(failed))/requested] if requested else None,
                numerical_gate_passed=bool(requested and not failed),
                performance_comparison_admissible=bool(requested and not failed),
                development_diagnostic_only=True,sealed_test_used=False)


def run_isolated(*, method, clusters, d_min_mm, seeds, out, duration_s=180., control_seed=42,
                 budget='fixed_total', anatomy='mca_m1_lvo', noise=.025, position_noise_mm=.02,
                 peer_sensing_mm=6., timeout_s=120., config=None):
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    parts=out.parent/(out.stem+'_parts')
    parts.mkdir(exist_ok=False)
    attempts=[]
    runner=Path(__file__).with_name('evaluate_multicluster.py')
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    env['PYTHONPATH']=str(runner.resolve().parents[1])+os.pathsep+env.get('PYTHONPATH','')
    with out.open('x',encoding='utf-8') as stream:
        for index,seed in enumerate(seeds):
            part=parts/f'episode_{index:04d}.jsonl'
            cmd=[sys.executable,'-X','faulthandler',str(runner),'--method',method,'--clusters',str(clusters),
                 '--d-min-mm',str(d_min_mm),'--episodes','1','--seed-base',str(seed),
                 '--control-seed',str(control_seed),'--budget',budget,'--anatomy',anatomy,
                 '--noise',str(noise),'--position-noise-mm',str(position_noise_mm),
                 '--peer-sensing-mm',str(peer_sensing_mm),'--duration-s',str(duration_s),'--out',str(part.resolve())]
            if config is not None: cmd.extend(['--config',str(Path(config).resolve())])
            start=time.monotonic()
            common=dict(seed=seed,method=method,clusters=clusters,control_seed=control_seed,
                        resource_budget=budget,min_spacing_threshold_mm=d_min_mm,sealed_test_used=False)
            try:
                p=subprocess.run(cmd,cwd=runner.resolve().parents[1],env=env,capture_output=True,
                                 text=True,timeout=timeout_s)
                (parts/f'episode_{index:04d}.stderr.txt').write_text(p.stderr,encoding='utf-8')
                if p.returncode:
                    row=dict(common,status='process_exit',returncode=p.returncode,stderr_path=str(parts/f'episode_{index:04d}.stderr.txt'))
                else:
                    row=json.loads(part.read_text().splitlines()[0])
                    if row.get('seed') != seed or row.get('status')!='completed':
                        raise ValueError('Child output identity/status mismatch')
            except subprocess.TimeoutExpired:
                row=dict(common,status='timeout',timeout_s=timeout_s)
            except (ValueError,OSError,IndexError) as exc:
                row=dict(common,status='invalid_output',detail=str(exc))
            row['process_walltime_s']=time.monotonic()-start
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            attempts.append(row)
            print(json.dumps({k:row[k] for k in ('seed','method','clusters','status','removal','cluster_safe_success') if k in row}),flush=True)
    summary=summarize_attempts(attempts)
    summary.update(method=method,clusters=clusters,d_min_mm=d_min_mm,budget=budget,
                   control_seed=control_seed,duration_s=duration_s,episode_isolated=True)
    out.with_name(out.stem+'_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    from scripts.evaluate_multicluster import METHODS
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--clusters',type=int,default=2)
    p.add_argument('--d-min-mm',type=float,default=2.)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-base',type=int,default=1302000000)
    p.add_argument('--control-seed',type=int,default=42)
    p.add_argument('--anatomy',default='mca_m1_lvo')
    p.add_argument('--config',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--noise',type=float,default=.025)
    p.add_argument('--position-noise-mm',type=float,default=.02)
    p.add_argument('--peer-sensing-mm',type=float,default=6.)
    p.add_argument('--budget',choices=('fixed_total','per_cluster'),default='fixed_total')
    p.add_argument('--duration-s',type=float,default=180.)
    p.add_argument('--timeout-s',type=float,default=120.)
    args=vars(p.parse_args())
    count,base=args.pop('episodes'),args.pop('seed_base')
    if count<1: p.error('episodes must be positive')
    result=run_isolated(seeds=list(range(base,base+count)),**args)
    print(json.dumps(result,indent=2))
    if not result['numerical_gate_passed']: raise SystemExit(2)


if __name__=='__main__': main()
