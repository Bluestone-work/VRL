"""Post-queue renderer, using reviewed GUI fixes without altering frozen training code."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/validation/EXP0060_VASCULAR_OPTION_RL_20261006'


def main():
    deadline=datetime(2026,10,6,8,45,tzinfo=timezone(timedelta(hours=8))).timestamp()
    env=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while time.time()<deadline:
        try:
            status=json.loads((OUT/'STATUS.json').read_text())
            if status['stage'] in ('complete','finished_with_failures'):break
            if status['stage']=='training' or status['stage'].startswith('training_seed'):
                # Refresh progress during the long training phase; avoid competing
                # report writers during queue evaluation and finalization.
                subprocess.run(['taskset','-c','0-5,8-23','/home/wj/.cache/vascular-research/cpu312-clean-20261004/bin/python',
                                '-m','scripts.report_vascular_option_study'],cwd=ROOT,env=env,check=True,timeout=120)
        except (OSError,ValueError,subprocess.SubprocessError) as e:
            print(repr(e),flush=True)
        time.sleep(60)
    trace=OUT/'final_replay_mca_m1_lvo_2810000000.trace.json'
    if trace.exists():
        subprocess.run(['taskset','-c','0-5,8-23','/home/wj/miniconda3/envs/v/bin/python','-m','scripts.view_vascular_option_rl',
                        '--trace',str(trace),'--headless','--out',str(ROOT/'research/figures/EXP0060_20261006/gui')],
                       cwd=ROOT,env=env,check=True,timeout=120)
    subprocess.run(['taskset','-c','0-5,8-23','/home/wj/.cache/vascular-research/cpu312-clean-20261004/bin/python',
                    '-m','scripts.report_vascular_option_study'],cwd=ROOT,env=env,check=True,timeout=120)
    (OUT/'MEDIA_FINALIZED.json').write_text(json.dumps(dict(time=datetime.now().isoformat(),
            renderer_sha256=hashlib.sha256((ROOT/'scripts/view_vascular_option_rl.py').read_bytes()).hexdigest(),
            final_trace_exists=trace.exists()),indent=2))


if __name__=='__main__':main()
