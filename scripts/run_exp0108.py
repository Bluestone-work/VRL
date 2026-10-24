"""Bounded 3D SAC smoke followed by paired development evaluation."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research/runs/EXP0108_SAC_3D_OBSTACLE_20261013/seed0'
EVAL = ROOT / 'research/validation/EXP0108_SAC_3D_OBSTACLE_20261013'


def main():
    EVAL.mkdir(parents=True, exist_ok=True)
    train = [sys.executable, '-m', 'scripts.train_finite_image_flow_sac',
             '--out', str(RUN), '--seed', '0', '--workers', '2',
             '--iterations', '20', '--rollout', '64', '--gradient-steps', '4',
             '--batch', '128', '--device', 'cpu', '--critic-warmup-iterations', '4',
             '--action-space', 'vector', '--dynamic-obstacles',
             '--dynamic-obstacle-count', '16', '--obstacle-features', '--team-reward']
    methods = ','.join([
        'parent', 'adaptive_settle', f'sac={RUN}/initial.pt', f'sac={RUN}/policy.pt',
        'sac_finite=research/runs/EXP0102_SAC_OBSFEATURE_DYNAMIC_20261011/seed4/policy.pt',
        'td3=research/runs/EXP0103_TD3_OBSFEATURE_DYNAMIC_20261011/seed4/policy.pt'])
    evaluate = [sys.executable, '-m', 'scripts.eval_image_flow_sac',
                '--out', str(EVAL), '--pool', 'screen', '--workers', '4',
                '--methods', methods, '--dynamic-obstacles',
                '--dynamic-obstacle-count', '16', '--attribution', '--traces']
    (EVAL / 'commands.json').write_text(json.dumps([train, evaluate], indent=2))
    # Never silently overwrite or resume a partial training run.
    if not (RUN / 'TRAINING_COMPLETE.json').exists():
        subprocess.run(train, cwd=ROOT, check=True)
    import torch
    policy = torch.load(RUN / 'policy.pt', map_location='cpu', weights_only=False)
    if policy['actor_gradient_updates'] <= 0:
        raise RuntimeError('Actor never updated: evaluation would not test RL')
    subprocess.run(evaluate, cwd=ROOT, check=True)
    failures = EVAL / 'failures.jsonl'
    if failures.exists() and failures.stat().st_size:
        raise RuntimeError('Evaluation failures recorded; inspect failures.jsonl')
    (EVAL / 'PIPELINE_COMPLETE.json').write_text(json.dumps({'completed': True}))


if __name__ == '__main__':
    main()
