"""EXP_0012 deconfound: control checkpoints (trained WITHOUT particles)
evaluated WITH particles at eval, using the sealed EXP_0002 manifest.

Isolates the evaluation-condition contribution from the training-with-
particles contribution measured in EXP_0012_PARTICLES_* arms:
  particles_arm  - control_no_particle_eval  = training + eval effect
  control_particle_eval - control            = eval-only effect
Same protocol as scripts/prospective_eval_0010_0013.py: deterministic
policy, per-episode identity with the same relaxed initial_state rule
for particle arms, sealed test untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.prospective_eval_0010_0013 import (
    ARMS, METRICS, ROOT, RUNS, STUDY, build_agent, episode, env_settings,
    mean_sd,
)

OUT = ROOT / 'research/runs/EXP_0012_PARTICLES_deconfound'
PARTICLE_COUNTS = (8, 16, 32)


def main():
    manifest = json.loads((ROOT / STUDY['prospective_validation']['manifest']).read_text())
    assert manifest.get('policy_evaluations') == 0, 'manifest already consumed'
    records_by_scenario = {}
    for row in manifest['records']:
        records_by_scenario.setdefault(row['scenario'], []).append(row)
    cfg = STUDY['training']
    device = 'cuda:0'
    OUT.mkdir(parents=True, exist_ok=True)
    for count in PARTICLE_COUNTS:
        arm_dir = OUT / f'CONTROL_EVAL_PARTICLES_{count}'
        arm_dir.mkdir(exist_ok=True)
        opts = dict(particles=count)
        for seed in STUDY['seeds']:
            # control checkpoints trained without particles
            checkpoint = RUNS / 'EXP_0012_CONTROL' / f'seed_{seed}' / 'final_policy.pt'
            agent = build_agent(checkpoint, cfg, device)
            rows = []
            for scenario, targets in sorted(records_by_scenario.items()):
                for target in targets:
                    particle_seed = 82000000 + (target['episode_seed'] % 100000)
                    from environments.vascular_3d_marl_env import Vascular3DMARLEnv
                    env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario],
                                            seed=target['episode_seed'],
                                            **env_settings(cfg, opts, particle_seed, target['episode_seed']))
                    try:
                        rows.append(episode(agent, env, target, opts, None))
                    finally:
                        env.close()
            summary = {'training_seed': seed, 'particles_at_eval': count,
                       'trained_with_particles': False,
                       **{key: mean_sd(rows, key) for key in METRICS}}
            (arm_dir / f'seed_{seed}_episodes.json').write_text(json.dumps(rows, indent=1))
            (arm_dir / f'seed_{seed}_summary.json').write_text(json.dumps(summary, indent=1))
            print(f'== CONTROL_EVAL_PARTICLES_{count} seed {seed}: '
                  f'success={summary["success"]["mean"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
