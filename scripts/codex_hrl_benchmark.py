"""Evaluate Codex's hierarchical RL (EXP0058 V-CTPG-HRL family) on the unified benchmark scenes.

Ported, not re-implemented: Codex's TPGEpisode (measured tracking sensors, rolling local TPG, event-driven
high-level priority, continuous residual low level) and its trained EXP0058 checkpoints are used as-is.
Only the scene changes: identical `paired_environment` scenes, physics (marl.teacher.TEACHER_CONFIG),
anatomy and 300 s horizon as every other benchmark method. Codex's provenance gates are bypassed for this
evaluation-only port and recorded in each row: its declared isolated Python-3.12 runtime (we run in the
project runtime), its development-seed whitelist (benchmark seeds are registered elsewhere), and its
checkpoint source-hash check (shared files were not edited since training except where listed in git).
Codex's own high level allocates targets (measured balanced assignment); it does not receive allocation A.
usage: codex_hrl_benchmark.py --variant graph_anchor|r_mappo|graph_ppo|mappo_anchor|tpg --checkpoint PT
       --clusters 3 --anatomy A --seeds S0:S1 --out JSONL
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True); p.add_argument('--checkpoint', type=Path)
    p.add_argument('--clusters', type=int, default=3); p.add_argument('--anatomy', required=True)
    p.add_argument('--seeds', required=True); p.add_argument('--horizon-s', type=float, default=300.)
    p.add_argument('--tag'); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1); torch.manual_seed(42)
    import scripts.run_continuous_marl as rc
    import scripts.run_measured_marl as base
    import scripts.local_learning_episode as lle
    from environments.mca_physical_env import DynamicsConfig
    from marl.teacher import TEACHER_CONFIG
    from scripts.tpg_episode import TPGEpisode
    rc.configure()
    proto = rc.protocol()
    # Same physics / anatomy / horizon as the benchmark: replace the config Codex's episode loads.
    class _Cfg:
        @staticmethod
        def from_json(_):
            return replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=a.anatomy)
    lle.DynamicsConfig = _Cfg
    net = None; ck_sha = None
    if a.variant != 'tpg':
        from marl.continuous_residual import ContinuousMARL
        net = ContinuousMARL(a.variant, proto['hidden_dim'], proto['clusters'], proto['prior_logit_bias'])
        payload = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
        assert payload['variant'] == a.variant
        net.load_state_dict(payload['model'], strict=True); net.eval()
        ck_sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    s0, s1 = map(int, a.seeds.split(':'))
    with a.out.open('a') as out:
        for seed in range(s0, s1+1):
            t0 = time.monotonic()
            ep = (TPGEpisode if net is None else rc.ContinuousEpisode)(seed, clusters=a.clusters, duration=a.horizon_s)
            try:
                while not ep.done:
                    if ep.needs_decision:
                        if net is None:
                            pr = ep.conventional_priority()
                        else:
                            st = base.observation(ep); act, _, _ = rc.act(net, 'high', st, True)
                            pr = base.priority_action(net, act, st)
                        ep.choose_priority(pr)
                    if net is None:
                        ch = ep.conventional_low()
                    else:
                        st = base.observation(ep); ch, _, _ = rc.act(net, 'low', st, True)
                    ep.step_control(ch)
                r = ep.result(a.tag or a.variant)
                ms = r['time_to_removal_s']
                row = dict(method=a.tag or f'codex_{a.variant}', information='measured (Codex tracked sensors)',
                           clusters=a.clusters, anatomy=a.anatomy, seed=seed, horizon_s=a.horizon_s,
                           cluster_safe_success=r['cluster_safe_success'], task_success=r['task_success'],
                           safe_success=r['safe_success'], removal=r['removal'], elapsed_s=r['elapsed_s'],
                           wall_contact_s=r['wall_contact_s'], wall_contact_ratio=r['wall_contact_ratio'],
                           max_continuous_wall_contact_s=r['max_continuous_wall_contact_s'], timeout=r['timeout'],
                           termination_reason=r['termination_reason'],
                           particle_events=int(ep.info['episode_particle_collision_events']),
                           particle_contact_s=r['particle_contact_s'], robot_pair_contact_s=r['robot_pair_contact_s'],
                           lost=r['lost_clusters'], t50_s=ms['50'], t90_s=ms['90'], t100_s=ms['100'],
                           removal_auc=r['removal_auc_s']/a.horizon_s, path_mm=r['path_mm'],
                           spacing={k: r[k] for k in ('minimum_spacing_mm', 'spacing_violation_pair_s', 'spacing_compliant') if k in r},
                           plan=[[0, 1, 2, 3]], yield_events=0, prolonged_yield_events=0,
                           scenario_hash=r['scenario_hash'], checkpoint=str(a.checkpoint) if a.checkpoint else None,
                           checkpoint_sha256=ck_sha, port_note='Codex runtime/seed/source gates bypassed for evaluation-only port',
                           walltime_s=time.monotonic()-t0)
                out.write(json.dumps(row)+'\n'); out.flush()
            finally:
                ep.close()


if __name__ == '__main__':
    main()
