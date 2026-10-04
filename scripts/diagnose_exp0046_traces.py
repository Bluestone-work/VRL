"""Retrospective symptom counts from preserved primary d=2 mm episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summaries = []
    for clusters in (1, 2, 3):
        method = 'single_sequential' if clusters == 1 else 'multi_parallel'
        files = sorted(args.directory.glob(f'{method}_n{clusters}_d2_s*_fixed_total.jsonl'))
        attempts = [json.loads(line) for file in files for line in file.read_text().splitlines()]
        rows = [r for r in attempts if r['status'] == 'completed']
        stagnation = []
        for row in rows:
            older = [t for t in row['trace'] if t['elapsed_s'] <= row['elapsed_s']-60]
            if row['removal'] < 1-1e-12 and older and row['removal']-older[-1]['removal'] < .001:
                stagnation.append(dict(scene_seed=row['seed'], control_seed=row['control_seed']))
        summaries.append(dict(method=method, clusters=clusters, recorded=len(attempts),
            completed=len(rows), terminal_reasons=dict(Counter(r['termination_reason'] for r in rows)),
            raw_success=sum(r['task_success'] for r in rows),
            cluster_safe_success=sum(r['cluster_safe_success'] for r in rows),
            wall_unsafe=sum(r['wall_contact_s'] >= 1 for r in rows),
            particle_contact=sum(r['particle_contact_s'] > 0 for r in rows),
            any_cluster_lost=sum(r['lost_clusters'] > 0 for r in rows),
            separation_violation=sum(not r['spacing_compliant'] for r in rows),
            prolonged_yield_events=sum(r['controller']['persistent_yield_events'] for r in rows),
            late_stagnation_count=len(stagnation), late_stagnation_cases=stagnation))
    output = dict(retrospective=True, performance_ranking_blocked=True,
        symptoms_are_nonexclusive=True,
        stagnation_definition='Uncleared episode; <0.001 removal gain since last logged point at least 60 s before termination. Not proven deadlock.',
        independent_scenes=10, controller_seeds=[42, 43, 44], arms=summaries)
    (args.directory/'failure_diagnostics.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = ['# EXP0046 v2 retrospective failure diagnostics', '',
        'The numerical gate failed (one native crash in the full matrix). These are symptom counts '
        'for debugging, not a performance ranking. Counts overlap. Each listed arm has 30 completed '
        'episodes but only 10 independent scenes; controller seeds 42/43/44 repeat each scene.', '',
        '| Arm, fixed total catalytic rate | Raw completions | Cluster-safe completions | Wall threshold failures | Separation violations | Late stagnation proxy |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in summaries:
        lines.append(f"| {arm['method']}, N={arm['clusters']}, d=2 mm | "
            +' | '.join(f"{arm[k]}/{arm['completed']}" for k in ('raw_success', 'cluster_safe_success',
                'wall_unsafe', 'separation_violation', 'late_stagnation_count'))+' |')
    lines += ['', 'No particle contacts or cluster losses occur in these three arms. '
        'Wall/spacing violations and incomplete progress therefore deserve separate investigation. '
        'Stagnation means <0.001 removal gain over at least the final 60 s of an uncleared episode; '
        'it can reflect inaccessible targets, local navigation traps, waiting, or other causes. '
        'The trace does not establish which one. Prolonged-yield counters also do not prove deadlock.', '',
        '## Two selected action replays', '',
        'Selection is retrospective: the largest wall-contact case and a second case whose '
        'minimum separation remained >6 mm, from N=3, d=2 mm, controller seed 42. '
        'The read-only action tap reproduces each original final-state hash exactly.', '',
        '- Scene 1302010003: 24.3785 cluster-s wall contact. Of 248 wall-contact agent-steps, '
        '244 also changed the nominal command through the spacing filter; 224 changed a nominal '
        'outward component <=0.05 to >0.05 in the measured radial direction. This motivates '
        'joint wall/separation constraints, but does not by itself quantify their causal benefit.',
        '- Scene 1302010005: 7.9323 cluster-s wall contact, 50% removal. The spacing filter '
        'never changed a command; all 165 wall-contact agent-steps already had a nominal outward '
        'component >0.05. Improving coordination alone cannot explain away this navigation symptom.', '',
        'Replay evidence: `../EXP0046_REPAIR_20261004/trace_seed1302010003/summary.json` '
        'and `../EXP0046_REPAIR_20261004/trace_seed1302010005/summary.json`; complete measured '
        'commands, clearances and evaluation-only contact increments are in their `trace.jsonl` files.', '',
        '## Next study decisions', '',
        '1. Keep the native fault open and preserve process-level failure accounting. No retry may '
        'overwrite the failed matrix row. Obtain a native stack/core if it recurs before asserting a repair.',
        '2. Prototype joint wall/separation action constraints using the same local observation. '
        'Test contradictory constraints, image noise and actuator saturation explicitly; report '
        'infeasibility instead of calling a stop command a safety guarantee.',
        '3. Improve the shared local navigation baseline separately (including N=1). Then compare '
        'coordination changes at fixed navigation, resources, starts, and sensing on fresh scenes. '
        'Any advantage that disappears under this stronger baseline is not a coordination contribution.',
        '4. Before learning, compare a conventional scheduling/memory baseline and identify actual '
        'conflict states from trajectories. Do not infer a need for RL merely from a timeout counter.',
        '5. Keep measured magnetic coupling, imaging error and deployment feasibility as explicit '
        'physical gates. No magnetic-field independence claim is supported by separation-only simulation.', '']
    (args.directory/'FAILURE_ANALYSIS.md').write_text('\n'.join(lines))
    print(json.dumps({**output, 'arms': [{k: v for k, v in a.items() if k != 'late_stagnation_cases'}
                                     for a in summaries]}, indent=2))


if __name__ == '__main__':
    main()
