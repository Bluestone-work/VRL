"""Resume an evaluation as isolated one-scene processes after simulator crashes."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--scene-list', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--centralized-critic', action='store_true')
    p.add_argument('--flow-feature', action='store_true')
    a = p.parse_args()
    payload = json.loads(a.scene_list.read_text()); rows = payload['rows']
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        scene = a.out.with_name(a.out.stem + f'_single_scene{i:04d}.json')
        result = a.out.with_name(a.out.stem + f'_single{i:04d}.json')
        if result.exists():
            try:
                old = json.loads(result.read_text())
                if old.get('overall', {}).get('complete') and len(old.get('episodes', [])) == 1:
                    continue
            except Exception:
                pass
        scene.write_text(json.dumps({**payload, 'rows': [row]}, indent=2))
        cmd = [sys.executable, str(ROOT/'scripts/evaluate_navigation_dataset.py'), '--model', str(a.model),
               '--kind', 'ppo', '--action-set', 'semantic_reduced', '--split', a.split, '--clusters', '3',
               '--scene-list', str(scene), '--out', str(result), '--device', 'cpu']
        if a.flow_feature: cmd.append('--flow-feature')
        if a.centralized_critic: cmd.append('--centralized-critic')
        done = False
        for _ in range(4):
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            try:
                subprocess.run(cmd, cwd=ROOT, env=env, timeout=240, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                data = json.loads(result.read_text())
                done = data.get('overall', {}).get('complete') and len(data.get('episodes', [])) == 1
                if done: break
            except Exception:
                pass
        print(i, 'ok' if done else 'FAILED', flush=True)
        if not done: raise SystemExit(2)

if __name__ == '__main__':
    main()
