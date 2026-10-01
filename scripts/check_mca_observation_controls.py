"""Digest old control observations and physical transitions under fixed actions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--config-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    from environments.mca_compiled import CompiledMCAPhysicalEnv
    from environments.mca_physical_env import DynamicsConfig
    from scripts.train_mca_compiled import reset_with_valid_particles
    from scripts.train_mca_physical import atomic_json
    if args.out.exists():
        raise ValueError('Preserve earlier audit artifacts')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ('TRAJECTORY','MASKED'):
        config = args.config_root/f'configs/experiments/EXP_0032_{name}_DYNAMICS.json'
        cfg = DynamicsConfig.from_json(config)
        digests = []
        for seed in range(940000000,940000005):
            env = CompiledMCAPhysicalEnv(cfg); obs, reset = reset_with_valid_particles(env,seed)
            digest = hashlib.sha256(json.dumps(reset,sort_keys=True).encode())
            rng = np.random.default_rng(seed)
            for _ in range(100):
                for key in sorted(obs):digest.update(obs[key].tobytes())
                action = rng.uniform(-.5,.5,(env.num_robots,3))
                obs,reward,term,trunc,_ = env.step(action)
                digest.update(np.array([reward,term,trunc],np.float64).tobytes())
                for key in ('positions_mm','velocity_mm_s','masses','active','path_mm'):
                    digest.update(getattr(env,key).tobytes())
                if term or trunc:break
            digests.append(dict(seed=seed,sha256=digest.hexdigest()))
        results[name] = digests
    atomic_json(args.out,dict(source_root=str(args.source_root.resolve()),results=results,
        scope='Five fixed action streams per EXP32 arm; entire observations, initialization, physics and reward byte comparison. Not policy performance.'))


if __name__ == '__main__':main()
