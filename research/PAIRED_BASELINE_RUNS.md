# Paired baseline runs

Run from the repository root in the VRL training environment. Save new results under
`runs/`. Use a separate output directory for each smoke test and full experiment.

## Three-cluster navigation

```bash
python -m unittest discover -s tests -p test_external_comparison_metrics.py
bash scripts/run_matched_actor_critic.sh 41 256 runs/actor_critic smoke
for seed in 41 42 43; do
  bash scripts/run_matched_actor_critic.sh "$seed" 50176 runs/actor_critic || break
done
```

The three arms use the same deployed local encoder, valid action set, flow input,
safety guards, reward, training curriculum rules and physical interaction budget.
IPPO and IA2C use local value functions. MAPPO-V2 uses a centralized critic in
training. IA2C uses the upstream Stable-Baselines3 A2C update with a 32-step
rollout. PPO uses 256 steps, four epochs and minibatches of 128. Training
trajectories can diverge under the adaptive curriculum. Set `TRAIN_DEVICE=cpu`
to force CPU training. The runner checks both physical steps and agent steps.

Every final checkpoint is evaluated on the frozen train-180 and holdout-100
manifests. Smoke mode takes the first scene from each split. By default the
runner starts a fresh simulator process per scene and reuses matching completed
scene files on retry. Keep `EVAL_CHUNK_SIZE` fixed throughout an evaluation.
Each seed produces `paired_summary.json` with strict success, navigation
success, paired differences and discordant counts. The exact McNemar p-value
describes the scene pairs for that training seed. Report seed-level variation
separately. Reserve holdout results for final comparisons.

## Planar single-agent navigation

```bash
CUDA_VISIBLE_DEVICES="" bash scripts/run_paired_mr_nav.sh --steps 256 --scenes 2 --out runs/mr_smoke
for seed in 41 42 43; do
  CUDA_VISIBLE_DEVICES="" bash scripts/run_paired_mr_nav.sh --seed "$seed" --steps 50176 --scenes 100 --out "runs/mr_s$seed" || break
done
```

The wrapper fetches the pinned yinghansun/mr-nav revision, including its released
VesselEnv checkpoint. It needs torch, matching torchvision, Gymnasium, SB3, OpenCV,
Pillow and tqdm. The experiment uses native vessel dynamics, native LiDAR and a
continuous radian heading. The three arms are the published deterministic actor
with its observation normalizer, a newly trained planar PPO with the VRL encoder,
and direct target heading. Each sees the same scene, starts, goals, disturbance
seed, and deadline. Inputs come from synthetic bifurcations with disjoint
training and evaluation map seeds. This measures transfer on a common task.
Training budgets for the published checkpoint and the retrained controller
differ. The two deterministic reference arms repeat across training seeds.

## Fixed-corridor scheduling

```bash
bash scripts/run_ccbs_baseline.sh --pairs 1 --out runs/ccbs_smoke
bash scripts/run_ccbs_baseline.sh --pairs 20 --out runs/ccbs_full
```

On Linux the wrapper builds the pinned PathPlanning/Continuous-CBS C++ solver
with CMake, a C++ compiler, Boost headers and the static runtime libraries used
by upstream. Its XML inputs, solver output and path logs are kept for inspection.
The VRL TPG, uncoordinated execution and upstream CCBS share corridor geometry,
endpoints, unit travel speed and 0.6 minimum centre separation. Each seed yields
one low-conflict and one high-conflict crossing. CCBS can wait at a vertex for a
continuous duration and move along either direction of its corridor. TPG advances
or waits at 0.2-second intervals. A common continuous-time trajectory scorer
includes occupancy of parked goals and reports safe completion, separation
violations and completion time capped at 30 seconds. Solver timeouts and
infeasibility are retained. This experiment measures scheduling on planar
crossings. The separate VRL three-dimensional fixed-route adapter has its own
results and execution assumptions.
