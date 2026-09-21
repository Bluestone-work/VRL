# EXP_0005_DIRECT_LOCAL — Direct Local GAT-MAPPO Navigation

## Preregistration

| Field | Value |
|---|---|
| Experiment ID | `EXP_0005_DIRECT_LOCAL` |
| Parent | `EXP_0004`, reference arm `geodesic_v` |
| Research question | Can GAT-MAPPO learn direct Frenet-frame navigation after removing the hand-authored flow-guided controller? |
| Main variable | Action execution semantics only: actor output is a direct `[u_parallel, u_normal, u_binormal]` Frenet action, then a basis transform |
| Algorithm | GAT-MAPPO, V critic, geometric36, geodesic contact, existing milestone reward |
| Seeds | 42, 43, 44 |
| Training budget | 1,000,000 real environment transitions per seed |
| Device | `cuda:0` |
| World model / MVE / MPC / Dreamer | Disabled |
| Prospective protocol | EXP_0004 prospective validation manifest; sealed test is not accessed |

The reference baseline is not redefined here: `EXP_0004/geodesic_v` is recorded as the existing strongest baseline, with the preregistered reported prospective success of 76.67%. Per-seed reference artifacts must be available before a quantitative paired comparison is claimed; otherwise the report labels the comparison as reference-only.

## Action semantics

The local policy path is:

```text
geometric36 observation (including blood-flow features and geodesic lookahead)
  -> GAT actor
  -> bounded local action [u_parallel, u_normal, u_binormal]
  -> Frenet basis transform using tangent/normal/binormal only
  -> environment.step(world action)
```

`route_guidance`, `flow_guidance`, `spread_flow_guidance`, residual addition, desired-velocity subtraction, radial flow offsets, and near-wall compensation are forbidden in `control_mode=local`. `residual_scale` remains in the parity configuration and checkpoint metadata but is intentionally ignored by the direct-local execution path. `local/world` actor heads use the existing standard network initialization; only controller-residual modes use zero mean-head initialization.

The transition dataset records both `policy_action` (Frenet local frame) and `executed_action` (world frame). The local-control tests assert that changing `residual_scale` cannot change the executed action and that controller functions are never called.

## Smoke gate

Before formal training, seed 4242 runs 16,384 real transitions with the same network, physics, reward, optimizer and curriculum settings. The gate requires finite observations/actions/rewards/losses/gradients/parameters, non-zero actor and critic updates, checkpoint save/load, evaluation execution, and an action audit showing `executed_action == FrenetTransform(policy_action)` with no guidance call.

## Formal evaluation

Each seed is trained independently for 1M transitions. The final policies are evaluated on the EXP_0004 prospective validation manifest only. Reported metrics are success, mass removal, episode length, completion time, first contact, wall contact/robot-step, pair collision/pair-step, mean action magnitude, mean flow exposure, and mean simultaneously contacted clots. Difficult territories are always reported separately: `femoropopliteal_pad`, `coronary_rca`, and `carotid_bifurcation`.

Failure labels are non-exclusive:

- `navigation_failure`: no clot contact;
- `contact_acquisition_failure`: near a clot but no sustained acquisition;
- `contact_maintenance_failure`: contact occurred but contact duty cycle is short;
- `completion_tail_failure`: removal exceeds 90% but episode is not complete;
- `wall_dominated_failure`: high wall-contact rate;
- `robot_crowding_failure`: high pair-collision rate;
- `timeout`: reaches horizon without another label.

Flow-learning diagnostics stratify local flow magnitude into low/medium/high bins and report each local action component plus `policy_action · flow_direction`. No learned compensation is assumed in advance; absent evidence is a result.

## Interpretation boundary

This experiment isolates direct model-free motion control. A poor result is retained and is not repaired by reward, optimizer, network, contact threshold, horizon, or budget tuning. Any such change requires a new experiment ID, for example `EXP_0006_DIRECT_LOCAL_CURRICULUM`. A successful result does not activate the world model; the next fair study is direct-local model-free RL versus direct-local plus a world model with identical local action semantics.

## Status

Preregistered in the current VRL repository before formal training. Existing EXP_0001–EXP_0004 records were not modified.

## Execution Record

The first CUDA smoke attempt received a native `SIGILL` in the existing vectorized environment step before the first PPO update. The failed attempt is retained at `research/runs/EXP_0005_DIRECT_LOCAL/smoke`; retrying with the established CPU affinity (`0-5,8-23`) passed. This was recorded as a native-signal retry, not a performance retry.

The first read-only evaluation pass is retained under `evaluation_metric_attempt1` and `evaluation_baseline_metric_attempt1`. Before final reporting, its flow-exposure definition was aligned with the existing protocol's cumulative per-robot `blood_flow_exposure`, and failure labels were restricted to failed episodes. The corrected pass reuses the same checkpoints and manifest; no training result changed.

The passing smoke used 16,384 transitions and completed two PPO updates plus deterministic evaluation. All observations, actions, rewards, losses, entropy, gradients, and checkpoint tensors were finite; actor and critic parameters changed between checkpoints; checkpoint reload succeeded; 16,384 transitions recorded both `policy_action` and `executed_action`. The independent audit confirmed the latter is the Frenet transform of the former and that changing `residual_scale` has no effect in local mode.

Formal training completed for seeds 42, 43, and 44 at exactly 1,000,000 real transitions each on `cuda:0`/`cuda:1` (one job per GPU). No training retry, world model, MVE, MPC, Dreamer, reward adjustment, optimizer adjustment, or curriculum tuning was used.

## Prospective Validation Result

The evaluator consumed 140 validation records per seed from the EXP_0004 prospective manifest (420 episodes total). The sealed test split was not accessed. Values below are means over the three training seeds; SD is sample SD over seed-level metrics.

| Metric | Flow-guided baseline | Direct Local | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| Success | 0.766667 +/- 0.017976 | 0.714286 +/- 0.091194 | -0.052381 | -6.83% |
| Mass removal | 0.945767 +/- 0.003796 | 0.878584 +/- 0.038851 | -0.067182 | -7.10% |
| Episode steps | 144.883333 +/- 7.925071 | 155.414286 +/- 21.085544 | +10.530952 | +7.27% |
| Completion time (successful only) | 97.757038 +/- 5.871125 | 97.797868 +/- 6.803460 | +0.040830 | +0.04% |
| First contact time | 7.969048 +/- 0.342956 | 5.645238 +/- 0.698260 | -2.323810 | -29.16% |
| Wall contact / robot-step | 0.343963 +/- 0.022855 | 0.612671 +/- 0.034126 | +0.268708 | +78.12% |
| Robot collision / pair-step | 0.076889 +/- 0.006230 | 0.124003 +/- 0.033925 | +0.047114 | +61.27% |
| Mean action magnitude | 0.965644 +/- 0.032659 | 0.918816 +/- 0.073191 | -0.046828 | -4.85% |
| Mean flow speed | 0.013810 +/- 0.000517 | 0.010966 +/- 0.000099 | -0.002844 | -20.60% |
| Blood-flow exposure (per robot cumulative proxy) | 2.157267 +/- 0.061644 | 1.844011 +/- 0.242868 | -0.313256 | -14.52% |
| Simultaneously contacted clots | 0.655413 +/- 0.054502 | 0.667972 +/- 0.043817 | +0.012559 | +1.92% |

The flow-guided baseline was evaluated from the existing `experiments/ladder_stage1/geodesic_v_seed{42,43,44}` checkpoints using the same manifest and environment settings; it was not redefined or selected after seeing Direct Local results.

## Difficult Territories

| Territory | Success | Removal | Steps | Wall contact | Pair collision |
|---|---:|---:|---:|---:|---:|
| `femoropopliteal_pad` | 56.67% | 99.47% | 166.97 | 89.10% | 30.21% |
| `coronary_rca` | 100.00% | 100.00% | 70.67 | 76.60% | 10.03% |
| `carotid_bifurcation` | 33.33% | 83.03% | 223.17 | 75.61% | 7.77% |

## Failure Diagnosis

Labels are non-exclusive and use the preregistered thresholds. Across failed episodes only, the labels were `wall_dominated_failure` (108), `robot_crowding_failure` (71), `contact_maintenance_failure` (72), `completion_tail_failure` (20), `contact_acquisition_failure` (19), and `timeout` (3). `navigation_failure` was 0: the trained policies reached at least one clot in every evaluated episode. Thus the main Direct Local loss is not global route acquisition; it is unsafe local motion and contact maintenance, with a smaller completion-tail component. Labels are non-exclusive, so their sum exceeds the number of failed episodes.

## Learned Flow Response

Flow magnitude was split into equal-count low/medium/high bins over recorded policy steps. The mean local action and action-flow alignment were:

| Flow bin | Flow magnitude | u_parallel | u_normal | u_binormal | action dot flow direction |
|---|---:|---:|---:|---:|---:|
| low | 0.001518 | -0.2112 | -0.3043 | -0.1732 | -0.1535 |
| medium | 0.009476 | -0.6774 | -0.1770 | +0.0298 | -0.6774 |
| high | 0.024602 | -0.4034 | -0.1573 | +0.0619 | -0.4034 |

The negative axial component and negative action-flow dot in medium/high bins are consistent with learned opposition to local flow, but this is descriptive evidence only. The experiment cannot separate flow compensation from route or contact features, and no controller term is present in the local execution path. The plot is `research/runs/EXP_0005_DIRECT_LOCAL/evaluation/flow_action_diagnostics.png`.

## Interpretation and Next Step

1. Removing the hand-authored flow-guided controller reduced success by 5.24 percentage points and removal by 6.72 points on the matched prospective validation.
2. Direct Local still reached clots reliably, but it incurred roughly 78% more wall contact and 61% more pair collision than the residual baseline.
3. The dominant bottleneck is wall-dominated local control and crowding, followed by contact maintenance; it is not a complete navigation failure.
4. The policy shows a flow-opposing action signature at medium/high flow, so RL learned some compensation-like behavior, but no causal claim is made.
5. These results do not establish that Direct Local is preferable to Direct World; that is a separate action-frame experiment.
6. A future world model should first target dynamics learning and contact stabilization/long-horizon prediction. Task allocation should remain a separate interface question; it is not justified by this experiment.
7. The next fair comparison is `Direct Local Model-Free RL` versus `Direct Local + World Model`, with imagined actions retaining the exact `[u_parallel, u_normal, u_binormal]` semantics. No curriculum or reward tuning is folded into EXP_0005.
