# EXP0075: target-local lateral residual intervention

## Evidence and scope

EXP0073 hard-condition screening shows no uniformly best method. Ours (flow_aux)
has excess wall contact in strong flow and loses to fixed Settle at high latency.
The existing route action offsets the carrot by up to 70% of the available map
radius in each normal/binormal direction. This study tests whether unnecessary
lateral offsets degrade target residence. It does not change axial speed authority,
the reward, network, sensing, shielding, metric definitions or termination.

Only the 9 registered training anatomies are used for development. The same four
conditions and paired scene jobs from EXP0073 are retained, 27 episodes per
condition. No held-out anatomy is used to choose the intervention.

## Stage 1: inference diagnostics

Both use the existing Ours seed-7101 checkpoint, not retrained models:

1. Global lateral multiplier 0.25, all states: EXP0075_LATERAL_PILOT_20261009.
2. Multiplier 0.25 only inside observed target distance 0.6 mm, otherwise 1:
   EXP0075_NEAR_LATERAL_PILOT_20261009. The radius is the existing PAC-NMPC
   final-approach radius, not a grid-search optimum. Distance uses registered
   target coordinates and image position estimates only.

Each pilot has 108 completed raw episodes. Global scaling improved high-delay
Strict from 18/27 to 20/27 but did not improve other condition success rates.
High-delay removal improved from 85.9% to 96.3%; wall time rose from 0.047 s
to 4.254 s. An isolated replay of iliac_may_thurner, N=1, explains a major outlier:
original Ours cleared only 0.9% with no wall contact, whereas scaled Ours cleared
75% then stalled en route to the last target, accumulating 113.7 robot-seconds.
Thus the extra wall time is not by itself evidence of impaired escape: the old
controller never reached this later segment. Neither pilot is a stable win.

## Stage 2: matched bounded continuation

Both arms initialize the same flow_aux_s7101/policy.pt checkpoint and run 120
PPO updates, 4 workers, continuation seed 7501, fixed 122880 environment steps.
Retained active-agent row counts can differ and must be reported separately.
Both use nominal flows .025/.05/.1, delay reset sampling 1/2/3, and the existing
variation distribution with s_max=1.25. This is continuation of the current
multi-dynamics training, not a first-stage fixed-response causal experiment.

- Control: unchanged lateral multiplier 1.
- Candidate: multiplier .25 within .6 mm only; same full axial residual scale 1.

Auxiliary weight is zero in BOTH continuations: no new privileged flow/response
labels are collected or optimized. The initializer still has privileged-supervision
history, so these cannot be labelled observable-only training. This common change
and extra PPO budget mean comparisons to frozen Ours are not pure mapping ablations;
only candidate vs matched continuation isolates the mapping in this short study.
Optimizer and schedules restart identically in both arms; this is a warm start,
not an exact optimizer-state resume.

Only the final predeclared 120-update checkpoints will be evaluated. No selection
from intermediate checkpoints, no additional hyperparameter sweep. Promotion
requires improvement over matched continuation without sacrificing difficult-condition
completion or hiding wall/spacing penalties. Otherwise retain the negative result
and do not start multi-seed expansion. A positive pilot would still require at
least three training seeds and later frozen-method generalization validation.

## Reproduction

```bash
python -m scripts.evaluate_hard_baselines --out research/validation/EXP0075_LATERAL_PILOT_20261009 --methods ours_lateral025 --policy research/runs/EXP0073_LONG1M_20261009/flow_aux_s7101/policy.pt --lateral-scale .25 --train-only --workers 16
python -m scripts.evaluate_hard_baselines --out research/validation/EXP0075_NEAR_LATERAL_PILOT_20261009 --methods ours_near_lateral025 --policy research/runs/EXP0073_LONG1M_20261009/flow_aux_s7101/policy.pt --lateral-scale .25 --lateral-near-radius .6 --train-only --workers 16
python -m scripts.train_lysis_nav --out research/runs/EXP0075_MATCHED_CONTINUATION_s7501 --init research/runs/EXP0073_LONG1M_20261009/flow_aux_s7101/policy.pt --matrix-arm flow_aux --updates 120 --workers 4 --seed 7501 --device cuda:0 --speed-prior --latency-max 3 --flow-levels .025,.05,.1 --scene-seed-base 2300000000 --aux-weight 0
python -m scripts.train_lysis_nav --out research/runs/EXP0075_NEAR_LATERAL_s7501 --init research/runs/EXP0073_LONG1M_20261009/flow_aux_s7101/policy.pt --matrix-arm flow_aux --updates 120 --workers 4 --seed 7501 --device cuda:1 --speed-prior --latency-max 3 --flow-levels .025,.05,.1 --scene-seed-base 2300000000 --aux-weight 0 --lateral-residual-scale .25 --lateral-near-radius .6
```

Use new output paths for reproduction; existing outputs are never overwritten.
The older action mapping is the default for checkpoints without the added config.
No new result is called a network, history-encoder, or auxiliary-task improvement.
