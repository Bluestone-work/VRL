# EXP0062 audit and delay mechanism report

Source: commit `2776f07`, `EXP0062_paired_v2` (1470 records). The source
records are complete and unique: 5 suites × 14 anatomies × N=1/2/3 × 7
methods, with paired scenario and variation hashes. All four checkpoints load;
their `causal` tensors contain `-inf` by design as Transformer masks, while all
learned parameters are finite. The actual training configs/logs are copied in
`training/` and checkpoint hashes are in `checkpoints.json`.

The nominal 0.0375 suite is an intermediate nominal flow, not a disjoint
unseen distribution: the per-scene multiplier means its realized support
overlaps the training support. The response parameters were fixed. C and D
were 8-step (0.8 s) GRU variants; D predicted the next observed velocity token
with PPO auxiliary weight 0.01. A and B were 32-step Transformer policies.

## Statistical correction

`stratified.json` adds `T90_300` for every episode, keeps conditional T90 and
its reach count, and retains AUC over all episodes. It stratifies by train vs
parameter-unseen anatomy and N. Bootstrap resampling uses the requested scene
(anatomy/seed) as the unit; all N and conditions from that scene remain in the
same draw. There is one scene per anatomy in this screening, so intervals are
exploratory and cannot estimate within-anatomy scene variance.

## Test-set access correction

The evaluation seeds were in the registered 2700000000 test range. Offset 11
was accessed for all 14 anatomies and is permanently reclassified as
development; no scene was deleted or replaced. The access ledger is
`access_ledger.json`. The remaining offsets are provisional sealed candidates,
not claims of an untouched test set. `multicluster_protocol.reserved_seed`
now rejects registered multi-cluster test seeds too.

## Delay mechanism

The new fixed-scene trace set uses offset 20, nominal flow 0.05, all 14
anatomies, N=1/2/3, and delays 1/2/3. It records each timestamped estimated
position, target, carrot, route progress, rule command, final sent command,
previous sent command, safety delta, active/clot state and wall contact.

For no-settle cruising:

| delay | all targets cleared | mean targets cleared | mean target departures |
|---:|---:|---:|---:|
| 1 | 36/42 | 3.81 | 117.4 |
| 2 | 0/42 | 0.00 | 122.9 |
| 3 | 0/42 | 0.00 | 279.7 |

Fixed Settle gives 11/42, 15/42, and 23/42 full clear for delays 1/2/3;
adaptive Settle is identical because response randomization is disabled.
The simple velocity damping candidate gives 6/42, 0/42, and 0/42 and is
rejected. It over-damps approach and does not solve delayed target switching.
The failure is therefore both delayed route/target tracking and inadequate
near-target residence; “entered any 0.3 mm neighborhood” is not a clearance
criterion. The trace uses 0.3 mm only as a diagnostic neighborhood.

## Checkpoint compatibility

The default for checkpoints without `prior_residual_scale` is restored to the
old 0.5 mapping. The full-scale mapping remains explicit in newer configs;
the A vs A_legacy comparison is an inference intervention on one checkpoint,
not an equal-budget retraining ablation.

## Reproduction

```bash
env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.audit_flow_delay
env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.diagnose_lysis_delay --out research/validation/FLOW_DELAY_AUDIT/trace_delay_005 --workers 12
env PYTHONPATH=. /home/wj/miniconda3/envs/v/bin/python -m scripts.diagnose_lysis_delay --out research/validation/FLOW_DELAY_AUDIT/classical_delay_005 --workers 20 --methods no_settle,settle,adaptive_settle,damped
```

The next candidate, if pursued, is one causal delay-aware feedback method
selected on training anatomies only. No GRU/auxiliary stacking or closed-test
selection is justified by this negative mechanism result.
