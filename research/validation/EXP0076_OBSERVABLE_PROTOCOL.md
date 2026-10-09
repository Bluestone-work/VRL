# Observable-history repair and revised primary latency scope

User steering: focus primary training/validation on 1–2 control-step imaging delay;
retain 3-step delay as a secondary stress condition. At the current dt=0.1 s these
are nominal 0.1/0.2/0.3 s delays, before additional frame-hold/exposure effects.
Changing scope is not an algorithmic gain. Old high-delay results are retained,
and old/new methods must be evaluated on identical primary conditions for claims.

## Why the previous Ours is not a no-privileged-supervision model

flow_aux and belief use true local flow/response labels from simulator position
and transport. Turning their loss off during fine-tuning does not remove that
information from pretrained weights. EXP0075 warm starts therefore remain
legacy privileged-pretrained diagnostics and are not eligible fair initializers.

The current image interface also returns env.active and env.masses > 0 as activity
and clot-visibility flags. These are idealized detection/status channels, not
pixel-derived detectors. All current methods share them, but a fully pixel-only
deployment claim requires a separate sensing implementation and validation.
Simulator rewards and evaluation scores remain unchanged; no new true flow,
response or robot-position supervision is introduced by this repair.

## Implemented opt-in repair

- LysisEpisode stores the final world-frame command after hold/spacing filtering
  and coordinate conversion, BEFORE hidden physical gains/noise/lag. It is the
  sent command, not the unknown realized plant velocity.
- ImageSensor attaches the acquisition timestamp from the actual delayed frame
  queue. Startup duplicates and held frames retain the old timestamp.
- --observable-history supplies this sent command in token columns 23:26 and
  appends frame age in seconds (40 inputs total). Velocity and sent command are
  both world-frame quantities. Train/evaluation paths use the same contract.
- --no-privileged-supervision rejects pretrained initialization, simulation ECG,
  legacy ABCD combinations, and nonzero privileged matrix auxiliary objectives.
  Combined with --matrix-arm nav_tf_v3 --aux-weight 0 it builds a fresh Transformer
  without dynamics-label collection. This name describes supervision, not a claim
  that the idealized status channels above have been replaced with pixel detectors.
- Existing checkpoints without these flags retain their original 39-input contract.
  Changes are opt-in; old and repaired training must not be presented as a pure
  history-network ablation.

## Verification

12 tests pass across test_image_sensing.py and test_lysis_abcd_eval.py, including
delayed acquisition timestamps, frame holds, final command after hold, 40-input
forward pass, old defaults, GAE/truncation and samplewise prior weighting.
An independent 2-update CPU smoke uses fresh initialization, delays 1/2, flow
.025/.05 and existing comprehensive dynamics sampling. It validates execution,
not performance. Artifacts: research/runs/EXP0076_OBSERVABLE_SMOKE_s7600.

```bash
python -m scripts.train_lysis_nav --out research/runs/EXP0076_OBSERVABLE_SMOKE_s7600 --matrix-arm nav_tf_v3 --updates 2 --workers 2 --seed 7600 --device cpu --speed-prior --latency-max 2 --flow-levels .025,.05 --aux-weight 0 --observable-history --no-privileged-supervision
```

## Next learning experiment (not yet a result)

First establish the repaired Transformer with no privileged auxiliary head, from
scratch and at a predeclared equal budget, in the revised delay-1/2 scope. Preserve
full action mapping and strong stateful classical comparators. Curriculum or
observable prediction should be tested one at a time, not together with action,
reward and model changes.

If adding prediction, predict observable displacement/motion from timestamped
image history and sent commands; align command/measurement times under delay,
mask repeated/missing frames, normalize by control time and robot speed. Compare
against zero-motion and constant-velocity/persistence predictors on validation
scenes. A low auxiliary error alone does not demonstrate useful dynamics learning.
Do not impose the simplistic target 'current action causes next received velocity'
when the next frame was acquired before that action.

Comprehensive-dynamics success could benefit from a staged curriculum: first
learn approach/residence under weaker variability, then increase response/flow
variation at reset while retaining easy replay cases. This is a hypothesis, not
an implemented improvement or evidence that increased network capacity is needed.
At least three training seeds and matched baseline comparisons are required for
stable-gain claims. The primary evaluation retains difficult flow/dynamics cases;
only latency scope has been revised by the user.
