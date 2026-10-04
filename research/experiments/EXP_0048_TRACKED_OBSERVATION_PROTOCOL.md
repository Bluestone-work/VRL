# EXP0048 — matched tracking observations for learning and heuristics

Registered 2026-10-04 following the user's explicit requirement that heuristic
controllers also receive no simulator-only navigation privilege.

## Why a new revision is required

EXP0047 policies have no environment handle, but their shared sensor adapter
still constructs observations from instantaneous true body/particle velocities,
exact clot mass fractions and exact Frenet frames. Its local geometry is also
an ideal reconstruction. A fair information boundary between policies alone is
insufficient to establish realistic sensing or sim-to-real accuracy.

EXP0047 data and 32k/64k checkpoints remain preserved as ideal-sensor development
results. They are not pooled with the new observation contract. Confirmation
seeds have not been accessed. Learning remains the active research direction.

## Shared observation contract

Both learned and heuristic controllers receive identical packets built from:

- noisy tracked centroids and finite-difference, smoothed velocity estimates;
- a calibrated fixed camera/actuator coordinate frame, not a true Frenet frame;
- only cropped local image-like geometry, with measured positions/radii;
- pre-operative target coordinates plus locally observed binary clot presence;
  no exact remaining mass or global instantaneous clearance notification;
- delayed frames, missing detections, explicit finite track lifetime, and
  repeated local absence observations before marking a target cleared.

Simulator truth is permitted inside the renderer and independent training
reward/evaluation instrumentation. The policy adapter cannot query true flow,
velocities, routes, assignments, body edge IDs, masses or global topology.
Tests must poison or vary those hidden quantities and verify invariance whenever
the rendered measurements are unchanged.

Engineering stress settings are assumptions, not measured hardware specs:
10 Hz frames, 0.1 s delay, 0.02 mm centroid error, 5% missing body detections,
0.02 mm local centerline error, 5% radius error, 0.05 mm pre-operative registration
error, 1% binary clot-classification error, three absent clot detections for
clearance, 0.3 s track lifetime.
Actual hardware still must demonstrate the required local 3D imaging and
camera-to-field registration. Cropped synthetic geometry is not a validated
image reconstruction pipeline, and actuator independence remains idealized.
Detection-to-identity associations are still idealized and explicitly require
validation; these experiments cannot yet establish sim-to-real success.

The rollout does not stop at simulator all-clear. It continues through delayed
observations until the shared measurement processor confirms every target absent,
all clusters exit, or 180 s elapses. The primary cluster-safe success requires
both true clearance and measured completion, with all safety conditions measured
through this observed stopping time. False visual completion and physically safe
clearance without observed confirmation are reported separately. Physical active
time, not the number of currently detected tracks, is the wall-contact denominator.

## Experiment sequence

1. Test the rendered-frame/policy separation and repeatability.
2. Compare matched heuristic controls on fresh development scenes; report every
   failure, raw clearing, cluster-safe success, wall and spacing contact.
3. Train the shared PPO maneuver selector from scratch with this same contract;
   compare three independent seeds against the strongest matched baseline.
4. Use validation only for choices; freeze before any fresh confirmation data.
   Do not claim a learning benefit from changing the sensor, reward, resource
   budget or shared hand-written filter in only one arm.

Primary metric remains cluster-safe success. Efficiency and safety tradeoffs
must be reported together, with paired scene-level uncertainty. No guaranteed
outperformance, clinical efficacy, magnetic independence or journal novelty.
