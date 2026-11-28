# Magnetic validation protocol (Phase I; NOT RUN — requires the physical platform)

No physical data exist in this repository. Every V3 result uses d_safe = 2 mm and a distance-only
interaction model; this protocol defines the measurements that would replace those assumptions.

## M1. Actuation degrees of freedom and independent control
* Record the coil/magnet configuration, field and gradient ranges, update rate and the number of
  independently addressable field patterns. State how two clusters receive different commands
  (selective field-free point, time multiplexing, heterogeneous clusters, or none).
* Output: `platform.json` with n_max_independent (replaces `resources.n_max` = 4 in `configs/v3/assumptions.json`).

## M2. Interference versus distance (calibrates d_safe)
* Two clusters of the size used in navigation, one actuated, one passive (field off for it if possible),
  in still fluid and in the perfused channel. Centre distances 0.5–6 mm in 0.5 mm steps, ≥ 10 repeats.
* Measure the displacement of the passive cluster and the deviation of the actuated one from its command.
* d_safe = smallest distance at which both deviations stay below the navigation tolerance (0.3 mm per
  1 s) in 95% of repeats. Report centre-to-centre and surface distance.
* Analysis: `analysis_interference.py` (to write) fitting deviation ~ a·d^-k with bootstrap intervals.

## M3. Catheter deployment
* Measure catheter advance and retraction speed in the phantom, the minimum lumen it enters, and the
  release interval needed before the tip is clear (replaces `deployment.*`).

## M4. Navigation and multi-cluster trials
* Phantom with ≥ 3 lesion marks; flow regimes matched to weak / strong / pulsatile; N chosen by the
  planner on the phantom's segmented centreline; ≥ 10 trials per condition; report the four metrics with
  the definitions in `FOUR_METRIC_DEFINITIONS.md` (contact from video at ≥ 10 fps).

## Data and analysis rules
* Raw videos and tracking CSVs kept with SHA-256; every exclusion listed with its reason.
* Simulation predictions for each trial are produced before the trial from the same map.
