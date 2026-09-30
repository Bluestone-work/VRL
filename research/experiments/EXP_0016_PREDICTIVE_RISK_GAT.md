# EXP_0016 Predictive Dynamic-Obstacle MAPPO

This experiment extends EXP_0015 with a causal short-horizon predictor. The
predictor extrapolates the currently sensed nearest obstacle using measured
relative velocity for 0.45 simulation steps, and supplies predicted clearance,
time-to-contact, uncertainty, and collision probability to the policy. The
allocator uses the same current-state constant-velocity estimate when scoring
candidate clot corridors. No future simulator state is read.

The experiment is registered as a new run and cannot overwrite EXP_0015. The
85% three-seed success threshold remains a pre-registered gate; failure must be
reported, and any later improvement needs a separate experiment.
