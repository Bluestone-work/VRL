# EXP0048 information and measurement audit

Recorded 2026-10-04 before the main tracked validation completes.

| Quantity | Controller source, identical for learned and heuristic arms | Remaining qualification |
|---|---|---|
| Current/relative velocity | Delayed centroid differences and EMA | Noise/latency are assumed, not calibrated |
| Ego coordinate frame | Fixed camera/actuator axes | Registration is assumed |
| Local lumen and paths | Cropped, noisy, voxelized segment reconstruction | Synthetic geometry is not real image segmentation |
| Clot coordinates | Noisy pre-operative locations, then local centroid updates | Pre-operative target inventory is assumed complete |
| Remaining clot state | Binary classifier and three visible negative observations | No exact mass; classifier errors are assumed |
| Remote clearance | None; previously unseen targets stay pending | Locally acquired observations are shared centrally |
| Peer states | Finite-lived tracked centroids within sensing range | Object identity association is still idealized |
| Global route, flow, true station/edge | Not read by the measurement processor or controllers | Simulator may use these internally for physics/rendering |
| Completion | Measured absence confirmation; never true all-clear alone | Independent truth verifies the reported outcome |
| Training reward and evaluation | Simulator outcome instrumentation | These labels never enter policy features |

Both sides receive the same 138-feature interface, shared maneuver definitions
and approximate constraint projection. Method 1 remains the N=1 local baseline
with the same sensing and aggregate catalytic-rate proxy. N=3 is pre-deployed;
insertion feasibility, total magnetic power and total material are not matched.

It would be incorrect to state that every sim-to-real assumption has been
removed. In particular, perfect identity association, synthetic local geometry,
fixed calibration and independent magnetic actuation are still material limits.
The present claim is narrower: removed identified navigation truth shortcuts
and subjected all controllers to the same explicit measurement stress model.

The initial preflight retained a native exit 139. A separate debugger repeat
completed; it does not repair or replace the original failure. Numerical
attempt accounting and controller performance are separate gates.
