# EXP_0015 Dynamic Risk-Aware Direct Local MAPPO

This is a new experiment and does not overwrite any earlier run. It combines
the validated `adaptive_edge_gat` actor with two explicitly registered changes:

1. `geometric_dynamic` appends eight local features describing the nearest
   currently observed dynamic obstacle: relative Frenet position, clearance,
   relative speed, overlap count, and two distance summaries. It does not expose
   future particle positions.
2. `risk_aware_connectivity` extends the existing connectivity allocator with
   current corridor clearance risk, current relative-speed risk, and an explicit
   geodesic path-length cost. It emits only clot assignments; Direct Local
   Frenet actions remain policy outputs.

The run is evaluated with success, removal, total robot path length, path length
per removed mass, wall contact, and robot collision. The 85% value is a
pre-registered acceptance gate, not an expected result. If the three-seed mean
does not reach it, the report must say so.
