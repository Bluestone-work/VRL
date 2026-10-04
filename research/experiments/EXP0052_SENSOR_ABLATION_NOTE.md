# EXP0052 paired observer ablation — exploratory registration

2026-10-04. Registered during EXP0052 validation, after its baseline results became visible and before executing the old-observer counterfactuals. This is explicitly post-hoc development diagnosis, not a preregistered confirmation test.

Run the frozen EXP0051 memory controller on ALL 12 EXP0052 development layouts (1820000000–1820000011), and compare against the already requested EXP0052 memory arm. Neither weights nor scenes are selected for favorable effects. The physical initialization snapshots, common source hashes, sensor noise/latency/dropout settings, shared supervisor settings, physical configurations and catalytic-rate proxy must match. The intended difference is command-aligned tracking. The extra high-level recommendation in EXP0052 is not used by memory control.

EXP0051's memory option nominally renews every 1 s; EXP0052's every 5 s. Both always select memory and recompute the identical low-level controller every 0.1 s. A direct equal-interval repeat must verify that this bookkeeping difference does not change the old-observer trajectory before interpreting the comparison.

All attempts are retained. Any failed attempt blocks an admitted paired estimator/control comparison. Result truth is used only for scoring. This diagnosis does not turn a common observation improvement into a learning contribution or establish hardware transfer.
