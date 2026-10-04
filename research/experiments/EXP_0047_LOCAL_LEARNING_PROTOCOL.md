# EXP0047 — learning local maneuvers under joint clearance constraints

Registered 2026-10-04 before training. Parent: EXP0046 v2. User explicitly
requests continued learning-based research and comparison with heuristics.

## Scientific question

Can a parameter-shared PPO selector learn when to continue, slow, wait, retreat,
or use another locally visible vessel direction, improving safe removal or
efficiency over conventional controllers with the same sensing, candidate
actions, actuator limits and joint wall/separation processing?

The learning mechanism is maneuver selection from measured local state and
recent progress. A better hand-written wall filter alone is not counted as a
learning contribution. The strongest shared-filter conventional baseline and
an untrained selector are required controls. Method 1 remains N=1; method 2 is
the main N=3 study, followed by N=2 transfer if the pilot is informative.

## Inputs and physical assumptions

Reuse EXP0046's sensor boundary: local noisy navigation, visible peer relative
positions/velocities within 6 mm, tracked clot IDs/status, previous commands and
history constructed from these measurements. No route lookup, true flow,
simulator edge ID, global topology or exact safety labels at policy inference.
Simulator contact/removal signals may define training rewards and evaluation
metrics; they are not added to actor observations.

All policies share ideal independent bounded actuators and the same fixed total
catalytic-rate proxy. Imaging and magnetic coupling remain uncalibrated. Joint
constraint handling is heuristic under noise and flow, never a certificate.

## Training and selection

Full settings and disjoint scene ranges are in
`configs/experiments/EXP_0047_LOCAL_LEARNING.json`. Three independent training
seeds; pilot 32,768 environment steps each, with optional predeclared extensions
to 65,536 and 131,072. Report environment steps separately from agent samples.
Snapshots, configuration, source hashes, checkpoint identity, episode attempts
and wall-clock cost must be recorded.

Validation uses only the registered validation pool. Select by cluster-safe
success, then common-horizon clearance AUC. Keep all checkpoints and negative
results. Confirmation seeds must remain unused until method/checkpoint freeze.
No registered repository sealed-test split is accessed.

## Comparisons and inference

Same paired scenes, nested starts, noise seeds, durations and resource budgets.
Report strict cluster-safe success, raw clearing, wall/particle/spacing contact,
removal AUC and milestone failure penalties. Compare learned vs joint heuristic,
memory heuristic, old spacing heuristic and the single-cluster baseline.
Average training-seed repetitions within scenes before paired scene bootstrap;
do not call repeated seeds independent anatomies. Distinguish predefined primary
evidence from exploratory wins on secondary metrics.

Native faults remain unresolved. Preserve failed attempts and any resumed
training budget; do not silently replace failed evaluation rows. A new clean
diagnostic batch does not retroactively repair EXP0046 or establish runtime
reliability. No journal-level novelty or acceptance probability is claimed.
