# EXP0047: literature boundaries and contribution hypothesis

Checked 2026-10-04 while the preregistered pilot trains. This is a targeted
primary-source check, not an exhaustive novelty review.

## Relevant primary work

1. An et al., *Autonomous navigation of intelligent microrobotic swarms in unknown environments*,
   Nature Machine Intelligence (published 2026-06-22), DOI
   `10.1038/s42256-026-01252-6`. The publisher's abstract describes Turbo learning
   collision avoidance in simulation and transferring to optical-feedback swarm
   navigation. Therefore generic learning-based magnetic swarm navigation,
   obstacle avoidance, or simulation transfer is not a new claim for this project.
2. Qin et al., *Learning Safe Multi-Agent Control with Decentralized Neural Barrier
   Certificates*, ICLR 2021, arXiv `2101.05436`, OpenReview `P6_q1BRxY8Q`.
   The primary paper jointly learns decentralized policies and barrier functions,
   with a neighbor encoder that supports changing agent counts. Neither local
   peer sensing nor jointly learned multi-agent safety is new by itself.
3. *Safe Reinforcement Learning Using Robust Control Barrier Functions*,
   arXiv `2110.05415` (2021; revised 2022). Its abstract describes a differentiable
   robust-barrier layer within model-based RL and modular reward learning. A generic
   RL-plus-barrier or learned-safety-filter combination is not itself a defensible
   novelty claim.
4. Johannink et al., *Residual Reinforcement Learning for Robot Control*,
   arXiv `1812.03201` (2018). Combining a conventional controller with learned
   corrections is established; calling such a combination "residual learning"
   does not establish algorithmic novelty.
5. Sun et al., *Minute-scale training for microrobot navigation*, Nature Machine
   Intelligence (published 2026-09-28), DOI `10.1038/s42256-026-01305-w`.
   The publisher's abstract reports vectorized vascular navigation training and
   task/shaping/regularization rewards with sim-to-real experiments. Fast
   training or reward shaping alone is therefore also a weak novelty claim.
   This note uses the abstract, not a full replication or supplement review.

## What this experiment can establish

The current PPO maneuver selector is an experimental learning baseline. Its
measurable hypothesis is useful local decisions under simultaneous wall and
inter-cluster separation constraints: waiting, retreating, choosing another
visible branch, or changing speed. Training and inference use the same explicit
local observation boundary. Strong conventional controls use the same action
library and joint projection, so improvements from the new wall filter alone
cannot be credited to learning.

A credible next contribution would need repeatable benefit over that control,
an informative learning ablation, independent scenes and training seeds, transfer
under changed N/sensing/flow, and physically justified actuator coupling. These
are evidence requirements, not completed results. No "first", safety guarantee,
hardware independence, publication acceptance probability, or clinical claim is
supported by the pilot.

## Distinguish evidence levels

- Engineering: observation boundary, paired resource-controlled comparisons,
  retained failures, actual spacing and wall measurements.
- Learning evidence: weights change through rewards and the trained selector
  improves a prespecified outcome beyond the identical untrained/shared-filter
  controller, with uncertainty and costs reported.
- Research novelty: requires a mechanism and generalizable conclusion not
  already covered by the cited navigation/residual/safety literature; pending.
