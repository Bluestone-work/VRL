# EXP0051 primary-source notes — 2026-10-04

Retrieved and checked the titles and author abstracts from the original arXiv records. This is a targeted method check, not an exhaustive novelty review or full replication.

1. Bacon, Harb and Precup, *The Option-Critic Architecture*, arXiv:1609.05140, `https://arxiv.org/abs/1609.05140`. The paper learns intra-option policies and termination along with option selection. EXP0051 uses persistent options but keeps low-level skills and their termination rules fixed; therefore it is not an Option-Critic reproduction.
2. Nachum, Gu, Lee and Levine, *Data-Efficient Hierarchical Reinforcement Learning*, arXiv:1805.08296, `https://arxiv.org/abs/1805.08296`. HIRO uses goal-conditioned layers and an off-policy correction for changing low-level behavior. EXP0051 borrows the separation of scheduling and execution; it uses on-policy PPO and a frozen shared controller, not HIRO's algorithm or sample-efficiency claim.
3. Achiam, Held, Tamar and Abbeel, *Constrained Policy Optimization*, arXiv:1705.10528, `https://arxiv.org/abs/1705.10528`. The work treats constraints separately from task reward. EXP0051 retains independent realized safety endpoints, but its weighted reward and approximate projection do not implement CPO or inherit its guarantees.

The intended learning hypothesis is temporal target/priority commitment under delayed, noisy observations. Temporal abstraction, PPO, target assignment, and safety projection are established ingredients. Only reproducible gains over matched strong schedulers and the flat learning ablation would support a useful project-specific result. Hardware actuation feasibility remains an additional unmet requirement.
