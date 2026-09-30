# EXP_0014_ADAPTIVE_EDGE_GAT

This experiment adds an edge-conditioned communication gate to Direct Local
GAT-MAPPO. For each robot pair, the existing 8-D relative position/velocity
edge descriptor produces both a per-head attention-logit bias and a sigmoid
message gate. The gate scales value messages after attention, allowing the
policy to suppress neighbours whose relative motion is unhelpful or unsafe.

The experiment keeps the Direct Local Frenet action, geometric36 observation,
reward, simulator, PPO settings, seeds, and 1M-transition budget unchanged.
The `gat` arm is the matched architecture baseline. This is an algorithmic
architecture comparison, not a claim of novelty until literature verification.

Formal training uses CPU affinity `0-5,8-23` because unrestricted execution on
this host has reproduced native bus errors inside the existing geometry kernel.
The smoke gate and all formal runs must retain their logs and exit codes.
