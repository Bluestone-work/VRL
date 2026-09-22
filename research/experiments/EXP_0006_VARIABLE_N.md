# EXP_0006_VARIABLE_N — Variable Agent Count via max_agents + agent_mask

## Preregistration

| Field | Value |
|---|---|
| Experiment ID | `EXP_0006_VARIABLE_N` |
| Parent | `EXP_0005_DIRECT_LOCAL` |
| Research question | Can one Direct Local GAT-MAPPO checkpoint serve different robot counts (3/5/8/10) through padding + masking, with padding agents excluded everywhere? |
| Main variable | Agent-count machinery only (`max_agents`, `agent_mask`) |
| Invariants | reward, MAPPO hyperparameters, direct-Frenet action semantics, physics all unchanged; world model disabled |

## Mechanism

- **Policy side.** `MAPPOAdvanced(max_agents=N)` records the capacity in checkpoint meta. GAT layers accept an `agent_mask`: padding slots are masked as both attention *keys* and *queries* (real agents never attend to padding; padding attends to nothing; fully-masked rows are NaN-rescued to zero). The PPO update zeroes padding in rewards/dones/terminals/values/bootstrap terms, masks the actor/critic/entropy losses, computes advantage statistics over real slots only, and divides every team term by the per-sample real-agent count — which is also what makes **mixed-N batches** (different active counts per sample) valid in one update.
- **Environment side.** `VectorVascularEnv(active_robots=a, num_robots=N)` simulates only the first `a` slots; padding slots have zero state, zero reward, and are excluded from pair terms, lysis contact, wall stats, peer features and the observation. The env's RNG stream is unchanged (noise is drawn for active slots only), so a padded env with the same seed reproduces the unpadded env **bit-identically**.
- **Parameter sharing** is retained: one weight set; agents differ only by observation.

## Validation

- `tests/test_variable_agents.py`: padded forward/update equivalence for **3/5/8** real agents in a 10-slot tensor; mixed-N batch (3/6/5/2) trains with finite losses; checkpoint save with `max_agents=10` reloads and runs at `n_agents=3`; env padding bit-identity vs the legacy env over 20 steps.
- Two real bugs were caught and fixed by these tests: (1) advantage normalization originally included padding zeros, silently coupling the update's scale to the padding ratio; (2) the mask was dropped from `ctx_all` in the vectorized reshape branch.
- Smoke: `scripts/run_scaling_smokes.sh variable_n` (8 slots / 5 active / capacity 10, 448 transitions, finite PPO update).

## Status

Capability implemented and tested. **No performance claim**: whether a checkpoint trained at one N transfers to other N's is a training question requiring new 1M-transition runs (not authorised in this round; see the long-run gating rule). EXP_0001–EXP_0005 reproduce unchanged because the default (`active_robots=None`, `max_agents=0`) is the legacy fixed-N path.
