# EXP0052 — common measured tracking and longer allocation options

Registered 2026-10-04 after EXP0051 completed all 184 main/stress attempts. All six EXP0051 learned deterministic policies reproduced the memory baseline on all 12 main scenes; changed weights alone were not a useful learning result. The stronger measured allocation baseline remains in the comparison.

## Specific changes and controls

1. A command-aligned observer subtracts the known requested-command integral before filtering unmodeled drift, and integrates known commands over image latency. It reads the same delayed noisy images, never true flow, velocity, routes or mass. In four shadow replays over two preflight scenes it reduced estimator errors without changing any reference trajectory. This is engineering evidence, not a learning gain, and all new arms share this observer.
2. High-level target/yield options last at most 5 seconds, while common measured safety/navigation still runs at 10 Hz. Targets retain their detection identity; locally observed disappearance ends an option early. The matched flat learner reselects every 0.1 second with identical rewards and physical step budget.
3. A measured balanced-allocation recommendation is the nominal prior. An untrained network reproduces this strong comparator exactly; training learns deviations. The prior is not privileged and initialization improvements are not counted as learning gains. It has low initial logit strength (0.5) for exploration.
4. Conventional allocation is evaluated at 0.1, 1 and 5 seconds, plus priority/yield at 1 second and memory at N=3 and N=1. A slower conventional decision rate is not used to manufacture a learning advantage.

## Frozen budget and pools

Two variants × three seeds (42/43/44) × 32,768 physical control steps = 196,608 training steps. Main evaluation uses 12 paired scenes 1820000000–1820000011, eight baseline/untrained arms and six trained arms: 168 attempts. Separate synthetic coupling stress uses four fixed scenes 1850000000–1850000003 and all 14 arms: 56 attempts. A two-scene preflight uses 1840000000–1840000001 and eight arms: 16 attempts.

Training base 1810000000 plus 100000 per seed. Confirmation 1830000000–1830000039 remains unopened. All requested and internally resampled reset seeds are guarded before actual environment reset. Initial physical snapshots, source files and checkpoints are retained.

The 2 mm physical separation threshold, original wall/particle/body-contact definitions, observed completion, and total catalytic-rate proxy are unchanged. The actuator stress remains synthetic alpha=0.1 with uncalibrated length 2 mm; no claim of magnetic independence or continuous-time safety is made. The existing native crash is not considered fixed just because EXP0051 completed.

All training seeds and attempts are reported. Failed processes block formal ranking and are never replaced. Generic hierarchical RL, PPO, shared filtering and heuristic initialization are not algorithmic novelty claims. Any project-specific contribution must survive all strong matched comparators and the flat ablation.
