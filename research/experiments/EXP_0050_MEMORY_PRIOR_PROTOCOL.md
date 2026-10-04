# EXP0050 — learn interventions from the strong common-memory baseline

Registered 2026-10-04 after EXP0048 64k and EXP0049 32k development evidence.
EXP0049 64k continues unchanged. This is a new development iteration, with
fresh training/validation scenes and no access to earlier confirmation pools.

## Mechanism

Both earlier learners started at joint-without-memory control, substantially
weaker than the best conventional memory comparator. In EXP0050 the exact same
measurement-only memory controller supplies one nominal candidate indicator.
The temporal candidate scorer learns deviations. Its initial nominal logit
bias is 5 (about 94% sampling probability with ten available choices); the
deterministic untrained model exactly reproduces the memory controller.
The recommendation is generated once per measured step, with no future or
simulator state input. Heuristics have access to the same observation history.

Sensor settings, reward, physical dynamics, ten candidate actions, projection,
observed completion and all success conditions remain unchanged. This is not
a claim that a conventional prior, recurrence or PPO is intrinsically novel.
Only improvement over the identical untrained/memory baseline counts as a
learning contribution. Baseline-initialization improvements do not count.

## Registered sequence

1. Confirm unchanged conventional memory trajectories and exact untrained
   equivalence in a separate preflight pool (1640000000 onward).
2. Three seeds 42/43/44 from scratch: 32,768, then at most 65,536 steps.
   Training bases 1610000000, 1610100000, 1610200000; continuation adds 10000.
3. Twelve paired validation scenes 1620000000–1620000011, with N=3 old spacing,
   joint and memory comparators, and N=1 joint and memory controls. All failures
   remain in the denominator. Evaluate all three learned seeds without selection.
4. Primary observed-and-physical cluster-safe success; jointly report clearing,
   AUC, wall contact, spacing, T90, false visual completion and command proxy.
   Less treatment activity does not establish a safety/efficacy improvement.
5. Do not open 1630000000–1630000039 confirmation scenes before freezing a
   method with a credible development signal. No guarantee of superiority.

The unresolved preflight native crash and unvalidated sensor/actuator assumptions
remain limitations. Preserve the complete previous negative experiments.
