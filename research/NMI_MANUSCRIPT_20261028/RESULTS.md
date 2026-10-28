# Results

## A single shared policy drives every cluster through the fixed global layer

We cast multi-cluster vascular navigation as local decision making under a fixed
global layer (Fig. 1). Geodesic allocation, route construction and the TPG precedence
schedule are computed once and are identical for every method we compare, and the
safety layer that gates wall proximity, inter-cluster spacing and actuator limits sits
downstream of every controller. What differs between methods is only the local command
chosen at each 0.1 s interval. The learned controller is a parameter-shared recurrent
actor that scores 13 semantic candidates and emits both direction and speed, so a
single set of weights serves N = 1, 2 and 3 clusters without retraining.

## Learning proceeds against a rising difficulty curriculum

Three independent training seeds reach whole-run training success of 59.7%, 55.7%
and 54.7% on the nine-anatomy pool (Fig. 2b). The success trace is
non-monotonic by construction: the curriculum promotes difficulty as the policy masters
the easier levels, so the mid-run dip coincides with the easy-to-hard transition rather
than with forgetting.

The curriculum does not advance uniformly across seeds, and this is a limitation of the
present training budget rather than a property of the task. Seed 41 progressed through
all four levels (16 easy, 10 medium, 27 hard
and 9 stress episodes) and its success fell from 69% on easy to
56% on stress. Seeds 42 and 43 spent most of their budget at the medium level
(41 and 38 episodes) and reached the hardest levels only a
handful of times (3 hard and 2 stress for seed 42;
3 hard and 1 stress for seed 43), so their stress-level
success cannot be estimated from this run. A difficulty-matched comparison across seeds is
therefore not available at this budget, and the pooled whole-run figures should be read as
mixtures over different difficulty distributions rather than as a common operating point.
This is the single clearest target for a longer training run.

## Navigation transfers to whole unseen vascular classes without degradation

We split anatomies by whole class rather than by scene. Nine territories supply the
training rows; five further territories appear only in the held-out split and are never
seen in any form during training or tuning. Evaluated deterministically on the frozen
`navigation_dataset_v1` manifest (180 rows per
seed on trained anatomies and 100 on held-out),
safe success is 50.2 ± 1.7% on trained anatomies and
50.3 ± 0.9% on whole-class held-out anatomies; task
completion is 52.0 ± 2.3% and
52.0 ± 1.4% respectively (mean ± s.d. over three
training seeds, Fig. 3b).

The two sides are statistically indistinguishable, which is the central generalisation
result: a policy trained on nine vascular territories does not lose performance when
moved to five vascular classes it has never encountered. Per-anatomy safe success spans
31.7-68.3% across trained territories and
35.0-70.0% across held-out ones, so the ranges overlap almost
completely; the easiest held-out territory (sma embolism) exceeds every
trained territory, and the hardest (coronary rca) sits below the trained
range. Mean wall contact is 7.37 robot·s on trained anatomies and
4.64 robot·s on held-out ones.

Two qualifications belong with these numbers. First, the absolute level around
50% reflects a deliberately severe protocol: the frozen manifest crosses
three flow regimes, three target distributions, four obstacle densities and two spatial
layouts, and includes the stress condition at 0.15 mm/s with 32 dynamic obstacles. It is
not comparable to figures obtained on small development sets of easier conditions.
Second, what generalises is the local control policy, not the whole system: allocation,
routing and TPG are recomputed from the geometry of each new territory, so the result
establishes that the learned component does not need anatomy-specific retraining, not
that the pipeline is anatomy-free.
