# Methods

## Task formulation

We formulate multi-cluster vascular navigation as a partially observed cooperative
control problem. A fixed global scheduler assigns one target site to each of
N ∈ {1, 2, 3} magnetically actuated robot clusters and supplies the route and the
time-parameterised-graph (TPG) coordination state. The learned policy is responsible
only for selecting a short velocity command that advances its assigned cluster while
avoiding vessel-wall contact, dynamic particulate obstacles and inter-cluster
interference. The physical simulator advances at a 0.1 s control interval and each
episode has a finite 300 s horizon. Global target allocation, route construction,
sensing, the obstacle safety layer and the physical dynamics are identical across
every method we compare, so differences in the reported metrics are attributable to
the local controller alone.

The policy never receives inlet flow, actuator response gain, simulator-true
position, anatomy identity, environment identity, obstacle identity or any future
state. It receives the deployable estimated observation, a bounded history of past
observations, and the commands that were actually executed after the safety layer.
This makes the policy a history-based local controller rather than an oracle
controller, and it is the property that makes the simulation-to-hardware claim
meaningful.

## Simulation environment and anatomy library

Fifteen vascular territories are registered as explicit synthetic trees with named
segments, per-segment tapering diameters, tortuosity, bend counts, branch angles,
registered target sites and optional stenoses. Fourteen territories form the named
scenario pools used by all pooled experiments. The fifteenth,
`hcc_portal_multifocal`, is registered for a dedicated study and is selected only by
name; it is deliberately excluded from the named pools so that the fourteen-territory
pools and their frozen evaluation splits remain unchanged.

The HCC territory is a procedural nominal portal-hepatic tree with multifocal
hepatocellular-carcinoma target sites: a 12 → 10 mm portal trunk over 42 mm, left and
right portal branches, anterior/posterior and medial/lateral divisions, and terminal
segmental branches of 3.0–3.7 mm tapering to 2.2–2.8 mm. Five target sites are
registered across hepatic segments VIII, VI, IV and II plus the portal bifurcation,
and two tumour-related narrowings (residual lumen fractions 0.65 and 0.70) act as
controlled difficulty features. The supplied reference construct describes the
distribution of lesions across hepatic segments and does not contain patient-specific
centrelines or radii, so the territory encodes that construct as a reproducible
synthetic tree rather than presenting an image as measured anatomy. No result in this
work is a patient-specific or clinical claim.

## Deployable observation

For cluster i the observation contains five groups of features.

**Motion history.** The most recent 24 control records, spanning 2.4 s at the 0.1 s
interval. Each record holds the estimated velocity (3), the command actually executed
after protection (3), the delayed position relative to the current local reference
(3), the observation frame age (1) and the TPG waiting flag (1). A history-validity
mask distinguishes missing or padded records from a genuine zero velocity.

**Route and map context.** A 32-dimensional vector containing route and map features,
forecast obstacle summaries, activity and waiting flags, elapsed-time fraction,
remaining-task fraction, control period and normalised cluster count. Route
directions are expressed in the local route frame; target distance and contact
decisions use the geometric target distance, while route progress follows the
assigned route coordinate.

**Image features.** The four-channel 32 × 32 local image representation produced by
the deployable sensing interface. The policy receives the representation available at
deployment, never simulator particle or flow state.

**Dynamic-obstacle and neighbour features.** At most eight obstacle tracks per
cluster, each with 25 deployable features and a validity flag. Tracks are associated
across successive observations and extrapolated with a causal constant-velocity model
at 0.2, 0.5 and 1.0 s; this forecaster is never fitted to future simulator positions.
Neighbours are represented by relative estimated position and velocity, frame
age/validity and activity/waiting indicators. The implementation shares imaging
information between clusters, so the deployment assumption is shared observation
rather than fully independent per-robot sensing.

**Candidate descriptors.** The candidate command matrix and its validity indicators,
computed from the current estimated state, route, target geometry and frozen obstacle
forecasts. No hand-designed scalar cost is supplied as the sole policy input.

## Reduced semantic action set

At each control interval the shared actor selects one candidate index for every active
cluster. The initial semantic set contained 18 commands. Selection statistics showed
that several high-amplitude slots were almost never chosen while low-amplitude lateral
motion remained necessary for obstacle avoidance, so the reduced set retains 13
candidates:

A₁₃ = {zero, rule, route-full, route-0.5, target-full, target-0.5, target-0.25,
settle, reverse, normal₊₀.₃₅, normal₋₀.₃₅, binormal₊₀.₃₅, binormal₋₀.₃₅}

These are complete local velocity pulses, not residual corrections around a learned
nominal action. Route and target commands are normalised in the local route frame and
the settle command is the distance-scaled target direction. The safety layer and the
TPG waiting logic remain active for every candidate. Inactive or duplicated candidates
are marked invalid, and the adapter canonicalises an invalid categorical choice to an
equivalent valid command before execution, so validity is exposed to the actor while a
deterministic safe execution rule is preserved.

## Policy architecture

The actor is a parameter-shared categorical policy: identical parameters are applied
independently to every active cluster, so the same network serves N = 1, 2 and 3
without retraining. A 32-unit GRU encodes the 24-step motion history. The image
branch uses two convolutional layers followed by adaptive pooling and a 32-unit
projection. Separate 32-unit projections encode the route context, the obstacle
tracks, the neighbour features and the flattened candidate matrix. The six
representations are concatenated and passed through a 128-unit layer-normalised GELU
trunk, which feeds two 128-unit policy layers and a categorical head over the 13
candidates. A separate 128-unit value branch provides each local critic. Actor and
critic consume the same deployable observation; no privileged state enters either
branch. The feature extractor holds approximately 41.0 × 10³ parameters and the
complete policy approximately 110 × 10³.

## Reward and episode handling

Let d_{i,t} be the estimated route distance from cluster i to its assigned target,
Δd_{i,t} = clip(d_{i,t} − d_{i,t+1}, −0.5, 0.5), and g_{i,t} indicate that cluster i
newly entered its target tolerance. With W, L, C, E and Q denoting the per-step
increments of wall contact, loss of control, particle contact, particle collision
events and particle proximity, the local reward is

r_{i,t} = 1.5 Δd_{i,t} + 8 g_{i,t} − 2 W_{i,t} − 8 L_{i,t} − 2 C_{i,t} − 5 E_{i,t}
− 0.5 Q_{i,t} − 0.02 ‖u_{i,t} − u_{i,t−1}‖² − 0.01

with a shared terminal credit of +20 when every cluster has reached its target and
none is lost. Every safety term is a per-step increment, so an earlier violation is
never re-charged on subsequent steps. Episodes terminate on joint success, on loss of
control, or at the 300 s horizon.

A makespan-optimal preoperative allocation can legitimately leave one cluster without
a target when concentrating the sites on fewer clusters shortens the makespan. Because
joint success is defined over robot indices, such a cluster would make success
unreachable for every controller. Both the training environment and the evaluation
harness therefore fall back to the physical geodesic assignment whenever the
preoperative plan leaves any cluster unassigned, so each cluster always carries one
target and all methods solve the identical task. With dissolution disabled for the
navigation task no target is ever retired mid-episode, which makes freezing the
assignment at reset equivalent to re-planning every step whenever the plan is
non-empty.

## Training protocol

Training uses independent PPO with full parameter sharing (IPPO) on the on-policy
environment reward, implemented on Stable-Baselines3 2.7.1 with learning rate 1 × 10⁻⁴,
n_steps 256, batch size 128, 4 epochs per update, γ = 0.995, GAE λ = 0.95, clip range
0.2 and entropy coefficient 0.01. Episodes draw an anatomy from the training pool, a
cluster count N ∈ {1, 2, 3}, one of three flow regimes, one of three target
distributions and one of four obstacle densities crossed with two spatial layouts.

A curriculum promotes the difficulty level as the policy masters the easier levels,
which makes the training-time success curve non-monotonic by construction: the mid-run
dip coincides with the easy-to-hard transition rather than with forgetting (Fig. 2b).

For the HCC study the anatomy pool is a mixture rather than a single territory. A
single-anatomy HCC run of 500 × 10³ physical steps reached only 23.2% training success
with a flat success quartile profile (26/18/20/24%) and 61.6% of episodes exhausting
the horizon, because the pool carried no cross-anatomy variation. The retraining
protocol therefore keeps the nine-anatomy pool and oversamples the HCC territory to
half of all episodes, and warm-starts the policy from the converged nine-anatomy
checkpoint. The anatomy sampler reshuffles the pool without replacement, so a repeated
entry oversamples that territory by exactly its repetition factor. Every run records
the anatomy pool, the oversampling factor, the warm-start checkpoint path and its
SHA-256 digest, so a reported number can be traced to the exact weights that produced
it.

## Evaluation protocol

All navigation results are measured on `navigation_dataset_v1`, a frozen manifest of
scene rows whose SHA-256 digest is recorded with every result file. A row fixes the
anatomy, cluster count, flow regime, target distribution, obstacle density, obstacle
layout and scene seed, so a condition is reproducible and two methods evaluated on the
same rows are paired by construction. The runner never synthesises scene seeds.

Anatomies are split by whole class, not by scene. The nine territories in the training
pool supply the train and validation rows; five further territories
(`basilar_vertebral`, `cerebral_venous_sinus`, `coronary_rca`, `femoropopliteal_pad`,
`sma_embolism`) appear only in `holdout_test` and are never seen in any form during
training or tuning. Generalisation is therefore measured against whole unseen vascular
classes rather than unseen seeds of familiar anatomies.

Policies are evaluated deterministically (arg max over valid candidates). We report
task completion (every cluster reached its assigned target), safe success (completion
under the safety criterion), wall contact in robot-seconds, loss rate and the fraction
of episodes containing a particle collision. Where a method is compared against
another on the same rows, the difference is reported with a bootstrap interval over
paired scene seeds (10⁴ resamples) and an exact McNemar test on the binary
success pairs; completion time is paired only over episodes that both methods solved,
because an unsolved episode is right-censored and averaging over solved episodes alone
would otherwise favour the method that fails more often.

## Baselines

Every baseline receives the identical global layer (geodesic allocation, route, TPG)
and the identical safety layer, so only the local controller differs. The comparison
set spans a deployable pursuit-and-settle rule; a switchable-TPG variant that
re-optimises the precedence of conflict zones no cluster has entered; a sampling model
predictive controller (MPPI); a control-barrier-function quadratic program (CBF-QP); a
probabilistically-constrained nonlinear MPC (PAC-NMPC); a transformer PPO policy
trained on the same task; and an online-estimator variant of the predictive selector
that replaces the learned motion predictor with a causal affine fit.

On the HCC territory the switchable-TPG baseline is bit-identical to the pursuit rule
on every episode. This is not a wiring error: the construct yields no TPG conflict
zone at three of the four difficulty levels, and at the fourth the switch search was
invoked 715 and 1,714 times in two representative episodes without ever reordering a
zone. We therefore report it as a degenerate baseline on this territory rather than as
an independent comparison.

## Statistics

Learning methods are reported as the mean over three independent training seeds with
the per-seed values shown; a single seed is treated as a pilot and never as a result.
Error bars are standard deviations across seeds unless stated otherwise. Paired
differences use the bootstrap interval and exact McNemar test described above. We do
not report a difference as an improvement when its interval contains zero. Scene
counts, seed identities, manifest digests and model digests accompany every reported
number in the source data files.
