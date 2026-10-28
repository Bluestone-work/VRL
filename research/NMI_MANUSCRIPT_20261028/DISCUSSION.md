# Discussion

We set out to establish what a learned local controller contributes to multi-cluster
vascular navigation when everything around it is held fixed. By freezing the global
layer — geodesic allocation, route construction, TPG precedence — and the downstream
safety layer, and by denying the policy any privileged simulator state, the comparison
isolates the local decision rather than the pipeline.

The result we consider solid is generalisation across whole vascular classes. A
parameter-shared recurrent actor trained on nine territories performs
indistinguishably on five territories it has never encountered in any form (safe
success 50.2% versus 50.3%, three training seeds, frozen scene manifest). Per-anatomy
ranges overlap almost completely, and the easiest held-out territory exceeds every
trained one. This is the property that matters for a controller intended to be deployed
beyond its development set: the learned component does not require anatomy-specific
retraining, even though allocation and routing are of course recomputed for each new
geometry.

## Limitations

**Absolute performance is moderate, and the protocol is the reason to take it
seriously rather than to discount it.** Safe success near 50% is measured on a frozen
manifest that crosses three flow regimes, three target distributions, four obstacle
densities and two spatial layouts, including a stress condition at 0.15 mm/s inlet flow
with 32 dynamic obstacles. Numbers obtained on smaller development sets of easier
conditions are not comparable, and we do not present them as such.

**Classical rules remain stronger on the hardest territory we tested.** On the HCC
portal-hepatic construct, a deployable pursuit-and-settle rule reaches 68.8% safe
success over twelve paired scene seeds while the learned policy reaches 12.5%
(difference −54.2 percentage points, bootstrap 95% interval [−68.8, −39.6], 0 wins
against 26 losses, exact McNemar p < 0.001). We investigated this gap rather than
setting it aside. It does not come from the training distribution (a mixed pool that
oversamples the territory to half of all episodes), the initialisation (warm start from
the converged multi-anatomy policy), the difficulty schedule (with and without
curriculum), the action space (identical reduced semantic set at training and
evaluation), the reward bookkeeping (all safety terms are per-step increments), or the
evaluation protocol (aligned with the training environment's target fallback). Three
training configurations leave HCC-episode success statistically unchanged at 21.7%,
23.2% and 23.7%. We therefore report this as a genuine limit of the present method on
this geometry, and as the clearest open problem the work leaves behind: a tapering
portal tree with four targets behind two tumour-related narrowings appears to demand
local manoeuvres that the current candidate set does not express.

**The curriculum does not advance uniformly across seeds.** One seed traversed all four
difficulty levels while two spent most of their budget at the intermediate level and
reached the hardest levels only a handful of times, so a difficulty-matched comparison
across seeds is not available at this training budget. The pooled training figures are
mixtures over different difficulty distributions.

**Sensing is shared rather than independent.** Clusters exchange imaging information,
so the deployment assumption is shared observation. Fully independent per-robot sensing
is untested.

**All navigation results are simulation.** The vascular territories are reproducible
synthetic trees with registered segment geometry, not patient-specific segmentations,
and the HCC territory encodes the distribution of lesions from a reference construct
rather than measured centrelines. No claim here is patient-specific or clinical.
In vitro validation on the magnetically actuated platform is reported separately and is
the necessary next step before any physical claim.

## Outlook

Two directions follow directly. First, the candidate action set is the component most
implicated by the HCC result; expanding it with manoeuvres suited to tapering,
stenosed branches — or replacing the discrete set with a bounded continuous
parameterisation under the same safety layer — is a concrete test of that hypothesis.
Second, the generalisation result invites the opposite experiment: train on fewer
territories and measure how quickly whole-class transfer degrades, which would
establish how much anatomical diversity the property actually requires.
