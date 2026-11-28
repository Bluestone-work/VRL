# Abstract and introduction, V3 draft (2026-11-28)

Numbers come from `CONFIGURATION_RESULTS/REPORT.md` (simulation, frozen test set of 120 tasks).
**[Bracketed]** text depends on phases not yet run. References need the verification listed in
`LITERATURE_AND_TRANSFER.md` before submission.

## Abstract

Delivering drugs to several lesions at once calls for several magnetic microrobot clusters, but
clusters that share vessels interfere with each other and with the operator's ability to steer them.
Aerial and ground swarms handle comparable problems by sizing the fleet, choosing depots, allocating
tasks and planning conflict-free routes before execution. We transfer this hierarchy to the
vasculature. A graph layer, computed from the vessel map, decides how many clusters to deploy, where a
catheter releases each one, which lesions it serves and in what order clusters pass shared vessels;
**[a local policy shared by all clusters handles flow, actuation variability and residual interactions
online]**. In 120 simulated tasks across 15 vascular territories, the graph layer raised navigation
success from 58% to 82%, contact-free success from 43% to 63% and cut the last lesion arrival from 194 s
to 139 s relative to a fixed three-cluster configuration, while eliminating inter-cluster spacing
violations (21% of tasks to 0%) and using two clusters on average. **[Local-layer and in vitro results.]**

## Introduction

Many therapeutic targets are not single sites. Multifocal liver cancer, metastatic deposits and
showers of emboli place lesions in several branches of one vascular tree, and a single microrobot or
swarm must visit them one after another. Magnetic microrobot swarms have been steered through vessels
and used for embolization and delivery [Wang Q 2021; Law 2022], and learned controllers now navigate a
single swarm through unknown and dynamic environments [An 2026; Medany 2025]. Serving several lesions in
parallel requires several clusters at once.

Several clusters bring problems that a single swarm does not have. They share vessels, so two clusters
can need the same segment at the same time; they respond to an external field generated outside the
body, so clusters that come close cannot be steered independently [Rahmer 2017]; and every additional
cluster costs one more catheter release and one more independently actuated channel. The question is
therefore not only how to move each cluster but how many to deploy, where to release them, which
lesions each should serve and in which order they should pass shared vessels.

Macro-scale multi-robot systems answer these questions with a standard hierarchy: fleet sizing and
depot selection, task allocation [Kuhn 1955; Choi 2009], multi-agent path finding with temporal plan
graphs [Hönig 2016; Andreychuk 2022; Jiang 2025] and decentralised local control. The vasculature
changes every element. Clusters can only be released where a catheter can reach, and deployment takes
time inside the mission; motion is confined to a tree, so routes are unique and upstream travel is
slowed by flow; clusters cannot pass each other within a safe distance inside a lumen. The vessel tree
and lesion sites, however, are known before the procedure, so the combinatorial part of the problem
can be solved on a graph before any cluster moves.

Here we build this global layer and measure its effect with the local controller held fixed. It
constructs a weighted vessel graph, and jointly chooses the number of clusters, their catheter release
stations, the ordered allocation of lesions and the release and passage schedule, under a lexicographic
objective that first enforces coverage and feasibility, then minimises predicted completion time,
waiting and risk, then uses the fewest clusters within 10% of the best. An exact best-first search
certifies optimality within the planning model for up to five lesions; a scalable local search comes
within 5% of it on every test task (median gap 0%). **[The local layer: a variable-size policy shared by all clusters, trained with
patient-level dynamics randomisation, that uses neighbour states and action-conditioned motion
prediction.]**

On a frozen test set the global layer improves all four metrics we use throughout, navigation safety,
success, delivery efficiency and inter-cluster conflict, against a fixed configuration. Its ablations
locate the gain: optimising release points contributes most; choosing N adaptively keeps performance
while using fewer clusters; and once release is optimised the configurations keep clusters apart, so
temporal coordination at execution has little left to do. The failures that remain, lost clusters,
wall contact and stalls in weak flow, are local, which motivates the second layer. **[Local-layer
comparison with pure pursuit, PAC-NMPC, IPPO and MAPPO; in vitro validation.]**
