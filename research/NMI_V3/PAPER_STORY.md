# Paper story (V3)

**Working title.** Hierarchical swarm intelligence for cooperative navigation of magnetic microrobot clusters.

**One-sentence claim (current evidence).** A graph-based global layer that decides, from the vessel map
alone, how many magnetic clusters to deploy, where the catheter releases them, which lesions each
serves and in which order they pass shared vessels makes multi-cluster navigation faster, more
successful and free of inter-cluster spacing violations, compared with a fixed configuration; most of
the gain comes from where clusters are released, and the number of clusters can be minimised without
losing performance.

**Second claim (to be tested, Phases D–G).** A local policy shared by all clusters handles what the map
cannot predict (flow, patient-specific actuation, moving obstacles, residual neighbour interactions).
Allowed only if it beats pure pursuit and PAC-NMPC on the same tasks and configurations.

**Story arc.**
1. Multifocal targets need several clusters; several magnetic clusters interact and share vessels.
2. Macro-scale swarms solve this with fleet sizing, depot choice, task allocation, MAPF and local
   control (Fig. 1a, transfer of ideas, not weights).
3. In vessels these become: N limited by actuation; release only where a catheter reaches; unique tree
   routes with flow-dependent costs; no passing within d_safe (Fig. 1b–e).
4. Global layer results (Fig. 2): configuration chosen per task; release optimisation dominates;
   adaptive N saves clusters; conflicts vanish at the configuration level, so TPG rarely acts.
5. What remains is local: lost clusters, wall contact, weak-flow stalls → local layer (Fig. 3–4).
6. System-level comparison against strong baselines on all four metrics (Fig. 5).
7. Physical validation (Fig. 6) — the platform must calibrate d_safe, n_max and catheter speed.

**Language rules.** Do not call the system medical-image-driven; CT/MR is only a possible source of the
map. Do not call `low_contact_success` safe. Do not claim magnetic isolation, drug delivery, or that the
learned policy is better than PAC-NMPC.
