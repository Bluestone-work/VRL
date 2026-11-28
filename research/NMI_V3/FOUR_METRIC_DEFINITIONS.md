# Four-metric definitions (V3, `metric_version = v3_four_metrics_1`)

Implementation: `marl/v3/metrics.py` (`FourMetricRecorder`), called by `marl/v3/execute.py` at the
end of every 0.1 s control step on **true simulator state**. Controllers never see these values.
Registered parameters: `configs/v3/assumptions.json`.

## Common definitions

| Quantity | Definition |
|---|---|
| Time origin | Catheter insertion, t = 0. Deployment, release, navigation and waiting are inside the mission. |
| Deadline | 300 s (`mission.deadline_s`). |
| Target delivered | The cluster **assigned** to the target first comes within 0.30 mm (centre to target station). Incidental passes by other clusters do not count. Arrival time is a delivery proxy: drug release and dose are not modelled. |
| Released / active cluster | Released from the catheter and not lost. Parked clusters (still in the catheter) are excluded from every safety and interaction quantity. |
| Lost cluster | Left the simulated domain through a boundary node (`exit_node >= 0`). |
| d_safe | 2.0 mm centre-to-centre (simulation assumption, not calibrated against magnetic measurements). Surface distance = d − 0.16 mm. |

## 1. Navigation safety

* **Primary: `safe_success`** = success AND wall contact = 0 robot·s AND dynamic-obstacle contact = 0 AND obstacle collision events = 0 AND no lost cluster. This is contact-free success.
* `low_contact_success` = success AND wall contact < 1 robot·s AND no lost cluster. This equals the legacy "strict success" (`scripts/benchmark_lysis.navigation_outcome`), which **allows up to 1 s of wall contact and does not check obstacle collisions**. It must not be called contact-free.
* Auxiliary: `wall_contact_robot_s`, `wall_contact_after_delivery_robot_s`, `obstacle_contact_robot_s`, `obstacle_collision_events`, `lost_clusters`, `min_wall_clearance_mm` (true lumen clearance = lumen radius − radial offset − body radius).

## 2. Navigation success

* **Primary: `success`** = every target delivered before the deadline. Defined on targets, so it is comparable when arms choose different N.
* Auxiliary: `target_coverage`, `per_target_arrived`, `timeout`, `failure_reason ∈ {lost_cluster, timeout}`.

## 3. Target-delivery efficiency

* **Primary: `T_last_deadline_scored_s`** = last target arrival if success, else the deadline (pre-registered scoring). Success-only times are reported separately (`T_last_success_s`).
* Auxiliary: `mean_target_arrival_s` (undelivered = deadline), `arrival_span_s` (synchronisation window, success only), `deployment_time_s` (last release), `total_cluster_time_s` (Σ over clusters of release → last own delivery; a lost cluster with targets counts to the deadline).

## 4. Inter-cluster interaction conflict

* **Primary: `conflict`** = at least one control step in which two released, active clusters are closer than d_safe. Task-level conflict rate = conflicting tasks / tasks.
* Auxiliary: `spacing_violation_pair_s` (step-sampled), `spacing_violation_pair_s_substep` (simulator substep integration), `exposed_pair_s` and `violation_per_exposed_pair_s` (normalised by pair exposure, needed when N differs), `min_inter_cluster_distance_mm`, `conflict_events`, `mean_conflict_duration_s`, `deadlock` (a cluster held ≥ 60 s continuously), `coupling_exposure_proxy` = ∫(d_safe/d)³ dt for d < 2·d_safe.
* N = 1 has no pairs: conflict = False, normalised rate undefined (reported as null, not 0).

## Separated interaction layers

| Layer | Source | Meaning |
|---|---|---|
| A. Planned potential conflict | planner, `configuration.predicted.potential_conflict_*` | violations the plan would have **without** coordination, on the planning model |
| B. Executed spacing violation | this evaluator | ground-truth violations during execution |
| C. Safety-layer intervention | this evaluator: `shield_intervention_steps`, `wall_projection_rewrite_steps`, `tpg_hold_robot_s` | counted at **every** step (the legacy classical evaluator stored only the last step's count) |
| D. Physical magnetic interference | not modelled | `coupling_exposure_proxy` is a distance proxy only |

## Motion bookkeeping

`mean_command_speed_mm_s` (sent command) and `mean_actual_speed_mm_s` (realised velocity) are reported separately; earlier figures plotted the sent command and labelled it accordingly.
