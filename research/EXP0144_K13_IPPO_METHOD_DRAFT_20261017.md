# Methods: History-aware semantic-candidate IPPO for local vascular navigation

## 2.1 Task formulation

We formulate local navigation as a partially observed cooperative control problem. A fixed global scheduler assigns targets to (N\in\{1,2,3\}) clusters and supplies the route and TPG coordination state. The local policy is responsible only for selecting a short velocity command that advances the assigned cluster while avoiding vessel-wall contact, dynamic obstacles, and inter-cluster interference. The physical simulator advances at a 0.1-s control interval and the finite episode horizon is 300 s. Global target allocation, route construction, sensing, the obstacle safety layer, and the physical dynamics are kept unchanged.

The policy does not receive inlet flow, response gain, simulator position, anatomy identity, environment identity, obstacle identity, or future state. It receives the deployable estimated observation, the history of observations, and the commands that were actually sent after the safety layer. This makes the policy a history-based local controller rather than an oracle controller.

## 2.2 Deployable observation

For cluster (i), the observation contains five groups of features.

1. **Motion history.** The most recent 24 control records are stored, corresponding to 2.4 s at the 0.1-s control interval. Each record contains the estimated velocity (3), the actually executed command after protection (3), delayed position relative to the current local reference (3), observation frame age (1), and TPG waiting flag (1). A history-validity mask distinguishes missing or padded records from a genuine zero velocity.

2. **Route and map context.** A 32-dimensional context vector contains the existing 19-dimensional route/map features, forecast obstacle summaries, activity and waiting flags, elapsed-time fraction, remaining-task fraction, control period, and normalized cluster count. Route directions are represented in the local route frame; target distance and contact decisions use the geometric target distance, while route progress follows the assigned route coordinate.

3. **Image features.** The observation includes the four-channel (32\times32) local image representation used by the current sensing interface. The policy receives the image representation available at deployment, rather than simulator particle or flow state.

4. **Dynamic-obstacle and neighbour features.** At most eight obstacle tracks are represented per cluster, each with 25 deployable features and a validity flag. Tracks are associated from successive observations and extrapolated with a causal constant-velocity model at 0.2, 0.5, and 1.0 s. The predictor is not trained with future simulator positions. Neighbours are represented by relative estimated position, relative estimated velocity, frame age/validity, and activity/waiting indicators. The current implementation uses shared imaging information, so the deployment assumption is shared observation rather than fully independent sensing.

5. **Candidate descriptors.** The policy receives the candidate command matrix and its validity indicators. Candidate descriptors are computed from the current estimated state, route, target geometry, and frozen obstacle forecasts. No artificial scalar cost is provided as the sole policy input.

## 2.3 Reduced semantic action set

At each control interval, the shared actor selects one candidate index for every active cluster. The initial semantic set contained 18 commands. Selection statistics showed that several high-amplitude slots were almost never selected, while low-amplitude lateral motion remained necessary for obstacle avoidance. The first reduced action set therefore contains 13 candidates:

\[
\mathcal A_{13}=\{\text{zero},\text{rule},\text{route-full},\text{route-0.5},
\text{target-full},\text{target-0.5},\text{target-0.25},\text{settle},
\text{reverse},\text{normal}_{+0.35},\text{normal}_{-0.35},
\text{binormal}_{+0.35},\text{binormal}_{-0.35}\}.
\]

The commands are complete local velocity pulses, not residual corrections around a learned action. The route and target commands are normalized in the local route frame; the settle command is the distance-scaled target direction. The safety layer and TPG waiting logic remain active. Inactive or duplicated candidates are marked invalid and the adapter canonicalizes an invalid categorical choice to an equivalent valid command before execution. Thus the current PPO implementation exposes validity to the actor and preserves a deterministic safe execution rule, although it does not yet use a native logit-level PPO action mask.

## 2.4 Policy architecture

The actor is a parameter-shared categorical policy: the same parameters are applied independently to all active clusters. A 32-unit GRU encodes the 24-step motion history. The image branch uses two convolutional layers followed by adaptive pooling and a 32-unit projection. Separate 32-unit projections encode route context, obstacle tracks, neighbour features, and the flattened candidate matrix. These six representations are concatenated and passed through a 128-unit layer-normalized GELU trunk.

The trunk feeds two 128-unit policy layers and a categorical head over the 13 candidates. A separate 128-unit value branch is used for each local IPPO critic. The actor and critic use the same deployable observation; no privileged state is supplied to either branch. The feature extractor has approximately 41.0k parameters for K13 (about 41.5k for K18), and the complete PPO policy has approximately 110k parameters.

## 2.5 Reward and episode handling

The local reward is computed from observable task progress and safety increments. Let (d_{i,t}) be the estimated route distance to the assigned target, 

\[
\Delta d_{i,t}=\operatorname{clip}(d_{i,t}-d_{i,t+1},-0.5,0.5),
\]

and let (g_{i,t}) indicate that cluster (i) newly entered its target tolerance. The reward is

\[
\begin{aligned}
r_{i,t}={}&1.5\Delta d_{i,t}+8g_{i,t}
-2W_{i,t}-8L_{i,t}\\
&-2C_{i,t}-5E_{i,t}-0.5Q_{i,t}
-0.02\|u_{i,t}-u_{i,t-1}\|^2-0.01,
\end{aligned}
\]

where (W) is new wall-contact time, (L) is a newly lost cluster indicator, (C) is particle-contact time, (E) is the new particle-collision event count, and (Q) is near-obstacle exposure. When all assigned targets are reached without a lost cluster, every active local learner receives an additional terminal credit of 20. The reward uses increments rather than repeatedly charging cumulative safety quantities.

Episodes terminate at task completion or at the 300-s finite horizon. Rollout-buffer boundaries are not task terminations. Reset clears the physical episode, history mask, obstacle tracks, and route state together. TPG waiting steps are represented as no-action/wait states and are not treated as free policy decisions.

## 2.6 PPO training

We use the existing stable-baselines3 PPO training flow. The first K13 study uses a learning rate of (10^{-4}), 256-step on-policy rollouts, minibatches of 128, four update epochs, (γ=0.995), GAE (λ=0.95), clip range 0.2, and entropy coefficient 0.01. Each run uses 100,000 physical environment steps with three synchronized clusters. The three training seeds are 41, 42, and 43. Patient/anatomy identity and physical flow remain hidden from the actor; the curriculum samples the registered training anatomies and the dynamic-obstacle difficulty levels.

The K13 runs are a candidate-space ablation and a multi-seed screening study. They are not compared as if they had the same action space as the K18 DQN or ISAC experiments. The current K13 training protocol uses zero sensing/command latency; therefore it does not establish high-latency robustness.

## 2.7 Evaluation protocol

The primary development evaluation uses 36 deterministic episodes with three clusters, fixed route/safety implementation, and zero latency. We report navigation success, Strict success, wall-contact time, collision events, loss, and spacing violations. The K13 transfer diagnostic additionally separates nine training anatomies (seen) from five holdout anatomies (unseen) using an explicitly unregistered development scene namespace. These holdout results are diagnostic and are not sealed-test claims. All raw per-episode records, training seeds, scene seeds, action statistics, and checkpoints are retained in the experiment directories.

## 2.8 Scope of the reported claim

Across the current three K13 seeds, deterministic development navigation success is 76.9% ± 1.6% over the 36-episode evaluations. The seen/unseen diagnostic gives 68.5% ± 6.4% and 73.3% ± 5.8% navigation success, respectively. These results identify K13 semantic IPPO as the current leading deployable candidate in this screening set. They do not yet establish superiority under latency, sealed tests, longer 3M training, or a parameter/update-matched comparison against DQN and ISAC.
