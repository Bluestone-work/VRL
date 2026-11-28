# Abstract and Introduction (draft, 2026-11-28)

Numbers in this draft come from result files listed in `DATA_SOURCES_*.md`. Text in **[brackets]** marks
a claim whose experiment has not been run yet; delete or fill it before submission. Verify every reference
(authors, year, venue) against the original paper before submission.

---

## Title (working)

Hierarchical graph planning and decentralized reinforcement learning for multifocal drug delivery with magnetic microrobot clusters

## Abstract (≈150 words, NMI limit)

Multifocal diseases such as multifocal liver cancer require drug delivery to several sites, but a single
microrobot serves them in sequence. Several magnetic microrobot clusters could serve them in parallel, yet
their routes and timing must keep them apart, because every cluster responds to the field applied to its
neighbours, and each must cope with pulsatile flow and patient-specific actuation that no preoperative scan
reveals. We split control along this line. A graph layer, computed from the preoperative vessel tree,
allocates lesions, routes clusters and orders their passage through shared segments. A local policy,
trained by independent proximal policy optimization and shared by all clusters, steers each cluster from its
own image history. Across 280 paired simulated scenes in 14 vascular territories, five held out as whole
classes, independent learning matched a centralized critic in 837 of 840 paired outcomes. The graph layer
let three clusters reach three lesions in 44 s versus 86 s for one. **[In vitro result.]**

## Introduction

Systemically administered drugs reach their target poorly: across a decade of nanoparticle studies, a
median of 0.7% of the injected dose arrived at the tumour [Wilhelm 2016]. Untethered microrobots steered
by external magnetic fields could instead carry a payload through the vasculature to a chosen site
[Nelson 2010; Li 2017; Nelson & Pané 2023], and magnetic microswarms have already been navigated through
blood vessels under ultrasound guidance and used for selective embolization [Wang Q 2021; Law 2022].
Many clinical targets are not single sites. Multifocal hepatocellular carcinoma, metastatic deposits and
showers of emboli present several lesions in different branches of one vascular tree. A single robot or
swarm must visit them in sequence, so the time to treat the last lesion grows with the number of lesions,
and the doses arrive at different moments. Several clusters deployed together could shorten the
treatment and bring the arrivals closer in time.

Controlling several clusters in a vessel is harder than controlling one. The magnetic field is generated
outside the body and acts on every cluster in the workspace, so clusters that come close respond to the
same field gradients and to each other; selective actuation of identical magnetic robots requires them to
be spatially separated [Rahmer 2017]. Motion is confined to the lumen, dominated by blood flow, and
observed only through delayed medical images. Robotics has a standard recipe for many robots that must
share space: allocate tasks to robots [Kuhn 1955; Choi 2009], plan conflict-free routes and the order in
which robots pass shared resources [Sharon 2015; Hönig 2016; Andreychuk 2022], and let each robot follow
its plan with a local controller, which in aerial swarms is increasingly learned [Hüttenrauch 2019; Batra
2021]. The recipe fits the vasculature for a specific reason. Before an intervention, computed tomography
angiography provides the vessel tree and the lesion sites, so the combinatorial part of the problem (which
cluster treats which lesion, along which branches, and who waits for whom at a bifurcation) can be solved
on a graph before the first cluster moves. Keeping clusters on separate routes, and letting them cross
shared segments one at a time, is also what keeps them out of each other's actuation neighbourhood.

What the preoperative scan does not provide is the dynamics. The flow rate and its pulsatile waveform,
the gain and lag with which a cluster follows the commanded field, the drag near the wall and the motion
of blood-borne obstacles differ between patients and change during the procedure. They can only be
inferred online from the history of commands and images. Hand-designed guidance laws handle the average
case but need retuning when these factors change, which is the setting in which learned feedback policies
trained over randomized simulated dynamics have succeeded [Tobin 2017; Peng 2018; Kaufmann 2023].
Reinforcement learning has been applied to microrobot navigation, but for a single robot or swarm, in two
dimensions or in open workspaces [Behrens 2022; Yang 2022; Sun (mr-nav)], and not to several clusters that
share one vessel tree and one actuation field.

Here we combine the two layers. The global layer computes geodesic distances on the vessel graph,
assigns lesions to clusters by minimizing the longest tour, and builds a temporal plan graph that fixes
the order of passage through shared segments. The local layer is one policy, trained with independent
proximal policy optimization (IPPO) [Schulman 2017; de Witt 2020] and shared by all clusters, that selects
every 0.1 s one of 13 candidate velocity commands from the cluster's own image-derived motion history,
route context, local image, obstacle tracks and an online estimate of flow and actuation gain. A safety
layer projects every command away from the wall and enforces a minimum spacing between clusters. The
hierarchy is what makes decentralized learning sufficient. Once the global layer has separated the
clusters in space and time, the local decisions of one cluster depend only weakly on the others, so each
cluster can act on local observations without communication and the same policy serves any cluster.
Consistent with this, a centralized critic (MAPPO) [Yu 2022] trained on the joint state gave the same
outcome as IPPO in all 280 paired evaluation scenes for two of three training seeds, and in 277 of 280 for
the third.

We evaluate the method in a simulator of magnetically actuated clusters in 14 vascular territories with
pulsatile flow, patient-level variation of actuation and moving obstacles, on a frozen manifest of
scenes that is identical for every method. We compare against published controllers for each layer:
sequential Hungarian assignment for allocation, switchable temporal plan graphs [Berndt 2020; Jiang 2025]
and continuous-time conflict-based search [Andreychuk 2022] for coordination, and pure pursuit,
PAC-certified nonlinear model predictive control [Gonzales 2025] and MAPPO for local control. With the
same lesions and deadline, and the local controller held fixed to isolate the global layer, three clusters reached all three lesions in 44.3 s on average against 86.5 s
for one cluster (paired difference −42.2 s, 95% interval −79.5 to −7.9 s, six scenes). The learned local
policy completed 85% of 180 scenes in the nine training territories and 80% of 100 scenes in five
territories never seen in training, the same rate as the pure-pursuit controller on the same scenes.
Switchable temporal plan graphs gave the same rates (152 of 180 and 80 of 100), and PAC-NMPC completed
151 of 180 and 84 of 100. **[Statement on where the learned policy is better, pending the experiments listed
below.] [In vitro validation on a magnetic actuation platform, Fig. 6.]**

---

## 故事线说明与当前数据的边界（给作者，非正文）

1. **分层的理由（用户给的三点）都已写进正文**：多病灶需要多集群（第1段），集群间受同一外场与彼此影响、需空间/时间分离（第2段，对应仿真里的 `coupling_exposure = ∫(d_min/d)^3 dt` 与 2 mm 间距护盾），CTA 术前可见几何与病灶 → 图论全局规划（第2段末）。
2. **强化学习的位置**：术前影像"看得见几何、看不见动力学"。`marl/physio_variation.py` 里随机化的正是 CT 看不到的东西（脉动流、执行增益/方向偏差/滞后/漂移、近壁阻力、黏附），所以底层用"在随机化动力学上训练、只用图像历史的反馈策略"。IPPO 而不是 MAPPO 的理由：高层已把集群在空间和时间上解耦，局部问题近似独立；数据支持这一点（IPPO 与 MAPPO 在 S42、S43 两个种子上 280 场逐场完全一致，S41 训练集 153 vs 152）。
3. **必须正视的事实**：在当前冻结 manifest 上，IPPO 与 pure pursuit 的成功率相同（训练 153/152/153 vs 152 /180；留出 80 vs 80 /100），按条件分解也重合（Fig. 4c）。原因在回放里能看到：IPPO 在 91.5% 的步上选择了"经典路由指令"候选（候选 1），7.7% 选择减速，0.8% 选择停止。也就是说策略学会了"大多数时候跟随规则，少数时候偏离"，偏离有赢有输（场景 3405052004 赢、3413006012 输），总体抵消。现在不能写"RL 优于经典方法"。
   - 2026-11-28 新跑的两个对比方法（同 280 场）：Switchable TPG 训练 152/180、留出 80/100，与 pure pursuit 完全相同；PAC-NMPC 训练 151/180、留出 **84/100**，在留出解剖上比 IPPO 高 4 个点（配对 4 胜 88 平 8 负，差异主要在强流和股腘动脉场景）。如果正文要宣称学习方法的泛化，必须先面对这个结果。
4. **要让 RL 站得住，建议补的实验**（按信息量排序）：
   - 在规则方法失效的区域训练和评估：强流（所有方法完成率 72%）、留出解剖 + 强流（17/33）、`variation_strength = 1.5` 的分布外动力学、图像延迟 2–3 帧。若 IPPO 在这些子集上显著高于 pursuit，故事就成立："规则在平均情况够用，学习在 CT 看不见的动力学偏移下才必要"。
   - Fig. 2d 的流估计分支消融（`--flow-feature` 开/关，各 3 种子），直接检验"在线辨识 CT 看不见的动力学"这一论点。
   - 用 IPPO 作局部控制器重跑 N=1/2/3 三靶点实验。目前 Fig. 5 和摘要里的 44 s vs 86 s 来自 pure pursuit 局部控制（为了只比较高层），而且 IPPO checkpoint 的观测宽度随 N 变化，N=1/2 需要各自训练。
   - 报告偏离率（选择非规则候选的比例）与偏离时的局部条件，作为可解释性分析。
5. **Fig. 2 与 pptx 的差异**：pptx 写的是"冻结运动预测器 Ensemble μ/σ"，当前 IPPO 代码实际用的是在线最小二乘流/增益估计（`ObservedMotion`），没有冻结集成预测器。图按代码画。若要保留预测器叙事，需要先把预测器接回并做消融。
6. **对比方法的引用核对**：`csn` 与 `yang_hier` 两个适配器在代码里没有写明出处。`yang_hier` 很可能对应 Yang et al., Nat. Mach. Intell. 2022（微机器人集群环境自适应导航），请核实后再放进图。mr-nav（Sun et al.，"Minute-Scale Training for Microrobot Navigation"）目前只在作者的二维地图上做了作者策略 vs 朝目标转向的对照（45/48 vs 42/48），没有本文方法在同一任务上的结果，只能放 Extended Data。

## References to verify

- Wilhelm S. et al. Analysis of nanoparticle delivery to tumours. Nat. Rev. Mater. 1, 16014 (2016).
- Nelson B. J., Kaliakatsos I. K., Abbott J. J. Microrobots for minimally invasive medicine. Annu. Rev. Biomed. Eng. 12, 55–85 (2010).
- Li J. et al. Micro/nanorobots for biomedicine: delivery, surgery, sensing, and detoxification. Sci. Robot. 2, eaam6431 (2017).
- Nelson B. J., Pané S. Delivering drugs with microrobots. Science 382, 1120–1122 (2023).
- Wang Q. et al. Ultrasound Doppler-guided real-time navigation of a magnetic microswarm for active endovascular delivery. Sci. Adv. 7, eabe5914 (2021).
- Law J. et al. Microrobotic swarms for selective embolization. Sci. Adv. 8, eabm5752 (2022).
- Rahmer J., Stehning C., Gleich B. Spatially selective remote magnetic actuation of identical helical micromachines. Sci. Robot. 2, eaal2845 (2017).
- Kuhn H. W. The Hungarian method for the assignment problem. Nav. Res. Logist. Q. 2, 83–97 (1955).
- Choi H.-L., Brunet L., How J. P. Consensus-based decentralized auctions for robust task allocation. IEEE Trans. Robot. 25, 912–926 (2009).
- Sharon G. et al. Conflict-based search for optimal multi-agent pathfinding. Artif. Intell. 219, 40–66 (2015).
- Hönig W. et al. Multi-agent path finding with kinematic constraints. Proc. ICAPS (2016).
- Andreychuk A. et al. Multi-agent pathfinding with continuous time. Artif. Intell. 305, 103662 (2022); IJCAI 2019.
- Berndt A. et al. A feedback scheme to reorder a multi-agent execution schedule by persistently optimizing a switchable action dependency graph. arXiv (2020).
- Jiang H., Lin, Li J. Speedup techniques for switchable temporal plan graph optimization. AAAI 39 (2025), doi:10.1609/aaai.v39i22.34487.
- Gonzales et al. Multi-agent feedback motion planning using probably approximately correct nonlinear model predictive control. IEEE CASE (2025), doi:10.1109/CASE58245.2025.11164063.
- Hüttenrauch M., Šošić A., Neumann G. Deep reinforcement learning for swarm systems. J. Mach. Learn. Res. 20, 1–31 (2019).
- Batra S. et al. Decentralized control of quadrotor swarms with end-to-end deep reinforcement learning. CoRL (2021).
- Tobin J. et al. Domain randomization for transferring deep neural networks from simulation to the real world. IROS (2017).
- Peng X. B. et al. Sim-to-real transfer of robotic control with dynamics randomization. ICRA (2018).
- Kaufmann E. et al. Champion-level drone racing using deep reinforcement learning. Nature 620, 982–987 (2023).
- Behrens M. R., Ruder W. C. Smart magnetic microrobots learn to swim with deep reinforcement learning. Adv. Intell. Syst. 4, 2200023 (2022).
- Yang L. et al. Autonomous environment-adaptive microrobot swarm navigation enabled by deep learning-based real-time distribution planning. Nat. Mach. Intell. 4, 480–493 (2022).
- Sun Y. et al. Minute-scale training for microrobot navigation. (code: github.com/yinghansun/mr-nav; venue to confirm).
- Schulman J. et al. Proximal policy optimization algorithms. arXiv:1707.06347 (2017).
- de Witt C. S. et al. Is independent learning all you need in the StarCraft multi-agent challenge? arXiv:2011.09533 (2020).
- Yu C. et al. The surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS Datasets and Benchmarks (2022).
