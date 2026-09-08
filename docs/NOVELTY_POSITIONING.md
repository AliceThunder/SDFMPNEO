# SDF-MPNEO 创新性定位（文献截止 2026-09-08）

本文用于约束论文中的 novelty claim，避免把已有技术重新命名为“首次”。正式投稿前应再次更新检索。

## 已有方向，不应单独宣称首次

### Physics-informed / solution-data-free operator learning

- Wang, Wang & Perdikaris, *Learning the solution operator of parametric partial differential equations with physics-informed DeepONets*, Science Advances 7 (2021), eabi8605, DOI: 10.1126/sciadv.abi8605。该工作已经明确展示在没有 paired input-output solution data 时通过物理约束学习参数 PDE solution operator。
- Li et al., *Physics-Informed Neural Operator for Learning Partial Differential Equations* (PINO), arXiv:2111.03794；后续正式版本。PINO 已覆盖 physics-constrained operator learning，并包含仅使用 PDE constraints 的情形。

因此 SDF-MPNEO 不应声称“首次无标签 operator learning”或“首次 physics-informed neural operator”。

### Laplace / pole-residue transient operator

- Cao, Goswami & Karniadakis, *Laplace neural operator for solving differential equations*, Nature Machine Intelligence 6, 631–640 (2024), DOI: 10.1038/s42256-024-00844-4。LNO 已使用 pole–residue transient representation，并讨论初值、瞬态和外推。
- Kim et al., *Physics-Informed Laplace Neural Operator for Solving Partial Differential Equations*, arXiv:2602.12706 (2026)。PILNO 已把 PDE/BC/IC residual 与 Laplace neural operator 结合，并使用 label-free virtual inputs 和 temporal-causality weighting。

因此 SDF-MPNEO 不应声称“首次用指数/Laplace 表示瞬态”“首次 physics-informed Laplace operator”或仅凭任意时间求值宣称独占创新。

### Greedy / residual-adaptive neural growth

- Siegel et al., *Greedy training algorithms for neural networks and applications to PDEs*, Journal of Computational Physics 484 (2023), 112084。已有基于 dictionary/greedy expansion 的 PDE neural solver 及条件下的收敛分析。
- Dang, Wang & Jiang, *Adaptive Growing Randomized Neural Networks for Solving Partial Differential Equations*, arXiv:2408.17225 (2024)。已有 residual-driven widening/deepening 的 adaptive growing network。
- Toscano et al., *A variational framework for residual-based adaptivity in neural PDE solvers and operator learning*, npj Artificial Intelligence 2, 32 (2026)。已有 residual-based adaptive weighting/sampling 的系统理论框架。

因此 SDF-MPNEO 不应把“残差驱动”“网络自动增长”本身作为唯一 novelty。

### Certified reduced basis / Maxwell ROM

- 经典 reduced-basis Maxwell / electromagnetic literature 已有 residual-based a posteriori estimators；例如 *Accelerated A Posteriori Error Estimation for the Reduced Basis Method with Application to 3D Electromagnetic Scattering Problems*, SIAM Journal on Scientific Computing, DOI: 10.1137/090760271。

因此 SDF-MPNEO 不应声称“首次 certified Maxwell ROM”或“首次 residual EM ROM”。

## SDF-MPNEO 可以主打的组合创新

截至上述检索，没有发现与当前实现**整条方法链同时一致**的工作：

```text
compatible Nedelec–P1 multiphysics discretization
+ reciprocal A-psi physical-energy certificate
+ solution-state-snapshot-free residual–Riesz joint-port EM basis
+ quasi-static elimination into a nonlinear thermal reduced vector field
+ shared topology-preserving geometry chart / thermal coordinates
+ exact dissipative analytic response DAG tied to thermal decay modes
+ residual-tangent-driven topology growth with full nonlinear acceptance
+ exact time derivative and direct arbitrary-time / stationary evaluation
+ typed a-posteriori propagation that keeps field/residual/state errors separate
```

论文的核心 novelty 应放在**这条编译式架构**，而不是某个单一关键词。

## 推荐的主贡献表述

可以安全使用接近下面的表述，但正式投稿仍应以最新系统检索为准：

> We introduce a physics-compiled neural evolution operator in which the electromagnetic field equations are first reduced by a certified residual–Riesz construction without solved-state snapshots, quasi-statically eliminated into a nonlinear reduced thermal vector field, and then represented by a topology-growing graph of exactly solvable dissipative response neurons. The graph embeds initial conditions and static geometry/operating parameters intrinsically, is trained only through the closed multiphysics residual and exact sensitivities, and supports direct arbitrary-time and stationary evaluation without thermal rollout.

进一步的 differentiators：

1. **网络核不是自由学习的 pole set。** response decay rate 与保留热物理衰减结构绑定；非线性交互通过 parent-product-response 结构显式生长。
2. **拓扑增长和物理 ROM 在同一 closed residual 上耦合。** candidate tangent 只提出结构，完整 nonlinear multiphysics residual 决定是否接受。
3. **EM basis 不用 solved-state snapshots。** enrichment 使用 certified residual-Riesz lift；这与常见 POD/snapshot ROM 的离线数据链不同。
4. **任意时间不是 rollout 技巧。** 每个有限 DAG 有精确状态空间 realization，可解析得到状态与时间导数，并定义稳定 stationary limit。
5. **误差类型不混加。** EM energy error 必须先传播成 heat/vector-field error；thermal projection state error 在动态 residual 传播之后再加。
6. **变几何 residual 保留真实 M_r(G), K_r(G)。** 不把共享参考坐标误当作所有几何上的质量正交坐标。

## 必须做的 novelty ablation

为了证明上述组合不是“把已有模块拼起来却没有收益”，论文至少报告：

- full SDF-MPNEO vs residual-only growth（不使用 physics seed）；
- full SDF-MPNEO vs physics seed only / no topology growth；
- adaptive seed vs fixed top-K seed；
- analytic DAG vs same EM/thermal ROM + Radau；
- shared geometry surrogate vs per-geometry retraining；
- 与 PI-DeepONet/PINO 类、LNO/PILNO 类外部 baseline 在相同参数域和计算预算下比较。

主要指标：held-out full-reference error、最大 physics residual、训练 wall-clock、online latency、长时稳定性、模型大小、所需 solution-label 数量（SDF-MPNEO 为 0）。

## 创新性结论的边界

当前可合理评价为“**system-level novelty high**”，但在完成上述公平 baseline/ablation 前，不应把“高创新性”写成已被实验完全证明的事实。论文最终 claim 应以可复现实验和投稿日前文献检索共同支撑。
