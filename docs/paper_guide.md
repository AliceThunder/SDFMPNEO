# SDF-MPNEO 论文撰写指导

> 用途：指导英文论文从 Introduction 到 Methods、Results、Discussion、Conclusion 的完整写作。本文档定义每一节应该证明什么、对应哪一部分代码和哪一类图表。

---

# 1. 论文核心命题

论文不是“提出一个神经网络预测 UWPT 参数”，而是证明：

> **A strongly coupled underwater-WPT multiphysics surrogate can be constructed without precomputed solution labels by learning admissible geometry-conditioned trial-space enrichments and solving the governing multiphysics equilibrium inside these spaces.**

核心科学问题分成四个：

1. 如何在没有 FEM/CFD 解标签时学习有用的 reduced space？
2. 如何在神经代理中保持 EM/flow admissibility、热守恒和多端口物理结构？
3. 如何处理 EM 与 thermal/flow 的强耦合和跨时间尺度？
4. 如何保证在线效率并知道 surrogate 何时不可靠？

整篇论文必须围绕这四个问题展开。

---

# 2. 推荐题目方向

工作名，不是最终标题：

> **Solution-Data-Free Physics-Embedded Neural Equilibrium Operator for Strongly Coupled Multiphysics Modeling of Underwater Wireless Power Transfer**

更强调 reduced space：

> **Physics-Seeded Neural Trial-Space Learning for Solution-Data-Free Multiphysics Surrogates of Underwater Wireless Power Transfer**

在 prior-art 完整排查前，不在标题中使用 `first`。

---

# 3. Abstract 结构

摘要按 5 句逻辑写：

## Sentence 1 — Problem

指出 UWPT 的 impedance/loss/temperature/flow 相互耦合，而 repeated high-fidelity simulation 昂贵。

## Sentence 2 — Gap

指出已有 data-driven surrogate 依赖 solution data；单纯 physics-informed fitting 又难以同时处理 hard admissibility、strong coupling 和 online cost。

## Sentence 3 — Method

必须包含：

```text
solution-data-free
physics-seeded neural trial-space enrichment
de Rham hard constraints
matrix-free EM inner solve
mixed-dimensional thermal + RANS-SST equilibrium
```

## Sentence 4 — Reliability/efficiency

写：

```text
solution-free operator cubature
independent residual certification
adaptive rank
```

## Sentence 5 — Results

最终实验完成后填：

- impedance error
- temperature error
- speedup
- label count = 0
- certificate coverage

摘要中不要堆网络层数、hidden dimension 等实现细节。

---

# 4. Introduction 建议结构

## Paragraph 1 — UWPT application and multiphysics need

讲：

- seawater conductive loss
- package thermal path
- ambient/current-driven flow
- temperature-dependent material properties

结尾落到：

> Accurate design therefore requires a strongly coupled EM–thermal–fluid model rather than independent single-physics evaluations.

## Paragraph 2 — High-fidelity multiphysics limitation

讲 FEM/CFD 的精度与 repeated evaluation 的成本矛盾。

重点不是批评 FEM，而是指出优化、参数扫描、实时设计需要 surrogate。

## Paragraph 3 — Data-driven surrogate limitation

说明常规 NN/ROM/operator learning 依赖 solution snapshots or labels。

问题：

- expensive offline data generation
- limited coverage of geometry/environment domain
- extrapolation confidence

## Paragraph 4 — Physics-informed learning limitation

不要泛泛说 PINN 不行。

具体指出：

- coupled residual scale imbalance
- soft divergence/boundary constraints
- geometry-dependent domains
- nonlinear coupled equilibrium
- residual evaluation itself can remain expensive

## Paragraph 5 — Our idea

第一次出现核心句：

> Instead of learning the physical solution or terminal impedance, SDF-MPNEO learns a geometry-conditioned admissible reduced trial space around deterministic physics seeds.

再说明控制方程仍然负责最终物理解。

## Paragraph 6 — Contributions

建议 4 条，不超过 5 条：

1. solution-data-free physics-seeded neural trial-space framework；
2. de Rham + mixed-dimensional conservative multiphysics structure；
3. multirate matrix-free strongly coupled equilibrium；
4. solution-free hyper-reduction + independent certification + adaptive rank。

贡献条目要写“能力”，不要只是列模块。

---

# 5. Methods 建议章节结构

推荐：

```text
II. Problem Formulation
III. SDF-MPNEO Architecture
IV. Solution-Data-Free Training and Certification
```

如果版面紧，可以把 IV 合入 III。

---

# 6. Section II — Problem Formulation

## A. Unified Geometry and Multiphysics Domain

定义：

```text
Gamma_c          conductor centerline
Omega_p          package solid
Omega_s          seawater domain
```

定义输入参数：

```text
xi = [geometry, pose, f, I, T_inf, S, sigma_ref, U_inf]
```

说明单/双线圈 topology templates 与 reference-domain geometry mapping。

## B. Electromagnetic Governing Problem

定义 seawater induced current `J_s`，而不是先从 NN 讲起。

给出 harmonic operator form：

```text
A_EM(T,S,omega) J_s = S_c I
```

强调 material dependence：

```text
sigma_sea = sigma(T,S)
rho_Cu = rho_Cu(T_c)
```

## C. Conjugate Heat Transfer

定义统一：

```text
Theta = [T_c, T_p, T_s]
```

给 1D conductor + 3D package/seawater equations。

给 conservative mortar：

```text
K_cp = E_cp^T W_cp E_cp
```

## D. Incompressible Flow and SST Closure

给 RANS-SST 控制形式。

说明 pressure 不是 surrogate state；velocity space 后续会 hard divergence-free。

## E. Coupling Graph

本节结尾必须给闭环：

```text
J -> Q_sea -> T -> sigma -> J
I -> Q_Cu -> T_c -> rho_Cu -> Q_Cu
T -> properties -> flow/SST -> heat transfer -> T
```

明确这是论文问题定义，不是网络结构。

---

# 7. Section III — SDF-MPNEO Architecture

## A. Multirate Reduced Equilibrium

先解释为什么 EM harmonic inner solve + slow thermal-flow outer equilibrium。

定义：

```text
z = [a_T,a_v,a_k,a_omega]
```

EM coefficient 被内层消元。

核心式：

```text
F_MP(z;xi,theta)=0
```

## B. Physics-Seeded Neural Trial Spaces

核心公式：

```text
B_p = orth_P [B_p^phys, (I-Pi_phys)B_p^NN]
```

说明：

- seed 不使用 solution snapshots；
- NN 只学习 orthogonal enrichment；
- seed 是所有 nested rank 的 mandatory prefix。

## C. Hard de Rham Constraints

给 exact sequence：

```text
V0 -> V1 -> V2 -> V3
D C = 0
```

写：

```text
B_J = [C_J,H_J] A_J
B_v = [C_v,H_v] A_v
```

说明多连通域 harmonic basis、gauge reduction、boundary-normal DOF removal。

## D. Geometry-Conditioned Neural Generator

写 shared encoder + physics-specific decoders。

强调 `slow_features` 不进入 EM head。

不要在主文过多展开普通 MLP 细节；把 hidden dim、层数放 implementation subsection 或 SI。

## E. Matrix-Free EM Projection

这是论文关键创新段之一。

不要写 full dense `R_s,L_s` 作为主实现。

定义：

```text
A(V), S(B), H(V), D(V)
```

然后：

```text
A_r = B_J^H A(B_J)
S_r = S(B_J)
A_r Y = S_r
Z_sea = H_r Y
Q_sea = Y^H D_r Y
```

说明环境 action 可以由 FFT/FMM/H2/DSE 等实现，方法不绑定某个 kernel。

## F. Coupled Thermal-Flow Reduced System

给 reduced thermal/RANS-SST residual。

说明每次 residual evaluation 都重新调用 current-temperature-dependent EM inner solve。

## G. Pseudo-Transient Newton

公式：

```text
(P/dtau + J_F) Delta z = -F
```

解释稳定性与靠近解后的 Newton convergence。

---

# 8. Section IV — Solution-Data-Free Training and Certification

## A. Why Galerkin residual cannot be the training loss

必须明确指出：

```text
B^H R = 0
```

可能是 projection 的天然结果，不能作为 trial-space quality 的独立监督。

## B. Riesz-Whitened Physics Residual

定义：

```text
P = blockdiag(P_J,P_T,P_v,P_k,P_omega)
L = 1/2 ||P^{-1/2} R_MP||^2
```

强调：避免大规模手工 `lambda_EM,lambda_T,lambda_F`。

## C. Implicit Differentiation

给：

```text
F(z*,theta)=0
F_z^H lambda = dL/dz
dL/dtheta = partial_theta L - lambda^H partial_theta F
```

说明 Newton history 不展开。

## D. Basis-Operator Cubature

定义 operator moments 和 positive weights。

强调：

- no solution snapshots；
- local thermal/flow nonlinear operators hyper-reduced；
- nonlocal EM 走 matrix-free action，不用 BOC。

## E. Independent Certification

说明 solve set 与 certifier set disjoint。

区别：

- EM：可给较严格 dissipative residual bound；
- nonlinear thermal-flow：a-posteriori indicator / dual-weighted residual。

不要把后者夸成 rigorous bound。

## F. Adaptive Rank

给 nested rank schedule 与 fail->increase->fallback 流程。

---

# 9. Results 章节必须回答的六个问题

Results 不按“网络训练结果 / 测试结果”这种普通 AI 论文组织。

应该按科学问题组织。

## A. Does the model preserve physics by construction?

展示：

- divergence defect
- reciprocity defect
- minimum dissipative eigenvalue
- heat-exchange conservation
- SST positivity

## B. Can it train with zero solution labels?

展示：

- residual loss vs iteration
- equilibrium convergence
- different parameter-domain cells
- zero labels statement

## C. How accurate is EM/thermal/flow prediction?

独立测试集：

- BFZI / Maxwell
- thermal FEM/COMSOL
- CFD
- experiments

所有这些**只用于最终 validation**。

## D. Is strong coupling necessary?

做 ablation：

```text
full strong coupling
freeze sigma(T)
freeze rho_Cu(T)
remove flow-to-thermal feedback
```

不是为了说简化模型“差”，而是量化每个反馈在不同工况下的物理贡献。

## E. Does adaptive rank work?

展示：

- rank distribution across parameter domain
- rank vs residual
- rank vs geometry/flow severity

## F. Is online speed actually improved?

必须分模块计时：

```text
geometry encoding
basis query
EM operator projection
reduced EM solve
thermal-flow BOC assembly
Newton
certification
optional field reconstruction
```

不要只给一个总 speedup。

---

# 10. 必做 Ablation

至少：

1. neural-only basis vs physics-seeded neural enrichment；
2. soft divergence penalty vs hard de Rham basis；
3. fixed rank vs adaptive rank；
4. full element assembly vs BOC；
5. same solve/certifier points vs independent certifier；
6. weak one-way coupling vs strong equilibrium；
7. dense reference EM vs matrix-free EM projection。

注意：ablation 是证明机制作用，不是为了贬低自己的基础版本。

---

# 11. 推荐主图

## Fig. 1
UWPT multiphysics problem and coupling graph。

## Fig. 2
Full SDF-MPNEO architecture。

## Fig. 3
Physics seed + neural enrichment + de Rham constraints。

## Fig. 4
Matrix-free EM + mixed-dimensional thermal + RANS-SST equilibrium。

## Fig. 5
Solution-data-free training + implicit adjoint + BOC + certifier。

## Fig. 6
Physics invariants / certification。

## Fig. 7
Accuracy against independent high-fidelity solvers。

## Fig. 8
Adaptive rank / efficiency scaling。

## Fig. 9
Representative strongly coupled temperature/flow/impedance case。

---

# 12. 推荐表格

## Table I — Parameter domain

列 geometry / electrical / seawater / thermal / flow ranges。

## Table II — Physical-state and reduced-space definitions

列 state、space、hard constraint、rank。

## Table III — Accuracy and certification

按 QoI 列 error / indicator / coverage。

## Table IV — Runtime breakdown

按模块列 full/reference vs SDF-MPNEO。

---

# 13. Discussion 应该讨论什么

Discussion 不要写成“limitations list”。

重点讨论：

1. 为什么 learned trial space 比 direct solution regression 更适合无标签训练；
2. 为什么强耦合与 multirate 并不矛盾；
3. 为什么 hard admissibility 减少了 loss balancing burden；
4. 为什么 operator-action abstraction 使方法不依赖某个 Green/FEM backend；
5. adaptive rank 与 physical certification 如何改变 surrogate 的使用方式。

如果要谈边界，只在物理适用性语境下写：

- RANS-SST model validity range；
- motional EM coupling activated by Rm criterion；
- geometry topology templates currently supported。

结尾回到方法能力，不以自我削弱结束。

---

# 14. Conclusion 写法

最后只保留三件事：

1. zero-solution-label multiphysics surrogate capability；
2. physics-by-construction strong coupling；
3. certified adaptive online efficiency。

不要在最后一句突然写 future work。

---

# 15. 论文中必须保持一致的术语

统一使用：

```text
solution-data-free
physics seed
neural enrichment
admissible trial space
multirate equilibrium
matrix-free operator action
Basis-Operator Cubature (BOC)
independent physical certifier
adaptive rank
physics fallback
```

避免混用：

```text
training data-free / data-free / no-data / unsupervised
```

优先统一成 `solution-data-free`。

---

# 16. 论文结果未完成前禁止写死的结论

在真实 benchmark 完成前，不写：

- `X× faster`
- `<1% error`
- `real-time`
- `rigorous thermal-flow error bound`
- `first`

这些必须由后续结果决定。
