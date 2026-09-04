# SDF-MPNEO PPT 指导稿

> 用途：组会、开题、中期、答辩、论文汇报。本文档不是简单目录，而是定义**每一页需要讲清楚的科学问题、图示和公式**。

---

# 1. 推荐主线

PPT 不应从“我们用了一个神经网络”开始，而应从 UWPT 的建模矛盾开始：

```text
真实水下 WPT = 几何 + 电磁 + 海水损耗 + 热 + 海流 + 材料反馈
                |
高保真求解准确但慢
传统代理快但依赖数据且缺少物理可信度
                |
SDF-MPNEO
```

整套 PPT 的叙事顺序固定为：

```text
为什么需要 -> 为什么现有方法不够 -> 我们的核心思想 -> 数学结构
-> 强耦合怎么实现 -> 无数据怎么训练 -> 怎么保证可信 -> 怎么验证
```

---

# 2. 建议 15 页标准版

## Slide 1 — Problem

标题建议：

> **Strongly Coupled Multiphysics Surrogate Modeling for Underwater WPT**

图：

- 双线圈 + 封装 + 海水；
- 箭头标出 EM、heat、flow；
- 标出 `T -> sigma_sea`, `T -> rho_Cu`, `v -> heat transfer`。

一句话：

> 水下 WPT 的阻抗、损耗与温升不是独立问题，而是由电磁–热–流体反馈共同决定。

---

## Slide 2 — Existing bottleneck

三栏：

| Method | Strength | Limitation |
|---|---|---|
| FEM/CFD | high fidelity | repeated solves expensive |
| data-driven surrogate | fast online | requires labeled solution data |
| PINN/operator residual fitting | physics informed | coupled residual balancing / hard constraints / geometry generalization remain difficult |

不要写“现有方法完全不行”，而强调：

> 现有路线难以同时满足 **zero solution labels + strong multiphysics coupling + online efficiency + physical admissibility**。

---

## Slide 3 — Central idea

必须放一句：

> **The neural network learns admissible trial spaces, not the final physical answers.**

主图：

```text
Geometry -> Neural trial-space generator -> Physics equations -> Equilibrium -> QoIs
```

与黑箱代理对比：

```text
Conventional: x -> NN -> Z,T,v
Ours: x -> NN -> B_J,B_T,B_v -> governing equations -> Z,T,v
```

---

## Slide 4 — Full architecture

使用 `docs/master_design.md` 中的总架构图。

必须突出三点：

1. shared geometry encoder；
2. physics seed + neural enrichment；
3. fast EM inner solve embedded in slow thermo-fluid equilibrium。

---

## Slide 5 — Why multirate

左：

```text
tau_EM ~ 1/f
```

右：热/流更慢。

结论：

> 不采用统一微秒时间步，而采用 harmonic EM inner solve + slow thermo-fluid equilibrium。

强调：

> 时间尺度分离不等于物理解耦；温度更新后 EM 在每次 residual evaluation 中重新求解。

---

## Slide 6 — Physics-seeded neural basis

核心公式：

```text
B_p = orth_P [ B_p^phys , (I-Pi_phys) B_p^NN ]
```

图：

```text
operator physics seed -> stable backbone
                     + neural enrichment -> remaining manifold
```

解释：

- 不从随机解空间起步；
- 不用 solution snapshots；
- NN 只补充 physics backbone 没覆盖的方向。

---

## Slide 7 — Hard physical admissibility

核心图：de Rham sequence

```text
V0 --G--> V1 --C--> V2 --D--> V3
```

公式：

```text
D C = 0
B_J = [C_J,H_J] A_J
B_v = [C_v,H_v] A_v
```

对应结果：

```text
div(J_s)=0
div(v)=0
```

再加：

- `k_t = k_ref exp(kappa)`
- `omega_t = omega_ref exp(varpi)`

标题可写：

> **Physics by construction rather than penalty tuning**

---

## Slide 8 — Matrix-free EM

不要画 dense matrix。

画：

```text
B_J -> A(B_J) -> B_J^H A(B_J) = A_r
B_J -> S(B_J) = S_r
```

核心：

```text
A_r Y = S_r
Z_sea = H_r Y
```

强调：

> 生产路径从不构造 full `[N_J,N_J]` 环境矩阵。

再画：

```text
T -> sigma(T,S) -> local diagonal weights
Green geometry kernel remains reusable
```

---

## Slide 9 — Mixed-dimensional thermal model

图：

```text
1D conductor centerline
inside
3D package + 3D seawater
```

公式：

```text
Theta = [T_c, T_package, T_sea] ~= B_T a_T
K_cp = E_cp^T W_cp E_cp
```

突出：

> conductor-package heat exchange is conservative by construction。

---

## Slide 10 — Flow and SST

图：安装盒绕流 + thermal boundary layer。

主模型：

```text
incompressible RANS + k-omega SST
```

强调：

- pressure 不作为 surrogate state；
- divergence-free velocity space；
- turbulence states 保证正值；
- 流体通过换热反馈到温度，再反馈到 EM。

---

## Slide 11 — Strong coupled equilibrium

必须用闭环图：

```text
T -> sigma -> EM -> Q_sea -> T
T_c -> rho_Cu -> Q_Cu -> T_c
T -> fluid properties -> flow/SST -> heat transfer -> T
```

核心 residual：

```text
F_MP(z)=0
z=[a_T,a_v,a_k,a_omega]
```

求解器：

```text
(P/dtau + J_F) Delta z = -F
```

---

## Slide 12 — Solution-data-free training

大字写：

```text
0 FEM/CFD/experimental solution labels
```

训练流程：

```text
sample physics parameters
-> generate basis
-> solve reduced equilibrium
-> evaluate independent physics residual
-> implicit adjoint gradient
```

Loss：

```text
L = 1/2 || P^{-1/2} R_MP ||^2
```

强调：

> 不使用大量人工 lambda 进行多物理 loss 拼接。

---

## Slide 13 — Hyper-reduction and certification

左：BOC

```text
all elements -> positive operator moments -> small cubature set
```

右：independent certifier

```text
S_BOC intersect S_cert = empty
```

说明：

- BOC 不使用 solution snapshots；
- certifier 不与 solve sampling 共用；
- 简单工况低 rank，复杂工况自动升 rank。

---

## Slide 14 — Adaptive rank

画阶梯：

```text
L1 -> certificate fail -> L2 -> L3 -> L4 -> fallback
```

表：

| level | rJ | rT | rv | rk | rw |
|---|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 12 | 4 | 4 |
| 2 | 16 | 16 | 20 | 8 | 8 |
| 3 | 24 | 24 | 28 | 10 | 10 |
| 4 | 32 | 32 | 36 | 12 | 12 |

一句：

> Model capacity is allocated by physical error, not a learned difficulty classifier.

---

## Slide 15 — Validation roadmap / contribution

验证必须分层：

```text
Level 1: mathematical invariants / CI
Level 2: independent deterministic physics backend
Level 3: BFZI / Maxwell / CFD comparison
Level 4: experiments
```

最后贡献建议写成四条：

1. solution-data-free physics-seeded neural trial-space learning；
2. de Rham constrained EM/flow spaces + mixed-dimensional thermal coupling；
3. matrix-free multirate strong multiphysics equilibrium；
4. solution-free hyper-reduction + independent certification + adaptive rank。

不要在最后突然加入“limitations/future work”弱化贡献。

---

# 3. 8 页精简版

```text
1 Problem
2 Existing gap
3 Core idea
4 Full architecture
5 Hard physics + multirate coupling
6 Solution-data-free training
7 Certification + efficiency
8 Contribution + validation
```

---

# 4. 必画的 6 张核心图

1. UWPT multiphysics coupling schematic
2. SDF-MPNEO overall architecture
3. physics seed + neural enrichment
4. de Rham hard-constraint diagram
5. multirate coupled equilibrium loop
6. adaptive rank + independent certification

这 6 张图也应优先成为论文方法图。

---

# 5. PPT 中禁止的表述

禁止：

- “NN directly predicts impedance accurately”
- “thermal and flow are auxiliary modules”
- “we neglect fluid effects”
- “no data are used at all”
- “our method is the first data-free neural operator”

应改为：

- NN learns admissible trial-space enrichments；
- EM/thermal/flow form one strongly coupled equilibrium；
- training uses no precomputed solution labels；
- novelty lies in the integrated structure, not a single generic concept。
