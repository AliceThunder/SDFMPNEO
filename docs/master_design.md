# SDF-MPNEO Master Design

> **Solution-Data-Free Multirate Physics-Embedded Neural Equilibrium Operator for Underwater WPT**
>
> 本文档是项目的**唯一主设计依据（single source of truth）**。PPT、论文和代码都必须从这里派生，避免三套叙述逐渐分叉。

---

# I. 一句话定义

SDF-MPNEO 不是一个从工况直接回归阻抗、温度或流场的黑箱神经网络，而是一个：

> **由神经网络生成几何条件下的多物理可行低维解空间，由真实电磁–热–流体控制方程在该空间内联合求解强耦合平衡，并通过独立物理残差实现自适应秩提升与可信度判定的无解数据神经平衡算子。**

核心关系：

```text
NN learns trial spaces, not physical answers.
Physics solves the states inside those spaces.
```

---

# II. 项目必须长期保持的六条原则

## P1. 多物理不能删

核心物理状态固定为：

- seawater induced current: `J_s`
- unified temperature state: `Theta = [T_c, T_package, T_sea]`
- seawater velocity: `v`
- SST turbulence states: `k_t, omega_t`

压力 `p` 只作为不可压缩流体内部约束/消元变量，不形成独立代理模块。

## P2. 无解数据训练

训练不得使用：

```text
Maxwell / COMSOL / CFD full solutions
experimental field / impedance / temperature labels
precomputed solution snapshots
```

允许使用：

```text
geometry samples
material/environment samples
operator actions
weak forms / energy forms
physics seed construction
residuals / adjoints / certificates
```

正式口径使用：

```text
solution-data-free
label-free
no precomputed solution labels
```

## P3. 神经网络不直接预测最终物理量

禁止主路径：

```text
x -> NN -> Z
x -> NN -> T_max
x -> NN -> v
```

主路径固定为：

```text
geometry + operating state
        |
neural trial-space generator
        |
physics admissibility maps
        |
reduced governing equations
        |
multiphysics equilibrium
        |
Z, loss, T, v
```

## P4. 强耦合必须保留反馈闭环

必须包含：

```text
J_s -> Q_sea -> T -> sigma_sea(T,S) -> J_s
I   -> Q_Cu  -> T_c -> rho_Cu(T_c)   -> Q_Cu
T   -> mu,rho,k -> v,k_t,omega_t -> turbulent heat transfer -> T
```

## P5. 能硬约束的物理不放进 loss

结构性保证：

- `div(J_s)=0`
- `J_s . n = 0`
- `div(v)=0`
- `k_t > 0`
- `omega_t > 0`
- conductor-package heat exchange conservation
- multiport shared EM trial space
- basis full rank / metric orthogonality

## P6. 在线模型必须真正快

禁止生产路径：

```text
construct full dense [N,N] EM matrices
materialize every full-field basis value when only QoIs are needed
assemble all thermal/fluid cells if hyper-reduction is available
```

生产路径只允许：

```text
matrix-free operator actions
reduced source projections
Basis-Operator Cubature
lazy/query basis evaluation
optional full-field reconstruction
```

---

# III. 整体架构

```text
                 Unified Geometry + Environment
                           |
                  Shared Geometry Encoder
                           |
          +----------------+----------------+
          |                |                |
        EM Head        Thermal Head      Flow-SST Head
          |                |                |
          +----------------+----------------+
                           |
                Physics Seed + Neural Enrichment
                           |
          +----------------+----------------+
          |                |                |
      de Rham J        H1/Mixed Heat      de Rham v + log SST
          |                |                |
          +----------------+----------------+
                           |
                 Metric Orthogonalization
                           |
                           v
                  Reduced Physical Spaces
                           |
             Fast Harmonic EM Inner Elimination
                           |
                    Q_Cu , Q_sea , Z
                           |
                           v
                Coupled Thermo-Fluid Equilibrium
                           ^
                           |
          T -> sigma, rho_Cu, mu, rho, k -> EM / Flow
                           |
                           v
                  Independent Physical Certifier
                           |
                           v
                Adaptive Rank / Physics Fallback
                           |
                           v
              Z, P_loss, T, v, QoI indicators
```

---

# IV. 多时间尺度设计

UWPT 电磁状态工作于 harmonic steady state，特征时间尺度约为：

```text
tau_EM ~ 1/f
```

热和海水流动明显更慢。

因此不把三个物理场强行放入同一个微秒级时间步，而采用：

```text
fast harmonic EM inner solve
inside
slow thermo-fluid equilibrium / transient solve
```

这不是弱耦合。每次慢变量迭代都重新更新温度相关材料参数并重新计算 EM reduced solve。

慢状态：

```text
z = [a_T, a_v, a_k, a_omega]
```

EM reduced coefficients `a_J` 通过内层物理方程消元，不进入慢 Newton unknown。

---

# V. Physics Seed + Neural Enrichment

纯随机 neural basis 会导致训练初期 reduced equilibrium 难以收敛，因此所有物理 head 都采用：

```text
B_p = orth_P [ B_p^phys , (I - Pi_phys) B_p^NN ]
```

其中：

- `B_p^phys`：不依赖 solution labels 的 deterministic operator modes；
- `B_p^NN`：geometry-conditioned neural enrichment；
- neural enrichment 必须先投影到 physics seed 的 metric-orthogonal complement。

推荐 physics seeds：

| Physics | Seed construction |
|---|---|
| EM | source/operator compression + rational Krylov / operator modes |
| Thermal | diffusion / energy generalized eigenmodes |
| Velocity | divergence-free Stokes modes |
| SST log states | constant + wall-distance + operator modes |

Physics seed rank 必须不大于第一级 nested rank，保证所有在线 rank 都保留完整 physics backbone。

---

# VI. 神经网络结构

## 6.1 输入信息分离

`global_features`：所有 head 都可见，只包含：

- geometry descriptors
- frequency
- salinity / reference conductivity
- ambient state
- imposed inflow descriptors

**禁止包含实际 port-current amplitudes。**

`slow_features`：仅 Thermal / Flow head 可见，可包含：

- `|I|^2`
- load descriptors
- thermal operating quantities

因此 EM trial space 保持 excitation-independent。

## 6.2 Query-conditioned basis decoder

不使用一次性输出完整 `[N,r]` 的巨大 MLP。

采用：

```text
basis_value = Decoder(local_query, global_latent, mode_embedding)
```

每个 head 在需要的物理 DOF / cubature point / visualization point 上按需查询 basis。

## 6.3 Encoder

V1 reference encoder 可以是简洁 message-passing / token encoder。

最终推荐：

```text
SE(3)-equivariant geometry encoder
```

原因：整体刚体旋转不应改变标量物理，矢量场应协变旋转。

---

# VII. de Rham 硬物理结构

参考域离散 exact sequence：

```text
V0 --G--> V1 --C--> V2 --D--> V3
```

满足：

```text
C G = 0
D C = 0
```

## 7.1 EM current basis

网络输出的是 curl-potential + harmonic coordinates：

```text
B_J_raw = [C_J, H_J] A_J
```

因此：

```text
D_J B_J_raw = 0
```

边界 normal-current DOF 从空间中删除，使：

```text
J_s . n = 0
```

## 7.2 Velocity basis

同理：

```text
B_v_raw = [C_v, H_v] A_v
D_v B_v_raw = 0
```

因此 pressure 不需要作为 neural state。

## 7.3 Harmonic space

安装盒使海水域成为多连通域，因此：

```text
ker(D) = range(C) + H
```

不能只用 `curl` 空间，必须显式补 harmonic basis。

## 7.4 Gauge reduction

backend 暴露给网络的 `C_g` 必须先去除 gauge-null directions。

运行时验证：

```text
D C_g = 0
D H   = 0
rank([C_g,H]) = number of supplied coordinates
```

---

# VIII. Metric Orthogonalization 与 rank guard

每个物理 basis 使用自己的物理 metric：

```text
B_p^H P_p B_p = I
```

在加入 numerical jitter 前必须检查原始 Gram matrix：

```text
lambda_min / lambda_max > rank_rtol
```

否则直接抛出 rank-collapse error。

Jitter 只允许用于数值稳定，不允许掩盖 basis collapse。

Decoder hidden dimension 必须满足：

```text
hidden_dim >= max requested rank
```

---

# IX. 电磁核心：通用 Matrix-Free Operator Action

SDF-MPNEO 不以某个具体 BFZI 实现为架构基础。

生产级 EM backend 必须满足通用接口：

```text
A(V)       : environment system action
S(B)       : direct reduced source projection B^H S
H(V)       : port feedback functional
D(V)       : dissipative action
Z0         : copper + air/background multiport impedance
```

对 trial basis `B_J`：

```text
A_r = B_J^H A(B_J)
S_r = S(B_J)
H_r = H(B_J)
D_r = B_J^H D(B_J)
```

然后求：

```text
A_r Y = S_r
```

并恢复：

```text
Z_sea = H_r Y
Z = Z0 + Z_sea
Q_sea = Y^H D_r Y
```

核心模型不需要知道 `A` 底层来自：

```text
FFT Green
DSE
FMM
H2
sparse PDE operator
other deterministic solvers
```

## 9.1 温度耦合

建议组织为：

```text
A_T(V) = V + G[ D_gamma2(T,S) P V ]
```

其中：

```text
gamma2(T,S) = j omega mu sigma(T,S)
```

Green geometry kernel 可缓存；每次慢变量更新只更新局部材料对角权重。

这使热–电强耦合保持高效。

## 9.2 BFZI 的定位

BFZI 只作为：

- deterministic reference
- optional backend implementation
- benchmark source

可参考的一般思想：

- geometry repetition compression
- source-side structural aggregation
- matrix-free Green action
- separation of geometry and local material weights

禁止复制：

- BFZI-specific classes/functions/naming
- branch formulas and implementation code
- FFT lattice implementation
- FV projector code
- cache/version scripts
- sample-specific configuration

详见 `docs/bfzi_reference_boundary.md`。

---

# X. 热模型：统一混合维状态

热状态固定为：

```text
Theta = [T_c(s), T_package(x), T_sea(x)]
```

只有一个 Thermal basis：

```text
Theta ≈ B_T a_T
```

不再把 1D conductor temperature 作为另一个独立 reduced block。

## 10.1 1D conductor

```text
rho_c c_c A_c dT_c/dt
- d/ds(k_c A_c dT_c/ds)
+ q_c->p
= q'_Cu
```

RMS convention：

```text
q'_Cu = R'_ac(f,T_c) |I|^2
```

## 10.2 package + seawater

统一共轭传热：

```text
rho cp dT/dt
+ rho cp v_Omega . grad(T)
- div(k_eff grad(T))
= Q
```

其中：

```text
v_Omega = 0       in package
v_Omega = v       in seawater
```

因此 package conduction 与 seawater convection 不需要两个热代理。

## 10.3 1D-3D conservative mortar

定义 jump operator：

```text
E_cp T = T_c - I_p->c T_p
```

耦合矩阵：

```text
K_cp = E_cp^T W_cp E_cp
```

`W_cp >= 0`。

这样导线失热与封装得热在离散层严格守恒。

---

# XI. 流体模型：incompressible RANS-SST

安装盒外流工况不能默认层流，因此 V1 主模型固定为：

```text
incompressible RANS + k-omega SST
```

动量方程：

```text
rho (dv/dt + v.grad(v))
- div[(mu+mu_t)(grad(v)+grad(v)^T)]
+ grad(p)
= rho g beta (T-T_inf)
```

```text
div(v)=0
```

pressure 通过 divergence-free test/trial space 消去，不作为 neural state。

## 11.1 SST positivity

网络学习：

```text
kappa = log(k_t/k_ref)
varpi = log(omega_t/omega_ref)
```

恢复：

```text
k_t = k_ref exp(kappa)
omega_t = omega_ref exp(varpi)
```

因此：

```text
k_t > 0
omega_t > 0
```

恒成立。

## 11.2 热流耦合

```text
nu_t -> k_eff = k_f + rho cp nu_t / Pr_t
```

形成：

```text
T -> fluid properties -> v,k_t,omega_t -> turbulent heat transfer -> T
```

## 11.3 Motional EM coupling

使用 magnetic Reynolds number：

```text
Rm = mu sigma U L
```

只有当 `Rm` 达到设定阈值时才启用显式 `v x B` motional EM term。

默认 UWPT 海流尺度下 `Rm` 通常很小，因此主路径保留：

```text
v -> T -> sigma(T) -> EM
```

但代码不能永久写死忽略，必须保留自动判据。

---

# XII. 强耦合 Reduced Equilibrium

慢状态：

```text
z = [a_T, a_v, a_k, a_omega]
```

EM 在每次 residual evaluation 内重新求解。

伪代码：

```text
function F_MP(z):
    reconstruct T/v/SST only at required points
    update sigma(T,S), rho_Cu(Tc), mu(T), rho(T), k(T)
    build matrix-free EM actions at current material state
    solve reduced harmonic EM
    compute Z, Q_sea, Q_Cu
    assemble reduced thermal residual
    assemble reduced RANS-SST residual
    return concatenated slow residual
```

稳态：

```text
F_MP(z) = 0
```

瞬态扩展：

```text
M_r dz/dt + F_MP(z) = 0
```

同一套 reduced physics 支持稳态与瞬态，不建立第二个代理。

---

# XIII. 非线性求解器

采用 pseudo-transient damped Newton：

```text
(P/dtau + J_F) Delta z = -F(z)
```

然后 line search：

```text
z_{k+1} = z_k + alpha Delta z
```

行为：

- 远离解：小 pseudo-time step 提供稳定性；
- 靠近解：`dtau -> infinity`，退化成 Newton；
- reduced state 只有几十维，可显式构建 Jacobian。

禁止把 neural fixed-point 作为主强耦合求解器。

---

# XIV. 无解数据训练

## 14.1 Training loss

训练不得使用 reduced Galerkin residual：

```text
B^H R = 0
```

因为它对任何 Galerkin solution 都可能天然为零。

必须使用：

```text
full-space residual action
or independent randomized / Riesz witness residual
```

先无量纲化，然后使用 block Riesz metric：

```text
P = blockdiag(P_J, P_T, P_v, P_k, P_omega)
```

训练目标：

```text
L = 0.5 || P^{-1/2} R_MP ||^2
```

不使用大量经验 loss weights。

## 14.2 Implicit differentiation

Newton 历史不进入 autograd graph。

在 equilibrium：

```text
F(z*,theta)=0
```

伴随方程：

```text
F_z^H lambda = dL/dz
```

总梯度：

```text
dL/dtheta = partial_theta L - lambda^H partial_theta F
```

因此训练内存不随 Newton 步数线性增长。

## 14.3 Two-pass basis evaluation

训练一步：

```text
1. no-grad basis -> solve equilibrium
2. recompute differentiable basis at converged z*
3. evaluate independent physics residual
4. implicit adjoint gradient
5. optimizer step
```

---

# XV. Solution-Free Hyper-Reduction

热和流体不能因为 reduced state 小就每次遍历所有 full-order elements。

采用：

> **Basis-Operator Cubature (BOC)**

对于 element operator moment：

```text
m_e(theta_basis, operator)
```

寻找：

```text
S subset of elements
w_e >= 0
```

使：

```text
sum_all m_e ≈ sum_{e in S} w_e m_e
```

特点：

- 不使用 full-order solution snapshots；
- 只使用 basis + governing operators；
- positive weights 有利于保留 diffusion/viscous dissipation positivity。

EM 非局部 Green operator 不使用 BOC，走 matrix-free environment action。

---

# XVI. Independent Certifier

在线 solve cubature 与 certifier 不能共用同一采样集合：

```text
S_BOC intersect S_cert = empty
```

输出 indicators：

```text
eta_Z
eta_T
eta_v
```

EM 线性子问题可以使用 dissipative residual bound。

RANS-SST 热流部分初期只称：

```text
residual-based a-posteriori indicator
or dual-weighted residual indicator
```

除非后续严格得到稳定常数，不得宣称全局 rigorous error bound。

---

# XVII. Nested Adaptive Rank

默认等级：

| level | rJ | rT | rv | rk | rw |
|---|---:|---:|---:|---:|---:|
| L1 | 8 | 8 | 12 | 4 | 4 |
| L2 | 16 | 16 | 20 | 8 | 8 |
| L3 | 24 | 24 | 28 | 10 | 10 |
| L4 | 32 | 32 | 36 | 12 | 12 |

训练时随机启用不同 prefix rank，使前序 mode 成为主低秩方向。

在线：

```text
solve L1
if certificate fails -> L2
if fails -> L3
if fails -> L4
if fails -> requires_fallback = True
```

因此模型容量由物理误差决定，而不是人为 hard/easy classifier。

---

# XVIII. Lazy Basis Evaluation

最终 online solver 不应默认生成完整：

```text
[N, r]
```

basis。

应只在：

- BOC cells
- certifier cells
- source/operator quadrature points
- requested visualization points

查询 basis。

如果用户只需要：

```text
Z
P_loss
T_max
```

可以完全不恢复全场。

只有要求云图时才执行：

```text
field(x) = B(x) a
```

---

# XIX. 几何处理

至少支持两套 topology template：

```text
T1: single package / single coil
T2: dual package / dual coil
```

同一 topology template 内几何变化通过 reference-domain map：

```text
x = chi_G(x_hat)
```

对 `J_s` 和 `v` 使用 contravariant Piola transform，保持 divergence conformity。

对 `T` 使用 H1 pullback。

统一线圈几何仍以当前项目的参数化连续中心线为基础。

---

# XX. 输出定义

默认工程输出：

```text
Z matrix
R1, R2
L1, L2
M / mutual impedance
P_Cu
P_sea
T_max
conductor/package/seawater temperature probes
flow probes / pressure-drop-derived QoIs if needed
eta_Z, eta_T, eta_v
active rank level
requires_fallback
```

可选全场：

```text
J_s(x)
T(x)
v(x)
k_t(x)
omega_t(x)
```

---

# XXI. 项目创新应如何表述

不要把单一概念写成“创新点”，因为 data-free、adaptive basis、neural operator、ROM 各自已有先行工作。

真正的方法贡献是以下机制的**统一组合**：

1. **Solution-data-free physics-seeded neural trial-space learning**
2. **Shared de Rham topology for seawater induced current and incompressible flow**
3. **Matrix-free projection of nonlinear multiphysics operators onto learned admissible spaces**
4. **Fast harmonic EM elimination embedded inside a slow strongly coupled thermo-fluid equilibrium**
5. **Mixed-dimensional conservative 1D-conductor/3D-package-seawater thermal formulation**
6. **Solution-free positive Basis-Operator Cubature**
7. **Independent physical certification and residual-driven adaptive rank**

论文中不得轻易使用 `first`。需要 prior-art review 后再决定。

---

# XXII. 代码模块映射

```text
src/sdfmpneo/
  contracts.py              # data contracts
  config.py                 # ranks/solver/tolerances
  topology.py               # de Rham runtime validation
  reduction.py              # seed + neural enrichment + metric QR
  seeds.py                  # seed-rank rules
  networks/
    base.py
    reference.py            # reference coordinate decoder
  physics/
    em.py                    # dense reference Schur only
    em_matrix_free.py        # production operator-action reduced solve
    thermal.py               # mixed-dimensional heat helpers
    flow.py                  # SST positivity / fluid helpers
  hyperreduction.py          # BOC
  solvers/
    equilibrium.py           # pseudo-transient Newton
  training.py                # solution-data-free implicit training
  backends/
    base.py                  # backend protocol
  model.py                   # adaptive-rank top-level forward
```

后续新增建议：

```text
geometry/
  coil_parameterization.py
  package_templates.py
  mappings.py

backends/reference/
  geometry_moments.py
  green_action.py
  thermal_operator.py
  rans_sst_operator.py
  certifier.py

backends/native/
  cxx bindings / CUDA bindings
```

---

# XXIII. 开发优先级

## Milestone A — Foundation

已完成/接近完成：

- contracts
- de Rham validation
- physics seed + neural enrichment
- metric QR + rank guard
- dense reference EM
- matrix-free EM abstraction
- mixed-dimensional thermal conservation helper
- SST positivity helper
- pseudo-transient Newton
- implicit adjoint
- BOC
- independent certifier interface
- CI invariant tests

## Milestone B — Independent reference physics backend

必须从零实现，不复制 BFZI 代码：

1. `GeometryMomentCompiler`
2. `GreenEnvironmentAction`
3. temperature-dependent seawater material weights
4. mixed-dimensional heat reference assembler
5. incompressible RANS-SST reference assembler
6. certifier operators

## Milestone C — First end-to-end solution-data-free training

最小参数域：

```text
single + dual coil templates
100-300 kHz
sigma / salinity
T_inf
U_inf
geometry parameters
```

目标：训练能稳定下降 residual，并通过未参与训练的 physics certifier。

## Milestone D — High-performance backend

- C++ / OpenMP / CUDA
- structured geometry moments
- fast Green actions
- BOC runtime
- lazy basis evaluation

## Milestone E — Validation

训练结束后才使用：

- BFZI reference
- Maxwell / COMSOL / CFD
- experiments

这些只用于验证，不回流为训练 labels。

---

# XXIV. 必须自动测试的物理不变量

每次 CI 至少验证：

```text
D C_g == 0
D H == 0
exact+harmonic generator full column rank
EM and velocity bases divergence free
metric Gram ~ I
basis collapse is rejected
slow_features cannot alter EM basis
multiport reciprocity
EM dissipation non-negative
port dissipation == reduced Joule quadratic form
1D-3D mortar symmetry / PSD / uniform-temperature null mode
SST reconstructed states positive
implicit gradient matches analytic toy problem
matrix-free EM == dense reference EM
matrix-free action receives only [N,r], never [N,N]
BOC weights non-negative
BOC and certifier sets disjoint
physics seed remains nested prefix
neural enrichment is metric-orthogonal to physics seed
```

---

# XXV. 三种使用方式

本主文档同时服务：

1. **PPT**：从“问题—架构—创新—关键公式—难点解决—验证路线”抽取；
2. **论文**：从“Problem Formulation—Method—Training—Certification—Experiments”组织；
3. **代码**：从“contracts—modules—tensor shapes—solver flow—tests—milestones”实现。

具体映射见：

- `docs/ppt_guide.md`
- `docs/paper_guide.md`
- `docs/implementation_plan.md`
- `docs/traceability_matrix.md`
