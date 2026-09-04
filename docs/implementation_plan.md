# SDF-MPNEO 代码实现方案

> 本文档用于直接指导代码开发。所有实现必须与 `docs/master_design.md` 保持一致。

---

# 1. 总体实现原则

## 1.1 Core 与 backend 分离

Core 只定义：

- trial-space contracts
- hard constraints
- reduced solves
- equilibrium solver
- implicit training
- certification flow

Backend 负责：

- geometry discretization
- topology construction
- physics seeds
- material laws
- operator actions
- element/operator moments
- field reconstruction

核心模型不得依赖某个具体 BFZI/FFT/FEM 实现。

## 1.2 Reference backend 与 native backend 分离

先实现完全独立的 Python reference backend，用来验证数学和接口。

之后再实现：

```text
C++/OpenMP
CUDA
other native kernels
```

二者必须通过同一 Protocol。

## 1.3 禁止隐藏 full-order cost

所有 production API 都必须能回答：

```text
是否构造了 [N,N] matrix?
是否遍历了所有 elements?
是否恢复了完整 [N,r] basis?
```

默认答案应为否，只有 reference/testing 或显式 full-field reconstruction 可以例外。

---

# 2. 目标目录结构

```text
src/sdfmpneo/
  config.py
  contracts.py
  model.py
  training.py
  topology.py
  reduction.py
  seeds.py
  state.py
  hyperreduction.py

  geometry/
    __init__.py
    coil_parameterization.py
    package_templates.py
    reference_domains.py
    mappings.py
    query_features.py

  networks/
    base.py
    reference.py
    equivariant.py
    mode_embeddings.py

  physics/
    em.py
    em_matrix_free.py
    thermal.py
    flow.py
    materials.py
    coupling.py

  solvers/
    equilibrium.py
    transient.py
    adjoint.py

  certification/
    em_bounds.py
    residual_indicators.py
    goal_oriented.py

  backends/
    base.py
    reference/
      topology.py
      seeds.py
      geometry_moments.py
      green_action.py
      thermal_operator.py
      rans_sst_operator.py
      cubature.py
      certifier.py
      backend.py
    native/
      bindings.py

  diagnostics/
    invariants.py
    timings.py
    rank_report.py
```

---

# 3. 数据契约

## 3.1 GeometryEncoding

必须包含：

```text
coil_tokens        [B,Nc,Dc]
package_tokens     [B,Np,Dp]
global_features    [B,Dg]
slow_features      [B,Ds]
basis_queries      BasisQueryFeatures
```

约束：

- actual current amplitudes 不得进入 `global_features`；
- `slow_features` 只被 thermal/flow heads 使用；
- geometry token 中不硬编码某一个样本 ID。

## 3.2 TopologyOperators

```text
curl_current          [Nj,Naj]
harmonic_current      [Nj,Nhj]
divergence_current    [Ndj,Nj]

curl_velocity         [Nv,Nav]
harmonic_velocity     [Nv,Nhv]
divergence_velocity   [Ndv,Nv]
```

生产路径进入 forward 前必须：

```text
validate_topology(topology)
```

## 3.3 PhysicsSeedBundle

```text
current       [B,Nj,sJ]
thermal       [B,Nt,sT]
velocity      [B,Nv,sV]
log_tke       [B,Nk,sK]
log_omega     [B,Nw,sW]
```

每个 seed rank：

```text
s_p <= first nested rank_p
```

## 3.4 MatrixFreeEMOperators

必须只暴露：

```text
system_action(V)
source_projection(B)
port_feedback(V)
dissipation_action(V)
z_background
```

`system_action` 和 `dissipation_action` 输入只允许：

```text
[B,N,r]
```

不得依赖 full dense `[B,N,N]`。

---

# 4. Milestone A — Foundation Core

## 已有模块

当前 foundation 已包含：

- contracts
- rank schedule
- de Rham validation
- physics seed + neural enrichment
- metric orthogonalization
- rank-collapse guards
- reference coordinate decoder
- dense reference EM Schur
- matrix-free EM reduced solve
- mixed-dimensional mortar helper
- SST positivity helpers
- pseudo-transient Newton
- implicit equilibrium gradients
- BOC
- adaptive-rank model
- CI tests

## 完成标准

CI 必须全绿且满足：

```text
dense EM == matrix-free reference adapter
all hard-invariant tests pass
```

---

# 5. Milestone B — Unified Geometry Layer

## 5.1 Coil parameterization

实现：

```python
class CoilGeometry:
    def centerline(self, quadrature_spec): ...
    def local_frame(self): ...
    def tokens(self): ...
    def source_moments(self, query): ...
```

V1 必须覆盖：

- circular degeneration
- rounded-square / superelliptic family
- turn number
- pitch
- package dimension
- single/dual coil relative pose

不复制 BFZI `_StrictFamily`。

## 5.2 Package templates

至少：

```text
single-package template
dual-package template
```

拓扑变化只允许通过 template 切换，不通过奇异几何映射模拟拓扑改变。

## 5.3 Reference maps

接口：

```python
class GeometryMap:
    def map_points(self, x_hat): ...
    def jacobian(self, x_hat): ...
    def det_jacobian(self, x_hat): ...
    def piola_hdiv(self, v_hat): ...
    def h1_pullback(self, scalar_hat): ...
```

## Tests

- circle / superellipse limit continuity
- pose rigid-transform consistency
- positive Jacobian determinant
- Piola divergence consistency
- topology unchanged inside same template

---

# 6. Milestone C — Reference de Rham Topology

## 6.1 Current space

实现 reference H(div)-compatible current DOF topology。

输出：

```text
C_g, H_J, D_J
```

步骤：

1. construct incidence complex；
2. remove boundary normal-current DOFs；
3. remove gauge nullspace from curl coordinates；
4. compute harmonic complement for multiply connected domain；
5. validate exact-sequence identities。

## 6.2 Velocity space

同样构造：

```text
C_v,g, H_v, D_v
```

solid/package wall no-through-flow DOF 必须硬移除。

## Acceptance

```text
||D C_g|| < tolerance
||D H|| < tolerance
rank([C_g,H]) full
```

---

# 7. Milestone D — Physics Seed Compiler

## 7.1 EM seed

目标不是求 full solution snapshot。

推荐：

```text
structured source moments
operator action probes
rational Krylov / block Krylov compilation
metric compression
```

接口：

```python
compile_em_seed(geometry, topology, em_action, target_rank)
```

不得读取：

```text
FEM current field
BFZI solved field snapshots
Maxwell impedance targets
```

## 7.2 Thermal seed

使用：

```text
generalized diffusion / energy eigenmodes
```

统一 state 维度：

```text
[Tc, Tp, Ts]
```

## 7.3 Flow seed

使用 divergence-free Stokes operator modes。

## 7.4 SST seed

至少包含：

- constant mode
- wall-distance mode
- operator modes

## Acceptance

- no solution snapshot dependency
- seeds metric-independent enough for numerical conditioning
- seed prefix preserved after neural enrichment

---

# 8. Milestone E — GeometryMomentCompiler

这是从 BFZI 一般思想中抽象出来、但必须独立实现的模块。

目标：不先物化大规模 source field，而直接生成 reduced source / feedback moments。

接口建议：

```python
class GeometryMomentCompiler:
    def project_source(self, basis_query, geometry, ports): ...
    def port_feedback(self, field_or_basis_query, geometry): ...
    def background_impedance(self, geometry, materials, omega): ...
```

优化方向：

- exploit turn-family affine structure when available；
- source moments direct-to-reduced accumulation；
- near-source accurate quadrature；
- terminal/lead geometry as ordinary geometry components, not ad-hoc corrections。

## 禁止

- 复制 BFZI source-family code；
- sample-specific branches；
- fitted correction factors。

---

# 9. Milestone F — GreenEnvironmentAction

## 9.1 抽象

实现：

```python
class GreenEnvironmentAction:
    def apply(self, field_block, material_weights): ...
```

推荐 mathematical form：

```text
A_T(V) = V + G[D_gamma2(T,S) P V]
```

geometry Green kernel 与 local material weights 分离。

## 9.2 Reference implementation

可以从零实现一种：

- regular-grid convolution；或
- direct matrix-free Green sum（小规模验证）；

但不得复制 BFZI `FFTGridGreenOperator`。

## 9.3 Native implementation

后续可实现：

- FFT
- FMM
- H2
- DSE-like action

核心接口不变。

## Tests

- linearity
- symmetry/reciprocity consistency
- temperature weight update leaves geometric kernel unchanged
- block action == column-wise action
- no dense `[N,N]` allocation in production path

---

# 10. Milestone G — Full EM Matrix-Free Backend

`assemble_reduced` 内：

```text
B_J
 -> system_action(B_J)
 -> A_r
 -> source_projection(B_J)
 -> S_r
 -> reduced solve
 -> port_feedback(response)
 -> Z_sea
 -> dissipation projection
 -> Q_sea
```

Material update：

```text
T -> sigma(T,S)
T_c -> rho_Cu(T_c)
```

`Z0` 包含：

```text
copper/background resistance
air inductive contribution
```

要求多端口 shared basis。

## Acceptance

- dense reference equivalence on toy problems
- reciprocity defect within tolerance
- non-negative dissipative spectrum
- Joule/port-loss consistency

---

# 11. Milestone H — Unified Mixed-Dimensional Thermal Backend

## State layout

固定：

```text
Theta = [Tc_1D, Tp_3D, Ts_3D]
```

## Operator components

```text
M_T
K_conductor
K_package
K_seawater
C_advection(v)
K_cp mortar
Q_Cu
Q_sea
```

## Mortar

```text
K_cp = E^T W E
```

must be symmetric PSD。

## BOC moments

预留 per-element/operator moment generation：

```python
thermal_operator_moments(basis, probe_states)
```

## Acceptance

- uniform temperature null mode for exchange
- global heat conservation
- positive diffusion dissipation
- correct solid/fluid advection masking

---

# 12. Milestone I — RANS-SST Backend

## State

```text
v = B_v a_v
k_t = k_ref exp(B_k a_k)
omega_t = omega_ref exp(B_w a_w)
```

## Required operators

- transient mass
- viscous diffusion
- convection
- SST production/destruction
- wall-distance terms
- buoyancy if enabled
- thermal turbulent conductivity

## Pressure

生产 reduced model 中不建立 neural pressure state。

若 reference discretization 内部需要 pressure，只能作为内部 constraint variable 使用。

## Motional coupling

提供：

```python
motional_emf_required(Rm, threshold)
```

不要永久删除 `v x B` pathway。

## Acceptance

- divergence-free velocity
- `k_t > 0`, `omega_t > 0`
- finite eddy viscosity
- stable zero/low-flow limit
- physically reasonable high-Re external-flow behavior

---

# 13. Milestone J — Coupled Equilibrium Backend

`assemble_reduced(geometry,bases,z)` 必须：

```text
1 reconstruct only needed T/v/SST values
2 update material laws
3 build EM matrix-free actions
4 solve reduced EM
5 compute Q_Cu,Q_sea
6 assemble thermal residual
7 assemble RANS-SST residual
8 concatenate F_MP(z)
```

不得：

- freeze EM outside Newton loop；
- only update EM after thermal convergence；
- sequentially call independent surrogate outputs as a weak coupling approximation。

## Acceptance

- coupled Jacobian finite
- pseudo-transient Newton converges on reference cases
- one-way decoupled limit reproduces expected special case

---

# 14. Milestone K — Basis-Operator Cubature

已有 core builder 后，backend 需要生成 operator moments。

流程：

```text
physics basis
+ operator moments
-> positive NNLS cubature
-> solve set
```

必须单独生成：

```text
certifier set
```

且：

```text
solve_set ∩ certifier_set = empty
```

## Acceptance

- non-negative weights
- target operator moments reconstructed within tolerance
- reduced dissipative operators remain PSD within tolerance

---

# 15. Milestone L — Training Loop

## Parameter sampler

采用 Sobol/on-the-fly sampling。

参数包括：

```text
geometry
pose
frequency
salinity/conductivity
ambient temperature
inflow velocity
operating load/current invariants
```

## Rank randomization

每个 step 随机选择 active nested level。

## Step

```text
sample xi
build detached basis
solve equilibrium
rebuild differentiable basis
compute independent/Riesz residual
implicit adjoint gradient
optimizer.step
```

## Hard rule

若 reduced equilibrium 不收敛：

```text
do not perform optimizer step on invalid equilibrium
```

记录 failure region，供 parameter-domain refinement 使用。

---

# 16. Milestone M — Adaptive Parameter Sampling

起始：Sobol covering。

后续维护 parameter cells：

```text
cell residual statistics
cell fallback frequency
cell active-rank statistics
```

高 residual / high fallback cells 自动细分或提高采样概率。

不使用高保真 solution error 作为 refinement criterion。

---

# 17. Milestone N — Certifier

## EM

实现：

- independent residual action
- dissipative norm indicator
- optional port-QoI bound

## Thermal/Flow

实现：

- independent residual indicator
- reduced adjoint / dual-weighted QoI indicator

QoIs 至少：

```text
T_max
Z11/Z12 or selected impedance entries
```

## Acceptance

validation 阶段统计：

```text
actual error <= indicator ?
coverage ratio
indicator sharpness
```

但训练不使用 actual high-fidelity error。

---

# 18. Milestone O — Lazy Basis Runtime

将当前 full basis materialization 逐步替换为：

```text
query basis at:
  EM operator points
  BOC elements
  certifier elements
  QoI probes
```

API 建议：

```python
basis_generator.query(head, coordinates, mode_range, context)
```

full field 只在：

```python
reconstruct_fields(..., requested=True)
```

时生成。

## Acceptance

在线 QoI-only 推理期间不得创建 full `[N,r]` basis tensor。

---

# 19. Milestone P — Native Backend

高性能阶段只替换 backend，不改 model/training API。

候选：

```text
C++17/OpenMP geometry moments
FFT/FMM/H2 Green actions
CUDA basis querying
CUDA/CPU BOC assembly
```

必须保留 Python reference backend 作为数值 oracle。

---

# 20. Validation Plan

## V1 — Unit / mathematical

CI invariants。

## V2 — Reference deterministic backend

小规模 full-order reference comparison。

## V3 — BFZI independent reference

只比较最终 QoIs / runtime / physical trends，不复制实现。

## V4 — Maxwell / COMSOL / CFD

完全独立 case set。

## V5 — Experiment

阻抗、损耗、温度等可测 QoIs。

---

# 21. Runtime instrumentation

每次正式 benchmark 必须输出：

```text
T_geometry
T_encoder
T_basis_query
T_seed
T_em_projection
T_em_reduced_solve
T_thermal_flow_assembly
T_newton
T_certifier
T_field_reconstruction
T_total
active_rank
newton_iterations
operator_action_count
basis_query_count
fallback
```

这样论文能解释 speedup 来源，而不是只给总时间。

---

# 22. Failure handling

必须明确区分：

## recoverable

- low-rank certificate failure -> increase rank
- Newton remote-state difficulty -> pseudo-time reduction / line search

## fallback

- max rank certificate failure
- topology/map invalid
- physical material law outside supported domain

## hard error

- `D C != 0`
- seed/generator rank deficiency
- negative Jacobian determinant
- illegal NaN/Inf
- hidden_dim < maximum rank

不得 silent clipping 这些结构错误。

---

# 23. Definition of Done for first publishable prototype

只有同时满足以下条件，才能称为第一版完整 SDF-MPNEO：

- [ ] single + dual topology templates available
- [ ] unified parameterized coil geometry available
- [ ] solution-free physics seeds implemented
- [ ] de Rham current and velocity spaces implemented
- [ ] matrix-free EM backend implemented
- [ ] mixed-dimensional thermal backend implemented
- [ ] RANS-SST backend implemented
- [ ] strong coupled Newton equilibrium implemented
- [ ] implicit solution-data-free training runs end-to-end
- [ ] BOC used online
- [ ] independent certifier used online
- [ ] adaptive rank works
- [ ] full-field reconstruction optional
- [ ] no solution labels enter training
- [ ] BFZI/Maxwell/CFD validation remains external
- [ ] runtime breakdown recorded
- [ ] all CI physics invariants pass

在此之前，不宣称“完整模型已实现”。
