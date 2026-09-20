# SDF-MPNEO — geometry→tensor + geometry-aware thermal ROM

当前生产实现只保留一条正式主链：

\[
\boxed{
 g
 \rightarrow
 \begin{cases}
 \Phi(g),\ M_r(g),\ K_r(g) & \text{deterministic thermal geometry map}\\
 Z_{\rm field}(g),\ D_{\rm vol}(g),\ H_j(g) & \text{neural EM tensor surrogate}
 \end{cases}
 \rightarrow
 \text{explicit current/circuit physics}
 \rightarrow
 \text{true reduced thermal ODE}
 \rightarrow T(t)
}
\]

神经网络只学习静态 `geometry -> EM tensor` 映射。thermal basis、thermal mass/stiffness、电流幅值/相位、线圈温度电阻、Joule 二次型、电路方程、热 ODE 和稳态方程都保留为显式物理。

在线推理不运行 Maxwell、neural Maxwell solver、Krylov/FGMRES 或 full-field correction。Maxwell 只用于离线 truth 与训练前/训练后的物理验证。

## 1. 运行

安装：

```bash
python -m pip install -e '.[gui,neural,dev]'
```

训练：

```bash
python run.py --mode train
```

纯控制台训练：

```bash
python run.py --mode train --headless
```

推理：

```bash
python run.py --mode predict
```

用户配置集中在 `run.py` 顶部。不需要单独运行 Gate；preflight、Physics Gate 和 final Go/No-Go 都属于同一个 `--mode train` 流程。

## 2. 训练执行顺序

生产训练严格按物理依赖执行：

1. 构建开放边界固定背景空间。
2. 在独立几何上执行 **pre-basis spatial truth preflight**：source/terminal、开放域、full-wave↔MQS、local self reference、corrected global mesh convergence。
3. 只有 preflight 通过后，才构建 geometry-aware canonical thermal library。
4. 对 thermal library 做 held-out resolvent 与 full-vs-ROM trajectory/steady audit。
5. 冻结 \(g\mapsto\Phi(g)\) 后，逐 geometry 生成 corrected `Z_field / D_vol / H_j` tensor truth。
6. 执行 post-basis Physics Gate：corrected Joule identities、local-defect Joule identities、Loewner、mesh/thermal/transport 等。
7. 只用 neural training split 构建 POD/normalization 并训练 MLP。
8. 训练完成后重新采样 **completely-held-out** 几何，执行 final Go/No-Go。
9. final audit 全部通过后才保存 `model.geometry_thermal.npz`。

任何硬 Gate 失败都会 fail closed；不会用 neural loss、physical projection 或后续优化掩盖底层物理问题。

## 3. 离线 Maxwell truth

固定 geometry 后：

\[
A_{\rm em}(g)X(g)=B(g),
\qquad B=-i\omega S,
\qquad Z_{\rm field}=-S^TX.
\]

采用 `e^{+i\omega t}`、峰值复相量约定。

### 3.1 Finite-cross-section stranded source

生产 source 使用：

```text
stranded_rectangular_cross_section_gauss3
```

每个 centerline segment 在真实 `conductor_width × conductor_thickness` 截面上做 3×3 Gauss 分布。centerline 采用固定 **1 mm physical sampling**，与 EM mesh 无关，因此 12 mm→9 mm mesh refinement 比较的是同一个物理 source、同一 wire length 和同一 terminal endpoint。

当前端口语义：

```text
impressed_port_path_with_endpoint_charge_balance
```

preflight 会检查 source/heat 守恒、两端分离、deposited path integral、material-fraction closure，以及线圈铜损没有与独立 `R_wire(T)` 重复计入。

### 3.2 Silver–Müller open boundary

离线 Maxwell 使用匹配海水介质的一阶开放阻抗边界：

\[
i\omega Y M_{\partial\Omega},
\qquad
Y=\sqrt{\frac{\epsilon-i\sigma/\omega}{\mu}},
\qquad \operatorname{Re}Y\ge0.
\]

体耗散与独立 outward power：

\[
D_{\rm vol}=X^HH_\sigma X,
\qquad
D_{\rm out}^{\rm phys}
=X^H\left(\operatorname{Re}Y\,M_{\partial\Omega}\right)X.
\]

并直接检查：

\[
\operatorname{Herm}(Z_{\rm field})
\approx D_{\rm vol}+D_{\rm out}^{\rm phys}.
\]

对有耗海水，人工边界外移会把一部分原本穿边界的功率重新归入新增海水体耗散，因此 **raw `D_out` 不要求跨人工边界保持数值不变**。域扩展 Gate 硬检查端口响应、体耗散、mutual Z，以及 `ΔD_out` 相对总端口耗散的 significance。

大尺寸 global truth 默认使用 compatible gradient + transverse ILU，并始终按原始物理矩阵的 true residual 做 `1e-9` 认证。少数合法 geometry 若在约 95k-edge production global grid 上无法由 iterative path 认证，且 `n_edges <= 100000`，允许一次共享 sparse-LU correctness fallback；118k/254k 的 local validation 系统明确在该上限之外，仍必须走 iterative/two-level 路径。fallback 不改变方程、source、boundary 或 Gate，只改变求解 backend。

## 4. Global coarse + canonical local self defect

实际 preflight 已确认：默认全局 12 mm 网格对 mutual/far-field 可以收敛，但毫米级 conductor 周围的 self response 无法直接解析。因此 production truth 使用结构化多尺度分裂：

\[
\boxed{
Z_{pp}^{\rm corr}
=Z_{pp}^{\rm global}
+\left(Z_{pp}^{\rm local,fine}-Z_{pp}^{\rm local,coarse}\right)
}
\]

同样的 per-port diagonal defect 一致应用于：

\[
D_{{\rm vol},pp},\qquad D_{{\rm out},pp},\qquad H_{j,pp}.
\]

mutual/off-diagonal 项始终来自完整 global Maxwell solve，不由 local correction 修改。

local problem 在当前 coil/package 的 canonical rigid frame 中求解；translation/rotation 被移除，真实 coil shape、尺寸、package、材料和 finite-cross-section source 保留。默认：

```python
"self_correction": {
    "enabled": True,
    "samples": 1,
    "fine_step": 0.003,
    "validation_fine_step": 0.00225,
    "core_padding": 0.006,
    "boundary_padding": 0.04,
    "growth": 1.5,
    "max_step": 0.02,
    "relative_tolerance": 1e-1,
    "joule_identity_tolerance": 1e-10,
}
```

### 4.1 三层空间验收

空间 truth 不是靠“有 correction 就自动通过”，而是分三层：

1. **raw global diagnostic**：报告 global coarse/refined 的 self-Z、Rself、Xself、self-D、mutual-Z 和 source-path invariance；raw unresolved self 允许作为诊断存在。
2. **local reference Gate**：在 held-out geometry/port 上直接比较 local `3 mm -> 2.25 mm`，检查 self Z、R/X、D_vol 和 outward-partition significance；默认必须 ≤10%。
3. **corrected global mesh Gate**：对 12 mm→9 mm global refinement 分别施加同一个经过认证的 local defect，再比较 corrected `Z/D/P_vol/D_out/mutual Z`。只有这一层也通过，production EM truth 才 certified。

source path invariance 是独立硬条件，默认要求相对差异 ≤`1e-10`；不能把“物理 source 变了”伪装成 mesh error。

### 4.2 Local defect 的 Joule/Poynting 证书

每个 local coarse/fine solve 都独立检查：

\[
D_{\rm vol,self}=2\sum_{\rm cells}q_{\rm cell},
\]

有 thermal basis 时还检查：

\[
H_{j,self}=2\,\phi_j^T q_{\rm cell}.
\]

local total/modal Joule identity 默认要求到 `1e-10`，local 与 corrected global Poynting balance 也进入 hard Gate。这样不能只修 `Z_self` 而破坏 `D/H` 的热功率闭合。

## 5. Pre-basis spatial truth preflight

在 thermal basis 构造之前自动执行：

- finite-cross-section source 与 terminal path conservation；
- material-fraction closure；
- wire/internal loss 与 seawater volume loss 分区；
- expanded-domain convergence；
- full-wave ↔ E-form MQS comparison；
- local self reference convergence；
- corrected EM mesh refinement convergence。

默认：

```python
"open_boundary_check": {
    "samples": 1,
    "padding": 0.12,
    "relative_tolerance": 5e-2,
},
"formulation_check": {
    "samples": 1,
    "relative_tolerance": 2e-2,
},
"mesh_check": {
    "samples": 1,
    "refinement_factor": 0.75,
    "relative_tolerance": 1e-1,
    "source_path_relative_tolerance": 1e-10,
},
```

正式物理域已经是 ±0.27 m，expanded reference 到 ±0.39 m。默认只做 1 个昂贵 open-domain certification sample，5% 阈值不变；这样避免每次 physical cache 重建重复做三组 8万/16万 DOF 级复数稀疏 LU。

\[
P_{\rm vol}(c)=\frac12c^HD_{\rm vol}c.
\]

preflight 会同时保留 raw 与 corrected diagnostics，便于区分 global mutual error、unresolved raw self、local-reference nonconvergence、local Joule identity failure 和 corrected-production nonconvergence。

## 6. Geometry-aware deterministic thermal ROM

生产 basis：

\[
\boxed{
\Phi(g)=
[\Phi_{\rm bg},\ \mathcal T_{\rm tx}(g)\Psi_{\rm tx},\ \mathcal T_{\rm rx}(g)\Psi_{\rm rx}]
}
\]

v14 修正了此前 geometry generalization 的根本结构问题：**被 rigid transport 的必须是局部 thermal state，而不是“局部 source 产生的整幅 full-domain solution”**。局部热源的解同样带有延伸到固定背景和人工边界的 diffusion tail；如果把整幅解随 coil 搬动，就会把本应固定的全局尾巴一起移动，之后只能靠 fixed-background modes 在训练几何上逐点抵消，形成明显的 train memorization。

现在每个真实 full-order forced thermal state \(u(g)\) 先使用与 moving supports 一致的平滑 partition-of-unity 做**状态分解**：

\[
u(g)
=
u_{\rm tx}(g)
+
u_{\rm rx}(g)
+
u_{\rm far}(g).
\]

其中 TX/RX state pieces 才允许通过 deterministic rigid-pose transport 进入 moving local atlas；far remainder 永远属于 fixed background。分解逐 cell 精确和回原始 full state。为了保持 energy greedy 的方程语义，对每个 state piece 定义 \(b_{\rm piece}=A(g)u_{\rm piece}\)，因此各 piece 仍是同一 SPD thermal resolvent 下的严格解，并且由线性性保持完整 equation/state decomposition。uniform initial condition 是全局量，始终全部留在 fixed background。

local state pieces 逆 rigid transport 到共同 reference pose 后，按相同 shift 使用 reference 'K_ref + s M_ref' 做 resolvent-energy greedy；不需要第二遍 Maxwell truth，也不再需要 canonical full-tail thermal solves。尺寸、turns、pitch、conductor/package 变化作为不同 localized state directions 留在同一 fixed-rank moving span 内。fixed background 不运输；不使用 neural basis、dynamic POD 或 Grassmann interpolation。

每个 geometry 仍从真实 full thermal operators 投影：

\[
M_r(g)=\Phi(g)^TM_T(g)\Phi(g),
\qquad
K_r(g)=\Phi(g)^TK_T(g)\Phi(g).
\]

'M_r/K_r' 不由网络预测。每个 production thermal context 检查 transported basis rank/support、conditioning、\(M_r\succ0\)、\(K_r\succ0\) 和 reduced operator condition number。

## 7. Thermal rank、resolvent 与 trajectory Gate

full truth anchors 仍由

\[
(K+sM)u=b
\]

自动构建，并包含 's=0' steady anchor。昂贵 geometry truth 的选点使用 v13 已引入的 hybrid coverage：默认 16 个 training geometries，其中 8 个来自完整 'encode_geometry' maximin，覆盖 absolute pose / fixed-background / boundary effects；另外 8 个来自 'intrinsic_relative_pose_v1' maximin，覆盖 shape / turns / outer size / pitch / conductor/package dimensions 与端口 relative pose。两组若重复，则在同一个 128-candidate pool 中按两个归一化空间的联合未覆盖距离补足。held-out validation、tensor dataset 和 final audit 使用独立固定 RNG 流，validation 从不参与 basis enrichment。

默认：

~~~python
"basis_samples": 16,
"basis_design_pool_multiplier": 8,
"thermal_basis_design": "hybrid_full_intrinsic_union_v1",
"thermal_basis_energy_tolerance": 5e-2,
"thermal_component_target_multiplier": 2.0,
"thermal_time_scales": [0.1, 1.0, 10.0]
~~~

background / TX-local / RX-local 只是完整 ROM 的初始化分块，默认 component target 仍为 '2 × 5% = 10%'。component greedy 后继续检查所有 training geometry 上 transported raw generator 的真实 weighted basis condition；'thermal_basis_conditioning_limit=1e10' 指 basis/generator condition，不是平方后的 Gram condition。低/中 condition 使用 Gram spectrum 的平方根快速估计，高 condition 自动切换到 weighted generator thin-QR + 小型 SVD certificate。5% Gate 与 '1e10' conditioning Gate 都没有放宽。

v14 同时修正第二个结构问题：**full-library residual 不再全部写入 fixed background**。每一轮会同时检查所有未达到 5% 的 training geometries，把各自最坏 full-ROM state error 用同一 TX/RX/far partition 拆开：

\[
e(g)
=
e_{\rm tx}(g)
+
e_{\rm rx}(g)
+
e_{\rm far}(g).
\]

'e_tx/e_rx' 逆 transport 回 reference local atlas 后分别 enrich moving TX/RX blocks；只有 'e_far' 可以 enrich fixed background。每一 sweep 批量处理所有 under-resolved training geometries，然后重新计算完整 full-library energy Gate，并重新检查所有 training geometry 的 raw-basis conditioning。若 conditioning 超过原 '1e10' Gate，整轮原子回滚并 fail closed。日志会直接报告：

~~~text
thermal_basis_stage = residual-driven-enrichment
thermal_basis_residual_sweeps
thermal_basis_residual_background_additions
thermal_basis_residual_local_additions
~~~

这避免了旧实现中“training sample 越多，fixed background residual rank 越线性增长”的世界坐标记忆行为，也把此前一条 residual 加一个 fixed mode 的串行 enrichment 改成多 geometry 批量 sweep。

昂贵的 thermal-anchor Maxwell port fields 继续使用独立 persistent cache：每个 geometry 的 \(X(g)\) 只有通过当前 'A(g)X=B(g)' 的 '1e-9' true-residual certificate 才写盘，命中时也会重新对当前 'A/B' 认证。thermal atlas/schema 改动不会清除此 EM-only cache；因此 v14 仍可复用此前已经算过的 16-point hybrid truth Maxwell fields。

held-out geometry 仍按“生成一个 truth anchor set → 立即做 resolvent energy Gate”的顺序流式检查；任意 geometry 超过 5% 就 fail closed，并跳过已经没有意义的昂贵 trajectory audit。只有全部 held-out resolvent Gate 通过才继续：

~~~python
"thermal_trajectory_times": [0.1, 1.0, 10.0, 100.0]
~~~

trajectory audit 覆盖 thermal-mass field error、'Tmin/Tmax'、wire-average temperature、uniform initial-condition evolution、forced steady field 和 steady reduced coordinate。100 s、1000 s 等长时间仍由同一个 reduced ODE 连续推进，不靠专用长时 basis。

## 8. Geometry-dependent Joule tensors

\[
[W_j(g)]_{mn}=\int_\Omega\sigma\,\phi_j(g)N_m\cdot N_n\,dx,
\qquad
H_j(g)=X(g)^HW_j(g)X(g).
\]

任意峰值复电流：

\[
q_{{\rm vol},j}(g,c)=\frac12\operatorname{Re}(c^HH_j(g)c).
\]

逐 geometry 计算 conductivity-loss support 上的 Loewner bounds：

\[
\phi_j^{\min}(g)D_{\rm vol}(g)
\preceq H_j(g)
\preceq\phi_j^{\max}(g)D_{\rm vol}(g).
\]

corrected truth 会在完整 Hermitian current span 上重新检查 total/modal Joule contractions，而不是假定 local defect 自动正确。

## 9. 神经网络只学 geometry→tensor POD coefficients

网络目标：

\[
g\mapsto\{Z_{\rm field}(g),D_{\rm vol}(g),H_1(g),\ldots,H_r(g)\}.
\]

输入不含时间、电流、热状态或 Maxwell residual。POD 与 normalization 只使用 neural training split。物理解码强制 reciprocal/passive `Z/D`、Hermitian `H_j` 和 geometry-dependent modal Loewner bounds。

## 10. Post-basis Physics Gate

训练 MLP 前继续检查：

- Maxwell algebraic residual / reciprocity；
- corrected `D_vol / D_out` passivity 与 Poynting balance；
- corrected global Joule total/modal identities；
- **local self coarse/fine Joule total/modal identities**；
- modal Loewner bounds；
- corrected full mesh audit：`Z/D/P_vol/D_out/H/Tmax/wire/a_*`；
- small geometry perturbation 下 source/material/\(\Phi\)/`Z/D/H` continuity；
- geometry-aware `M_r/K_r` SPD/conditioning；
- held-out thermal resolvent + trajectory/steady。

## 11. Completely-held-out final Go/No-Go

MLP 训练结束后，用独立 seed 重新采样 final geometries。它们不参与 thermal rank、tensor dataset、POD、validation、early stopping 或 optimizer。

final audit 同时覆盖：

1. full thermal truth vs geometry-aware ROM；
2. corrected truth tensors vs neural tensors，以及复电流方向的 `Zc/P/q` contractions；
3. truth-tensor ROM vs surrogate-tensor ROM 的 current-controlled 和 circuit-controlled trajectories/steady；
4. production `etd2_adaptive` vs 高精度 BDF reduced-ODE reference。

只有 final audit 通过才原子保存新 artifact。报告标记：

```text
frozen_held_out_numerical_validation
```

它是冻结 held-out 集上的数值验证，不宣称连续参数域的严格数学 certificate。

## 12. 在线阶段

给定静态 geometry：

1. 检查 production-domain 与几何合法性；
2. 确定性生成 \(\Phi(g),M_r(g),K_r(g)\)；
3. MLP 一次 forward 得到 `Z_field / D_vol / H_j`；
4. 用当前 geometry modal bounds 做 hard physical decode；
5. current-controlled 直接使用 prescribed complex current，或由显式 circuit system 求 current；
6. 显式更新 `R_wire(T)` 与 wire heat；
7. 推进 reduced thermal ODE 或求 stable steady state。

local Maxwell correction **不在线执行**。它只属于离线 truth/certification；网络已经学习 corrected tensors。

## 13. 缓存与 artifact

```text
results/uwpt/unified.cache.json
results/uwpt/unified.geometry_thermal.npz
results/uwpt/unified.tensor_dataset.npz
results/uwpt/model.tensor_training.pt
results/uwpt/model.geometry_thermal.npz
```

当前物理：

```text
CACHE_FORMAT = 18
```

cache 只有在以下条件全部成立时才可复用：physical signature 相同、preflight 已 certified、local self reference 已 converged、self-correction model 与当前 production model 一致。修改 `self_correction` 的 mesh/padding/tolerance 会使物理 cache 自动失效。

当前 unified model artifact：

```text
FORMAT_VERSION = 13
```

`FORMAT_VERSION` 表示二进制 schema；local self correction 属于 release-physics semantics，因此由正式 prediction release gate 检查。`python run.py --mode predict` 要求：

- pre-basis truth preflight certified；
- local self correction convergence certified；
- post-basis Physics Gate certified；
- `self_correction_model = canonical_local_fine_minus_coarse_self_defect_v1`；
- completely-held-out final audit certified；
- production-integrator audit passed；
- `certificate_level = frozen_held_out_numerical_validation`。

缺任一项都会 fail closed。低层 `UnifiedNeuralElectroThermalModel.load()` 只负责库级 schema/round-trip，不代替 production release gate。

## 14. 关键测试

```bash
python -m pytest -q \
  tests/test_unified_geometry_physics.py \
  tests/test_unified_open_boundary.py \
  tests/test_unified_source_mesh_invariance.py \
  tests/test_unified_self_correction.py \
  tests/test_unified_self_correction_audit.py \
  tests/test_unified_truth_preflight.py \
  tests/test_unified_physics_gate.py \
  tests/test_unified_thermal.py \
  tests/test_unified_thermal_trajectory_gate.py \
  tests/test_unified_tensor_surrogate.py \
  tests/test_unified_final_audit.py \
  tests/test_unified_release_artifact.py \
  tests/test_unified_end_to_end.py \
  tests/test_run_neural_user_defaults.py \
  tests/test_package_metadata.py
```

生产训练与测试不依赖 GitHub Actions。
