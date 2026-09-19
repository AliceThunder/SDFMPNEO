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

首版只使用确定性的 rigid translation/rotation transport 和连续插值；不使用 neural basis、dynamic POD 或 Grassmann interpolation。

canonical local block 包含 wire heat，以及对应端口 unit-port self-volume Joule source 的**pose-following 近/中场部分**。pure-port volume source 用随 coil 尺寸缩放的平滑局部窗严格拆成 \(q_{\rm moving}+q_{\rm far}=q\)：窗口覆盖 package 以及数个 coil radius 的 conductive-seawater Joule lobe，只运输 \(q_{\rm moving}\) 的 resolvent；真正靠近固定人工边界的 \(q_{\rm far}\) 与 uniform initial response 留在 fixed background block。对 \(e_i+e_j\) / \(e_i+i e_j\) 产生的组合热源，background 不再直接保存整份正热源，而是先减去两个 diagonal self source，只保留真正的 signed Hermitian cross component，避免 self hotspot 在 local/background 两边重复出现。这个分解不删除或重标定任何 Joule heat；held-out Gate 仍然对原始完整物理 current directions 做验证。

每个 geometry 从真实 full thermal operators 投影：

\[
M_r(g)=\Phi(g)^TM_T(g)\Phi(g),
\qquad
K_r(g)=\Phi(g)^TK_T(g)\Phi(g).
\]

`M_r/K_r` 不由网络预测。每个 production thermal context 检查 transported basis rank/support、conditioning、\(M_r\succ0\)、\(K_r\succ0\) 和 reduced operator condition number。

## 7. Thermal rank、resolvent 与 trajectory Gate

canonical rank 用：

\[
(K+sM)u=b
\]

的 resolvent anchors 自动构建。thermal basis 的昂贵 geometry truth 不再直接取少量 iid 随机点。默认先生成廉价候选池，在归一化 `encode_geometry` 空间中用 farthest-point/maximin 选择 12 个覆盖点；held-out validation 仍然由独立随机样本组成，不参与 basis enrichment。

默认：

```python
"basis_samples": 12,
"basis_design_pool_multiplier": 16,
"thermal_time_scales": [0.1, 1.0, 10.0]
```

并始终包含 `s=0` steady anchor。同一 geometry/shift 的多个 thermal RHS 共用一次 full factorization 和一次 reduced solve；background residual enrichment 会缓存各训练 geometry 的 transported-local span，只增量加入新的 fixed-background direction，不会每升一阶 rank 都重新运输并正交化整套 local basis。

held-out geometry 按“生成一个 truth anchor set → 立刻做一个 resolvent energy Gate”的顺序流式检查；任意一个 geometry 超过目标误差就立即 fail closed，不再生成剩余 held-out Maxwell truth，并跳过更昂贵且已不可能改变结论的 full-vs-ROM trajectory audit。只有全部 held-out resolvent Gate 通过时才继续：

```python
"thermal_trajectory_times": [0.1, 1.0, 10.0, 100.0]
```

包括 thermal-mass field error、`Tmin/Tmax`、wire-average temperature、uniform initial-condition evolution、forced steady field 和 steady reduced coordinate。

100 s、1000 s 等长时间不要求相同时间尺度的专用 basis，而是由同一个 reduced ODE 连续推进。

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
