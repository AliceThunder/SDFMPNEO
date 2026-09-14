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

用户配置集中在 `run.py` 顶部。不需要单独运行 Gate；所有 preflight、Physics Gate 和 final Go/No-Go 都属于同一个 `--mode train` 流程。

## 2. 训练执行顺序

生产训练严格按物理依赖执行：

1. 构建开放边界固定背景空间。
2. 在独立几何上执行 **pre-basis spatial truth preflight**。
3. 只有 preflight 通过后，才构建 geometry-aware canonical thermal library。
4. 对 thermal library 做 held-out resolvent 与 full-vs-ROM trajectory/steady audit。
5. 冻结 \(g\mapsto\Phi(g)\) 后，逐 geometry 生成最终 `Z_field / D_vol / H_j` tensor truth。
6. 执行 post-basis Physics Gate：Joule identities、Loewner、thermal mesh/transport 等。
7. 只用 neural training split 构建 POD/normalization 并训练 MLP。
8. 训练完成后重新采样 **completely-held-out** 几何，执行 final Go/No-Go。
9. final audit 全部通过后才保存 `model.geometry_thermal.npz`。

任何硬 Gate 失败都会 fail closed；不会用 neural loss、physical projection 或后续优化掩盖底层物理问题。

## 3. 离线 Maxwell truth

固定 geometry 后：

\[
A_{\rm em}(g)X(g)=B(g),
\qquad
B=-i\omega S.
\]

采用 `e^{+i\omega t}`、峰值复相量约定。端口场阻抗使用 negative source reaction：

\[
Z_{\rm field}=-S^TX.
\]

### 3.1 Finite-cross-section stranded source

生产 source 不再使用“网格尺寸充当隐式 wire radius”的零半径 line source。`OpenBoundaryBackground` 使用

```text
stranded_rectangular_cross_section_gauss3
```

即每个 centerline segment 在真实 `conductor_width × conductor_thickness` 截面上做 3×3 Gauss 分布。截面 quadrature 总权重为 1，因此 refinement 改变的是同一个物理 source 的解析度，而不是 ampere-turns。

当前端口语义为：

```text
impressed_port_path_with_endpoint_charge_balance
```

每个端口会数值检查：

- conductor width/thickness 为真实正值；
- heat/source 权重守恒；
- open path 两端确实分离；
- deposited edge source 的定向积分重现真实 centerline endpoint displacement；
- terminal-path relative error 默认要求不高于 `1e-12`。

铜的 EM conductivity 在 `em=True` 时被排除，wire/internal ohmic loss 由独立 `R_wire(T)` 处理。preflight 会独立重组非线圈材料的 \(\sigma_{\rm em}\) 来验证没有重复计入铜损。

### 3.2 开放边界与独立 Poynting 功率

离线 Maxwell 使用匹配海水介质的一阶 Silver–Müller / Sommerfeld 开放阻抗边界：

\[
i\omega Y M_{\partial\Omega},
\qquad
Y=\sqrt{\frac{\epsilon-i\sigma/\omega}{\mu}},
\qquad \operatorname{Re}Y\ge0.
\]

volume loss：

\[
D_{\rm vol}=X^HH_\sigma X.
\]

开放边界独立给出：

\[
D_{\rm out}^{\rm phys}
=X^H\left(\operatorname{Re}Y\,M_{\partial\Omega}\right)X.
\]

truth 直接检查：

\[
\operatorname{Herm}(Z_{\rm field})
\approx D_{\rm vol}+D_{\rm out}^{\rm phys}.
\]

禁止用 `Herm(Z)-D_vol` 反定义 outward power 后再声称 Poynting balance 已验证。

## 4. Pre-basis spatial truth preflight

在 thermal basis 构造之前自动执行：

- finite-cross-section source 与 terminal path conservation；
- material-fraction closure；
- wire/internal loss 与 seawater volume loss 分区；
- expanded-domain convergence；
- full-wave ↔ E-form MQS comparison；
- EM mesh refinement convergence。

默认设置：

```python
"open_boundary_check": {
    "samples": 3,
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
},
```

expanded-domain Gate 比较 `Z_field / D_vol / D_out_phys / mutual Z`。mesh Gate 除了这些矩阵量，还在完整 Hermitian current-space basis 上直接比较

\[
P_{\rm vol}(c)=\frac12 c^H D_{\rm vol}c,
\]

并报告 `relative_p_vol_error`；post-basis mesh audit 还会继续比较 `H_j`、steady `Tmax`、wire temperature 和 projected steady coordinate。

## 5. Geometry-aware deterministic thermal ROM

生产 basis：

\[
\boxed{
\Phi(g)=
[\Phi_{\rm bg},\ \mathcal T_{\rm tx}(g)\Psi_{\rm tx},\ \mathcal T_{\rm rx}(g)\Psi_{\rm rx}]
}
\]

首版只使用确定性的 rigid translation/rotation transport 和连续插值；不使用 neural basis、dynamic POD 或 Grassmann interpolation。

每个 geometry 都从真实 full thermal operators 投影：

\[
M_r(g)=\Phi(g)^TM_T(g)\Phi(g),
\qquad
K_r(g)=\Phi(g)^TK_T(g)\Phi(g).
\]

`M_r/K_r` 不由网络预测。每个 production thermal context 都直接检查：

- transported basis support/full rank；
- basis conditioning；
- \(M_r(g)\succ0\)；
- \(K_r(g)\succ0\)；
- reduced operator condition number 未超过配置上限。

## 6. Thermal rank、resolvent 与 trajectory Gate

canonical rank 用 resolvent anchor 自动构建：

\[
(K+sM)u=b.
\]

默认 resolvent time scales：

```python
"thermal_time_scales": [0.1, 1.0, 10.0]
```

并始终包含 `s=0` steady anchor。

resolvent error 使用真实 Galerkin energy error：

\[
\frac{\|u-u_r\|_{K+sM}}{\|u\|_{K+sM}}.
\]

resolvent 不是 trajectory certificate。held-out geometry 还会直接比较 full thermal 与 geometry-aware ROM，默认：

```python
"thermal_trajectory_times": [0.1, 1.0, 10.0, 100.0]
```

比较内容包括：

- field thermal-mass relative error；
- `Tmin/Tmax`；
- wire-average temperature；
- uniform initial-condition homogeneous evolution；
- forced steady field；
- steady reduced coordinate \(a_*\)。

full/reduced 线性审计使用 matrix exponential action，避免把普通 time-step tolerance 混入 ROM error。

## 7. Geometry-dependent Joule tensors

当前 geometry 的 thermal mode \(\phi_j(g)\) 对应：

\[
[W_j(g)]_{mn}
=\int_\Omega \sigma\,\phi_j(g)N_m\cdot N_n\,dx,
\qquad
H_j(g)=X(g)^HW_j(g)X(g).
\]

任意峰值复端口电流：

\[
q_{{\rm vol},j}(g,c)
=\frac12\operatorname{Re}(c^HH_j(g)c).
\]

逐 geometry 计算 conductivity-loss support 上的 bounds：

\[
\phi_j^{\min}(g)D_{\rm vol}(g)
\preceq H_j(g)
\preceq\phi_j^{\max}(g)D_{\rm vol}(g).
\]

truth 生成还会显式审计 cell Joule total power、`D_vol` contraction 与每个 `H_j` modal contraction 的一致性。

## 8. 神经网络只学 geometry→tensor POD coefficients

目标：

\[
g\mapsto\{Z_{\rm field}(g),D_{\rm vol}(g),H_1(g),\ldots,H_r(g)\}.
\]

输入不含时间、电流、热状态或 Maxwell residual。POD 与 normalization 只使用 neural training split。

物理解码强制：

- `Z_field` complex symmetric；
- `D_vol` Hermitian PSD；
- `Herm(Z_field)-D_vol` PSD；
- `H_j` Hermitian；
- geometry-dependent modal Loewner bounds。

`Z/D` 与 `H` projection correction 单独报告；final audit 对 projection correction 也设 Go/No-Go 上限。

## 9. Post-basis Physics Gate

thermal library 和 tensor truth 完成后，训练 MLP 前继续检查：

- Maxwell algebraic residual；
- raw reaction reciprocity；
- `D_vol / D_out_phys` passivity；
- independent Poynting matrix balance；
- Joule total/modal identities；
- modal Loewner bounds；
- full mesh audit：`Z/D/P_vol/D_out/H/Tmax/wire/a_*`；
- small geometry perturbation 下 source/material/\(\Phi\)/`Z/D/H` continuity；
- geometry-aware `M_r/K_r` SPD/conditioning；
- held-out resolvent + full-vs-ROM trajectory/steady。

small perturbation 默认：

```python
"geometry_continuity_check": {
    "samples": 1,
    "translation_step": 1e-4,
    "angle_step": 1e-3,
    "relative_change_limit": 2e-1,
}
```

## 10. Completely-held-out final Go/No-Go

MLP 训练完成后，系统使用单独 seed 重新采样 final audit geometries。这些 geometry 不参与：

- thermal canonical rank/enrichment；
- transport 参数选择；
- tensor dataset；
- POD/normalization；
- validation/early stopping；
- optimizer update。

只有 final audit 通过才会保存模型 artifact。

默认 final audit：

```python
"final_audit": {
    "samples": 2,
    "times": [0.1, 1.0, 10.0, 100.0],
    "full_vs_rom_thermal_tolerance": 5e-2,
    "tensor_relative_tolerance": 2e-1,
    "current_space_relative_tolerance": 2e-1,
    "outward_relative_tolerance": 2e-1,
    "projection_correction_limit": 2e-1,
    "reduced_dynamic_relative_tolerance": 1e-1,
    "integrator_relative_tolerance": 1e-4,
    "integrator_rtol": 1e-7,
    "integrator_atol": 1e-9,
    "integrator_max_step": 10.0,
    "circuit_condition_limit": 1e8,
    "operating_cases": [
        {"name": "current-controlled", "operating": [5.0, 0.0]},
        {"name": "circuit-controlled", "drive": {
            "voltage": [10.0, 0.0],
            "series_impedance": [0.1, 0.1],
        }},
    ],
}
```

final audit 分四层：

1. full thermal truth vs geometry-aware thermal ROM：`T(x,t) / Tmin / Tmax / wire / steady a_*`；
2. truth tensors vs neural tensors：`Z/D/H`、physical outward loss、decoder correction，以及复电流方向 `e_i`、`e_i±e_j`、`e_i±i e_j` 的 `Zc/P/q` contractions；
3. truth-tensor ROM vs surrogate-tensor ROM：current-controlled 与 circuit-controlled 的 finite-time trajectories、`Z(t)`、currents、wire temperature、steady residual、closed-loop local stability 和 circuit condition number；
4. 实际生产 `etd2_adaptive` vs 同一 surrogate-tensor reduced ODE 的高精度 BDF reference，单独报告 `maximum_integrator_relative_error`，避免把时间积分误差混入 surrogate error。

报告明确标记为：

```text
frozen_held_out_numerical_validation
```

它是冻结有限 held-out 集上的数值验证，不宣称覆盖整个连续参数域的严格数学 certificate。

## 11. 在线阶段

给定一个静态 geometry：

1. 检查 production-domain 与几何合法性；
2. 确定性生成 \(\Phi(g),M_r(g),K_r(g)\)；
3. MLP 一次 forward 得到 raw `Z_field / D_vol / H_j`；
4. 用当前 geometry 的 modal bounds 做 hard physical decode；
5. current-controlled 直接使用 prescribed complex current；
6. circuit-controlled 由显式端口 circuit system 求 current；
7. 显式更新 `R_wire(T)` 和 wire heat；
8. 推进 reduced thermal ODE 或求 stable steady state。

current-controlled 默认：

```python
PREDICTION["operating"] = [5.0, 0.0]
```

voltage-driven 示例：

```python
PREDICTION["drive"] = {
    "voltage": [10.0, 0.0],
    "series_impedance": [0.1, 0.1],
}
```

一次 query 内 geometry 默认静止。若未来支持 \(g=g(t)\)，必须显式加入 moving-basis transport term

\[
\Phi(g)^TM(g)\dot\Phi(g)a,
\]

不能简单逐 time step 更换 basis。

## 12. 默认网格的重要说明

当前默认：

```python
"fine_step": 0.012
```

即约 12 mm，而 conductor width/thickness 的 production range 可以明显小于该尺度。finite-cross-section Gauss source 解决的是“source support 必须绑定真实 conductor geometry”，**并不意味着 12 mm 网格自动解析了 1 mm 级 conductor self physics**。

因此新版训练强制执行 EM/full mesh refinement Gate。若默认网格对 `Z/D/P_vol/D_out/H/T` 未达到配置收敛阈值，`python run.py --mode train` 会直接停止。正确处理方式是收细 `fine_step/max_step` 或进一步改进局部离散/自项物理模型，而不是放宽 Gate 来获得一个伪 certified 模型。

## 13. 缓存与 artifact

```text
results/uwpt/unified.cache.json
results/uwpt/unified.geometry_thermal.npz
results/uwpt/unified.tensor_dataset.npz
results/uwpt/model.tensor_training.pt
results/uwpt/model.geometry_thermal.npz
```

当前物理 cache format 已升级到包含 pre-basis spatial truth preflight 的版本；旧 cache 不能跳过新 Gate。

当前 unified model artifact `FORMAT_VERSION = 12`。没有 finite-support source / pre-basis spatial truth preflight / post-basis Physics Gate / completely-held-out release audit 这一整套冻结语义的旧模型会被 `load()` fail closed，需要重新训练。

物理 cache signature 不包含 MLP network/optimizer/device，也不包含 final release audit 的阈值或 operating cases；只改 neural optimizer 或 final Go/No-Go 阈值会复用已经冻结的 thermal/tensor truth。改变以下任一上游对象则会使 downstream truth/POD/model 失效：

- source/terminal convention；
- open boundary/domain；
- mesh/material/loss masks；
- geometry domain；
- canonical thermal library/basis generator；
- thermal time-scale/trajectory policy；
- tensor schema/modal-bound convention。

## 14. 关键测试

```bash
python -m pytest -q \
  tests/test_unified_geometry_physics.py \
  tests/test_unified_open_boundary.py \
  tests/test_unified_thermal.py \
  tests/test_unified_thermal_trajectory_gate.py \
  tests/test_unified_tensor_surrogate.py \
  tests/test_unified_end_to_end.py \
  tests/test_run_neural_user_defaults.py
```

生产训练与测试不依赖 GitHub Actions。