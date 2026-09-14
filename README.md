# SDF-MPNEO — geometry→tensor + geometry-aware thermal ROM

当前生产主链只保留一条路线：

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

核心原则：**神经网络只学习静态 geometry→EM tensor 映射。** thermal basis、thermal mass/stiffness、电流幅值/相位、线圈温度电阻、Joule 二次型、电路方程、热 ODE 和稳态方程全部保留为显式物理。

在线推理不运行 Maxwell、neural Maxwell solver、Krylov/FGMRES 或 full-field correction。Maxwell 只用于离线 truth 生成。

## 运行

```bash
python -m pip install -e '.[gui,neural,dev]'
python run.py --mode train
python run.py --mode predict
```

纯控制台训练：

```bash
python run.py --mode train --headless
```

用户配置集中在 `run.py` 顶部。

## 1. 离线开放域 Maxwell truth

固定几何后求端口基激励场：

\[
A_{\rm em}(g)X(g)=B(g).
\]

离线 Maxwell 使用匹配海水介质的一阶 Silver–Müller / Sommerfeld 开放阻抗边界。边界切向 edge DOF 被保留，弱式加入

\[
i\omega Y M_{\partial\Omega},
\qquad
Y=\sqrt{\frac{\epsilon-i\sigma/\omega}{\mu}},
\qquad \operatorname{Re}Y\ge0.
\]

端口场阻抗采用 negative source reaction：

\[
Z_{\rm field}=-S^TX.
\]

总物理海水导电耗散矩阵为

\[
D_{\rm vol}=X^H H_\sigma X,
\qquad
P_{\rm vol}(c)=\frac12c^HD_{\rm vol}c.
\]

开放边界独立给出

\[
D_{\rm out}^{\rm phys}
=X^H\left(\operatorname{Re}Y\,M_{\partial\Omega}\right)X,
\]

因此 truth 直接检查

\[
\operatorname{Herm}(Z_{\rm field})
\approx D_{\rm vol}+D_{\rm out}^{\rm phys},
\]

而不是用 `Herm(Z)-D_vol` 反定义 outward power 后自证。

线圈温度相关 AC 电阻独立处理：

\[
Z_{\rm tot}=Z_{\rm field}+\operatorname{diag}(R_{\rm wire}(T)).
\]

## 2. Geometry-aware deterministic thermal ROM

不再假设整个 production geometry domain 共用一个固定 \(\Phi\)。生产 thermal basis 是

\[
\boxed{
\Phi(g)=
[\Phi_{\rm bg},\ \mathcal T_{\rm tx}(g)\Psi_{\rm tx},\ \mathcal T_{\rm rx}(g)\Psi_{\rm rx}]
}
\]

首版只实现确定性的 rigid translation / rotation transport；不使用 neural basis、dynamic POD 或 Grassmann interpolation。局部 canonical blocks 在 reference pose 中构建，查询 geometry 时通过连续三线性插值 transport 到当前 pose。mode ordering 始终固定。

每个 geometry 都从真实 full thermal operators 投影：

\[
M_r(g)=\Phi(g)^TM_T(g)\Phi(g),
\qquad
K_r(g)=\Phi(g)^TK_T(g)\Phi(g).
\]

`M_r/K_r` 不由网络预测。每次生成 \(\Phi(g)\) 都检查 support、rank 和 transport conditioning；每次组装生产 thermal context 还会直接检查真实 `M_r(g)`、`K_r(g)` 的正定性与 condition number，超限即 fail closed。

canonical library 当前由三部分组成：

- `background_modes`：覆盖代表性 geometry 的 volume-Joule / uniform-initial-condition 慢热响应；
- `local_modes[0]`：TX canonical wire/local thermal block；
- `local_modes[1]`：RX canonical wire/local thermal block。

尺寸/shape 变化会进入 canonical-mode truth 构造与真实 \(M(g),K(g)\)；首版不额外引入 scale-aware transport。只有 held-out audit 证明 rigid transport 不够时才增加该结构。

## 3. Geometry-dependent modal Joule tensors

第 \(j\) 个 modal Joule tensor 使用当前 geometry 的 mode \(\phi_j(g)\)：

\[
[W_j(g)]_{mn}
=\int_\Omega\sigma\,\phi_j(g)N_m\cdot N_n\,dx,
\qquad
H_j(g)=X(g)^H W_j(g)X(g).
\]

任意峰值复端口电流的 reduced volume heat 为

\[
q_{{\rm vol},j}(g,c)
=\frac12\operatorname{Re}(c^HH_j(g)c).
\]

由于 \(\Phi\) 随 geometry 变化，Loewner bounds 也必须逐 geometry 计算：

\[
\boxed{
\phi_j^{\min}(g)D_{\rm vol}(g)
\preceq H_j(g)
\preceq
\phi_j^{\max}(g)D_{\rm vol}(g)
}
\]

所以 tensor dataset 对每个 geometry 同时保存自己的 `phi_min / phi_max`。训练 penalty 和在线 hard physical decoder 都使用该 geometry 的 bounds，不再保存一套全局 `phi_min/max`。

## 4. 神经网络只学 geometry→tensor POD coefficients

神经目标仍然只有

\[
g\mapsto\{Z_{\rm field}(g),D_{\rm vol}(g),H_1(g),\ldots,H_r(g)\}.
\]

输出 tensor 仅用 training split 做 POD/SVD 压缩，residual MLP 预测 POD coefficients：

```python
"network": {
    "width": 128,
    "blocks": 3,
    "activation": "silu",
}
```

输入只有固定宽度 geometry encoding，不输入时间、电流、热状态或 Maxwell residual。姿态用 `sin/cos` 编码，coil/package pose 都进入 encoding。

POD 解码后物理层强制：

- `Z_field` complex symmetric；
- `D_vol` Hermitian PSD；
- `Herm(Z_field)-D_vol` PSD；
- 每个 `H_j` Hermitian；
- 当前 geometry 的 modal Loewner bounds。

`Z/D` 与 `H` projection correction 单独报告；大 correction 不能被安全层掩盖。

## 5. Thermal basis 时间尺度与独立 trajectory 验证

thermal canonical rank 由 resolvent energy error 自动决定。统一 anchor：

\[
(K+sM)u=b,
\]

并用真正的 Galerkin energy error

\[
\frac{\|u-u_r\|_{K+sM}}{\|u\|_{K+sM}}
\]

作为 canonical block 构造与第一层 held-out 验收指标。

默认不再强迫当前约 12 mm thermal mesh 表示 1 ms 下远小于网格的局部扩散。默认可解析 resolvent 时间尺度为：

```python
"thermal_time_scales": [0.1, 1.0, 10.0]
```

并始终加入 `s=0` steady anchor。

**resolvent audit 不是 trajectory certificate。** 新版训练还会在完全 held-out geometries 上独立比较 full thermal system 与 geometry-aware ROM。默认时间为：

```python
"thermal_trajectory_times": [0.1, 1.0, 10.0, 100.0]
```

constant volume-Joule 与 wire source directions 从零初值推进；声明的 uniform initial-condition family 做 homogeneous transient。full/reduced 两边直接使用线性 matrix-exponential action，因此该 Gate 不混入普通 time-step tolerance 误差。

每个 held-out case 检查并记录：

- full-field thermal-mass relative error；
- `T_min / T_max` relative error；
- 每个 wire-average temperature relative error；
- steady full field；
- steady reduced coordinate \(a_*\) 相对当前 geometry 的 `M`-projection error；
- uniform initial-condition 在 `t=0` 的 projection error。

trajectory 与 steady 的 composite error 必须和 resolvent error 一样低于 `thermal_basis_energy_tolerance`，否则 `stop_reason=validation_trajectory_target_not_met`，训练直接拒绝继续。

1000 s、10000 s 等更长查询不需要一一增加同长度的 resolvent anchor；同一个稳定 reduced ODE 可以继续积分，`t=inf` 独立求 equilibrium 并验证稳定性。100 s trajectory audit 则专门检查“超出 resolvent anchor 时间”的实际动力学表现。

held-out geometry 从不参与 canonical library enrichment。失败时会明确报告最差 geometry、source/case、时间、field/extrema/wire/steady-coordinate error，而不是只给一个不可解释的总数。

full initial temperature 输入在当前 geometry 上做

\[
[\Phi(g)^TM(g)\Phi(g)]a_0
=\Phi(g)^TM(g)\theta_0
\]

的 `M`-正交投影。

## 6. Matrix-aware training

数据严格分成 `train / validation / test / audit`；POD、归一化只使用 train split。训练 loss 在 POD 解码后的物理对象上分别计算 `Z_field / D_vol / H_j` 矩阵相对误差，并加轻量 passivity / geometry-dependent Loewner penalty。

默认 optimizer：

```python
"optimizer": {
    "epochs": 240,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "patience": 40,
    "validation_interval": 2,
    "pod_relative_tail_tolerance": 1e-4,
    "physics_penalty_weight": 0.05,
    "z_weight": 1.0,
    "d_weight": 1.0,
    "h_weight": 1.0,
    "dtype": "float32",
}
```

电流幅值/相位不进入网络，因此改变 current phasor 不需要重新训练。

## 7. 几何与离散连续性

训练/推理 geometry 先做最低必要物理合法性检查：

- coil 连同导体截面必须位于自己的 package 内；
- package 不允许相交；
- geometry 必须位于背景域内。

line source / line heat 使用线性 cloud-in-cell；package fraction 使用固定 3×3×3 Gauss 子单元积分，避免 midpoint assignment 造成明显 staircasing。

thermal rigid transport 使用连续三线性 interpolation，并在 transport 后做 mode support、rank、conditioning 和 weighted orthonormality 检查。

## 8. 在线阶段

给定一个静态 geometry：

1. 确定性生成当前 `Phi(g)`；
2. 用真实 `M(g), K(g)` 投影 `Mr(g), Kr(g)`，并验证 SPD/conditioning；
3. MLP 调用一次得到 raw `Z_field / D_vol / H_j`；
4. 用当前 `Phi(g)` 的 conductivity-loss-support `phi_min/max` 做 hard physical decode；
5. 给定 current phasor，解析 contraction 得到 volume power / modal heat；
6. 从当前 temperature field 显式计算 `R_wire(T)` 和 wire heat；
7. 推进 reduced thermal ODE 或求 stable steady state。

也支持 voltage-driven：

```python
PREDICTION["drive"] = {
    "voltage": [10.0, 0.0],
    "series_impedance": [0.1, 0.1],
}
```

每个 thermal stage 只解端口级小 circuit system，不回到 Maxwell。

一次 query 内 geometry 默认为静止，因此没有 moving-basis 项。未来若支持 \(g=g(t)\)，必须显式加入

\[
\Phi(g)^TM(g)\dot\Phi(g)a
\]

而不能简单每个 time step 更换 basis。

`t=inf` root 收敛后还会检查 reduced closed-loop Jacobian spectral abscissa，并单独报告 `stable`。

## 9. Physics / ROM Gate

`python run.py --mode train` 自动 fail-fast 检查：

- Maxwell algebraic residual；
- raw reaction reciprocity；
- `D_vol` PSD；
- 独立边界 Poynting quadratic form PSD；
- `Herm(Z_field) = D_vol + D_out_phys` 的矩阵功率闭合；
- 当前 geometry 的 modal Loewner bounds；
- open-boundary domain expansion convergence；
- geometry-aware transported basis support / rank / conditioning；
- 每个 geometry 的真实 `M_r(g), K_r(g)` SPD / conditioning；
- held-out thermal resolvent energy error；
- held-out full-vs-ROM short/intermediate/long trajectory outputs；
- held-out forced steady field 与 \(a_*\) consistency。

开放边界默认：

```python
"open_boundary_check": {
    "samples": 3,
    "padding": 0.12,
    "relative_tolerance": 5e-2,
}
```

任一关键检查失败都会拒绝 surrogate training。

当前 EM source 仍是 regularized line/filament approximation；physical wire-radius/self-impedance convergence 仍属于独立 conductor-model Gate，不能由 algebraic residual 替代。

## 10. 缓存与 artifact

```text
results/uwpt/unified.cache.json
results/uwpt/unified.geometry_thermal.npz
results/uwpt/unified.tensor_dataset.npz
results/uwpt/model.tensor_training.pt
results/uwpt/model.geometry_thermal.npz
```

模型 artifact 保存 canonical thermal library（BG/local blocks + reference geometry），而不是一张固定 `thermal_basis`。

trajectory Gate 接入后物理 cache format 已升级，旧 cache 不允许跳过新验证；model artifact format 也同步升级，旧的 pre-trajectory-certification geometry-aware model 会 fail closed，需要重新训练生成。

改变 MLP optimizer 时可复用通过当前版本 Gate 的物理 cache；改变背景、材料、geometry domain、canonical thermal library 定义、时间尺度、trajectory audit 时间、开放边界或 tensor schema 会使物理 cache 失效。

## 11. 关键测试

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

生产入口不依赖 GitHub Actions。
