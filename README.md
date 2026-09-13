# SDF-MPNEO — geometry→tensor electrothermal ROM

当前生产主链只保留一条路线：

\[
\boxed{
\text{geometry } g
\rightarrow
\{Z_{\rm field}(g),D_{\rm vol}(g),H_1(g),\ldots,H_r(g)\}
\rightarrow
\text{explicit current/circuit physics}
\rightarrow
\text{shared thermal ROM}
\rightarrow
T(t)
}
\]

核心原则：**神经网络只学习静态几何到低维电磁 tensor 的映射。** 电流幅值/相位、线圈温度电阻、Joule 二次型、热 ODE 和稳态方程全部保留显式物理。

在线推理不运行 neural Maxwell solver、Krylov residual training 或 FGMRES。Maxwell 只用于离线生成 truth tensors。

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

配置集中在 `run.py` 顶部。

## 1. 离线电磁 truth

固定几何后先求端口单位激励场：

\[
A_{\rm em}(g)X(g)=B(g).
\]

Maxwell 只作为离线 truth solver。由完整场构造：

\[
Z_{\rm field}(g),
\qquad
D_{\rm vol}(g),
\qquad
H_j(g),\ j=1,\ldots,r.
\]

采用峰值 phasor，因此

\[
P_{\rm vol}(c)=\frac12 c^H D_{\rm vol} c,
\]

\[
q_j(c)=\frac12\operatorname{Re}(c^H H_j c).
\]

端口场阻抗采用统一的 **negative source reaction** 约定

\[
Z_{\rm field}=-S^TX.
\]

线圈自身温度相关 AC 电阻不塞进网络，而是在在线阶段显式加入

\[
Z_{\rm tot}=Z_{\rm field}+\operatorname{diag}(R_{\rm wire}(T)).
\]

truth label 在做 reciprocity 投影之前先检查 raw reaction matrix；同时检查 Maxwell algebraic residual 和当前有限 PEC 离散模型内部的 reaction/Joule power balance，避免“先投影正确再自证”。

## 2. 神经网络只学 geometry→tensor POD coefficients

输出 tensor 先在 **training split** 上做 POD/SVD 压缩，普通 residual MLP 只预测 POD coefficients：

```python
"network": {
    "width": 128,
    "blocks": 3,
    "activation": "silu",
}
```

输入只有固定宽度 geometry encoding；不输入时间、电流、热状态或 Maxwell residual。姿态使用 `sin/cos` 编码；coil 和 package pose 都进入 encoding，避免隐藏几何变量。

POD 容差显式配置：

```python
"pod_relative_tail_tolerance": 1e-4
```

POD 解码后再恢复物理矩阵结构：

- `Z_field`：复对称；
- `D_vol`：Hermitian；
- `H_j`：Hermitian；
- `D_vol >= 0`；
- `Herm(Z_field)-D_vol >= 0`；
- `phi_min[j] D_vol <= H_j <= phi_max[j] D_vol`（Loewner 序）。

结构投影的 `Z/D` correction 和 `H` correction 会单独报告；不能用大幅投影掩盖差的 raw surrogate。

## 3. Matrix-aware training

训练损失在 POD 解码后的物理对象上分别计算

\[
Z_{\rm field},\quad D_{\rm vol},\quad H_j
\]

的矩阵相对误差，并加入轻量 passivity / Loewner penalty。数据严格分为 `train / validation / test / audit` 四组；POD 和归一化只用 train split。

默认：

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

电流幅值/相位不属于训练输入，因此改变 current phasor 不需要重新训练。

## 4. Transient-aware thermal ROM

thermal rank 自动决定，判据使用与目标动力学一致的 Galerkin energy error。

对每个 geometry，热源覆盖完整有限维 port-current quadratic span，并显式加入 wire-heat directions。然后在多个 resolvent shift

\[
A_s=K+sM
\]

上构造 anchors，对实际 Galerkin 解误差

\[
\frac{\|u-u_r\|_{A_s}}{\|u\|_{A_s}}
\]

做 greedy enrichment。

默认时间尺度：

```python
"thermal_time_scales": [1e-3, 1.0, 1000.0]
```

同时保留独立 geometry validation set。full initial temperature 输入在当前几何的真实 thermal mass 上做

\[
(\Phi^TM\Phi)a_0=\Phi^TM\theta_0
\]

的 `M`-正交投影，而不是简单体积投影。

## 5. 几何与离散连续性

训练/推理几何现在先做最低必要的物理合法性检查：

- coil 连同导体截面必须位于自己的 package 内；
- package 不允许相交；
- 几何必须位于背景域内。

line source / line heat 不再采用“segment midpoint 落在哪个 cell 就全给哪个 cell”的跳变分配，而采用线性 cloud-in-cell；package fraction 使用固定 3×3×3 Gauss 子单元积分。这样减少人为 geometry staircasing，同时保持实现简单。

## 6. 在线阶段

给定新 geometry：

1. MLP 只调用一次，得到 `Z_field / D_vol / H_j`；
2. 给定 current phasor，直接做矩阵二次型得到 volume power 和 modal heat；
3. 由当前温度显式计算 `R_wire(T)`；
4. 求 reduced thermal ODE 或 steady state；
5. 输出 impedance、current、volume/wire/outward power、temperature。

也支持简单 voltage-driven 模式：

```python
PREDICTION["drive"] = {
    "voltage": [10.0, 0.0],
    "series_impedance": [0.1, 0.1],
}
```

此时每个 thermal stage 只解一个端口级小线性系统，不回到 Maxwell。

`t=inf` 不再把任意 nonlinear root 直接称为稳定稳态：root 收敛后还会计算闭环 reduced vector field Jacobian 的 spectral abscissa，并单独报告 `stable`。

## 7. Physics Gate 当前状态

正式 tensor truth 会 fail-fast 检查：

- Maxwell algebraic residual；
- **raw** reaction reciprocity；
- `D_vol` PSD；
- 当前有限 PEC 离散模型内部的 reaction/Joule power balance；
- modal Loewner bounds。

但当前固定背景电磁外边界仍是**有限 PEC 截断**，还没有独立 open-boundary / PML / Poynting-flux reference。因此 artifact 会明确保持：

```text
open_boundary_verified = false
independent_outward_power_verified = false
certified = false
status = internal_truth_passed_open_boundary_provisional
```

这不是隐藏误差。真正做 domain/open-boundary convergence 后才能升级该 Gate；本次不为了“形式完整”过度设计一个假的 PML。

另外，当前 EM source 仍属于 regularized line/filament approximation；物理 wire-radius/self-impedance convergence 仍需后续 Physics Gate 验证，不能仅凭当前 mesh residual 宣称 full conductor fidelity。

## 8. 缓存与检查点

```text
results/uwpt/unified.cache.json
results/uwpt/unified.thermal_basis.npy
results/uwpt/unified.tensor_dataset.npz
results/uwpt/model.tensor_training.pt
```

改变 MLP 宽度/优化器时，可以复用 thermal basis 和 tensor truth 数据；改变背景、材料、geometry domain、thermal basis 定义或 tensor schema 时会使物理缓存失效。

## 9. 关键测试

```bash
python -m pytest -q \
  tests/test_unified_geometry_physics.py \
  tests/test_unified_thermal.py \
  tests/test_unified_tensor_surrogate.py \
  tests/test_unified_end_to_end.py \
  tests/test_run_neural_user_defaults.py
```

生产入口不依赖 GitHub Actions。
