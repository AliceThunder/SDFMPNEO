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

## 1. 离线开放域 Maxwell truth

固定几何后先求端口单位激励场：

\[
A_{\rm em}(g)X(g)=B(g).
\]

离线 Maxwell 使用匹配海水介质的一阶 Silver–Müller / Sommerfeld 开放阻抗边界。边界切向 edge DOF 不再像旧实现那样删除为 PEC，而是在弱式中加入

\[
i\omega Y M_{\partial\Omega},
\qquad
Y=\sqrt{\frac{\epsilon-i\sigma/\omega}{\mu}}
\]

的被动边界项，其中采用与本项目一致的 \(e^{+i\omega t}\) 相量约定，并选择 \(\operatorname{Re}Y\ge0\) 的 outgoing-wave 分支。

由完整场构造：

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

开放边界还直接给出独立的 outward-power quadratic form：

\[
D_{\rm out}^{\rm phys}
=
X^H\left(\operatorname{Re}Y\,M_{\partial\Omega}\right)X.
\]

因此 truth 侧可以直接检查矩阵级功率恒等式

\[
\operatorname{Herm}(Z_{\rm field})
\approx
D_{\rm vol}+D_{\rm out}^{\rm phys},
\]

而不是用 `Herm(Z_field)-D_vol` 定义 outward power 后再自证。

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

训练/推理几何先做最低必要的物理合法性检查：

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

## 7. Physics Gate

`python run.py --mode train` 会自动 fail-fast 检查：

- Maxwell algebraic residual；
- **raw** reaction reciprocity；
- `D_vol` PSD；
- 独立边界 Poynting quadratic form 的 PSD；
- `Herm(Z_field) = D_vol + D_out_phys` 的矩阵级功率闭合；
- modal Loewner bounds；
- **开放边界域扩展收敛**。

域扩展检查会对独立几何样本再建一个更大的开放边界背景，并分别比较总阻抗、`Re(Z)` 和 `Im(Z)`。默认配置：

```python
"open_boundary_check": {
    "samples": 3,
    "padding": 0.12,
    "relative_tolerance": 5e-2,
}
```

只有所有检查通过，训练才继续，并记录：

```text
open_boundary_verified = true
independent_outward_power_verified = true
certified = true
status = certified
```

任何一项失败都会直接拒绝 surrogate training，不会再生成 `provisional` 模型。

当前 EM source 仍属于 regularized line/filament approximation；物理 wire-radius/self-impedance convergence 是独立于本次开放边界修正的导体模型问题，不能仅凭 mesh residual 宣称 full solid-conductor fidelity。

## 8. 缓存与检查点

```text
results/uwpt/unified.cache.json
results/uwpt/unified.thermal_basis.npy
results/uwpt/unified.tensor_dataset.npz
results/uwpt/model.tensor_training.pt
```

改变 MLP 宽度/优化器时，可以复用 thermal basis 和 tensor truth 数据；改变背景、材料、geometry domain、thermal basis 定义、开放边界设置或 tensor schema 时会使物理缓存失效。

## 9. 关键测试

```bash
python -m pytest -q \
  tests/test_unified_geometry_physics.py \
  tests/test_unified_open_boundary.py \
  tests/test_unified_thermal.py \
  tests/test_unified_tensor_surrogate.py \
  tests/test_unified_end_to_end.py \
  tests/test_run_neural_user_defaults.py
```

生产入口不依赖 GitHub Actions。