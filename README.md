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

在线推理不再运行 neural Maxwell solver、Krylov residual training 或 FGMRES。Maxwell 只用于离线生成 truth tensors。

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

配置仍集中在 `run.py` 顶部。

## 1. 离线电磁 truth

固定几何后先求端口单位激励场：

\[
A_{\rm em}(g)X(g)=B(g).
\]

这里 Maxwell 只作为离线 truth solver。由完整场构造：

\[
Z_{\rm field}(g),
\qquad
D_{\rm vol}(g),
\qquad
H_j(g),\ j=1,\ldots,r.
\]

其中

\[
P_{\rm vol}(I)=\frac12 I^H D_{\rm vol} I,
\]

且 thermal modal Joule forcing 为

\[
q_j(I)=\frac12 I^H H_j I.
\]

端口场阻抗采用统一的 **negative source reaction** 约定；线圈自身温度相关 AC 电阻不塞进网络，而是在在线阶段显式加到

\[
Z_{\rm tot}=Z_{\rm field}+\operatorname{diag}(R_{\rm wire}(T)).
\]

## 2. 神经网络只学 geometry→tensor

网络是普通 residual MLP：

```python
"network": {
    "width": 128,
    "blocks": 3,
    "activation": "silu",
}
```

输入只有固定宽度 geometry encoding；不输入时间、电流、热状态或 Maxwell residual。

输出按矩阵结构编码：

- `Z_field`：复对称；
- `D_vol`：Hermitian；
- `H_j`：Hermitian。

推理时再做结构投影：

- `D_vol >= 0`；
- `Herm(Z_field)-D_vol >= 0`；
- 对每个 thermal mode，利用训练 basis 的 `phi_min/phi_max` 强制 Loewner bounds。

因此网络不需要学习 Hermitian/reciprocity/passivity 这些本来就已知的结构。

## 3. Matrix-aware training

训练目标不是任意 flat-vector MSE，而是分别对

\[
Z_{\rm field},\quad D_{\rm vol},\quad H_j
\]

计算相对 Frobenius 误差，再加入轻量 passivity / Loewner penalty。

默认：

```python
"optimizer": {
    "epochs": 240,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-6,
    "patience": 40,
    "validation_interval": 2,
    "physics_penalty_weight": 0.05,
    "z_weight": 1.0,
    "d_weight": 1.0,
    "h_weight": 1.0,
    "dtype": "float32",
}
```

电流幅值/相位不属于训练输入，因此改变 current phasor 不需要重新训练。

## 4. Transient-aware thermal ROM

thermal rank 仍然自动决定，但判据已改成与目标动力学一致的 Galerkin energy error。

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

同时保留独立 geometry validation set，避免只在 basis construction geometries 上自洽。

## 5. 在线阶段

给定新 geometry：

1. MLP 只调用一次，得到 `Z_field / D_vol / H_j`；
2. 给定 current phasor，直接做矩阵二次型得到 volume power 和 modal heat；
3. 由当前温度显式计算 `R_wire(T)`；
4. 求 reduced thermal ODE / steady state；
5. 输出 impedance、current、volume/wire/outward power、temperature。

也支持简单 voltage-driven 模式：

```python
PREDICTION["drive"] = {
    "voltage": [10.0, 0.0],
    "series_impedance": [0.1, 0.1],
}
```

此时每个 thermal stage 只解一个端口级小线性系统，不回到 Maxwell。

## 6. Physics Gate 0 当前状态

代码现在会显式检查并记录：

- reaction impedance sign；
- reciprocity；
- `D_vol` PSD；
- modal Loewner bounds；
- implied dissipative remainder。

但当前固定背景电磁外边界仍是**有限 PEC 截断**，还没有独立 open-boundary / PML / Poynting-flux reference。因此训练报告会写：

```text
status = provisional_until_open_boundary_gate
certified = false
```

这不是隐藏误差，而是当前理论体系里明确保留的 Physics Gate 0 缺口。后续真正做 open-boundary reference 时再补这一层，不在本次为了“形式完整”过度设计一个假的 PML。

## 7. 缓存与检查点

```text
results/uwpt/unified.cache.json
results/uwpt/unified.thermal_basis.npy
results/uwpt/unified.tensor_dataset.npz
results/uwpt/model.tensor_training.pt
```

改变 MLP 宽度/优化器时，可以复用 thermal basis 和 tensor truth 数据；改变背景、材料、geometry domain、thermal basis 定义或 tensor sample 定义时会使物理缓存失效。

## 8. 正式测试

```bash
python -m pytest -q \
  tests/test_unified_geometry.py \
  tests/test_unified_background.py \
  tests/test_unified_thermal.py \
  tests/test_unified_tensor_surrogate.py \
  tests/test_unified_end_to_end.py \
  tests/test_run_neural_user_defaults.py
```

生产入口不依赖 GitHub Actions。
