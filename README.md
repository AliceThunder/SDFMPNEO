# SDF-MPNEO — 统一几何 sparse-neural Maxwell–thermal solver

正式实现只保留一条模型路线：

\[
\boxed{
\text{隐式几何}
\rightarrow \text{固定背景物理}
\rightarrow \text{full sparse Maxwell}
\rightarrow \textbf{sparse neural residual solver}
\rightarrow \text{FGMRES true-residual closure}
\rightarrow \text{exact Joule}
\rightarrow \text{automatic thermal ROM}
}
\]

核心原则：**神经网络是真正的 Maxwell 加速主体；物理算子提供结构与监督，FGMRES 只负责最终精度闭环。**

网络不直接预测阻抗、Joule heat、温度或最终 Maxwell 场。它读取当前真实 sparse Maxwell operator 的复数耦合图和当前 residual，直接输出 full edge-space correction。最终结果只由

\[
\frac{\|B-A_{\rm em}X\|_2}{\|B\|_2}\le \varepsilon_{\rm EM}
\]

决定。

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

没有 fast/general/legacy 模式，没有 Maxwell solution basis / Maxwell rank，也没有 ILU、AMG 或 polynomial 多套求解路径。

## Full-space Maxwell

完整电磁方程始终在固定背景 edge space 中求解：

\[
A_{\rm em}(g,T)X=B(g),
\qquad
A_{\rm em}=C^T H_{\mu^{-1}}C-\omega^2H_\epsilon+i\omega H_\sigma.
\]

正式主链不包含全局 Maxwell basis、`unified.em_basis.npy`、dense $Q/S$ 或 reduced-coefficient 网络。海水电导率、介电常数和体积电场直接进入真实 operator，因此海水体积 Joule loss 不会被代理绕过。

## Sparse neural Maxwell solver

每条 edge 的节点输入包括：

- 当前 residual 的实部/虚部；
- $D^{-1}r$ 的实部/虚部，只作为归一化 reference；
- diagonal magnitude/phase/loss ratio；
- row coupling strength 与 sparse degree；
- edge 物理位置与方向。

网络的关键不是这些标量特征，而是每层都显式使用当前真实 Maxwell 非对角复耦合：

\[
m^{(l)}=\widehat A h^{(l)},
\]

其中 $\widehat A$ 是按行归一化后的 complex off-diagonal coupling。$\Re(m)$ 和 $\Im(m)$ 分别送入 message block，所以耦合的符号和复相位不会被 row-sum 等统计量抹掉。

网络最终直接输出

\[
\Delta X_\theta\in\mathbb C^{N_E\times N_{\rm rhs}},
\]

而不是全局系数或 polynomial coefficient。

Jacobi 只作为输入尺度/reference，以及网络输出为 NaN/Inf 或近零退化方向时的数值安全 fallback；它不是正式神经求解器主体。

## Solution-label-free shared unroll

不需要 Maxwell solution label。训练 residual seed 来自端口 RHS 和随机 mixed-port RHS。后续 residual 由同一个神经网络自己产生：

\[
R_0=R,
\qquad
\Delta X_k=N_\theta(A,R_k),
\qquad
R_{k+1}=R_k-A\Delta X_k.
\]

训练与推理共享同一个 `solver_steps`。默认是 3 步。每一步都通过真实 matrix-free Maxwell operator 更新 residual，训练目标对后续 residual 给予更高权重：

\[
L=\frac{1}{\sum_{k=1}^{K}2^{k-1}}
\sum_{k=1}^{K}2^{k-1}
\frac{\|R_k\|_2^2}{\|R_0\|_2^2}.
\]

所以网络直接学习“连续几步怎样消除真实 Maxwell residual”，而不是学习传统预条件器参数。

## 推理与 FGMRES closure

推理先连续应用 `solver_steps` 次共享 neural correction。每一步都检查真实 sparse residual，只有 residual 下降的 RHS 列才接受该步。

如果仍未达到目标容差，FGMRES 使用同一个 neural solver 作为可变 preconditioner，只闭合剩余误差。达到 `maxwell_max_iterations` 后仍不满足真实 residual 时明确报错，不返回 surrogate answer。

## 默认训练设置

```python
TRAINING = {
    "n_operator_samples": 96,
    "residual_training_steps": 3,  # mixed-port residual seed diversity
    "device": "cuda",
    "network": {
        "width": 32,
        "message_passing_steps": 3,
        "solver_steps": 3,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 120,
        "batch_size": 1,
        "learning_rate": 2e-3,
        "patience": 20,
        "validation_interval": 2,
        "min_relative_improvement": 1e-3,
    },
}
```

`batch_size=1` 是有意的：每个 operator sample 内已经同时处理多个 RHS，而且 shared unroll 会保留多步 sparse-message-passing autograd graph。

## Solver benchmark

训练结束后 validation/test 会记录：

- neural rollout 后 relative residual 的 median/p90/max；
- FGMRES closure iterations 与 restarts；
- neural rollout wall time；
- 总 Maxwell solve wall time；
- true-residual success rate 与最大最终 residual。

因此正式判断标准不是单独的 neural loss，而是网络先消掉多少 PDE residual，以及最终 closure 还剩多少工作。

## Joule 与 thermal ROM

只有通过真实 Maxwell residual 检查后的完整场 $X$ 才进入 Joule 与阻抗计算。体积导电区域使用

\[
q^{\rm volume}=\frac12\sigma|E|^2,
\]

线圈 AC resistance heating 也作为物理项加入。

Maxwell 不做全局 ROM，但 thermal ROM 保留。thermal basis 由真实 Joule anchor 和热方程 residual 自动增广；thermal rank 是 residual 目标的结果，不是手工输入。时间演化继续求解

\[
M_r(g)\dot a=-K_r(g)a+q_r(a,g,u),
\]

时间不是神经网络输入。

## 缓存与检查点

```text
results/uwpt/unified.cache.json
results/uwpt/unified.thermal_basis.npy
results/uwpt/unified.residual_dataset.npz
results/uwpt/model.training.pt
```

神经检查点身份包含 feature schema、network config、training config、edge topology 和采样设置。旧 polynomial / 旧 edge-MLP checkpoint 不会静默续训。

## 正确性边界

小 algebraic residual 只说明离散 Maxwell 系统被充分求解，不自动保证连续 PDE 或空间离散误差足够小。最终精度还依赖背景网格、edge discretization、sub-cell geometry/source representation、材料本构和 thermal residual target。

## 正式测试

```bash
python -m pytest -q \
  tests/test_unified_geometry.py \
  tests/test_unified_background.py \
  tests/test_unified_thermal.py \
  tests/test_unified_residual.py \
  tests/test_unified_end_to_end.py \
  tests/test_run_neural_user_defaults.py
```
