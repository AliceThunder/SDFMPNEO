# SDF-MPNEO — 统一几何 sparse-neural Maxwell–thermal solver

SDF-MPNEO 现在只保留一条正式模型路线：

\[
\boxed{
\text{隐式几何}
\rightarrow
\text{固定背景物理}
\rightarrow
\text{full sparse Maxwell}
\rightarrow
\textbf{sparse neural residual solver}
\rightarrow
\text{FGMRES true-residual closure}
\rightarrow
\text{exact Joule}
\rightarrow
\text{automatic thermal ROM}
}
\]

核心原则：**神经网络是真正的 Maxwell 加速主体；物理算子提供结构与监督，FGMRES 只负责最终精度闭环。**

网络不直接预测阻抗、Joule heat、温度或最终 Maxwell 场。它读取当前真实 sparse Maxwell operator 的复数耦合图和当前 residual，直接输出 full edge-space correction。

最终 Maxwell 解是否接受，只由

\[
\frac{\|B-A_{\rm em}X\|_2}{\|B\|_2}\le \varepsilon_{\rm EM}
\]

决定。

## 直接运行

正常使用只需要根目录 `run.py`：

```bash
python -m pip install -e '.[gui,neural,dev]'
python run.py --mode train
python run.py --mode predict
```

纯控制台训练：

```bash
python run.py --mode train --headless
```

没有 fast/general/legacy 模式，没有 Maxwell solution basis，没有 Maxwell rank，也没有 ILU/AMG/polynomial 多套求解路径。

## Maxwell 物理空间

完整 Maxwell 方程始终在固定背景 edge space 中求解：

\[
A_{\rm em}(g,T)X=B(g),
\]

其中

\[
A_{\rm em}=C^T H_{\mu^{-1}}C-\omega^2H_\epsilon+i\omega H_\sigma.
\]

正式主链不包含：

- 全局 Maxwell solution basis \(V_E\)
- Maxwell rank
- `unified.em_basis.npy`
- dense \(Q=(AV)^H(AV)\) / \(S=(AV)^HB\)
- reduced-coefficient neural network

海水电导率、介电常数和体积电场保留在真实 Maxwell operator 中，因此海水体积 Joule loss 不会被代理模型绕过。

## Sparse neural Maxwell solver

网络对每个 edge 使用：

- 当前 residual 的实部和虚部；
- \(D^{-1}r\) 的实部和虚部，作为尺度/reference feature；
- diagonal magnitude/phase/loss ratio；
- sparse row coupling strength 与 degree；
- edge 物理位置和方向。

更关键的是，网络不是只读取这些节点统计量。每一层都会使用当前真实 Maxwell 非对角耦合：

\[
h_i^{(l+1)}=
\phi_\theta\!\left(
 h_i^{(l)},
 \sum_{j\ne i}\widehat A_{ij}h_j^{(l)}
\right),
\]

其中 \(\widehat A\) 是按行归一化后的真实复数 off-diagonal Maxwell coupling。实现中实部和虚部 message 分开送入共享 message block，所以耦合的符号和复相位不会被 row-sum 等标量统计丢掉。

网络最终直接输出

\[
\Delta X_\theta\in\mathbb C^{N_E\times N_{\rm rhs}},
\]

而不是少量 polynomial coefficient。

Jacobi 只作为输入归一化参考，以及网络输出为 NaN/Inf 或退化为零方向时的安全 fallback；它不是正式神经求解器的主体。

## Solution-label-free 多步训练

不需要 Maxwell solution label。

训练 residual seed 只包含端口 RHS 和随机 mixed-port RHS。后续 residual 不再由 Jacobi/polynomial 人工生成，而是由同一个神经网络自己产生：

\[
R_0=R,
\]

\[
\Delta X_k=N_\theta(A,R_k),
\]

\[
R_{k+1}=R_k-A\Delta X_k.
\]

默认共享权重 unroll 3 步。每一步都通过真实 matrix-free Maxwell operator 更新 residual，训练目标对后续步骤加更高权重：

\[
L=\frac{1}{\sum_k 2^k}
\sum_{k=1}^{K}2^{k-1}
\frac{\|R_k\|_2^2}{\|R_0\|_2^2}.
\]

因此网络直接学习“连续几步怎样把真实 Maxwell residual 消掉”，而不是学习传统预条件器的参数。

## 推理：神经网络先解，FGMRES 后闭环

推理首先连续应用若干次共享 neural correction。每个 neural step 都用真实 sparse residual 检查；只有 residual 下降的列才接受该步。

随后如果仍未达到目标容差，FGMRES 使用同一个 neural solver 作为可变 preconditioner，只负责把剩余误差闭合到

\[
\varepsilon_{\rm EM}=10^{-7}
\]

等用户指定值。

FGMRES 不提供 surrogate answer。达到 `maxwell_max_iterations` 后仍未满足真实 residual，就明确报错。

## 默认神经训练设置

```python
TRAINING = {
    "n_operator_samples": 96,
    "residual_training_steps": 3,  # mixed residual seed diversity
    "device": "cuda",
    "network": {
        "width": 32,
        "message_passing_steps": 3,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 120,
        "batch_size": 1,
        "learning_rate": 2e-3,
        "patience": 20,
        "validation_interval": 2,
        "min_relative_improvement": 1e-3,
        "unroll_steps": 3,
    },
}
```

batch size 默认 1 是有意的：每个 operator sample 内部已经同时处理多个 RHS，并且 unroll 会保留多步 sparse-message-passing autograd graph；不再通过大 batch 堆显存。

## Solver benchmark

训练完成后，validation/test 会报告：

- neural rollout 后的 relative residual median/p90/max；
- 后续 FGMRES iterations 与 restarts；
- neural rollout 时间；
- 总 Maxwell solve 时间；
- `1e-7` true-residual success rate 与最大最终 residual。

这比只看 neural loss 更重要：正式目标是让神经网络先把 PDE residual 大幅消掉，同时减少最终 closure 的工作量。

## Joule 与 thermal ROM

只有通过真实 Maxwell residual 检查后的完整场 \(X\) 才进入 Joule 和阻抗计算。

体积导电区域使用

\[
q^{\rm volume}=\frac12\sigma|E|^2,
\]

线圈 AC resistance heating 也作为物理项加入。

Maxwell 不做全局 ROM，但 thermal ROM 保留。thermal basis \(\Phi_T\) 由真实 Joule anchor 和热方程 residual 自动增广，thermal rank 是 residual 目标的结果，不是手工输入。

热动力学继续求解

\[
M_r(g)\dot a=-K_r(g)a+q_r(a,g,u),
\]

时间不是神经网络输入。

## 缓存和检查点

物理缓存：

```text
results/uwpt/unified.cache.json
results/uwpt/unified.thermal_basis.npy
results/uwpt/unified.residual_dataset.npz
```

神经训练检查点：

```text
results/uwpt/model.training.pt
```

检查点身份包含 feature schema、network config、training config、edge topology 和采样设置。旧 polynomial/旧 edge-MLP checkpoint 不会静默续训。

## 正确性边界

小 algebraic residual 只说明离散 Maxwell 系统被充分求解，不自动保证连续 PDE 或空间离散误差足够小。最终精度还取决于背景网格、edge discretization、sub-cell geometry/source representation、材料本构和 thermal residual target。

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

`test_unified_residual.py` 会检查 sparse complex coupling、full-edge neural output、零初始化安全 fallback、NaN fallback，以及最终 true-residual closure。
