# SDF-MPNEO — multiscale sparse-neural Maxwell–thermal solver

正式实现只保留一条模型路线：

\[
\boxed{
\text{隐式几何}
\rightarrow \text{固定背景物理}
\rightarrow \text{full sparse Maxwell}
\rightarrow \textbf{multiscale sparse neural solver}
\rightarrow \text{FGMRES true-residual closure}
\rightarrow \text{exact Joule}
\rightarrow \text{automatic thermal ROM}
}
\]

核心原则：**神经网络是真正的 Maxwell 求解主体；物理算子提供结构与无标签 residual 监督，FGMRES 只负责最终真实 residual 闭环。**

网络不直接预测阻抗、Joule heat、温度或降阶 Maxwell 系数。它读取当前真实 sparse Maxwell operator 与当前 residual，直接输出 full edge-space correction。最终电磁结果必须满足

\[
\frac{\|B-A_{\rm em}X\|_2}{\|B\|_2}\le \varepsilon_{\rm EM}.
\]

## 运行

```bash
python -m pip install -e '.[gui,neural,dev]'
python run.py --mode train
python run.py --mode predict
```

纯控制台：

```bash
python run.py --mode train --headless
```

没有 fast/general/legacy 模式，没有 Maxwell solution basis / Maxwell rank，也没有 ILU、AMG 或 polynomial 多套主路径。

## Full-space Maxwell

完整电磁方程始终在固定背景 edge space 中求解：

\[
A_{\rm em}(g,T)X=B(g),
\qquad
A_{\rm em}=C^T H_{\mu^{-1}}C-\omega^2H_\epsilon+i\omega H_\sigma.
\]

海水电导率、介电常数和体积电场直接进入真实 operator。正式主链不包含全局 Maxwell basis、`unified.em_basis.npy`、dense $Q/S$ 或 reduced-coefficient 网络。

## Multiscale sparse neural Maxwell solver

每条 edge 的输入包括当前 residual、$D^{-1}r$、局部 operator statistics、edge 位置与方向。网络显式保留复数 Maxwell coupling 的符号和相位。

与旧的单尺度 3-hop message passing 不同，当前网络同时构造：

- fine edge graph；
- stride-2 coarse edge graph；
- stride-4 coarse edge graph；
- stride-8 coarse edge graph。

固定 edge group 保留方向，coarse coupling 由真实 Maxwell sparse operator 聚合得到。每个尺度都进行 complex sparse message passing，随后 coarse hidden state broadcast 回 fine edges，与 fine branch 融合，再输出

\[
\Delta X_\theta\in\mathbb C^{N_E\times N_{\rm rhs}}.
\]

因此网络既保留 fine local physics，又能通过 coarse branches 快速传播长程 Maxwell 信息。Jacobi 只用于输入尺度/reference，以及 NaN/Inf 或近零输出时的数值安全 fallback；它不是正式求解器主体。

默认网络：

```python
"network": {
    "width": 32,
    "fine_message_steps": 2,
    "coarse_levels": 3,
    "coarse_message_steps": 2,
    "fusion_message_steps": 1,
    "solver_steps": 3,
    "activation": "silu",
}
```

## Full-space residual training

训练仍然不需要 Maxwell solution label，但 residual seed 不再局限于端口 RHS 的低维线性张成空间。

每个 operator 默认使用：

```text
2 port RHS
2 full-space random complex residuals
2 multiscale smooth residuals
```

因此网络同时看到真实物理激励、一般 full-edge Krylov-like 方向和低频/长程误差模式。后续 residual 全部由同一个网络和真实 Maxwell operator 产生：

\[
R_0=R,
\qquad
\Delta X_k=N_\theta(A,R_k),
\qquad
R_{k+1}=R_k-A\Delta X_k.
\]

训练与推理共享 `solver_steps=3`。默认 loss 对三步 residual 使用

\[
L=0.1L_1+0.2L_2+0.7L_3,
\qquad
L_k=\frac{\|R_k\|_2^2}{\|R_0\|_2^2},
\]

让最终第 3 步 residual 成为主要优化目标，同时保留前两步稳定监督。

## 优化器与学习率

默认保留物理 `batch_size=1`，并恢复为每个 operator 都更新一次参数：

```python
"optimizer": {
    "epochs": 160,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-3,
    "lr_decay_factor": 0.5,
    "lr_plateau_patience": 8,
    "minimum_learning_rate": 2.5e-4,
    "patience": 32,
    "validation_interval": 2,
    "min_relative_improvement": 5e-4,
    "random_residual_vectors": 2,
    "smooth_residual_vectors": 2,
    "final_step_loss_weight": 0.7,
}
```

学习率最多按 plateau 逐级走：

\[
2\times10^{-3}\rightarrow10^{-3}\rightarrow5\times10^{-4}\rightarrow2.5\times10^{-4}.
\]

LR plateau 判定被刻意放慢，避免仍在有效下降时过早减小步长。发生实际降 LR 时 early-stop stale 计数会清零，让新学习率获得独立观察窗口。

## Solver benchmark 与验收

训练报告同时给出：

- full-space train / validation / test residual loss；
- validation/test port-only residual loss；
- residual seed composition；
- neural rollout 后 relative residual 的 median/p90/max；
- FGMRES closure iterations 与 restarts；
- neural rollout / total solve wall time；
- true-residual success rate 与最大最终 residual。

训练 loss 不能单独定义成功。当前目标是 validation/test full-space loss 至少进入 `0.01` 量级，同时 physical-port neural rollout 明显降低、FGMRES iterations 不再全部撞上限，并最终满足 `1e-7` true residual。

## 推理与 FGMRES closure

推理先连续应用 `solver_steps` 次 neural correction。每一步都检查真实 sparse residual，只有 residual 下降的 RHS 列才接受。

如果仍未达到目标容差，FGMRES 使用同一个 multiscale neural solver 作为可变 preconditioner，只闭合剩余误差。达到 `maxwell_max_iterations` 后仍未达到目标时明确报错，不返回 surrogate answer。

## Joule 与 thermal ROM

只有通过真实 Maxwell residual 检查后的完整场 $X$ 才进入 Joule 与阻抗计算。体积导电区域使用

\[
q^{\rm volume}=\frac12\sigma|E|^2.
\]

Maxwell 不做全局 ROM；thermal ROM 保留。thermal basis 由真实 Joule anchor 和热方程 residual 自动增广，thermal rank 是 residual 目标的结果。时间演化继续求解

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

神经检查点身份包含 feature schema、multiscale network config、training config、edge topology 和 residual seed composition。旧 single-scale / polynomial checkpoint 不会静默续训。网络/优化器改动不会要求重建 thermal basis 与 operator dataset 物理缓存。

## 正确性边界

小 algebraic residual 只说明离散 Maxwell 系统被充分求解，不自动保证连续 PDE 或空间离散误差足够小。最终精度仍依赖背景网格、edge discretization、sub-cell geometry/source representation、材料本构和 thermal residual target。

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
