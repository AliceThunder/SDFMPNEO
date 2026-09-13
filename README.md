# SDF-MPNEO — 统一几何 multiscale neural Maxwell–thermal solver

正式实现只保留一条模型路线：

\[
\boxed{
\text{隐式几何}
\rightarrow \text{固定背景物理}
\rightarrow \text{full sparse Maxwell}
\rightarrow \textbf{multiscale neural residual solver}
\rightarrow \text{FGMRES true-residual closure}
\rightarrow \text{exact Joule}
\rightarrow \text{automatic thermal ROM}
}
\]

核心原则：**神经网络是 Maxwell 加速主体；真实 sparse Maxwell operator 提供结构与训练残差，FGMRES 只负责最终精度闭环。**

网络不直接预测阻抗、Joule heat、温度或最终 Maxwell 场。最终结果必须满足

\[
\frac{\|B-A_{\rm em}X\|_2}{\|B\|_2}\le \varepsilon_{\rm EM}.
\]

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

正式主链没有 fast/general/legacy 模式，没有 Maxwell solution basis / Maxwell rank，也没有 ILU、AMG 或 polynomial 多套求解路径。

## Full-space Maxwell

完整电磁方程始终在固定背景 edge space 中求解：

\[
A_{\rm em}(g,T)X=B(g),
\qquad
A_{\rm em}=C^T H_{\mu^{-1}}C-\omega^2H_\epsilon+i\omega H_\sigma.
\]

海水电导率、介电常数和体积电场直接进入真实 operator，因此海水体积 Joule loss 不会被代理绕过。

## Multiscale sparse neural Maxwell solver

每条 edge 的动态输入包括：

- 当前 residual 的实部/虚部；
- $D^{-1}r$ 的实部/虚部，作为归一化 reference；
- diagonal magnitude/phase/loss ratio；
- row coupling strength 与 sparse degree。

固定输入包括 edge 物理位置与方向。

网络同时使用四个真实 sparse Maxwell 耦合尺度：

- fine edge graph；
- stride-2 coarse edge graph；
- stride-4 coarse edge graph；
- stride-8 coarse edge graph。

coarse coupling 由真实 Maxwell operator 聚合得到。每个尺度都执行 complex sparse message passing，coarse hidden state 再 broadcast 回 fine edges，与 fine branch 融合，最终直接输出

\[
\Delta X_\theta\in\mathbb C^{N_E\times N_{\rm rhs}}.
\]

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

Jacobi 只用于输入尺度/reference，以及网络输出为 NaN/Inf 或近零退化方向时的数值安全 fallback；它不是正式主求解器。

## 训练 residual：物理端口 + FGMRES Krylov 方向

训练仍然不需要 Maxwell solution label。

此前使用任意 full-space 白噪声 residual 会把训练目标变成与实际 FGMRES 输入分布不一致的困难问题，并且会稀释物理端口 RHS。当前实现改为：

```text
2 physical port RHS
2 Jacobi-Arnoldi Krylov directions per port
```

默认两端口系统因此每个 operator 有 6 条 residual seed。

Jacobi-Arnoldi seed 按与 FGMRES Arnoldi 过程一致的结构构造：

\[
v_0=\frac{b}{\|b\|},
\qquad
z_j=D^{-1}v_j,
\qquad
w_j=A z_j,
\]

随后对已有 Arnoldi basis 做两次正交化并归一化得到后续 $v_{j+1}$。这些方向比任意白噪声更接近 FGMRES 实际调用 neural preconditioner 时看到的输入分布。

训练 objective 对 residual 类型显式加权：

\[
L_{\rm seeds}
=
0.6\,L_{\rm port}
+
0.4\,L_{\rm Krylov}.
\]

因此不会因为增加 Krylov 泛化训练而把物理端口任务稀释到只占三分之一。

## Shared 3-step residual unroll

每个 seed 都使用同一个网络做 3 步 shared unroll：

\[
R_0=R,
\qquad
\Delta X_k=N_\theta(A,R_k),
\qquad
R_{k+1}=R_k-A\Delta X_k.
\]

三步 loss 默认使用：

\[
L=0.1L_1+0.2L_2+0.7L_3,
\qquad
L_k=\frac{\|R_k\|_2^2}{\|R_0\|_2^2}.
\]

最终第 3 步 residual 是主要优化目标。

## 默认训练配置

```python
TRAINING = {
    "n_operator_samples": 96,
    "device": "cuda",
    "network": {
        "width": 32,
        "fine_message_steps": 2,
        "coarse_levels": 3,
        "coarse_message_steps": 2,
        "fusion_message_steps": 1,
        "solver_steps": 3,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 160,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-3,
        "weight_decay": 1e-6,
        "lr_decay_factor": 0.5,
        "lr_plateau_patience": 8,
        "minimum_learning_rate": 2.5e-4,
        "patience": 32,
        "validation_interval": 2,
        "min_relative_improvement": 5e-4,
        "krylov_vectors_per_port": 2,
        "port_loss_weight": 0.6,
        "final_step_loss_weight": 0.7,
        "benchmark_samples_per_split": 4,
    },
}
```

`batch_size=1` 且 `gradient_accumulation_steps=1`，所以每个 operator 都进行一次 optimizer update。

学习率按 validation plateau 最多逐级：

\[
2\times10^{-3}
\rightarrow 10^{-3}
\rightarrow 5\times10^{-4}
\rightarrow 2.5\times10^{-4}.
\]

## 训练监控与可比指标

每次 validation 都分别输出：

```text
val=...  port=...  krylov=...  lr=...
```

这里：

- `port` 才能和早期仅使用物理/端口 residual 的训练结果直接比较；
- `krylov` 衡量网络作为 FGMRES preconditioner 对 Arnoldi 方向的泛化；
- `val` 是按 60% port / 40% Krylov 得到的综合指标。

因此不要再把不同 residual 分布下的总 validation loss 直接横向比较。

最终 `training.report.json` 同时记录：

- weighted train/validation/test residual loss；
- validation/test port-only residual loss；
- validation/test Krylov-only residual loss；
- residual seed composition；
- effective batch size 与最终学习率；
- solver benchmark。

## Solver benchmark

validation/test benchmark 记录：

- 3-step neural rollout 后 relative residual median/p90/max；
- FGMRES closure iterations 与 restarts；
- neural rollout wall time；
- 总 Maxwell solve wall time；
- true-residual success rate；
- maximum final relative residual。

训练是否成功不能只由神经 loss 定义。真正目标是：

1. physical-port neural rollout 明显降低 residual；
2. Krylov residual loss 同时下降；
3. FGMRES iterations 不再全部撞 `maxwell_max_iterations`；
4. success rate 最终达到 100%；
5. 最终 true residual 满足默认 `1e-7`。

## Joule 与 thermal ROM

只有通过 Maxwell true-residual 检查后的完整场 $X$ 才进入 Joule 与阻抗计算。

体积导电区域使用

\[
q^{\rm volume}=\frac12\sigma|E|^2.
\]

线圈 AC resistance heating 同样作为物理项加入。

Maxwell 不做全局 ROM；thermal ROM 保留。thermal basis 由真实 Joule anchor 与热方程 residual 自动增广，thermal rank 是 residual target 的结果，不是手工输入。时间演化继续求解

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

神经检查点身份包含 feature schema、network config、training config、edge topology 和 residual seed composition。

更改 residual 训练策略时旧 neural checkpoint 会自动判为不兼容并重新训练；thermal basis 和 96 个 operator 的物理缓存仍可复用。

## 正确性边界

小 algebraic residual 只说明离散 Maxwell 线性系统被充分求解，不自动保证连续 PDE 或空间离散误差足够小。最终物理精度还依赖背景网格、edge discretization、sub-cell geometry/source representation、材料本构和 thermal residual target。

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
