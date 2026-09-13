# SDF-MPNEO — 统一几何、operator-polynomial neural-FGMRES 电磁–热求解器

SDF-MPNEO 只有一条正式模型路线：

\[
\boxed{
\text{解析/隐式几何}
\rightarrow
\text{固定多尺度背景物理空间}
\rightarrow
\text{full sparse Maxwell}
\rightarrow
\text{MR-Jacobi + neural operator polynomial}
\rightarrow
\text{FGMRES 真 residual 闭环}
\rightarrow
\text{严格 Joule 热源}
\rightarrow
\text{自动 thermal ROM}
\rightarrow
\text{结构保持热动力学}
}
\]

核心原则：**只学习“怎么更快地解 PDE”，不学习“PDE 的答案是什么”。**

神经网络不直接预测阻抗、Joule tensor、温度或最终 Maxwell 场。Maxwell 的空间修正方向由真实 sparse operator 本身生成；小网络只预测少量复数 polynomial 系数。最终电磁解是否接受，只由真实方程

\[
\frac{\|B-A_{\rm em}X\|_2}{\|B\|_2}\le \varepsilon_{\rm EM}
\]

决定。

## 直接运行

所有正常使用参数都集中在根目录 `run.py`：

```bash
python -m pip install -e '.[gui,neural,dev]'
python run.py --mode train
python run.py --mode predict
```

训练默认打开 PyQt 窗口；纯控制台：

```bash
python run.py --mode train --headless
```

没有 fast/general/legacy 模式，没有 domain probe、Gate 或另一套代理模型。

## 统一几何与固定背景

`DEFAULT_GEOMETRY` 是默认查询几何。线圈支持 `circle`、`rounded_square`、`polyline`、`spline`。几何可以改变线圈尺寸、匝数、pitch、线宽、厚度、三维平移和姿态，封装尺寸/姿态也属于同一描述。

几何变化只改变固定背景中的材料占据、线圈源和当前物理算子，不会创建另一套模型。`GEOMETRY_SAMPLING` 仅用于训练求解加速器，不是模型有效域。真正的几何硬边界是 `BACKGROUND['bounds']`。

背景采用固定非均匀 Cartesian edge space。完整 Maxwell 算子为

\[
A_{\rm em}=C^T H_{\mu^{-1}}C-\omega^2H_\epsilon+i\omega H_\sigma.
\]

海水电导率、介电常数和三维体积电场都进入真实算子，因此海水涡流和海水体积 Joule 发热不会被“离线圈远就删除”的规则忽略。

## Maxwell 不做全局 solution ROM

正式主链已经删除：

- 全局 Maxwell solution basis \(V_E\)
- Maxwell rank
- `unified.em_basis.npy`
- dense \(Q=(AV)^H(AV)\)
- dense \(S=(AV)^HB\)
- 网络输出 reduced coefficient \(C\)

广几何下 Maxwell solution family 不具备足够强的全局低秩性。因此当前 Maxwell 始终求解完整固定背景方程：

\[
A_{\rm em}(g,T)X=B(g).
\]

## MR-Jacobi + neural operator polynomial

对当前真实 residual \(r\)，先构造 Jacobi 方向

\[
q_0=D^{-1}r,
\]

并求一个复数标量 \(\alpha\)，使

\[
\|r-\alpha A q_0\|_2
\]

最小。这个 minimum-residual Jacobi 修正记为 \(z_{\rm MR}\)。因此即使网络完全为零，物理 baseline 的一步 residual 也不会比原 residual 更差。

在此基础上继续由真实算子构造短阶 polynomial 空间：

\[
q_{k+1}=D^{-1}Aq_k.
\]

默认使用 3 阶。网络不再输出 25,200 个 edge correction，而只输出三个有界复数系数：

\[
\boxed{
z_\theta=z_{\rm MR}+\sum_{k=0}^{2} c_k(\mathcal F(A,r))\,\widehat q_k
}
\]

其中 \(\widehat q_k\) 做 RMS 归一化，避免不同 polynomial 阶之间尺度失衡。

网络输入仍包含真实 residual、MR-Jacobi correction、MR 后 residual、operator diagonal/phase/row coupling、edge orientation 和物理位置。局部 edge 特征经过 orientation-preserving multiscale aggregation，最后只汇总为少量全局 polynomial 系数。

这意味着：**空间结构由真实 \(A\) 生成，神经网络只决定这些物理方向应该怎样组合。**

最后一层零初始化，因此未训练模型严格退化到 MR-Jacobi baseline。

## solution-label-free 训练

训练没有 Maxwell solution label。对网络产生的方向 \(z_\theta\)，训练目标与 FGMRES 的实际使用方式一致：FGMRES 会自己选择该方向的最优复数幅值，所以训练直接最小化

\[
\boxed{
L_{\rm EM}
=
\min_{\alpha\in\mathbb C}
\frac{\|r-\alpha A z_\theta\|_2^2}{\|r\|_2^2}
}
\]

而不是强迫网络同时学准方向和绝对幅值。

训练 residual 包含 unit-port RHS、端口组合和基础物理迭代过程中出现的中间 residual。训练缓存只保存几何、材料温升、split 和 residual-generation 设置，不保存 Maxwell solution label 或 dense reduced matrices。

## FGMRES 真 residual 闭环

同一 polynomial preconditioner 既用于初始方向，也用于每一步 flexible Krylov 预条件。因为预条件器随 residual 改变，正式求解器使用 **FGMRES**。

FGMRES 每一步都用当前真实 sparse operator 重新计算 residual。唯一停止条件是

\[
\frac{\|B-AX\|_2}{\|B\|_2}\le\varepsilon_{\rm EM}.
\]

网络若输出 NaN/Inf，该次 correction 自动退化到 MR-Jacobi。若在 `maxwell_max_iterations` 内真实 residual 仍未达到目标，则明确报错，不会用 surrogate 或 silent fallback 返回答案。

`run.py` 中：

```python
PHYSICS = {
    "maxwell_residual_tolerance": 1e-7,
    "maxwell_max_iterations": 200,
    "maxwell_restart": 40,
}
```

## 训练报告直接 benchmark 真正的 solver

单个 directional loss 不能等价代表 Krylov 加速，因此训练结束后会在 validation/test 子集上直接比较：

- MR-Jacobi baseline
- neural 3 阶 operator-polynomial preconditioner

`training.report.json` 中的 `solver_benchmark` 会记录：

- FGMRES iterations 的 median / p90 / max
- restart 的 median / p90 / max
- success rate
- maximum final true residual
- 总 wall time 与每 system wall time
- median iteration speedup
- wall-time speedup

因此是否值得保留神经模块最终以**真实 FGMRES 性能**判断，而不是只看 training loss。

## Joule 与阻抗

只有通过真实 Maxwell residual 检查后的多端口场 \(X\) 才进入输出层。

体积导电区域使用

\[
q^{\rm volume}=\frac12\sigma|E|^2,
\]

因此海水体积损耗完整保留。对 thermal basis 第 \(j\) 个模式：

\[
G_j=\Re(X^HH_jX),\qquad q_j=\zeta^HG_j\zeta.
\]

线圈 AC resistance heating 同样由物理项加入。阻抗也从 corrected field 与真实 source functional 计算，不由网络直接预测。

## thermal rank 自动决定

Maxwell 全局解基被删除，但 thermal ROM 保留。thermal basis \(\Phi_T\) 由真实 Joule source anchor 自动增广，同时控制

\[
KT=q,\qquad M\dot T=q,
\]

直到 steady/dynamic anchor residual 都低于 `thermal_basis_anchor_residual`。thermal rank 是 residual 目标的结果，不是用户填写的 rank。

热动力学始终由

\[
M_r(g)\dot a=-K_r(g)a+q_r(a,g,u)
\]

数值积分；时间不是网络输入。

## 默认训练设置

```python
TRAINING = {
    "thermal_basis_anchor_residual": 5e-2,
    "em_temperature_rise_bounds": [0.0, 80.0],
    "n_operator_samples": 96,
    "residual_training_steps": 3,
    "device": "cuda",
    "network": {
        "width": 32,
        "levels": 3,
        "blocks_per_level": 1,
        "activation": "silu",
        "polynomial_order": 3,
        "coefficient_limit": 2.0,
    },
    "optimizer": {
        "epochs": 200,
        "patience": 24,
        "validation_interval": 2,
        "min_relative_improvement": 5e-4,
        "benchmark_samples_per_split": 6,
    },
}
```

`epochs=200` 只是上限。默认 early stopping 不再把第 5、6 位小数的抖动当成有效进展，从而避免平台期继续浪费大量训练时间。

这里不存在任何 Maxwell rank 配置。

## 缓存与继续训练

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

检查点身份包含 network config、training config、固定 edge topology、sample count、residual-generation 设置和 feature schema。polynomial architecture/schema 改变时旧检查点会自动失效并重新训练；thermal basis 和 residual dataset 仍可复用。

## 推理日志

推理会打印真正有物理意义的指标：

```text
Maxwell initial=... -> final=...
FGMRES iterations=...
restarts=...
Tmax=... K
```

其中 `final` 是真实 sparse Maxwell residual。

## 正确性边界

小 algebraic residual 说明离散 Maxwell 方程被充分求解，但不自动保证连续模型或空间离散误差很小。最终精度还取决于固定背景分辨率、curl-compatible edge discretization、sub-cell geometry/source representation、材料本构、thermal residual target 和频率模型假设。必要时应做背景空间收敛检查。

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

`test_unified_residual.py` 检查：operator-action feature、三阶 polynomial direction、零初始化退化到 MR-Jacobi、FGMRES 最终满足真实 full-space residual，以及非有限神经输出不会污染最终物理解。
