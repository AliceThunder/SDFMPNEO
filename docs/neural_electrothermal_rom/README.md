# 几何参数化电磁张量神经电热 ROM

本目录冻结一条生产理论主线：

```text
validated single-frequency Maxwell truth physics
-> spatial Joule/source truth bank
-> transient-aware shared thermal basis
-> geometry-dependent Z_field / D_vol / Hermitian Joule tensors H_j
-> neural geometry surrogate
-> joint energy-feasible physical decode
-> exact complex-current or circuit physics
-> true thermal ROM / ODE
-> T(t), Z(t), power, steady state
```

神经网络只学习几何到低维电磁参数的映射。它不学习 Maxwell field、时间轨迹、thermal mass/stiffness、端口 current law 或外部 circuit law。

## 冻结的相量和功率约定

采用 `exp(+i wt)` 与**峰值复相量**。因此所有 cycle-averaged power 都带 `1/2`：

\[
P_{\rm in}=\frac12\operatorname{Re}(c^H v).
\]

体耗散矩阵定义为

\[
D_{\rm vol}=X^H H_\sigma X,
\qquad
P_{\rm vol}=\frac12c^H D_{\rm vol}c.
\]

thermal modal Joule tensor 为

\[
H_j=X^H W_jX,
\qquad
q_{{\rm vol},j}=\frac12\operatorname{Re}(c^HH_jc).
\]

禁止混用峰值与 RMS phasor convention。

## 当前材料模型下的关键简化

当前正式模型中，体 Maxwell constitutive parameters 不随温度变化，因此

\[
A_{\rm em}=A_{\rm em}(g).
\]

固定几何只需一组 Maxwell truth response。thermal state 只通过明确的 conductor-temperature functional 和集中 wire resistance 进入：

\[
Z(a,g)=Z_{\rm field}(g)+R_{\rm wire}(a,g).
\]

## 生产 surrogate 输出

神经代理学习

\[
g\rightarrow \beta_\theta(g)
\rightarrow
\{\widehat Z_{\rm field},\widehat D_{\rm vol},\widehat H_j\}.
\]

output POD 只是压缩。结构化解码和固定联合可行层必须保证：

\[
Z_{\rm field}^T=Z_{\rm field},
\qquad
D_{\rm vol}=D_{\rm vol}^H\succeq0,
\qquad
H_j=H_j^H,
\]

以及

\[
D_{\rm out}
=\operatorname{Herm}(Z_{\rm field})-D_{\rm vol}\succeq0.
\]

因此 port passivity、海水体耗散和开放域 outward loss 被放进同一矩阵级能量框架。联合投影的 correction norm 必须报告；不能用大幅投影掩盖差的 raw surrogate。

## Frequency-transient 前提

热模型使用 cycle-averaged electromagnetic loss，要求载波周期远短于 thermal/operating-envelope 时间尺度。若 excitation 出现快速调制、多谐波、宽带或强电磁非线性，应改用多频或全时域 EM，当前单频理论不适用。

当前 passivity 也只针对冻结工作频率；未来如果 frequency 成为 surrogate input，必须增加 broadband positive-real / causality structure，不能只逐频率做 PSD 检查。

## Port source 不是默认“闭合线圈”

实际 spiral path 可能是 two-terminal open conductor。source 必须有明确 terminal orientation、normalization、return/feed path 或 charge/current continuity interpretation：

\[
\nabla\cdot J_s=-i\omega\rho_s.
\]

只有真实闭合 loop 才要求 divergence-free。省略 feed/return path 是模型近似，必须验证，不能在理论上自动忽略。

## Thermal basis 的依赖顺序

`H_j` 依赖 frozen thermal mode `phi_j`，所以顺序必须是：

```text
validated spatial Maxwell/Joule/source truth bank
-> build and freeze Phi
-> project spatial Joule bilinear forms onto Phi
-> generate final Z_field / D_vol / H_j tensor dataset
-> POD + neural training
```

固定 geometry 下，volume heat 对 complex current 是 Hermitian quadratic form，因此完整 current-induced heat-source span 的实维数至多为 `n_ports^2`。basis builder 使用确定性的 Hermitian port-space basis combinations 完整覆盖该 span，不依赖大量随机 current samples。

thermal basis 还必须覆盖目标 transient time scales。使用

\[
(K+sM)u=Q
\]

的 geometry/source/shift resolvent anchors，并检查真实 Galerkin residual；最终仍以 completely held-out full-vs-ROM transient/steady audit 为准。

## 热系统

规定电流模式：

\[
M_r(g)\dot a
=-K_r(g)a
+\widehat q_{\rm vol}(g,c)
+q_{\rm wire}(a,g,c)
+f_T(g).
\]

wire resistance 不能把 `a` 当黑箱输入；必须先由 reduced temperature field 得到明确的 conductor-temperature functional，再进入 constitutive resistance law。

若工作点由电压源、补偿网络或负载决定，则显式求解 circuit equations，例如

\[
[\widehat Z_{\rm field}+R_{\rm wire}+Z_{\rm ext}]c=v_{\rm src},
\]

再把得到的 `c(a,t)` 送入 quadratic Joule layer。current-controlled 与 circuit-controlled 结果必须明确区分。

## Physics Gate

正式 dataset/training 前，至少必须通过：

1. Maxwell formulation、开放域/域扩展和 mesh convergence；
2. phasor convention、port units/sign/orientation 与 reciprocity；
3. `D_vol = X^H H_sigma X` 和 `D_out = Herm(Z_field)-D_vol` 的矩阵级非负/功率闭合；
4. spatial Joule、`D_vol` 和 modal `H_j` 的统一 Hodge 能量一致性；
5. terminal/source/return-path 或 charge-continuity consistency；
6. 几何无穿透、包封、边界裕量和连续守恒 source/material deposition；
7. thermal boundary/domain sensitivity；
8. complete current-induced source span + transient-aware thermal ROM；
9. wire-temperature functional 与 full-vs-ROM conductor-temperature audit；
10. filament conductor、AC resistance、feed/return path、海水 thermal convection 等模型假设的适用性；
11. frequency-transient 时间尺度与 operating-mode/circuit conditioning。

没有通过这些 Gate 时，降低 neural loss 没有物理意义。

## 训练与验证

只用 neural training split 构造 normalization 和 POD。validation/audit 至少报告：

- raw/POD latent reconstruction；
- decoded `Z_field / D_vol / H_j` error；
- joint-feasible projection correction；
- arbitrary complex-current voltage/Joule/total-loss error；
- matrix-level `D_out` consistency；
- thermal trajectory / steady-state error；
- circuit-controlled error（若启用）；
- completely held-out end-to-end geometry audit。

## 推理

```text
geometry
-> one neural forward
-> structured + joint energy-feasible decode
-> cache Z_field / D_vol / H_j
-> prescribed current or deterministic circuit solve
-> exact quadratic current contraction
-> true R_wire(temperature functional)
-> true thermal ROM integration / steady solve
```

同一 geometry 的多个时间、电流、外部电路和初值查询复用同一组 neural EM tensors。

## 理论文档

主文档：`main.tex`。理论只保留上述生产主链，不并列维护其他神经求解架构。
