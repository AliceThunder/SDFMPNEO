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

当前正式 source basis `S(g)` 是实值几何电流形状，所有端口幅值和相位都放在复向量 `c` 中。`v_field = -S^T E` 首先是与 source mode 功率共轭的 generalized reaction voltage；只有 source/terminal/feed-return 定义经过校准后，才把它解释为物理 terminal voltage。

体耗散矩阵定义为

\[
D_{\rm vol}=X^H H_\sigma X,
\qquad
P_{\rm vol}=\frac12c^H D_{\rm vol}c.
\]

thermal modal Joule tensor 必须从同一有限元 basis 和 quadrature 直接组装加权 loss bilinear form：

\[
[W_j]_{mn}=\int_\Omega \sigma\,\phi_j\,N_m\cdot N_n\,dx,
\qquad
H_j=X^H W_jX,
\]

\[
q_{{\rm vol},j}=\frac12\operatorname{Re}(c^HH_jc).
\]

不能对一般 consistent `H_sigma` 做任意单侧逐点加权后假设 Hermitian/energy consistency 自动保持；mass-lumped diagonal Hodge 只是上述统一定义的特殊实现。取 `phi = 1` 时，对应 weighted operator 必须退化为 `H_sigma`。

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

以及 surrogate implied outward-loss

\[
D_{\rm out}^{\rm imp}
=\operatorname{Herm}(Z_{\rm field})-D_{\rm vol}\succeq0.
\]

注意：`D_out^imp` 只用于 surrogate feasibility。truth 侧的 `D_out^phys` 必须从独立 Poynting-flux / absorbing-boundary / PML absorption bilinear form 计算，再验证

\[
\operatorname{Herm}(Z_{\rm field})
\approx D_{\rm vol}+D_{\rm out}^{\rm phys}.
\]

不能把差值定义成 truth outward loss 后再声称“功率闭合”。联合投影的 correction norm 必须报告；不能用大幅投影掩盖差的 raw surrogate。投影只修耗散块，不无故改变 reciprocal reactive block。

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

固定 geometry 下，volume heat 对 complex current 是 Hermitian quadratic form，因此完整 current-induced heat-source span 的实维数至多为 `n_ports^2`。basis builder 使用确定性的 Hermitian port-space combinations 完整覆盖该 span，不依赖大量随机 current samples。

当前 thermal theory 假定生产域内

\[
M_T(g)\succ0,\qquad K_T(g)\succ0.
\]

若 pure Neumann 等导致 `K` 有零模，必须单独处理 nullspace、能量积累和 steady-state solvability，不能继续原样使用 `K^-1` / 唯一稳态理论。

thermal basis 还必须覆盖受迫响应、允许的 initial-condition family 与目标 transient time scales。统一 resolvent 写成

\[
A_su=b,\qquad A_s=K+sM,
\]

source anchor 取 `b = Q`，initial-condition anchor 取 `b = M theta0`。自动 rank 的理论 residual 指标使用

\[
\eta_s^2
=\frac{r^TA_s^{-1}r}{b^TA_s^{-1}b}
=\frac{\|u-\tilde u\|_{A_s}^2}{\|u\|_{A_s}^2},
\]

而不是未经尺度化的 Euclidean residual。最终仍以 completely held-out full-vs-ROM transient/steady audit 为准。

如果输入 full initial temperature，默认做 `M`-orthogonal projection；如果直接输入任意 reduced `a0`，只有属于已定义并审计 initial-condition family 的状态才带 full-order accuracy claim。

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

`t = inf` 不是“找到任意 root”。带 temperature-dependent wire heating 或 circuit feedback 时可能存在多稳态、无稳定平衡或 thermal runaway；候选 steady state 必须通过 residual、constitutive-domain 和闭环 Jacobian stability 检查。

## Physics Gate

正式 dataset/training 前，至少必须通过：

1. Maxwell formulation、开放域/域扩展和 mesh convergence；
2. peak-phasor、port units/sign/orientation、reaction-voltage 与 terminal calibration；
3. truth `D_out^phys` 的独立计算，以及 `Herm(Z_field) ≈ D_vol + D_out^phys` 的矩阵级功率闭合；
4. `H_sigma / W_j / D_vol / H_j` 的统一有限元双线性型和 quadrature 一致性；
5. terminal/source/return-path 或 charge-continuity consistency；
6. 几何无穿透、包封、边界裕量和连续守恒 source/material deposition；
7. thermal boundary/domain sensitivity 与 `M,K` coercivity/nullspace policy；
8. complete current-induced source span + initial-condition family + transient-aware thermal ROM；
9. energy/dual-norm Galerkin resolvent error 与 full-vs-ROM trajectory；
10. wire-temperature functional 与 conductor-temperature audit；
11. filament conductor、AC resistance、feed/return path、海水 thermal convection 等模型假设的适用性；
12. frequency-transient 时间尺度与 operating-mode/circuit conditioning。

没有通过这些 Gate 时，降低 neural loss 没有物理意义。

## 训练与验证

只用 neural training split 构造 normalization 和 POD。validation/audit 至少报告：

- raw/POD latent reconstruction；
- decoded `Z_field / D_vol / H_j` error；
- joint-feasible projection correction；
- arbitrary complex-current voltage/Joule/total-loss error；
- independent truth `D_out^phys` 与 surrogate `D_out^imp` consistency；
- thermal resolvent/trajectory/steady-state error；
- initial-condition projection/homogeneous-response error；
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
-> true thermal ROM integration / stable steady solve
```

同一 geometry 的多个时间、电流、外部电路和初值查询复用同一组 neural EM tensors。

## 理论文档

主文档：`main.tex`。理论只保留上述生产主链，不并列维护其他神经求解架构。
