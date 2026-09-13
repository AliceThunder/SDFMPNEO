# 几何参数化电磁张量神经电热 ROM

本目录冻结唯一生产理论主线：

```text
validated single-frequency Maxwell truth physics
-> spatial Joule/source truth bank
-> transient/initial-condition-aware shared thermal basis
-> geometry-dependent Z_field / D_vol / Hermitian H_j
-> neural geometry surrogate
-> joint energy/modal-feasible physical decode
-> exact complex-current or circuit physics
-> true thermal ROM / ODE
-> T(t), Z(t), power, stable steady state
```

神经网络只学习几何到低维电磁参数的映射。它不学习 Maxwell field、时间轨迹、thermal operators、端口 current law 或外部 circuit law。

## 冻结的相量、端口和功率约定

采用 `exp(+i wt)` 与**峰值复相量**，所以所有 cycle-averaged power 带 `1/2`：

\[
P_{\rm in}=\frac12\operatorname{Re}(c^H v).
\]

正式 source basis `S(g)` 是实值几何电流形状，所有端口幅值/相位位于复向量 `c`。`v_field=-S^T E` 首先是与 source mode 功率共轭的 generalized reaction voltage；只有 terminal/feed-return/source normalization 已独立校准时才解释为物理 terminal voltage。

reciprocity 只在 reciprocal constitutive law 与 reciprocal boundary treatment 下硬编码；若未来材料/边界破坏 reciprocity，必须取消 `Z^T=Z` 约束。

## 统一电磁耗散与 Joule 定义

\[
D_{\rm vol}=X^H H_\sigma X,
\qquad
P_{\rm vol}=\frac12c^H D_{\rm vol}c.
\]

thermal modal Joule operator 从同一 FE basis/quadrature 直接组装：

\[
[W_j]_{mn}=\int_\Omega\sigma\,\phi_j\,N_m\cdot N_n\,dx,
\qquad
H_j=X^H W_jX,
\]

\[
q_{{\rm vol},j}=\frac12\operatorname{Re}(c^HH_jc).
\]

取 `phi = 1` 时 weighted operator 必须退化为 `H_sigma`。如果 EM/thermal 使用不同网格，则 cross-space transfer 必须冻结并验证 constant-preservation、积分一致性与 mesh convergence。

## 生产 surrogate 的联合物理可行集

神经代理学习

\[
g\rightarrow\{\widehat Z_{\rm field},\widehat D_{\rm vol},\widehat H_j\}.
\]

固定物理解码层要求

\[
D_{\rm vol}\succeq0,
\qquad
\operatorname{Herm}(Z_{\rm field})-D_{\rm vol}\succeq0,
\]

并利用同一空间非负 Joule density 推出的必要 Loewner bounds：

\[
\boxed{
\phi_j^{\min}D_{\rm vol}
\preceq H_j
\preceq
\phi_j^{\max}D_{\rm vol}
}
\]

其中 `phi_j_min/max` 在 conductivity-loss quadrature/support 上计算并随 thermal basis 版本冻结。可行投影保持 reciprocal reactive block 不变；`(Z,D)` 和 `H_j` 的 projection correction 必须单独报告。大投影意味着 raw surrogate 不合格，不能用安全层掩盖。

这些有限矩阵约束是强必要条件，但仍不是 exact spatial Joule realizability 的充分证明，因此 held-out truth audit 不能省略。

## outward power 必须独立验证

truth 侧的

\[
D_{\rm out}^{\rm phys}
\]

必须从独立 Poynting-flux、absorbing-boundary 或 PML absorption bilinear form 得到，然后验证

\[
\operatorname{Herm}(Z_{\rm field})
\approx D_{\rm vol}+D_{\rm out}^{\rm phys}.
\]

不能先定义 `D_out = Herm(Z)-D_vol` 再声称功率闭合。

surrogate 可以定义 implied outward loss

\[
D_{\rm out}^{\rm imp}
=\operatorname{Herm}(\widehat Z_{\rm field})-\widehat D_{\rm vol},
\]

但必须在 held-out geometry 上与独立 truth `D_out^phys` 比较。

PML/人工吸收只属于 outward/domain-truncation power，不进入真实 thermal heat；真实海水 loss、wire/internal loss 和 outward loss 使用互斥 region/physics masks。

## source / conductor 模型边界

实际 spiral 可能是 two-terminal open conductor，而不是数学上的闭合 loop。source 必须有明确 terminal orientation、normalization、return/feed path 或 charge/current continuity interpretation：

\[
\nabla\cdot J_s=-i\omega\rho_s.
\]

零半径 filament 的 self impedance 不能由 mesh size 隐式正则化。必须使用与物理 wire radius 绑定的 finite-support impressed current、stranded-conductor approximation 或明确的 regularized self-term；`Z_pp` 必须在固定物理 regularization 下 mesh-converge。

若 wire ohmic loss 由独立 `R_wire` 处理，同一铜损不得同时进入 `D_vol`。如果 internal reactance、proximity coupling 或非均匀 conductor heating 不可忽略，先升级 truth wire-impedance/heat model，再生成 dataset。

## 当前材料模型与时间尺度

当前体 Maxwell constitutive parameters 不随温度变化：

\[
A_{\rm em}=A_{\rm em}(g).
\]

固定几何只需一组 EM tensors。thermal state 只通过明确的 conductor-temperature functional 和 wire model 进入。

frequency-transient 分层要求 carrier/field-settling time 远短于 thermal/envelope time scale。若快速调制、宽带、多谐波或强 EM 非线性出现，当前单频理论失效。

circuit-controlled 模式还要求 circuit ring-up time 远短于 thermal/envelope time scale；否则必须显式引入 circuit dynamic states。

## Thermal basis 与初值

理论假定生产域内

\[
M_T(g)\succ0,\qquad K_T(g)\succ0.
\]

若 `K` 有 nullspace，必须重新处理 steady-state solvability 和 transient theory。

固定 geometry 下，current-induced spatial heat span 的实维数至多为 `n_ports^2`，因此用确定性的 Hermitian port-space combinations 完整覆盖，不依赖随机 current sampling。

统一 resolvent anchor：

\[
A_su=b,\qquad A_s=K+sM.
\]

source anchor 用 `b=Q`，initial-condition anchor 用 `b=M theta0`。rank criterion 使用

\[
\boxed{
\eta_s^2
=\frac{r^TA_s^{-1}r}{b^TA_s^{-1}b}
=\frac{\|u-\tilde u\|_{A_s}^2}{\|u\|_{A_s}^2}
}
\]

或有已知等价常数的 validated estimator，而不是裸 Euclidean residual。

full initial field 默认做 `M`-orthogonal projection；任意 reduced `a0` 只有属于已冻结/审计 initial-condition family 时才带 full-order accuracy 声明。

## Thermal 能量、正性与稳态

零源 ROM 在 `M_r,K_r > 0` 下有

\[
V=\frac12a^TM_ra,
\qquad
\dot V=-a^TK_ra\le0.
\]

这保证能量耗散，但**不自动保证 pointwise temperature positivity 或 maximum principle**。必须审计 `T_min/T_max` 与 conductor-temperature outputs；若工程要求严格点值正性/上界，需要 positivity/bound-preserving ROM 或 limiter。

`t=inf` 不是“找任意 root”。带 wire-temperature 或 circuit feedback 时可能出现多稳态、无稳定平衡或 thermal runaway。候选 equilibrium 必须通过 residual、constitutive-domain 和 Jacobian stability；若有多个稳定 equilibrium，`t=inf` 必须与给定初值/动态 circuit state 的 basin 一致，否则报告 ambiguity。

## current-controlled 与 circuit-controlled

规定电流时：

\[
M_r\dot a=-K_ra+q_{\rm vol}(g,c)+q_{\rm wire}(a,g,c)+f_T.
\]

若工作点由电压源、补偿网络或负载决定，则显式求解 circuit equations，例如

\[
[Z_{\rm field}+R_{\rm wire}+Z_{\rm ext}]c=v_{\rm src}.
\]

此时误差必须按 `Z -> c -> Joule -> temperature` 传播，并检查 near-resonance/high-Q conditioning。

## validated production domain

必须随模型冻结：

- geometry domain `G_prod` 与几何可制造性约束；
- current/circuit operating domain；
- thermal/constitutive state domain；
- initial-condition family；
- frequency、source regularization、loss-region masks 和 physics/basis versions。

域外输入不能继承域内 validated accuracy 声明。

## Physics Gate

正式 dataset/training 前必须覆盖：开放域/formulation/mesh convergence、端口 reaction-voltage calibration、独立 matrix power balance、统一 `H_sigma/W_j/D_vol/H_j` assembly、modal Loewner bounds、source/terminal continuity、physical filament regularization、无双重计损、thermal coercivity/positivity policy、energy-norm ROM error、initial-condition response、full-vs-ROM trajectories、wire-temperature model、stable-equilibrium branches、frequency/circuit timescale 与模型适用范围。

没有通过这些 Gate 时，降低 neural loss 没有物理意义。

## 推理

```text
check production domain
-> geometry neural forward
-> joint energy/modal-feasible decode
-> cache Z_field / D_vol / H_j
-> prescribed-current or deterministic circuit physics
-> exact quadratic Joule contraction
-> true wire-temperature constitutive law
-> true thermal ROM integration / validated steady-state logic
```

理论只保留上述生产主链，不并列维护其他神经求解架构。
