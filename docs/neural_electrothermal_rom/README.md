# 几何参数化电磁张量神经电热 ROM

本目录冻结唯一生产理论主线：

```text
validated single-frequency Maxwell truth physics
-> canonical geometry-aware thermal basis generator Phi(g)
-> geometry-dependent Z_field / D_vol / Hermitian H_j
-> neural geometry surrogate
-> joint energy/modal-feasible physical decode
-> exact complex-current or circuit physics
-> geometry-aware true thermal ROM / ODE
-> T(t), Z(t), power, stable steady state
```

核心分工固定为：**复杂的 geometry→EM tensor 映射交给神经网络；已知的几何平移/旋转对 thermal modes 的作用由确定性坐标变换处理。** 网络不学习 Maxwell field、thermal trajectory、thermal operators、端口 current law、外部 circuit law，也不学习 thermal basis 本身。

## 冻结的相量、端口和功率约定

采用 `exp(+i wt)` 与峰值复相量，因此 cycle-averaged power 均带 `1/2`：

\[
P_{\rm in}=\frac12\operatorname{Re}(c^H v).
\]

正式 source basis `S(g)` 是实值几何电流形状，端口幅值/相位全部位于复向量 `c`。`v_field=-S^T E` 首先是 generalized reaction voltage；只有 terminal/feed-return/source normalization 已独立校准时才解释为物理 terminal voltage。

## Geometry-aware thermal ROM

完整热模型为

\[
M_T(g)\dot\theta+K_T(g)\theta
=Q_{\rm vol}(g,c)+Q_{\rm wire}(\theta,g,c)+F_T(g).
\]

不再假设一个固定全局 basis 能覆盖大范围移动几何。生产 ROM 改为

\[
\boxed{\theta(x,t;g)\approx\Phi(x;g)a(t;g)}
\]

并采用固定列语义的 canonical 分块结构

\[
\boxed{
\Phi(g)=
[\Phi_{\rm bg},\;\mathcal T_{\rm tx}(g)\Psi_{\rm tx},\;\mathcal T_{\rm rx}(g)\Psi_{\rm rx}].
}
\]

其中 `background` modes 负责远场/慢扩散，TX/RX local modes 在各自局部坐标中定义；`T_tx/T_rx` 首版只执行已知的 rigid translation/rotation 与守恒插值。尺寸变化先由真实 `M_T(g),K_T(g)` 投影和 held-out audit 吸收；只有验证证明仍不足时才增加 scale-aware transport，而不是先引入 neural basis、Grassmann interpolation 或其他复杂结构。

mode 数量与索引语义在所有 geometry 上固定。例如 `1..r_bg` 永远是 background modes，随后是 TX local modes、RX local modes。允许 mode 的空间位置/姿态随 `g` 变化，但禁止 mode index 在不同 geometry 间交换物理含义。

对静态 geometry：

\[
M_r(g)=\Phi(g)^TM_T(g)\Phi(g),\qquad
K_r(g)=\Phi(g)^TK_T(g)\Phi(g),
\]

并要求 `Phi(g)` 满列秩、`M_r(g),K_r(g)` 正定且条件数在生产域内可接受。真实 reduced operators 每个 geometry 确定性组装，不由网络预测。

若未来允许 `g=g(t)`，则由于 `theta=Phi(g(t))a(t)`，动力学会出现额外 transport term

\[
\Phi^TM_T\dot\Phi\,a,
\]

因此当前静态 geometry ROM 不能直接冒充运动几何模型。

## Thermal 时间尺度策略

resolvent anchor 仍写成

\[
(K+sM)u=b,\qquad s=1/\tau,
\]

但 anchor 的最短时间尺度必须与 thermal spatial discretization 和目标输出相容。若

\[
\ell_d(\tau)=\sqrt{\alpha\tau}
\]

远小于可解析网格尺度，则要求 ROM 精确拟合该 anchor 主要是在学习离散局部 source placement，而不是可解析的热扩散，不应默认纳入生产 basis target。

当前推荐先覆盖可解析的短/中时尺度（例如约 `0.1, 1, 10 s`，最终以 mesh/diffusion audit 为准）并保留 `s=0` steady anchor。长时间查询无需增加 `1000 s`、`10000 s` 等专门 basis anchor；稳定 reduced ODE 通过连续/链式时间推进自然达到长时间状态，`t=inf` 直接求经稳定性验证的 equilibrium。

## 统一电磁耗散与 geometry-dependent modal Joule

\[
D_{\rm vol}(g)=X(g)^H H_\sigma(g)X(g),
\qquad
P_{\rm vol}=\frac12c^H D_{\rm vol}c.
\]

对当前 geometry 的第 `j` 个 thermal mode `phi_j(g)`，直接由同一 FE basis/quadrature 组装

\[
[W_j(g)]_{mn}
=\int_\Omega\sigma(x;g)\phi_j(x;g)N_m\cdot N_n\,dx,
\]

\[
\boxed{H_j(g)=X(g)^H W_j(g)X(g)},
\qquad
q_{{\rm vol},j}=\frac12\operatorname{Re}(c^HH_j(g)c).
\]

因此 NN 仍学习固定维度的

\[
g\rightarrow\{Z_{\rm field}(g),D_{\rm vol}(g),H_1(g),\ldots,H_r(g)\},
\]

但 `H_j` 的定义使用当前 geometry 的 canonical mode `phi_j(g)`。

## 生产 surrogate 的联合物理可行集

固定物理解码层要求

\[
Z_{\rm field}^T=Z_{\rm field},\qquad
D_{\rm vol}\succeq0,
\qquad
\operatorname{Herm}(Z_{\rm field})-D_{\rm vol}\succeq0.
\]

对每个 geometry，从 `Phi(g)` 在 conductivity-loss quadrature/support 上确定性计算

\[
\phi_j^{\min}(g),\qquad\phi_j^{\max}(g),
\]

并施加

\[
\boxed{
\phi_j^{\min}(g)D_{\rm vol}(g)
\preceq H_j(g)
\preceq
\phi_j^{\max}(g)D_{\rm vol}(g).
}
\]

这些 bounds 不再是一个全局固定常数表，而是 geometry-aware basis generator 的确定性输出。可行投影 correction 必须单独报告；大 correction 不能被安全层掩盖。

## outward power 独立验证

truth 侧的 `D_out^phys` 必须由独立 Poynting-flux、absorbing-boundary 或 PML absorption bilinear form 得到，并验证

\[
\operatorname{Herm}(Z_{\rm field})
\approx D_{\rm vol}+D_{\rm out}^{\rm phys}.
\]

禁止把 `Herm(Z)-D_vol` 作为 truth outward power 的定义后再自证。surrogate 可定义 implied outward loss，但必须在 held-out geometry 上与独立 truth 比较。

## 初值、wire temperature 与 ODE

full initial field 使用当前 geometry 的 mass projection

\[
\boxed{
a_0=[\Phi(g)^TM_T(g)\Phi(g)]^{-1}\Phi(g)^TM_T(g)\theta_0.}
\]

wire-temperature functional 同样使用 `Phi(g)`：

\[
\bar\theta_p=\ell_p^{(T)T}(g)\Phi(g)a.
\]

规定电流时：

\[
\boxed{
M_r(g)\dot a=-K_r(g)a
+q_{\rm vol}(g,c)
+q_{\rm wire}(a,g,c)+f_T(g).
}
\]

电压源/补偿网络/负载控制时，显式求解 circuit equations，例如

\[
[Z_{\rm field}(g)+R_{\rm wire}(a,g)+Z_{\rm ext}]c=v_{\rm src}.
\]

## 数据依赖与验证

正确顺序固定为：

1. Physics Gate 与 validated Maxwell/Joule truth physics；
2. 构建 canonical background/TX/RX thermal mode library，并冻结 deterministic basis generator `B:g->Phi(g)`、固定 mode ordering 和 transport/interpolation rule；
3. 在每个 geometry 构造 `Phi(g), M_r(g), K_r(g), phi_min/max(g)`，再生成相应 `H_j(g)` labels；
4. 对 `Z/D/H` 做结构编码、POD 和 neural training；
5. 用完全未参与 thermal-library tuning 与 neural training 的 held-out geometries 做 end-to-end audit。

thermal validation 不允许只看一个 aggregate worst number。必须按 `geometry / source type / mode block / time scale` 报告 resolvent error，同时报告 anchor 的绝对能量/功率尺度，避免一个几乎无功率的方向仅因相对误差大就主导诊断。

geometry-aware ROM 必须在 held-out geometries 上直接比较 full-vs-ROM trajectories，至少覆盖可解析短时、中间时间、长时间和 steady outputs，以及 `T_min/T_max`、wire temperature 与安全相关量。

## Physics Gate

正式 dataset/training 前必须覆盖：开放域/formulation/mesh convergence、端口 reaction-voltage calibration、独立 matrix power balance、统一 `H_sigma/W_j/D_vol/H_j` assembly、geometry-dependent modal bounds、source/terminal continuity、physical filament regularization、无双重计损、thermal coercivity、basis transport continuity/full-rank/conditioning、可解析 time-scale selection、initial-condition response、full-vs-ROM trajectories、wire-temperature model、stable-equilibrium branches、frequency/circuit timescale 与模型适用范围。

## 推理

```text
check production domain
-> deterministic Phi(g), M_r(g), K_r(g), modal bounds
-> geometry neural forward
-> joint energy/modal-feasible decode
-> cache geometry thermal + EM reduced objects
-> prescribed-current or deterministic circuit physics
-> exact quadratic Joule contraction
-> true wire-temperature constitutive law
-> true geometry-aware thermal ROM integration / validated steady-state logic
```

理论只保留上述生产主链，不并列维护固定全局 thermal basis、neural thermal basis、trajectory NN 或 neural Maxwell 架构。