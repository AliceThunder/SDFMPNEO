# SDF-MPNEO 理论修正版与论文级验收协议

本文是当前科研实现的**理论勘误与验收基线**。如 `SDFMPNEO_theory.tex` 中旧表述与本文冲突，以本文和可执行的 `certification/proof_composer.py` 为准，直到主理论稿下一次整体重排。

当前物理范围固定为：**磁准静态电磁场 + 固体/静止介质导热**。海水保留为显式导电、导热空间材料；本模型有意不包含流体速度、自然/强迫对流或 CFD。因而本文所有正确性结论都针对这一已声明模型，不把未建模的流动误差吸收到代理误差里。

## 1. 统一误差传播：修正 thermal-ROM 与 residual 的量纲混合

旧理论稿曾把 thermal spectral tail、热动力学 residual 和由电磁误差传播得到的热源误差直接相加再做时间积分。该写法量纲不一致，现正式废止。

设全离散温度与保留热子空间解满足

\[
T_h=T_b+\Phi_r a_r+T_{\rm tail},
\]

解析网络给出 `a_theta`。总状态误差应拆为

\[
\boxed{
\|T_h(t)-T_\theta(t)\|
\le \eta_{T\text{-ROM}}(t)+C_\Phi\|a_r(t)-a_\theta(t)\|.
}
\]

其中 `eta_T-ROM` 是**直接状态误差项**，不进入 residual 的时间积分。

保留坐标的动力学误差满足

\[
\dot e=F(a_r)-F(a_\theta)-R_\theta+\delta_F,
\]

`R_theta = dot(a_theta)-F_model(a_theta)`，`delta_F` 是 constitutive / algebraic / EM-ROM / mesh / outer-domain 等误差先经过有证明的灵敏度传播后得到的**热 vector-field / residual 单位**误差。

若在所声明域中存在 contractivity 下界 `kappa>0`，则

\[
\boxed{
\|e(t)\|
\le
\int_0^t e^{-\kappa(t-s)}
\left(\|R_\theta(s)\|+\eta_F(s)\right)\,ds.
}
\]

从而对一致上界 `eta_res` 有

\[
\|e(t)\|\le \frac{1-e^{-\kappa t}}{\kappa}\eta_{res},
\]

长期统一界为 `eta_res/kappa`。若 `kappa=0`，增益连续退化为 `t`；若 `kappa<0`，只能返回有限时间 Gronwall/比较方程界，不能声称长期认证。

当前代码中的 `compose_electrothermal_error_certificate()` 已按这一**typed propagation**实现：residual 节点先统一到 thermal-residual norm，thermal projection/tail 作为 state 节点在积分之后相加。任何 field-energy、heat-source、thermal-state 三种量都不能因为“都是有限数字”而直接相加。

## 2. 跨几何 residual：质量形式与当前 vector-field 形式严格等价

跨几何保留热方程是

\[
M_r(G)\dot a+K_r(G)a=q_r(G,a,U),
\]

其中 `M_r(G)` 对整个合法 chart 均为对称正定。当前训练 residual 为

\[
r_v=\dot a-M_r(G)^{-1}\{-K_r(G)a+q_r(G,a,U)\}.
\]

原始质量形式 residual 为

\[
r_M=M_r(G)\dot a+K_r(G)a-q_r(G,a,U).
\]

因此严格有

\[
\boxed{r_M=M_r(G)r_v},\qquad r_v=0\Longleftrightarrow r_M=0.
\]

`AffineTetrahedralGeometryChart.certify_box()` 已对**整个连续几何参数盒**证明 P1 mass quadratic-form 比例

\[
\alpha_M M_c\preceq M(G)\preceq\beta_M M_c.
\]

限制到共享热子空间后 Löwner 顺序保持：

\[
\alpha_M M_{r,c}\preceq M_r(G)\preceq\beta_M M_{r,c}.
\]

因此若

\[
m_- = \alpha_M\lambda_{\min}(M_{r,c}),\qquad
m_+ = \beta_M\lambda_{\max}(M_{r,c}),
\]

则对整个连续几何盒严格成立

\[
\boxed{
\frac{\|r_M\|_2}{m_+}\le \|r_v\|_2\le\frac{\|r_M\|_2}{m_-}.
}
\]

这解决了“GUI 的 absolute residual 是否与真实质量形式方程同一个零点、尺度如何关联”的问题。当前 `1e-5` 仍是 modal vector-field residual 的数值目标，不应直接解释成 Kelvin 误差；但它现在有明确的、可计算的质量形式范数等价常数。

可执行接口：

```python
from sdfmpneo.certification import certify_geometry_mass_residual_equivalence
certificate = certify_geometry_mass_residual_equivalence(model)
```

## 3. 变几何稳定性：必须使用 M_r(G) 能量范数

对固定查询几何 `G`，`M_r(G)` 在整条热轨迹上不随时间改变。令

\[
J_F=\frac{\partial F}{\partial a}.
\]

在热质量范数

\[
\|e\|_M^2=e^TM_r e
\]

下，正确的 logarithmic growth rate 是

\[
\boxed{
\mu_M(J_F)=
\lambda_{\max}\left(
M_r^{-1/2}\,\operatorname{sym}(M_rJ_F)\,M_r^{-1/2}
\right).
}
\]

局部 contractivity margin 定义为

\[
\boxed{\kappa_M=-\mu_M(J_F)}.
\]

`kappa_M>0` 表示该点在质量能量度量下局部收缩。参考质量正交坐标 `M=I` 时退化为原来的 `-lambda_max(sym(J_F))`。

当前实现新增 exact pointwise diagnostic 与独立 held-out sampled report。**抽样全为正仍不是连续域证明**。要声称 `t=infinity` 具有统一物理误差证书，必须进一步由连续 thermal/current/geometry derivative bounds 或 branch-and-bound 给出

\[
\inf_{(G,a,U)\in\mathcal D}\kappa_M(G,a,U)>0.
\]

在这一步闭合之前：

- 任意有限/无限时间的解析网络求值在计算上仍是定义良好的；
- `infinity` 的物理解读仅在真实 closed thermal system 具有稳定平衡时成立；
- sampled contractivity 是回归证据，不允许命名为 continuous certificate。

## 4. 收敛性定理的正确强度

### 4.1 EM residual-Riesz reduction

有限候选集合上的 residual upper bound 是证书；有限锚点密度本身不能变成连续参数盒证书。连续域结论必须由 `certification/em_domain.py`、`nonlinear_em_domain.py`、`geometry_domain.py` 一类 interval/analytic bound 闭合。

关于 residual-Riesz basis enrichment 的全局收敛，应表述为**条件命题**：若参数域紧致、enrichment 方向在每个未解析点非零、嵌套空间在解流形上稠密，并且每一步残差上界严格闭合，则最坏 residual certificate 收敛到独立离散/本构误差地板。没有这些假设的实现不得宣称无条件全局收敛。

### 4.2 Analytic residual-grown topology

同理，analytic response dictionary 的 convergence 目前是**under-density-assumptions proposition**，不是已经对当前有限 `max_degree/max_parent_responses/max_realization_dimension` 配置证明的无条件定理。

当前工程算法仍保持严格的单步性质：

1. candidate tangent 只提出方向/初始权重；
2. 完整 nonlinear residual 重新计算；
3. 只有实际 objective 下降才接受；
4. budget exhausted / stalled 不会写成 converged。

论文中应把“单步单调接受性质”与“字典极限稠密性”分开陈述。

## 5. 跨几何 held-out full-EM transient 验证

几何代理现在必须能够在**完全未参与训练/EM anchor/seed fit**的几何上做独立时间轨迹对照：

```python
from sdfmpneo.certification import validate_geometry_trajectory

result = validate_geometry_trajectory(
    model,
    [0.0, 1.0, 100.0, 1e4, 1e5],
    geometry=held_out_geometry,
    a0=[...],
    operating=[...],
    full_electromagnetics=True,
)
```

参考路径为

```text
queried geometry
 -> full sparse nonlinear EM equilibrium at every RHS evaluation
 -> same geometry M_r(G), K_r(G)
 -> independent Radau thermal integration
```

这个验证**永远不参与训练**。

它隔离并检查：

- analytic DAG evolution error；
- shared EM-ROM error 对热轨迹的影响；
- geometry conditioning 的联合泛化。

它**不**检查：

- thermal-rank truncation；
- mesh discretization；
- outer-domain truncation；
- 未建模物理。

所以论文必须另做 thermal-rank、mesh-size、outer-radius convergence study。

## 6. 论文级验证矩阵

正式宣称方法可靠前，至少执行以下矩阵。

### 6.1 Held-out parameter cases

- 几何中心附近、各轴附近和多个随机内部点；
- 不得与 EM basis anchors、seed axis probes、training/validation Halton points 重合；
- 多组 `a0`、双端口电流组合；
- 早期 logarithmic time、中期、训练窗末端、适度时间外推、稳态诊断。

记录：maximum coordinate error、maximum nodal-temperature error、full-EM vs EM-ROM discrepancy、physics residual、local M-contractivity margin、wall-clock。

### 6.2 Spatial/reduction convergence

至少比较：

- `thermal_rank = 2, 4, ...` 直到关注输出稳定；
- 网格细化至少三级；
- 海水外域半径至少三级；
- EM energy target 至少 `1e-4, 1e-5, 1e-6` 或更严格。

不能用更小的 analytic residual 掩盖 thermal-rank/mesh/outer-domain error。

### 6.3 Ablation

内部消融至少包含：

- full SDF-MPNEO；
- 关闭 physics seed，仅 residual growth；
- physics seed 后不允许 topology growth；
- 固定节点预算 vs adaptive seed；
- analytic DAG + EM ROM 与 ROM + Radau time marching；
- shared geometry surrogate 与每几何独立模型的离线/在线成本比较。

任何消融都保持相同物理模型、参数域和误差指标，不能通过放宽 tolerance 获得速度优势。

### 6.4 External baselines

论文层面建议对比 PI-DeepONet/PINO 类 physics-informed operator、Laplace/pole-residue neural operator，以及标准 ROM + implicit integrator。外部方法不是本仓库核心依赖；若实现不可复现，应明确版本、网络规模、训练点数、优化预算、硬件和停止准则。

SDF-MPNEO 的创新主张不应是“首次无标签”“首次跨几何”“首次指数/Laplace”。应限定为系统组合：

```text
snapshot-free certified EM ROM
+ quasi-static multiphysics closure
+ shared geometry/thermal coordinates
+ exact analytic response DAG
+ residual-grown topology
+ direct arbitrary-time / stationary evaluation
+ typed a-posteriori error propagation
```

## 7. 现在可以与不可以声称什么

可以声称：

- 电磁 physical-energy coercivity theorem 与 residual-to-state bound 数学成立；
- 当前 geometry chart 的非退化与 P1 mass-form ratio 可覆盖连续几何盒；
- vector-field residual 与 mass residual 零点严格一致，并有连续盒范数等价界；
- analytic DAG 保持初值、具有解析时间导数、无需在线 thermal time marching；
- training 不使用 transient solution labels；
- held-out full sparse EM + Radau 验证可以完全独立于训练执行。

在没有对应 proof/run 之前不可以声称：

- 默认 10D 几何盒的 shared EM basis 已在连续域被 `1e-6` 全域认证；
- 默认 `thermal_rank=2` 已有 thermal-tail certificate；
- 默认 `1e-5` physics residual 等于 `1e-5 K` 或某个固定相对误差；
- sampled positive `kappa_M` 等于连续域稳定性证书；
- 当前有限 analytic dictionary 已有无条件全局 convergence theorem；
- 未实际运行的 full UWPT wall-clock / baseline 结果已经成立。

这组限制不是弱化理论，而是把“已证明”“已数值验证”“条件性命题”“待验收项”分层，避免论文中的强结论超过实现证据。
