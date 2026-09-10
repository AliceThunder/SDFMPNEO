# 自动热秩 + Finite-Horizon Fixed Analytic Response Network

当前 SDF-MPNEO 的生产训练路径只有一条：

\[
\boxed{
\text{自动热秩}
\rightarrow
\text{固定最大解析响应网络}
\rightarrow
\text{有限时间物理残差训练}
\rightarrow
\text{restart consistency}
\rightarrow
\text{独立验证剪枝}
\rightarrow
\text{分段 rollout}
}
\]

训练不使用瞬态解标签。热动力学目标始终来自真实降阶物理场

\[
\dot a=F(a,G,U),
\qquad
R_{phys}=\dot{\hat a}-F(\hat a,G,U).
\]

## 1. 热空间

温度场写为

\[
T(x,t)\approx T_{ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),
\qquad
K\phi_j=\lambda_jM\phi_j.
\]

默认 `THERMAL_RANK=None`。自动选择器通过全阶电磁平衡在确定性锚点上估计模态响应包络

\[
E_j\approx \max\frac{|q_j|}{\lambda_j}.
\]

探测会继续扩大到更高热谱，必要时一直到完整离散热空间；不存在“探测到 32 阶仍不确定就直接把 full rank 带进网络”的固定上限。只有完整诊断后确实无法满足尾部标准，才使用全热空间。

有限锚点 envelope 只是一种可复现的数值/物理选秩判据，报告中保持 `certified_continuous_domain=false`。当严格 thermal-tail certificate 所需输入齐全时，仍可使用严格证书路径。

自动选秩同时生成 restart-state box：初始扰动允许量与安全放大的稳态响应包络共同给出每个保留模态的 `initial_lower/upper`。因此第二段及以后使用上一段终态作为初值时，不会默认只在环境附近的任意小盒子上训练。

## 2. 单段有限时间解析网络

网络只定义

\[
\hat\Phi_t(a_0,G,U),\qquad 0\le t\le H,
\]

其中 `H=max_response_time`，默认入口使用 `100 s`。

每个响应通道绑定真实热衰减率：

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}=S_{jc}^{(\ell)},
\qquad h_{jc}^{(\ell)}(0)=0.
\]

第一层 source 包含

\[
S=b+W_xx+\sum_r g_r^{xx}\alpha_r(u_r^Tx)(v_r^Tx),
\]

后续层增加上一层响应、输入–响应和响应–响应作用：

\[
S=b+W_hH+\Phi_{xH}(x,H)+\Phi_{HH}(H,H).
\]

每层都有显式 bias；动态层不使用 ReLU。非线性由低秩乘法项提供，时间记忆由

\[
\mathcal R_\lambda=(\partial_t+\lambda)^{-1}
\]

提供。

网络内部只允许有限单段时间。`FixedAnalyticResponseNetwork.evaluate()` 对 `t>H` 或 `t=inf` 明确拒绝；长时间组合和稳态由更高层模型负责。

## 3. 两阶段无标签训练

第一阶段只训练 governing-equation residual，直到

\[
\max\|R_{phys}\|\le\varepsilon.
\]

然后开启 restart/semigroup consistency：

\[
R_{sg}=\frac{
\hat\Phi_{t_1+t_2}(a_0)
-\hat\Phi_{t_2}(\hat\Phi_{t_1}(a_0))
}{H},
\qquad t_1+t_2\le H.
\]

使用 rate defect 是为了与物理残差保持相同的状态/时间量纲。参数 Jacobian 使用解析链式法则

\[
J_{sg}=J_\theta\Phi_{t_1+t_2}-\left(J_\theta\Phi_{t_2}+J_{a_0}\Phi_{t_2}J_\theta\Phi_{t_1}\right).
\]

先物理、后联合约束可以避免随机初始网络因为“错误动力学也可能自洽”而妨碍物理学习。

训练和独立验证都要求 physics residual 与 restart-rate defect 不超过同一用户容差，报告同时保留未经除以 `H` 的 state defect。

## 4. 长时间 rollout

对任意有限查询时间 `T`，模型自动分段：

\[
T=N H+\Delta t,
\qquad 0\le\Delta t<H,
\]

\[
\hat\Phi_T=\hat\Phi_{\Delta t}\circ\underbrace{\hat\Phi_H\circ\cdots\circ\hat\Phi_H}_{N\text{ 次}}.
\]

例如 `H=100 s`、查询 `350 s`：

```text
0 -> 100 -> 200 -> 300 -> 350
```

每段结束后，终态成为下一段新的 `a0`。除非显式设置 `allow_extrapolation=True`，每次 restart 前都会检查状态仍在训练的 restart-state box 内。

## 5. 稳态

真正的稳态不通过网络的 `t=inf` 求值，而是直接对当前物理场做 damped Newton：

\[
F(a_\infty,G,U)=0.
\]

接口为 `model.steady_state(...)`。为方便调用，`model.predict(float("inf"), ...)` 只作为路由语法，内部仍调用物理稳态求解器，不调用无限时间网络表达。

## 6. 连续结构稀疏化

训练开始时最大网络已经固定。channel gate、`xx/xH/HH` component gate、bias、投影和低秩因子都是连续参数，不进行候选结构枚举。

达到联合容差后，用残差 Jacobian 估计 gate 的局部影响，尝试将低影响结构置零，并在 physics + restart 的训练/验证集合上重新计算完整非线性残差。只有仍满足容差的剪枝才保留。

因此不存在 `candidate_search`、Grow、Enrich 或 Split。

## 7. 几何族

连续几何族共享参考热坐标、跨几何 EM 基和同一个 finite-horizon network，但每个几何仍使用自己的

\[
M_r(G),\quad K_r(G),\quad A_{EM}(G,a).
\]

几何模型的长时间 rollout 与稳态求解同样分别使用网络分段组合和该几何的真实降阶物理场。

## 8. 当前格式

当前模型格式只支持 segmented fixed-network 架构；历史 DAG、旧 fixed-network 参数排列和旧无限时间模型不转换。当前格式的 checkpoint 可以正常 resume。
