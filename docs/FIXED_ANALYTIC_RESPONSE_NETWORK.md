# 自动热秩 + Fixed Analytic Response Network

当前 SDF-MPNEO 训练架构只有一条路径：

\[
\boxed{\text{自动热秩}\rightarrow\text{固定最大解析响应网络}\rightarrow\text{连续残差训练}\rightarrow\text{独立验证剪枝}}
\]

控制方程残差为

\[
R(a,\dot a,G,U)=\dot a-F(a,G,U).
\]

训练不使用瞬态解标签。只有训练配点和独立验证配点的最大残差都低于用户容差，才报告 `numerically_converged`。

## 1. 热空间选择

温度场写为

\[
T(x,t)\approx T_{ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x).
\]

`r` 是空间热降阶维数，与网络 depth 无关。默认 `THERMAL_RANK=None`。

`automatic_physics_envelope` 从低阶热谱逐步扩展，并在确定性的物理锚点上用全阶电磁平衡估计

\[
E_j\approx \max\frac{|q_j|}{\lambda_j}.
\]

接受部分秩必须同时满足已解析尾部和探测边界衰减条件。若到 `max_probe_rank` 仍无法确认尾部，结果回退完整离散热空间，而不是接受一个未解析的部分秩。

默认 envelope 是有限锚点判据，所以报告明确写 `certified_continuous_domain=false`。如果 `initial_temperature_deviation_free`、`source_dual_bound`、`requested_state_tolerance` 三项严格证书输入同时存在，则直接使用 theorem-level thermal-tail certificate。

## 2. 解析响应层

记归一化输入

\[
x=(a_0,G,U).
\]

每个响应通道绑定一个真实热衰减率：

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}=S_{jc}^{(\ell)},
\qquad h_{jc}^{(\ell)}(0)=0.
\]

第一层 source 为

\[
S=b+W_xx+\sum_r g_r^{xx}\alpha_r
(u_r^Tx)(v_r^Tx).
\]

后续层增加上一层响应、输入–响应和响应–响应作用：

\[
\begin{aligned}
S={}&b+W_hH\\
&+\sum_r g^{xH}_{\ell r}\beta_r(p_r^Tx)(q_r^TH)\\
&+\sum_r g^{HH}_{\ell r}\gamma_r(s_r^TH)(t_r^TH).
\end{aligned}
\]

每一层都有显式 bias。动态层不使用 ReLU；非线性由低秩乘积产生，时间记忆由解析响应算子

\[
\mathcal R_\lambda=(\partial_t+\lambda)^{-1}
\]

产生。

## 3. 不再进行离散结构搜索

训练开始时最大网络已经固定。所有 channel gate、`xx/xH/HH` component gate、bias、投影和低秩因子都是连续参数。

训练循环只做：

\[
\theta_{k+1}=\theta_k+\Delta\theta,
\]

其中 `Δθ` 由硬残差加权的 LM/Gauss–Newton 子问题得到。接受步必须重新计算完整非线性物理残差，并优先改善最大残差。

不存在：

```text
candidate enumeration
Grow
Enrich
Split
candidate_search
```

## 4. 残差验证剪枝

达到目标后，对当前 gate 的残差 Jacobian 列估计局部影响：

\[
I_i\approx |g_i|\max_p\left\|\frac{\partial R_p}{\partial g_i}\right\|_2.
\]

程序从低影响 gate 开始尝试置零，并在训练集 + 独立验证集上重新计算完整非线性残差。只有仍满足用户容差的剪枝才保留。

因此最终有效 depth、channel 数和低秩 rank 是残差验证后的结果，而不是预先指定的最终网络大小。

## 5. 解析时间闭包

网络所有动态信号保持在有限指数–多项式闭包：

\[
t^k e^{-\mu t}.
\]

加法、乘法和一阶响应保持闭合；共振产生更高阶 `t^k` 因子。有限超长时间采用指数对数域保护，`t=inf` 直接计算稳态极限。

因此训练窗口只定义残差采样范围，不是推理时间上限。

## 6. 几何族

连续几何族使用同一个固定网络和同一个参考热坐标系。几何变化后，物理残差仍使用对应几何的

\[
M_r(G)\dot a+K_r(G)a=q_r(G,a,U).
\]

同时使用对应几何的电磁算子；不会把参考几何的 `M/K/EM` 算子错误复用到其他几何。

## 7. 持久化

当前模型只保存：

- 当前 fixed analytic network metadata；
- 当前参数向量；
- 热/电磁空间基和物理离散数据；
- 当前训练域、训练报告和自动热秩报告。

加载时 model format 和 fixed-network format 必须精确匹配当前实现。历史 DAG、历史 fixed-network 参数排列和旧 checkpoint 不做升级或转换。

同一当前格式的 checkpoint 可以正常继续训练，这是 resume 功能，不是历史兼容层。
