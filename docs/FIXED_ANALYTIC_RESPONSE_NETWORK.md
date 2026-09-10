# 自动热秩 + 固定最大解析响应网络

fresh SDF-MPNEO 训练采用 **自动热空间选择 + 固定最大容量解析响应网络 + 残差驱动连续稀疏化**。正常训练不再枚举候选神经元，也不再执行 Grow / Enrich / Split / `candidate_search`。

物理残差不变：

\[
R(a,\dot a,G,U)=\dot a-F(a,G,U).
\]

只有训练点和独立验证点的最大残差都达到用户设置的容差，才报告数值收敛。

## 1. 热秩不再写死

三维温度场写成

\[
T(x,t)\approx T_{ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x).
\]

`r` 是空间物理降阶维数，不是神经网络层数。默认 `run.py` 设置 `THERMAL_RANK=None`，程序从低阶热谱开始逐步扩大，并用真实全阶电磁平衡得到各模态的稳态响应包络

\[
E_j\approx \max_{anchors}\frac{|q_j|}{\lambda_j}.
\]

选择最小前缀，使已解析的高阶尾部低于

\[
\varepsilon_{abs}+\varepsilon_{rel}\|E\|_2,
\]

并要求当前探测谱的最高几阶已经明显衰减。若达到 `max_probe_rank` 仍无法确认尾部，程序不会把一个未解析的部分阶数当成可靠结果，而会退回完整热空间路径。

默认物理包络使用有限的电流/温度锚点，因此报告中明确写 `certified_continuous_domain=false`。如果用户能够提供严格的 `initial_temperature_deviation_free`、`source_dual_bound` 和 `requested_state_tolerance`，原有基于首个舍弃特征值下界的 thermal-tail certificate 路径仍然保留。

自动选出 `r` 后，训练初态模态域也自动展开到 `r` 维；默认无需再写两个固定的 `a0`。

## 2. 响应神经元

静态输入记为

\[
x=(a_0,G,U).
\]

每个响应通道绑定一个物理热衰减率并满足

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}=S_{jc}^{(\ell)},
\qquad h_{jc}^{(\ell)}(0)=0.
\]

第一层包含偏置/线性输入和低秩二次项；后续层包含显式 bias、上一层响应、输入–响应和响应–响应非线性：

\[
\begin{aligned}
S={}&b+W_xx+W_hH\\
&+\sum_r \alpha_r(u_r^Tx)(v_r^Tx)\\
&+\sum_r \beta_r(p_r^Tx)(q_r^TH)\\
&+\sum_r \gamma_r(s_r^TH)(t_r^TH).
\end{aligned}
\]

因此一个响应神经元仍然拥有多个 source coefficient；旧模型“一条响应状态可以有多个系数”的表达能力没有丢失。

动态层不使用 ReLU。非线性由 `x^2`、`xH`、`H^2` 提供，而

\[
\mathcal R_{\lambda}=(\partial_t+\lambda)^{-1}
\]

就是带物理时间尺度和记忆的解析响应算子。

## 3. 没有结构搜索，但仍然自适应

训练开始时一次性建立一个稍大的最大网络。每个响应通道以及每个 `x^2 / xH / H^2` 低秩分量都有一个连续 gate：

\[
g_{\ell c},\quad g^{xx}_r,\quad g^{xH}_{\ell r},\quad g^{HH}_{\ell r}.
\]

这些 gate 与普通权重、bias、低秩因子一起通过控制方程残差连续优化。网络从不询问“下一个候选神经元是谁”。

达到残差容差后，程序用完整残差 Jacobian 估计各已有 gate 的局部影响，将影响最小的一批 gate 置零，并重新执行完整非线性物理残差检查。只有训练点和验证点仍满足容差的剪枝才保留；失败的批次会缩小或放弃。

因此自适应机制变成：

\[
\boxed{\text{固定最大容量}\rightarrow\text{连续残差训练}\rightarrow\text{残差验证剪枝}}
\]

而不是：

\[
\text{候选枚举}\rightarrow\text{Grow/Split}\rightarrow\text{再训练}.
\]

热模态与响应通道的选择准则不同：热模态由空间物理截断准则决定；每个已保留热模态内部需要多少响应通道，由残差和验证剪枝决定。

## 4. 最大容量不是最终结构

最大 depth、每模态最大 channel 数以及低秩上限会根据热秩和输入维数给一个保守容量。它们只是安全上限，不代表最后网络必须全部使用。

训练完成后 `run.py` 写出：

```text
results/uwpt/thermal.rank.json
results/uwpt/network.structure.json
```

前者记录最终热秩、自动选择方法和尾部响应信息；后者记录实际有效 depth、各层有效通道以及有效 `xx/xH/HH` rank。

高级实验仍可通过 `SDFMPNEO_FIXED_NETWORK_*` 环境变量覆盖最大容量，但正常使用不需要手工决定“4 层、2 通道”之类的内部结构。

## 5. 解析时间表示

所有动态信号仍直接表示在

\[
t^k e^{-\mu t}
\]

闭包中。加法、乘法和一阶响应算子保持解析闭合；共振使用 confluent polynomial 分支。有限超长时间使用对数域评估，因此支持例如 `1e300`；`t=inf` 直接计算稳态极限。

这意味着取消结构搜索并没有牺牲任意时间直接查询能力。

## 6. 训练与兼容性

fresh 模型正常阶段大致为：

```text
thermal_rank_selection
initial_residual
weight_refinement
...
validation
structure_pruning
saving
```

fresh 日志不应出现 `candidate_search`。

fixed-network checkpoint 使用 research model format v3，网络内部 metadata 当前为 fixed-network format v2。早期 fixed-network v1 参数可以自动升级：旧参数原样保留，新 bias 初始化为 0，新 gate 初始化为 1，从而保持旧函数。

历史 v1/v2 非空解析 DAG checkpoint 仍可加载，并故意继续使用旧 DAG 训练器；不会把一个已训练旧结构静默解释成新 fixed network。
