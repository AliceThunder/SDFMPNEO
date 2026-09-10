# 自动热秩 + Finite-Horizon Fixed Analytic Response Network

当前生产训练路径只有一条：自动热秩 → 固定最大解析响应网络 → 有限时间物理残差训练 → restart consistency → 独立验证剪枝 → 分段 rollout。训练不使用瞬态解标签，热动力学目标始终来自真实降阶物理场

\[
\dot a=F(a,G,U),\qquad R_{phys}=\dot{\hat a}-F(\hat a,G,U).
\]

## 1. 热空间与有限时间

温度场写为

\[
T(x,t)\approx T_{ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),\qquad K\phi_j=\lambda_jM\phi_j.
\]

默认 `THERMAL_RANK=None`。自动选秩会继续扩大热谱，必要时检查完整离散热空间，再选满足响应尾部判据的最小前缀。有限锚点 envelope 保持 `certified_continuous_domain=false`；严格证书输入齐全时仍可使用严格 thermal-tail certificate。

网络只定义单段

\[
\hat\Phi_t(a_0,G,U),\qquad 0\le t\le H,
\]

默认 `H=100 s`。长时间由 restart rollout 组合，真正稳态直接解 `F(a_\infty,G,U)=0`。

## 2. 高热秩可扩展网络

每个响应通道绑定真实热衰减率：

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}=S_{jc}^{(\ell)},\qquad h_{jc}^{(\ell)}(0)=0.
\]

高热秩下不能使用 `width × width` dense hidden coupling，也不能让两个包含全部热坐标的 dense projection 直接相乘，否则参数量和解析指数项都会按二次规模增长。当前实现将最大容量按 thermal rank 分段设置：`r>=96` 时默认 `depth=1`、`channels_per_mode=1`；因此例如 `r=198` 时只有 198 个响应通道，而不是旧设计的 1584 个。

第一层 source 使用四类项：

\[
S=b+S_{lin}+S_{dyn\times static}+S_{square}.
\]

其中线性项采用低秩输入/输出因子；`dyn × static` 允许温度状态与几何/电流静态参数发生非线性交互，但避免形成所有热模态两两组合；另外显式保留低秩逐模态平方通道

\[
S_{square}=W_{sq}\left(\sum_j c_{rj}\,x_j^2\right),
\]

用于表示温度非线性中的 `a_j^2` 成分。低热秩时仍可使用更深的解析响应层，深层线性映射也采用低秩因子而不是 dense `width × width` 矩阵。

指数多项式内部只保存参与乘积的模态索引 signature，而不是长度为 `r` 的 dense 计数向量，因此符号 bookkeeping 随实际非线性次数增长，而不是随完整 thermal rank 线性复制到每个 term。

动态层不使用 ReLU；时间记忆仍由

\[
\mathcal R_\lambda=(\partial_t+\lambda)^{-1}
\]

提供。

## 3. 高热秩 Jacobian 与 Gauss–Newton

完整 residual 始终在全部训练配点上评估，并决定 trial 是否真正接受；但参数 Jacobian 只对当前 residual 最大的 hard points 联合构造。默认线性化点数还会随 thermal rank 自动收紧，使

\[
N_{jac}\,r
\]

保持有界，避免 `r≈200` 时形成多 GB dense Jacobian / normal equation。

Gauss–Newton/LM 仍是**联合参数更新**，不是逐小块 block-coordinate 优化。求解时优先使用较小的 primal/dual 正定系统，并先允许 bias/输出幅值块做 warm start，再开放全部连续参数。所有候选步最后都重新计算完整训练 residual；hard-point Jacobian 只负责提出方向，不负责替代验收标准。

默认可配置：

```python
TRAINING = {
    "max_network_depth": None,
    "max_channels_per_mode": None,
    "max_linear_rank": None,
    "max_hidden_rank": None,
    "max_quadratic_rank": None,
    "max_square_rank": None,
    "max_cross_rank": None,
    "max_state_rank": None,
    "jacobian_point_budget": 8,
    "semigroup_jacobian_point_budget": 4,
}
```

`None` 表示根据 thermal rank 自动选择安全的最大容量。

## 4. restart consistency

物理残差达到目标后开启

\[
R_{sg}=\frac{\hat\Phi_{t_1+t_2}(a_0)-\hat\Phi_{t_2}(\hat\Phi_{t_1}(a_0))}{H},\qquad t_1+t_2\le H.
\]

参数 Jacobian 使用解析链式法则

\[
J_{sg}=J_\theta\Phi_{t_1+t_2}-\left(J_\theta\Phi_{t_2}+J_{a_0}\Phi_{t_2}J_\theta\Phi_{t_1}\right).
\]

训练和独立验证同时要求 physics residual 与 restart-rate defect 满足用户容差。

## 5. 监控与格式

`initial_residual`、`physics_jacobian`、`restart_residual` 和 `restart_jacobian` 都向 monitor 发布逐点 `work_completed/work_total`。因此首个 RMS 点尚未形成时，GUI 也会显示当前处理进度，而不是长期保持灰色空图且无法区分“正在计算”和“卡住”。

当前 `FixedAnalyticResponseNetwork` metadata format 为 **v6**。历史 fixed-network 参数布局不做迁移；需要重新训练。当前格式 checkpoint 可以正常 resume。
