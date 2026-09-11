# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

SDF-MPNEO 面向 UWPT 磁准静态电磁–瞬态热耦合代理。训练不使用瞬态解标签，而是直接最小化控制方程残差；温度相关电导率、电磁平衡、焦耳热和热扩散始终由物理模型计算。

当前生产路径只有一条：

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
\text{验证剪枝}
\rightarrow
\text{长时间分段 rollout}
}
\]

不再存在候选神经元枚举、Grow / Enrich / Split、`candidate_search` 或历史 DAG 训练器。

## 安装

```bash
python -m pip install -e '.[cad,gui]'
```

开发测试：

```bash
python -m pip install -e '.[dev,cad,gui]'
```

## 推荐入口：`run.py`

训练：

```bash
python run.py --mode train
```

无界面训练：

```bash
python run.py --mode train --headless
```

推理：

```bash
python run.py --mode predict
```

通常只需要修改 `run.py` 顶部的几何、材料、误差目标、训练域和推理输入。

## 1. 自动热秩与缓存

默认 `THERMAL_RANK=None`。温度写为

\[
T(x,t)\approx T_{\rm ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),
\qquad K\phi_j=\lambda_jM\phi_j.
\]

自动选择器只做一次完整离散热谱和一次完整电磁模态响应包络诊断，再从完整 envelope 中代数选择满足误差目标的最小前缀 `r`。默认保存 thermal spectrum、physics envelope 和最终 rank/report 三层独立缓存。只修改 thermal tolerance/safety factor 时仅重新扫描 envelope；修改频率、电学材料、端口或工况域时重算 envelope 但复用 thermal spectrum；只有 mesh 或热材料变化才重算 thermal spectrum。

自动选秩同时把允许的初始扰动与安全放大的热响应包络组合，生成每个保留热模态的 restart-state box。有限锚点 physics envelope 不会被标记成连续域严格证明；严格 thermal-tail certificate 输入齐全时仍走严格证书路径。

## 2. finite-horizon 解析响应网络

网络只学习单段

\[
\hat\Phi_{\Delta t}(a_0,G,U),\qquad0\le\Delta t\le H,
\]

默认 `MAX_RESPONSE_TIME=100.0` s。每个响应通道满足

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}=S_{jc}^{(\ell)},\qquad h_{jc}^{(\ell)}(0)=0.
\]

source 包含显式 bias、低秩线性项、温度×静态参数交互和逐模态平方项。动态层不使用 ReLU。高热秩时容量自动收紧；默认 `r>=96` 时使用 `depth=1`、`channels_per_mode=1`，因此 `r=198` 时只有 198 个响应通道。

## 3. 高热秩物理残差加速

高热秩训练不再逐个热模态重复组装全局焦耳热 loss operator。当前实现把电磁状态恢复为每个四面体上的 Nedelec 电场，一次性积分

\[
m_i^{(e)}=\int_{K_e}\sigma(T)|E|^2\lambda_i\,dV,
\]

再通过一次 dense contraction 投影到全部 thermal test modes。它与逐模态计算

\[
q_j=x^H H_j(a)x
\]

代数等价，但每个配点不再组装 `r` 个全局 sparse loss matrices。

训练的 Gauss–Newton 方向使用冻结焦耳热反馈的 dissipative thermal Jacobian。固定几何使用

\[
J_F^{GN}\approx-\Lambda,
\]

几何族使用

\[
J_F^{GN}\approx-M_r(G)^{-1}K_r(G).
\]

这只是搜索方向近似；每个 trial、训练 residual 和验证 residual 都仍使用完整温度相关电导率与真实电磁焦耳热，因此最终误差目标不变。这样避免 `r≈200` 时对每个 hard point 组装约 `r^2` 个 `loss_operator_derivative_sparse(j,k)`。

参数 Jacobian 只在当前 residual 最大的少量 hard points 上构建，但 trial 是否接受始终由完整训练配点的真实 residual 决定。输出幅值只做一次 warm start；之后强制联合更新完整连续参数，避免 amplitude-only 的微小改进长期阻塞非线性因子训练。

## 4. 两阶段无标签训练

第一阶段最小化

\[
R_{\rm phys}=\dot{\hat a}-F(\hat a,G,U).
\]

物理残差达标后再开启 restart/semigroup consistency：

\[
R_{\rm sg}=\frac{\hat\Phi_{t_1+t_2}(a_0)-\hat\Phi_{t_2}(\hat\Phi_{t_1}(a_0))}{H},
\qquad t_1+t_2\le H.
\]

训练和独立验证都必须满足同一残差目标。

## 5. 长时间 rollout

例如查询 350 s、`H=100 s` 时，内部自动执行

```text
0 -> 100 -> 200 -> 300 -> 350
```

每段终态成为下一段 `a0`，默认检查 restart state 仍在训练域内。用户仍直接调用 `model.predict(350.0, ...)`。

## 6. 稳态

真正稳态直接解

\[
F(a_\infty,G,U)=0.
\]

`model.predict(float("inf"), ...)` 只路由到该物理稳态求解器。高热秩时稳态迭代同样使用完整真实 residual + 冻结焦耳热反馈的 dissipative Newton 方向和回溯验收，不再组装完整 `r\times r` 电磁热源导数。

## 7. 连续几何族

开启 `GEOMETRY_FAMILY["enabled"]=True` 后，不同几何共享参考热坐标、跨几何 EM 基和同一个 finite-horizon network，但每个几何残差仍使用自己的

\[
M_r(G),\quad K_r(G),\quad A_{\rm EM}(G,a).
\]

## 8. 关键训练配置

```python
TRAINING = {
    "initial_lower": [],
    "initial_upper": [],
    "operating_lower": [0.0, 0.0],
    "operating_upper": [10.0, 10.0],
    "max_response_time": 100.0,
    "residual_tolerance": 1e-5,
    "sample_count": 64,
    "validation_count": 64,
    "semigroup_sample_count": 8,
    "semigroup_validation_count": 8,
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

用户主要控制物理误差目标和单段时间，不需要手工搜索神经元数量。

## 9. 输出

默认训练输出包括：

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/train.settings.json
results/uwpt/training.report.json
results/uwpt/thermal.rank.json
results/uwpt/network.structure.json
results/uwpt/geometry.domain.json    # 几何族启用时
results/uwpt/logs/
```

## 10. 模型格式

当前模型格式只支持当前 segmented fixed-network 架构。历史 DAG、旧 fixed-network 参数排列和旧无限时间模型不会转换，需要重新训练。同一当前格式的 `.stopped.npz` 可以继续训练。
