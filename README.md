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

## 1. 自动热秩与分层缓存

默认：

```python
THERMAL_RANK = None
```

热空间写为

\[
T(x,t)\approx T_{\rm ref}(x)+\sum_{j=1}^{r}a_j(t)\phi_j(x),
\qquad
K\phi_j=\lambda_jM\phi_j.
\]

自动选择器现在不再按 `4 -> 8 -> 16 -> ...` 重复构建多个不同阶数的热/电磁探测模型。首次运行会：

1. 构建一次完整离散热谱，得到 `Phi` 与 `lambdas`；
2. 在完整热谱上只做一次确定性的全阶 EM response-envelope 诊断；
3. 从完整 envelope 中纯代数扫描满足尾部误差要求的最小 `rank`；
4. 最终代理网络只保留该前缀，不会因为完整谱参与诊断就训练全部热自由度。

模态稳态响应包络采用

\[
E_j\sim \max\frac{|q_j|}{\lambda_j}.
\]

零热态下所有 operating anchors 共用同一个 EM 分解；每个模态的 loss operator 只组装一次，再同时评估全部 operating anchors，避免重复组装。

默认启用三层缓存：

```text
model.config.thermal_rank.cache.spectrum.npz   # 完整热谱 Phi/lambdas
model.config.thermal_rank.cache.envelope.npz   # 完整物理 response envelope
model.config.thermal_rank.cache.json            # 当前 tolerance 下的最终 rank/report
```

缓存按内容指纹自动失效：

- mesh、热导率或热容量变化：热谱与后续缓存全部失效；
- 频率、电学材料、端口映射、电流范围或温度 probe 策略变化：只重算 EM envelope，热谱可复用；
- 只修改 `relative_tolerance`、`absolute_tolerance`、`source_bound_safety_factor` 或 restart safety factor：直接复用完整 envelope，只重新扫描 rank，通常接近瞬时完成。

因此第一次自动选秩仍需要实际物理计算；相同物理配置下后续训练不会重复做这部分工作。

如果已经生成稳定网格，建议：

```python
MESH["generate"] = False
```

避免每次重新生成 `.msh` 导致 mesh 内容指纹变化。

有限锚点物理 envelope 是可复现的数值/物理选秩判据，不会被标记成连续几何域的严格数学证书。若提供 strict thermal-tail certificate 所需全部输入，则仍走严格证书路径。

## 2. restart 状态域

分段 rollout 中，第 2 段以后网络输入的 `a0` 是上一段终态，因此训练域必须覆盖可达热状态，而不只是环境初态附近。

默认：

```python
TRAINING["initial_lower"] = []
TRAINING["initial_upper"] = []
```

留空后，自动热秩阶段会将允许的初始扰动与安全放大的热响应 envelope 组合，生成每个保留热模态的 restart-state box。预测时每次进入下一段前都会检查当前状态是否仍在训练盒内；默认不静默外推。

## 3. 有限时间解析响应网络

网络只学习单段：

\[
\hat\Phi_{\Delta t}(a_0,G,U),
\qquad
0\le\Delta t\le H.
\]

默认：

```python
MAX_RESPONSE_TIME = 100.0
```

每个响应通道满足

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}
=
S_{jc}^{(\ell)},
\qquad
h_{jc}^{(\ell)}(0)=0.
\]

source 包含显式 bias、低秩线性项、动态热态×静态几何/工况项，以及逐模态热态平方项。动态层不使用 ReLU；解析响应算子负责时间记忆。

高 thermal rank 时网络容量自动收紧。当前默认在 `rank >= 96` 时使用 `depth=1`、`channels_per_mode=1`，避免响应通道和 Jacobian 随 thermal rank 二次膨胀。例如 `rank=198` 时默认产生 198 个响应通道，而不是旧结构中的 1584 个。

## 4. 两阶段无标签训练

第一阶段只最小化控制方程残差：

\[
R_{\rm phys}
=
\dot{\hat a}-F(\hat a,G,U).
\]

达到物理残差目标后，再加入 restart/semigroup consistency：

\[
R_{\rm sg}
=
\frac{
\hat\Phi_{t_1+t_2}(a_0)
-
\hat\Phi_{t_2}(\hat\Phi_{t_1}(a_0))
}{H},
\qquad
t_1+t_2\le H.
\]

训练使用 hard-point 联合 Gauss–Newton：Jacobian 只在当前最难配点上构造以限制内存，但每个候选更新仍用完整训练 residual 集合接受或拒绝。GUI 在首个 RMS 点出现前会显示逐点 residual/Jacobian 组装进度。

## 5. 长时间自动 rollout

用户仍然直接查询总时间：

```python
result = model.predict(
    350.0,
    a0=a0,
    operating=U,
)
```

如果 `H=100 s`，内部自动执行：

```text
0 -> 100
100 -> 200
200 -> 300
300 -> 350
```

数学上：

\[
\hat\Phi_{350}
=
\hat\Phi_{50}
\circ
\hat\Phi_{100}
\circ
\hat\Phi_{100}
\circ
\hat\Phi_{100}.
\]

结果会报告 `segment_count`、`segment_durations` 和 `max_response_time`。

## 6. 稳态

真正稳态不通过网络做 `t=\infty` 外推，而是直接求解

\[
F(a_\infty,G,U)=0.
\]

调用：

```python
steady = model.steady_state(
    a0=a0,
    operating=U,
)
```

`model.predict(float("inf"), ...)` 只是方便路由到同一个物理稳态 Newton 求解器。

## 7. 连续几何族

开启：

```python
GEOMETRY_FAMILY["enabled"] = True
```

不同几何共用参考热坐标系、跨几何 EM 基和同一个 finite-horizon network，但每个几何残差仍使用自己的

\[
M_r(G),\quad K_r(G),\quad A_{\rm EM}(G,a).
\]

## 8. 关键训练配置

```python
MAX_RESPONSE_TIME = 100.0

TRAINING = {
    "initial_lower": [],
    "initial_upper": [],
    "operating_lower": [0.0, 0.0],
    "operating_upper": [10.0, 10.0],
    "max_response_time": MAX_RESPONSE_TIME,
    "residual_tolerance": 1e-5,
    "sample_count": 64,
    "validation_count": 64,
    "semigroup_sample_count": 8,
    "semigroup_validation_count": 8,
    "max_network_depth": None,
    "max_channels_per_mode": None,
    "max_quadratic_rank": None,
    "max_cross_rank": None,
    "max_state_rank": None,
}
```

用户主要控制精度目标与单段时间，而不是内部神经元数量。

## 9. 输出

默认训练会写出：

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/train.settings.json
results/uwpt/training.report.json
results/uwpt/thermal.rank.json
results/uwpt/model.config.thermal_rank.cache.json
results/uwpt/model.config.thermal_rank.cache.spectrum.npz
results/uwpt/model.config.thermal_rank.cache.envelope.npz
results/uwpt/network.structure.json
results/uwpt/geometry.domain.json    # 几何族启用时
results/uwpt/logs/
```

`thermal.rank.json` 会报告 `selection_cache_hit`、`envelope_cache_hit`、`spectrum_cache_hit` 和各缓存路径，便于判断本次是否真的复用了昂贵计算。

## 10. 模型格式

当前模型格式只支持当前 segmented fixed-network 架构。历史 DAG、旧 fixed-network 参数排列和旧无限时间模型不会被转换；需要重新训练。

同一当前格式的 `.stopped.npz` 可以继续训练。
