# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

SDF-MPNEO 面向 UWPT 磁准静态电磁–瞬态热耦合代理。训练不使用瞬态解标签，而是直接最小化控制方程残差。温度相关电导率、电磁平衡、焦耳热和热扩散始终由物理模型计算。

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

## 1. 自动热秩

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

自动选择器从低阶热谱开始扩展，通过真实电磁平衡估计模态稳态响应包络

\[
E_j\sim \max\frac{|q_j|}{\lambda_j}.
\]

探测不再在固定的 32 阶上限处直接回退到全热空间，而是继续倍增直到找到可信尾部，必要时一直检查完整离散热谱。完整谱只用于**选秩诊断**；最终网络只保留满足目标的最小前缀 `r`。只有完整谱本身仍不能截断时，才真正使用全热空间。

如果提供严格 thermal-tail certificate 所需的全部输入，则使用严格证书路径；有限锚点物理包络不会被标记成连续域严格证明。

## 2. restart 状态域

分段 rollout 中，第 2 段以后网络输入的 `a0` 是上一段的终态，因此训练域必须覆盖可达热状态，而不只是环境初态附近。

默认把：

```python
TRAINING["initial_lower"] = []
TRAINING["initial_upper"] = []
```

留空。自动热秩阶段会把允许的初始扰动与安全放大的热响应包络组合，生成每个保留热模态的 restart-state box。预测时每次进入下一段前都会检查当前状态是否仍在该训练盒内；默认不静默外推。

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

每个响应神经元满足

\[
(\partial_t+\lambda_j)h_{jc}^{(\ell)}=S_{jc}^{(\ell)},
\qquad
h_{jc}^{(\ell)}(0)=0.
\]

source 包含显式 bias、线性项以及低秩 `x²`、`xH`、`H²` 交互。动态层不使用 ReLU；解析响应算子本身承担时间记忆，乘法项提供非线性。

网络最大容量在训练开始时固定，channel/component gate 与其余系数一起连续优化。达到误差目标后，仅保留重新验证仍满足容差的剪枝。

## 4. 两阶段无标签训练

第一阶段只最小化控制方程残差：

\[
R_{\rm phys}
=
\dot{\hat a}-F(\hat a,G,U).
\]

物理残差达到目标后，再开启 restart/semigroup consistency：

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

这样避免随机初始网络因为“零/错误动力学也可能自洽”而阻碍物理残差优化。

semigroup 参数 Jacobian 使用解析链式法则：

\[
J_{\rm sg}
=
J_\theta\Phi_{t_1+t_2}
-
\left(
J_\theta\Phi_{t_2}
+
J_{a_0}\Phi_{t_2}\,
J_\theta\Phi_{t_1}
\right).
\]

训练和独立验证都必须同时满足物理残差与 restart-rate defect 的目标。

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

因此训练窗口不再等于最大可查询时间。网络从不单次外推到 `1e300`。

## 6. 稳态单独解物理方程

真正的稳态不由网络做 `t=\infty` 外推，而是直接解：

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

为了方便，`model.predict(float("inf"), ...)` 也会路由到同一个物理稳态 Newton 求解器，而不是调用网络的无限时间表达。

## 7. 连续几何族

开启：

```python
GEOMETRY_FAMILY["enabled"] = True
```

不同几何共用参考热坐标系、跨几何 EM 基和同一个 finite-horizon network，但每个几何的残差仍使用自己的

\[
M_r(G),\quad K_r(G),\quad A_{\rm EM}(G,a).
\]

长时间 rollout 和稳态物理解对几何族使用完全相同的接口。

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

用户主要控制的是精度目标和单段时间，而不是内部神经元数量。

## 9. 输出

默认训练会写出：

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

`training.report.json` 同时报告 physics training/validation residual、semigroup training/validation rate defect、原始 semigroup state defect、最终有效网络结构和单段最大时间。

## 10. 模型格式

当前模型格式只支持当前 segmented fixed-network 架构。历史 DAG、旧 fixed-network 参数排列和旧无限时间模型不会被转换；需要重新训练。

同一当前格式的 `.stopped.npz` 可以继续训练。
