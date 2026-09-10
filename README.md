# SDF-MPNEO — 电磁–热多物理代理模型

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

SDF-MPNEO 面向水下 WPT 的磁准静态电磁–瞬态热耦合。训练不使用瞬态解标签；网络通过控制方程残差训练，电磁平衡、温度相关电导率、焦耳热和热方程仍来自物理模型。

当前 fresh 训练主架构是：

\[
\boxed{\text{自动热秩}\rightarrow\text{固定最大解析响应网络}\rightarrow\text{连续残差训练}\rightarrow\text{验证剪枝}}
\]

正常 fresh 训练不再逐个搜索响应神经元，不执行 Grow / Enrich / Split / `candidate_search`。完整原理见 [`docs/FIXED_ANALYTIC_RESPONSE_NETWORK.md`](docs/FIXED_ANALYTIC_RESPONSE_NETWORK.md)。

## 1. 安装

```bash
python -m pip install -e '.[cad,gui]'
```

测试环境：

```bash
python -m pip install -e '.[dev,cad,gui]'
```

Linux 的 Gmsh wheel 可能还需要：

```bash
sudo apt-get install libgl1 libglu1-mesa
```

## 2. 推荐入口：`run.py`

训练：

```bash
python run.py --mode train
```

无界面：

```bash
python run.py --mode train --headless
```

推理：

```bash
python run.py --mode predict
```

指定模型路径：

```bash
python run.py --mode predict --model results/uwpt/model.npz
```

几何、材料、频率、端口、误差目标和推理输入都集中在 `run.py` 顶部。默认一次训练覆盖 10 个连续几何参数，域内换几何仍使用同一个保存模型。

## 3. 用户主要指定误差，而不是内部阶数

默认不再写死：

```python
THERMAL_RANK = 2
```

而是：

```python
THERMAL_RANK = None
THERMAL_TRUNCATION = {
    "mode": "automatic_physics_envelope",
    "relative_tolerance": 1e-3,
    "absolute_tolerance": 0.0,
    "initial_coordinate_bound": 0.1,
    "probe_start_rank": 4,
    "max_probe_rank": 32,
    "source_bound_safety_factor": 2.0,
    "boundary_fraction": 0.25,
    "temperature_probe_axes": 4,
    "initial_temperature_deviation_free": None,
    "source_dual_bound": None,
    "requested_state_tolerance": None,
    "prefer_partial_thermal_spectrum": True,
}
```

程序从低阶热谱开始扩大，用真实全阶 EM 平衡估计各热模态的稳态响应包络。只有高阶尾部已经足够小才截断。默认包络是有限物理锚点收敛判据，程序会在报告中明确写 `certified_continuous_domain=false`，不会把它冒充严格连续域证书。

如果能提供严格的：

```text
initial_temperature_deviation_free
source_dual_bound
requested_state_tolerance
```

则继续使用原有 theorem-level thermal-tail certificate。

自动选出最终 rank 后，训练初态域自动展开：

```python
TRAINING = {
    "initial_lower": [],
    "initial_upper": [],
    ...
}
```

空列表表示使用 `initial_coordinate_bound` 生成最终 `r` 维模态坐标盒。因此不会再发生“自动选成 6 阶，但配置仍只有两个 a0”的情况。

## 4. 网络结构如何自动适应

网络先创建一个根据最终热秩和输入维数确定的**最大容量**。每个热模态有若干候选解析响应通道；每个通道满足

\[
(\partial_t+\lambda_j)h=S.
\]

source 包含：

\[
S=b+W_xx+W_hH+\Phi_{xx}+\Phi_{xH}+\Phi_{HH}.
\]

所以每个响应神经元仍然拥有多个系数，并且有显式 bias。动态层不使用 ReLU；`x^2`、`xH`、`H^2` 提供非线性，而解析响应算子提供物理时间尺度和记忆。

每个响应通道和低秩 `xx/xH/HH` 分量都有连续 gate。所有 gate 与普通权重一起由物理残差优化。达到残差目标后，程序根据完整残差 Jacobian 对已有 gate 做影响排序，尝试将低影响 gate 置零，并重新执行完整非线性训练/验证残差检查。只有仍满足容差的剪枝才保留。

因此仍然是“残差决定需要哪些响应神经元”，但不再通过候选结构搜索完成。

高级实验可以用 `SDFMPNEO_FIXED_NETWORK_*` 环境变量修改最大容量；正常使用无需手工指定最终 depth、每模态 channel 数或低秩 rank。

## 5. 训练配置

默认核心设置类似：

```python
TRAINING = {
    "initial_lower": [],
    "initial_upper": [],
    "operating_lower": [0.0, 0.0],
    "operating_upper": [10.0, 10.0],
    "time_horizon": 100000.0,
    "residual_tolerance": 1e-5,
    "time_sampling": "mixed_log",
    "time_min": 1e-6,
    "include_steady_state": True,
    "sample_count": 64,
    "validation_count": 64,
    "max_nodes": 256,
    "max_degree": 3,
    "max_parent_responses": 1,
    "max_realization_dimension": 64,
}
```

最后四个结构预算只用于旧 v1/v2 非空 DAG checkpoint 的兼容继续训练；fresh fixed network 不使用它们进行 candidate search。

fresh 训练阶段大致为：

```text
thermal_rank_selection
initial_residual
weight_refinement
...
validation
structure_pruning
saving
```

如果 fresh 日志出现 `candidate_search`，应检查是否实际上加载了旧的非空 DAG checkpoint。

## 6. 输出文件

默认训练后生成：

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/geometry.domain.json
results/uwpt/thermal.rank.json
results/uwpt/network.structure.json
results/uwpt/train.settings.json
results/uwpt/training.report.json
results/uwpt/logs/...
```

其中：

- `thermal.rank.json`：最终热秩、选择方法、探测阶数和模态响应尾部信息；
- `network.structure.json`：最大/有效 depth、各层有效通道数、各热模态有效通道数以及有效 `xx/xH/HH` rank；
- `training.report.json`：训练/验证最大残差及是否真正达到容差。

达到残差目标时退出码为 `0`；未达到时仍保存当前模型和报告，但退出码为 `2`，不会伪装成成功。

## 7. 推理

自动热秩后默认不再手写两个零初态：

```python
PREDICTION = {
    "a0": None,
    "operating": [5.0, 0.0],
    "times": [0.0, 0.001, 1.0, 1000.0, 100000.0, 1000000.0, "inf"],
    "geometry": None,
    "allow_time_extrapolation": True,
    "initial_temperature_file": None,
    "state_only": False,
    "allow_extrapolation": False,
}
```

`a0=None` 自动生成保存模型最终 thermal rank 对应的全零初态。也可以显式传入最终 `r` 维模态坐标，或通过 `initial_temperature_file` 提供全节点开尔文温度 `.npy` 后自动投影。

`geometry=None` 查询保存几何域中心；指定几何时必须给出完整参数字典且位于保存域内。

## 8. 任意时间和稳态

解析时间表示保持为

\[
t^k e^{-\mu t}.
\]

因此不需要时间步进，可以直接查询：

```python
PREDICTION["times"] = [0.0, 1.0, 1e5, 1e6, 1e300, "inf"]
```

- `1e300` 等超长有限时间使用稳定解析计算；
- `"inf"` 直接计算稳态极限；
- `allow_time_extrapolation=False` 可禁止有限时间超出训练窗。

有限训练窗外的查询可以计算，但训练残差本身不自动构成域外精度证书。

## 9. Python API：换几何直接查询

```python
import numpy as np
from sdfmpneo import ResearchElectroThermalModel

model = ResearchElectroThermalModel.load("results/uwpt/model.npz")
geometry = dict(zip(
    model.geometry_names,
    ((model.lower + model.upper) / 2).tolist(),
))
geometry["rx_gap"] = 0.0101
geometry["rx_offset_x"] = 0.0001

a0 = np.zeros(model.graph.n_modes)
for t in [0.0, 1.0, 1e6, np.inf]:
    result = model.predict(
        t,
        geometry=geometry,
        a0=a0,
        operating=[5.0, 0.0],
        diagnostics=True,
        allow_time_extrapolation=True,
    )
    print(t, result.maximum_temperature)
```

改变几何后不会重新训练；仍使用同一个解析网络、共享电磁基和共享热坐标。

## 10. 继续训练和兼容性

```python
FILES["resume_model"] = "results/uwpt/model.npz"
```

新版 fixed-network checkpoint 继续优化同一组连续参数和 gate。历史 v1/v2 非空解析 DAG checkpoint 仍能加载，并故意继续走旧 DAG trainer，避免静默改变旧模型含义。

默认 `FILES["resume_model"] = None`，因此 `python run.py --mode train` 是 fresh 自动热秩 + fixed-network 训练。

## 11. 物理范围

当前一次几何族训练内保持线圈形状类别、匝数、材料拓扑和频率固定。几何参数可以连续变化，但不是“任意换一个 CAD 拓扑都无需重训”。

电磁变量没有被删除：对于当前固定频率时谐 MQS 问题，它们作为快速平衡变量在每个物理残差点求解并生成焦耳热；网络代理的是慢时间热状态。若未来研究开关瞬态、变频或电路包络动态，再把 EM/circuit 动态状态显式加入演化系统。
