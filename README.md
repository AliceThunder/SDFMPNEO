# SDF-MPNEO

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

SDF-MPNEO 用于 UWPT 磁准静态电磁–瞬态热耦合代理。训练不使用瞬态解标签，而是直接最小化控制方程残差；温度相关电导率、电磁平衡、焦耳热和热扩散都由物理模型计算。

当前代码只有一条训练主路径：

\[
\boxed{\text{自动热秩}\rightarrow\text{固定解析响应网络}\rightarrow\text{连续残差优化}\rightarrow\text{独立验证与剪枝}}
\]

不再存在候选神经元枚举、Grow / Enrich / Split 或 `candidate_search`。历史 DAG/checkpoint 不做兼容转换；继续训练只支持当前 fixed-network 格式。

## 安装

UWPT 几何和训练窗口：

```bash
python -m pip install -e '.[cad,gui]'
```

开发测试：

```bash
python -m pip install -e '.[dev,cad,gui]'
```

Linux 的 Gmsh wheel 如缺少 OpenGL/GLU，可安装：

```bash
sudo apt-get install libgl1 libglu1-mesa
```

## 推荐入口：`run.py`

通常只修改 `run.py` 顶部的几何、材料、误差目标、训练域和推理输入。

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

指定模型路径：

```bash
python run.py --mode predict --model results/uwpt/model.npz
```

默认训练结果包括：

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/train.settings.json
results/uwpt/training.report.json
results/uwpt/thermal.rank.json
results/uwpt/network.structure.json
results/uwpt/geometry.domain.json    # 启用几何族时
results/uwpt/logs/
```

训练只有在训练配点和独立验证配点的最大物理残差都达到 `residual_tolerance` 时返回成功；否则保存当前结果并返回非零退出码。

## 自动热秩

默认：

```python
THERMAL_RANK = None
```

并由 `THERMAL_TRUNCATION` 指定物理截断目标。默认 `automatic_physics_envelope` 从低阶热谱开始扩展，用全阶电磁平衡估计各热模态的稳态响应包络。只有已解析尾部足够小且探测边界已经衰减时才接受部分秩。

如果到 `max_probe_rank` 仍不能确认尾部，程序不会把未解析的部分秩当成成功，而是回退到完整离散热空间。

如果同时提供：

```text
initial_temperature_deviation_free
source_dual_bound
requested_state_tolerance
```

则使用 theorem-level thermal-tail certificate 路径；三项必须同时给出。

自动得到最终热秩 `r` 后，`TRAINING["initial_lower"]` / `initial_upper` 可以留空，程序会按 `initial_coordinate_bound` 自动扩展为 `r` 维。

## Fixed analytic response network

网络不是普通时间步进 MLP。每个解析响应通道绑定一个热衰减率：

\[
(\partial_t+\lambda_j)h=S,\qquad h(0)=0.
\]

source 由显式 bias、输入线性项、低秩 `x^2`、`xH` 和 `H^2` 组成。网络最大结构在训练开始时一次性建立，所有普通参数与 channel/component gate 一起连续优化，不执行离散结构搜索。

默认容量会根据最终热秩和输入维数确定；也可以在 `TRAINING` 中直接限制：

```python
TRAINING = {
    ...
    "max_network_depth": None,
    "max_channels_per_mode": None,
    "max_quadratic_rank": None,
    "max_cross_rank": None,
    "max_state_rank": None,
    "max_iterations": 36,
    "max_validation_epochs": 5,
    "gate_shrink": 0.0,
    "prune_relative_budget": 0.10,
    "prune_rounds": 3,
}
```

达到残差目标后，程序根据完整残差 Jacobian 对 gate 做影响排序；只有置零后训练点和独立验证点仍满足目标的剪枝才会保留。

详细数学结构见 [`docs/FIXED_ANALYTIC_RESPONSE_NETWORK.md`](docs/FIXED_ANALYTIC_RESPONSE_NETWORK.md)。

## 任意时间与稳态查询

解析网络中的时间信号保持在有限指数–多项式闭包中：

\[
t^k e^{-\mu t}.
\]

因此推理不是把时间截断到训练窗，也不需要逐步时间积分。有限超长时间直接稳定评估，`t=inf` 直接求解析稳态极限。

例如：

```python
PREDICTION = {
    "a0": None,
    "operating": [5.0, 0.0],
    "times": [0.0, 1.0, 1e5, 1e6, "inf"],
    "geometry": None,
    "allow_time_extrapolation": True,
    ...
}
```

## 连续几何族

`GEOMETRY_FAMILY["enabled"] = True` 时，不同几何共用：

- 同一参考热坐标系；
- 同一跨几何 EM 降阶基；
- 同一个 fixed analytic response network。

每个几何仍使用自己的质量矩阵、导热矩阵和电磁算子计算物理残差。训练后，域内换几何直接查询同一个模型，不需要重新训练。

几何输入必须位于保存模型的非退化 affine chart 内。超出训练域不会默认静默外推。

## 暂停和继续训练

训练窗口支持暂停、恢复和停止。停止时如果已经存在 fixed network，会保存 `.stopped.npz` 检查点。

继续训练时：

```python
FILES["resume_model"] = "results/uwpt/model.stopped.npz"
```

resume 只接受**当前 model format + 当前 fixed-network format**。历史格式会明确报错，要求重新训练；代码中不保留转换器或旧训练器。

## 直接 Python 调用

```python
from sdfmpneo import ResearchElectroThermalModel, ResearchTrainingConfig
from sdfmpneo.research import model_from_config

model, config = model_from_config("examples/configs/uwpt_research.json")
report = model.train(config)
model.save("model.npz")

loaded = ResearchElectroThermalModel.load("model.npz")
result = loaded.predict(
    1e6,
    a0=[0.0] * loaded.network.n_modes,
    operating=[5.0, 0.0],
)
```

几何族使用：

```python
from sdfmpneo.geometry_research import geometry_model_from_config

model, config = geometry_model_from_config("case.json")
report = model.train(config)
model.save("geometry_model.npz")

result = model.predict(
    float("inf"),
    geometry={name: value for name, value in zip(model.geometry_names, model.geometry_reference)},
    a0=[0.0] * model.network.n_modes,
    operating=[5.0, 0.0],
)
```

## 代码边界

当前包保留通用解析代数、热/电磁/网格和误差证书模块，因为它们是独立的数学/物理内核。已经废弃的 DAG 搜索训练器、runtime monkey-patch、DAG 专用 C++ 后端和历史 checkpoint 迁移代码不属于当前架构，不再保留。
