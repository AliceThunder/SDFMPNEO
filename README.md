# SDF-MPNEO — 电磁–热多物理代理模型

**Solution-Data-Free Multirate Physics-Embedded Neural Evolution Operator**

0.10.0 面向科学研究，提供从三维材料/网格到无解标签训练、模型保存与加载、任意时间推理及独立数值对照的完整工作流。

研究对象是水下 WPT 的磁准静态电磁场与瞬态热场：铜、封装和海水保留为空间材料，温度改变电导率，电磁焦耳热反馈到热方程。当前范围是电磁–热耦合。

## 推荐调用方式：训练一次，换几何直接查询

根目录的 [`run.py`](run.py) 是默认入口。它把网格、几何族、材料、电磁参数、训练域、保存路径和推理输入集中在文件顶部；通常只需修改这些配置块，不需要维护额外 JSON。

**新的 fresh 训练默认使用固定深度、低秩的解析响应网络。** 网络拓扑和全部连续参数从训练开始就存在，训练只做控制方程残差的连续 LM/Gauss–Newton 优化和独立验证，不再逐个搜索响应神经元，也不再执行正常的 Grow / Enrich / Split / `candidate_search`。时间表示仍是解析的，有限任意时间和 `t=inf` 都直接计算。实现细节见 [`docs/FIXED_ANALYTIC_RESPONSE_NETWORK.md`](docs/FIXED_ANALYTIC_RESPONSE_NETWORK.md)。

### 1. 安装

完整 UWPT 几何训练推荐安装 CAD 和 GUI 依赖：

```bash
python -m pip install -e '.[cad,gui]'
```

只运行测试可额外安装 `dev`：

```bash
python -m pip install -e '.[dev,cad,gui]'
```

Linux 下 Gmsh Python wheel 可能还需要系统 OpenGL/GLU 运行库，例如 Ubuntu：

```bash
sudo apt-get install libgl1 libglu1-mesa
```

### 2. 配置 `run.py`

最常修改的是下面几组配置：

| 配置块 | 用途 |
|---|---|
| `FILES` | 模型、预测结果、日志与继续训练模型路径 |
| `TRANSMITTER` / `RECEIVER` / `ENVIRONMENT` | 参考 UWPT 几何 |
| `GEOMETRY_FAMILY` | 一次训练覆盖的连续几何参数域 |
| `PHYSICS` / `MATERIALS` / `PORTS` | 频率、材料与端口电流映射 |
| `THERMAL_RANK` | 热空间截断阶数 |
| `TRAINING` | 初态、电流、时间窗、残差目标和训练/验证配点 |
| `PREDICTION` | 已保存模型的查询输入 |
| `MONITOR` | 训练窗口、日志与计算线程 |

默认 `GEOMETRY_FAMILY` 同时覆盖 10 个连续几何参数。形状类别、匝数、材料拓扑和频率固定；域内改变线圈尺寸、厚度、接收位置、封装尺寸和海水半径时，使用的是**同一个保存模型**，不需要重新训练。

固定解析网络的默认容量为 4 层、每个热模态 2 个通道、输入二次低秩 8、输入–响应交互秩 4、响应–响应交互秩 3。通常不需要在 `run.py` 增加额外配置；高级容量实验可通过文档中列出的 `SDFMPNEO_FIXED_NETWORK_*` 环境变量覆盖。

### 3. 训练并保存模型

默认训练入口：

```bash
python run.py --mode train
```

默认会打开 PyQt 训练窗口，点击“启动”开始；窗口支持暂停、恢复和停止。

无界面训练：

```bash
python run.py --mode train --headless
```

临时覆盖模型保存路径：

```bash
python run.py --mode train --headless --model results/uwpt/model.npz
```

fresh 固定网络训练的正常阶段应主要是：

```text
initial_residual
weight_refinement
weight_refinement
...
validation
saving
```

fresh 训练日志中不应出现 `candidate_search`。如果出现，说明加载的是旧版非空解析 DAG checkpoint，程序正在使用兼容的历史训练器，而不是把旧模型静默解释成另一种网络。要使用新的无搜索结构，请从 fresh 模型开始训练。

默认输出：

```text
results/uwpt/model.npz              # 保存模型
results/uwpt/model.config.json      # 实际物理配置
results/uwpt/geometry.domain.json   # 几何参数名、参考值、上下界和网格证书
results/uwpt/train.settings.json    # 本次训练设置
results/uwpt/training.report.json   # 残差训练报告
results/uwpt/logs/...               # 训练日志
```

继续训练已有模型时，在 `run.py` 中设置：

```python
FILES["resume_model"] = "results/uwpt/model.npz"
```

继续训练会使用 NPZ 中保存的物理模型、空间基、几何域和已有网络，只采用当前 `TRAINING` 作为新的训练设置；不会重新生成网格。新版 fixed-network checkpoint 会继续优化同一组连续参数；旧版 v1/v2 非空 DAG checkpoint 仍可加载，并继续使用历史 DAG 训练器。

训练达到 `TRAINING["residual_tolerance"]` 时退出码为 `0`。连续优化和独立验证确实仍未达到目标时会保存当前模型和报告并返回退出码 `2`，不会把未收敛写成成功。

### 4. 加载模型并查询

最简单的推理：

```bash
python run.py --mode predict
```

也可显式指定模型：

```bash
python run.py --mode predict --model results/uwpt/model.npz
```

`run.py` 的查询输入全部位于 `PREDICTION`：

```python
PREDICTION = {
    "a0": [0.0, 0.0],
    "operating": [5.0, 0.0],
    "times": [0.0, 0.001, 1.0, 1000.0, 100000.0, 1000000.0, "inf"],
    "geometry": None,
    "allow_time_extrapolation": True,
    "initial_temperature_file": None,
    "state_only": False,
    "allow_extrapolation": False,
}
```

其中：

- `a0`：质量正交热模态初始坐标，长度等于热秩；也可用 `initial_temperature_file` 提供全节点开尔文温度 `.npy`，程序会自动投影。
- `operating`：静态工况 `U`；默认端口映射为 `I=U`，单位为峰值相量 A。
- `times`：任意非负有限时间；可以超过训练时间窗。`"inf"` 表示解析稳态极限，不是一个人为设置的超大有限时间。
- `geometry=None`：查询保存几何域的中心。
- `geometry={...}`：查询指定几何；必须给出保存模型的完整几何参数字典，并位于有效几何域内。
- `state_only=True`：只执行解析网络和温度重构，不组装查询点的电磁诊断；适合大量快速温度查询。
- `allow_extrapolation`：只控制初态/工况是否允许超出训练域；几何仍必须位于保存的有效网格参数域。

推理结果默认写入：

```text
results/uwpt/predictions.json
results/uwpt/predict.settings.json
```

完整推理会返回温度、最高温度、阻抗/电感、各材料焦耳损耗、物理残差和电磁误差诊断。`state_only=True` 时只保留快速状态推理所需路径。

### 5. 查询不同几何

训练完成后，实际几何参数名、参考值和上下界保存在：

```text
results/uwpt/geometry.domain.json
```

默认 10 维几何参数为：

| 参数 | 含义 | 默认训练范围 |
|---|---|---|
| `tx_planar_scale` / `rx_planar_scale` | 发射/接收线圈局部平面缩放 | 0.97–1.03 |
| `tx_thickness_scale` / `rx_thickness_scale` | 各线圈厚度缩放 | 0.95–1.05 |
| `rx_offset_x` / `rx_offset_y` | 接收线圈横向偏移，m | −0.0002–0.0002 |
| `rx_gap` | 两线圈中心的 z 间距，m | 0.0098–0.0102 |
| `tx_package_scale` / `rx_package_scale` | 各封装外表面缩放 | 0.98–1.02 |
| `seawater_radius` | 实际海水外球半径，m | 参考网格半径的 0.98–1.02 倍 |

`planar_scale` 会联动改变线圈外径、匝距和导体宽度；厚度、接收位置、封装尺寸和海水半径独立变化。

若直接使用 Python API，可以从保存模型自动取得几何域中心，再只修改感兴趣的参数：

```python
import numpy as np
from sdfmpneo import ResearchElectroThermalModel

model = ResearchElectroThermalModel.load("results/uwpt/model.npz")

# 保存几何域中心；避免手工填写 seawater_radius 等参考值。
geometry = dict(zip(
    model.geometry_names,
    ((model.lower + model.upper) / 2.0).tolist(),
))

# 在训练几何域内修改任意参数。
geometry["rx_gap"] = 0.0101
geometry["rx_offset_x"] = 0.0001
geometry["tx_planar_scale"] = 1.01

for t in [0.0, 1.0, 1e6, np.inf]:
    result = model.predict(
        t,
        a0=[0.0, 0.0],
        operating=[5.0, 0.0],
        geometry=geometry,
        diagnostics=True,
        allow_time_extrapolation=True,
    )
    print(t, result.maximum_temperature)
```

这段代码不会重新训练模型。改变 `geometry` 后仍使用同一个固定解析网络、共享电磁基和共享热坐标。

### 6. 长时间与稳态查询

默认训练时间窗是 `0–100000 s`，但推理不会把时间截断到训练上限：

```python
PREDICTION["times"] = [0.0, 1.0, 1e5, 1e6, 1e300, "inf"]
```

- `1e6`、`1e300` 等有限时间直接计算解析响应。
- `"inf"` 直接计算稳态极限。
- `allow_time_extrapolation=False` 可强制有限时间不得超过训练窗。
- 超出训练时间窗属于时间外推；程序会计算，但有限训练残差检查并不自动构成域外精度证书。

### 7. 另一套命令行入口

如果更适合用 JSON 配置和纯 CLI，也可以使用 `python -m sdfmpneo`：

```bash
python -m pip install -e '.[dev,cad]'
python examples/create_uwpt_mesh.py examples/configs/uwpt_research.json
python -m sdfmpneo train \
  --config examples/configs/uwpt_research.json \
  --output results/uwpt.npz

python -m sdfmpneo predict results/uwpt.npz \
  --a0 0 0 \
  --operating 5 0 \
  --times 0 0.1 1 1000 1000000 inf \
  --output results/uwpt_predictions.json
```

几何模型还可通过 `--geometry geometry.json` 指定保存几何域内的**完整参数字典**：

```bash
python -m sdfmpneo predict results/uwpt.npz \
  --a0 0 0 \
  --operating 5 0 \
  --times 0 1 1000000 inf \
  --geometry geometry.json \
  --output results/uwpt_predictions.json
```

`--state-only` 跳过完整电磁诊断；`--strict-time-window` 禁止有限时间超出训练窗。旧固定几何模型仍可通过该入口使用 `validate` 做独立 Radau / 全阶稀疏电磁对照。

## 训练窗口与日志

训练默认打开 **PyQt6 实时窗口**，点击“启动”后执行任务；窗口提供暂停、恢复和停止按钮。`MONITOR` 配置控制日志周期、曲线刷新、显示点数和计算线程数。使用 `python run.py --mode train --headless` 可仅训练并记录日志，推理模式保持命令行输出。

实时曲线包括 MSE、训练 RMS/最大残差、独立检查最大残差、固定解析网络逻辑响应通道数和训练配点数。后台训练进程的日志线程周期性写入 JSONL；另一个 `QThread` 增量读取文件，通过信号通知主线程绘图。每次任务的日志保存在 `results/uwpt/logs/<时间_编号>/`。完整操作和日志格式见 [训练监控说明](docs/TRAINING_MONITOR.md)。

训练仅评估给定输入上的控制方程残差，不使用瞬态轨迹、FEM/Maxwell/COMSOL 或实验解标签。原有 `python -m sdfmpneo` 接口也保留，其中固定几何的 `validate` 才调用独立 Radau 积分和全阶稀疏电磁求解；一键脚本不自动执行这类验证。

## 真实线圈、封装与海水网格

`run.py` 默认会自动生成参考 UWPT 网格。设置 `MESH["generate"]=False` 时可导入已有网格；相对路径统一以 `run.py` 所在目录为基准。

配置包含几何尺寸、接收线圈平移/旋转、材料参数、频率、实体端子、热空间截断及训练域。将线圈 `shape` 改为 `rounded_square` 可生成圆角方形线圈。自有 Gmsh 2.2 ASCII 四面体网格也可以直接通过材料和端子物理标签导入。

当前几何代理的形状类别、匝数和材料连接拓扑在一次训练内保持固定；圆形与圆角方形应分别建立自己的参考几何族。它不是“任意换一个 CAD 拓扑都无需重新训练”的代理。

## 模型如何工作

1. 同一四面体网格上的 Nédélec/P1 离散提供电磁算子和热质量/刚度矩阵。
2. 温度相关铜电阻率、海水损耗、热源及其导数由同一材料定义计算。
3. 残差–Riesz 增广建立电磁降阶空间，不采集全阶电磁解快照。
4. 几何族使用共享热坐标，电磁平衡消元后得到 `M_r(G) da/dt = -K_r(G) a + q_em(G,a,U)`；固定几何谱坐标是其特例。
5. fresh 训练使用固定深度低秩解析响应网络表示 `(G,a0,U,t) -> a(t)`；所有网络参数同时连续优化，结构不随训练动态生长。
6. 时间信号直接保持在 `t^k exp(-mu t)` 解析闭包中，因此不需要瞬态时间步进，也不需要为结构候选构造大规模矩阵指数。
