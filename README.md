# SDF-MPNEO — 结构保持神经电磁–热 ROM

当前主路线是结构保持神经电热降阶模型：

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u)+f_T,
\]

其中神经网络只学习电磁 Joule 二次型系数随热状态/几何的变化：

\[
(a,g)\rightarrow \beta_\theta\rightarrow G_\theta,
\qquad
q_{\theta,j}=\zeta^T G_{\theta,j}\zeta.
\]

热质量矩阵、热扩散矩阵、电流二次结构和连续时间动力学都保留为显式物理结构。

## 推荐用法：只使用 `run.py`

不需要自己写 JSON，也不需要调用底层 CLI。

修改根目录 [`run.py`](run.py) 顶部配置后直接运行：

```bash
python run.py --mode train
python run.py --mode predict
```

`run.py` 会自动生成底层 `model.config.json`、Joule tensor 数据集、训练报告和最终模型。

## 1. 安装

```bash
python -m pip install -e '.[cad,neural,dev]'
```

若只使用已有网格，可不安装 CAD 额外依赖；若使用 CUDA，请安装与你机器 CUDA 版本匹配的 PyTorch。

Linux 下 Gmsh wheel 可能还需要：

```bash
sudo apt-get install libgl1 libglu1-mesa
```

## 2. `run.py` 里需要改什么

常用配置块：

| 配置 | 用途 |
|---|---|
| `FILES` | 模型、预测结果和结果目录 |
| `MESH` | 是否重新生成网格、网格尺寸 |
| `TRANSMITTER` / `RECEIVER` / `ENVIRONMENT` | UWPT 参考几何 |
| `GEOMETRY_FAMILY` | 连续几何参数范围 |
| `PHYSICS` / `MATERIALS` / `PORTS` | 电磁、材料和端口参数 |
| `THERMAL_RANK` | 热降阶维数 |
| `TRAINING` | 状态域、工况域、snapshot/POD/MLP 训练参数 |
| `PREDICTION` | 推理初态、工况、时间、几何和积分参数 |

### `TRAINING`

示例：

```python
TRAINING = {
    "initial_lower": [-0.1, -0.1],
    "initial_upper": [0.1, 0.1],
    "operating_lower": [0.0, 0.0],
    "operating_upper": [10.0, 10.0],

    "n_snapshots": 2048,
    "seed": 17,
    "snapshot_workers": 1,

    "pod_rank": None,
    "pod_relative_tail_tolerance": 1e-4,

    "device": None,       # None=自动；也可 "cuda" / "cpu"

    "network": {
        "width": 256,
        "blocks": 4,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 500,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "heat_loss_weight": 0.25,
        "patience": 50,
        "seed": 17,
        "dtype": "float32",
        "mixed_precision": False,
    },
}
```

为兼容原来的 `run.py`，仍使用 `initial_lower/initial_upper` 这个字段名；在新模型中它表示 **NN 学习的热状态坐标范围**，应覆盖实际轨迹，而不仅仅是初始状态。

`THERMAL_RANK=r` 时，`initial_lower/initial_upper` 都必须包含 `r` 个元素。

## 3. 训练

```bash
python run.py --mode train
```

程序自动执行：

```text
按 run.py 生成/读取 UWPT 网格
        ↓
自动写 results/uwpt/model.config.json
        ↓
构建 thermal ROM + reduced EM
        ↓
采样 (a,g)
        ↓
一次 multi-RHS EM solve 构造 G(a,g)
        ↓
POD 压缩 Joule tensor
        ↓
普通 residual MLP 学习 (a,g) -> beta
        ↓
保存 results/uwpt/model.npz
```

默认输出：

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/quadratic_joule_dataset.npz
# 数据较大时上面会自动变成 quadratic_joule_dataset.store/
results/uwpt/train.settings.json
results/uwpt/training.report.json
```

snapshot 生成支持断点文件；如果完整 tensor dataset 已存在且配置一致，重新执行 `train` 会直接复用，不重新计算最昂贵的 EM 标签。

第一次只是检查整条链时，可以先把：

```python
TRAINING["n_snapshots"] = 512
TRAINING["optimizer"]["epochs"] = 50
```

跑通后再增加样本和 epoch。

## 4. 推理

配置 `run.py`：

```python
PREDICTION = {
    "a0": [0.0, 0.0],
    "operating": [5.0, 0.0],
    "times": [0.0, 0.001, 1.0, 1000.0, 100000.0, "inf"],
    "geometry": None,

    "method": "etd2_adaptive",
    "max_step": 100.0,
    "initial_step": 0.001,
    "rtol": 1e-5,
    "atol": 1e-8,
    "allow_extrapolation": False,
}
```

然后：

```bash
python run.py --mode predict
```

输出：

```text
results/uwpt/predictions.json
results/uwpt/predict.settings.json
```

`"inf"` 使用独立稳态求解，不是设置一个很大的有限时间。

### 几何参数

`geometry=None` 表示几何训练盒中心。

也可以直接按物理参数名填写，例如：

```python
PREDICTION["geometry"] = {
    "rx_gap": 0.0101,
    "rx_offset_x": 0.0001,
    "tx_planar_scale": 1.01,
}
```

未填写的几何参数使用参考值；`run.py` 会自动转换为网络内部的 `[-1,1]` 归一化坐标。

## 5. 模型实际学习什么

网络不直接学习时间轨迹，也不学习热扩散：

```text
(a,g)
  -> ordinary residual MLP
  -> beta
  -> POD contraction
  -> exact current-quadratic layer
  -> q_theta
  -> M_r(g) a_dot = -K_r(g) a + q_theta + f_T
  -> ETD2 / adaptive ETD2 / IMEX
  -> a(t)
```

因此：

- 初始条件由 ODE 直接给定；
- `M_r(g),K_r(g)` 是真实热算子；
- 电流依赖严格保持二次型；
- 神经网络只负责平滑的 `(a,g) -> G(a,g)` 非线性部分。

## 6. 测试

新架构核心测试：

```bash
python -m pytest -q \
  tests/test_quadratic_joule_tensor.py \
  tests/test_quadratic_joule_direct_reduced.py \
  tests/test_tensor_dataset_pod.py \
  tests/test_resumable_tensor_generator.py \
  tests/test_out_of_core_neural_tensor_pipeline.py \
  tests/test_neural_tensor_physical_layer.py \
  tests/test_neural_training_smoke.py \
  tests/test_neural_integrators.py \
  tests/test_neural_model_persistence.py \
  tests/test_neural_geometry_persistence.py \
  tests/test_neural_batch_inference.py \
  tests/test_neural_thermal_forcing.py
```

完整回归：

```bash
python -m pytest -q
```

## 7. 底层 CLI

仓库仍保留 `sdfmpneo-neural` 作为调试/脚本化底层入口，但**普通使用不需要它**。推荐入口始终是：

```bash
python run.py --mode train
python run.py --mode predict
```
