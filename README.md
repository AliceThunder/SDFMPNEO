# SDF-MPNEO — 结构保持神经电磁–热 ROM

当前主路线：

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u)+f_T,
\]

神经网络只学习 Joule 二次型系数随热状态/几何的变化；热质量、热扩散、电流二次结构和连续时间动力学都保留为显式物理结构。

## 推荐入口：只使用 `run.py`

不需要手写额外 JSON，也不需要直接调用底层 CLI。

```bash
python -m pip install -e '.[cad,gui,neural,dev]'

python run.py --mode train
python run.py --mode predict
```

所有常用配置都集中在 [`run.py`](run.py) 顶部。

## PyQt 非阻塞训练窗口

默认执行：

```bash
python run.py --mode train
```

会打开原来的 PyQt 训练窗口。数值训练在独立 `QProcess` 中运行，所以 GUI 不阻塞。

窗口保留四个按钮：

- **启动**：启动后台训练进程；
- **暂停**：在当前不可拆分数值操作结束后的安全边界暂停；
- **恢复**：同一后台进程原地继续，GPU 模型和 AdamW 状态不重新创建；
- **停止**：安全停止并保存续训状态。

无 GUI：

```bash
python run.py --mode train --headless
```

显式要求 GUI：

```bash
python run.py --mode train --gui
```

## 停止后的自动续训

`run.py` 默认配置：

```python
FILES = {
    "model": "results/uwpt/model.npz",
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",
    "resume_model": None,
    "training_checkpoint": "results/uwpt/model.training.pt",
}
```

训练停止时分两种情况：

1. **Joule tensor snapshot 生成阶段停止**：已有 `quadratic_joule.partial.npz` / packed sidecar 会保留；再次运行 `python run.py --mode train` 会从未完成样本继续。
2. **MLP/AdamW 阶段停止**：自动保存 `model.training.pt`，其中包含同一个 POD、网络权重和 AdamW 状态；再次运行 `python run.py --mode train` 会自动继续。

正常训练完成后，`model.training.pt` 会自动删除。

## 从已有 `.npz` 模型继续训练

若已经有完整神经 ROM，希望追加 epoch，设置：

```python
FILES["resume_model"] = "results/uwpt/model.npz"
```

然后：

```bash
python run.py --mode train
```

此模式会：

- 加载已有模型的网络权重；
- 复用已有模型的 POD；
- 复用 `results/uwpt/quadratic_joule_dataset.npz` 或 `.store/`；
- 不重新生成 EM 标签；
- 使用当前 `TRAINING["optimizer"]` 作为本次追加训练参数。

因此 `epochs` 在续训时表示**本次最多追加的 epoch 数**。

## `TRAINING`

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

为兼容原 `run.py`，仍使用 `initial_lower/initial_upper` 名称；在新模型里它表示 NN 覆盖的热状态坐标范围，应覆盖实际轨迹。

## 训练流程

```text
run.py 配置
   ↓
自动生成/读取 UWPT 网格和内部 model.config.json
   ↓
thermal ROM + reduced EM
   ↓
生成/恢复 G(a,g) snapshots
   ↓
POD
   ↓
普通 residual MLP: (a,g) -> beta
   ↓
quadratic-current physical layer
   ↓
保存 model.npz
```

默认输出：

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/quadratic_joule_dataset.npz
# 大数据时自动为 quadratic_joule_dataset.store/
results/uwpt/train.settings.json
results/uwpt/training.report.json
results/uwpt/logs/...
```

## 推理

仍然只改 `run.py` 的 `PREDICTION`：

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

`"inf"` 使用独立稳态求解。

## 测试

核心新架构测试包括：

```bash
python -m pytest -q \
  tests/test_quadratic_joule_tensor.py \
  tests/test_tensor_dataset_pod.py \
  tests/test_resumable_tensor_generator.py \
  tests/test_neural_training_smoke.py \
  tests/test_neural_training_control.py \
  tests/test_neural_integrators.py \
  tests/test_neural_model_persistence.py
```

完整回归：

```bash
python -m pytest -q
```

普通使用不需要 `sdfmpneo-neural`；推荐入口始终是 `run.py`。
