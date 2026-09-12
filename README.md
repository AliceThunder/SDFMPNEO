# SDF-MPNEO — 结构保持神经电磁–热 ROM

当前主路线：

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u)+f_T,
\]

神经网络只学习 Joule 二次型随热状态/几何的变化；热质量、热扩散、电流二次结构和时间动力学仍是显式物理结构。

## 直接运行

普通使用只改根目录 `run.py`：

```bash
python -m pip install -e '.[cad,gui,neural,dev]'

python run.py --mode train
python run.py --mode predict
```

不需要额外手写 JSON。

## PyQt 非阻塞训练

默认：

```bash
python run.py --mode train
```

打开 PyQt 窗口，数值训练在独立 `QProcess` 中执行。支持：

- **启动**：启动后台训练；
- **暂停**：在安全数值边界暂停，进程、CUDA 网络和 optimizer 都保留；
- **恢复**：原地继续；
- **停止**：安全停止并保存续训状态。

纯控制台：

```bash
python run.py --mode train --headless
```

训练输出会持续显示百分比，例如：

```text
生成 UWPT 网格……8%
组装物理模型并构建电磁降阶空间……30%
生成 Joule tensor 标签……47%  (963/2048)
拟合 Joule tensor POD……100%  rank=32
训练神经网络…… 62.0%  epoch=310/500  train=...  val=...
保存神经 ROM……99%
训练完成……100%
```

## CUDA 与训练速度

`run.py` 默认：

```python
TRAINING["device"] = "cuda"
TRAINING["snapshot_workers"] = 4
TRAINING["optimizer"]["batch_size"] = 512
TRAINING["optimizer"]["mixed_precision"] = True
TRAINING["optimizer"]["validation_interval"] = 5
TRAINING["optimizer"]["preload_to_device"] = True
TRAINING["optimizer"]["enable_tf32"] = True
```

神经训练会优先使用：

- CUDA；
- mixed precision；
- fused AdamW（PyTorch/设备支持时）；
- TF32；
- 将 POD 后的小型 input/β tensor 常驻 GPU，避免每个 mini-batch 重复 CPU→GPU 拷贝；
- 降低全量 validation 频率。

若 CUDA 不可用会明确提示并退回 CPU。

## 初始状态范围与训练状态范围

这两个概念现在分开：

```python
TRAINING = {
    # 只描述 a(0)
    "initial_lower": [-0.1, -0.1],
    "initial_upper": [0.1, 0.1],

    # 描述整个加热轨迹中 NN 需要覆盖的 thermal coordinates
    "state_lower": [-2.0, -2.0],
    "state_upper": [2.0, 2.0],
    ...
}
```

以前把 `initial_*` 同时当作整条轨迹训练盒，会造成长时间推理离开 `[-0.1,0.1]` 后直接报错。现在 snapshot 使用 `state_*`，而旧物理构建接口仍使用 `initial_*`。

`EM_CANDIDATE_STATES=None` 时，`run.py` 自动使用 `state_lower / center / state_upper` 构建 EM 热状态锚点。

## 停止后自动续训

默认文件：

```python
FILES = {
    "model": "results/uwpt/model.npz",
    "settings_dir": "results/uwpt",
    "resume_model": None,
    "training_checkpoint": "results/uwpt/model.training.pt",
    ...
}
```

### snapshot 阶段停止

保留：

```text
results/uwpt/quadratic_joule.partial.npz
results/uwpt/quadratic_joule.partial.npz.packed.npy
```

再次运行 `python run.py --mode train` 会继续未完成的物理标签。

### MLP 阶段停止

保存：

```text
results/uwpt/model.training.pt
```

其中包含：

- frozen dataset identity；
- 原 POD 坐标；
- 网络权重；
- AdamW optimizer 状态。

再次执行 `python run.py --mode train` 自动继续。旧 v1/v2/v3 training checkpoint 都能读取；信息不足或网络结构变化时会打印 `[续训兼容]` 提示并安全降级到“复用 dataset/POD、重新初始化不兼容部分”，而不是直接崩溃。

正常训练完成后 `model.training.pt` 自动删除。

## 从已有模型继续训练

```python
FILES["resume_model"] = "results/uwpt/model.npz"
```

然后：

```bash
python run.py --mode train
```

会复用原冻结 Joule tensor dataset，不重新做最昂贵的 EM 标签生成，并尽量恢复原 POD 和网络权重。

## 推理

```bash
python run.py --mode predict
```

默认使用 CUDA，并打印物理可读结果，例如：

```text
加载神经 ROM 推理：...  device=cuda
t=1 s，Tmax=293.16 K (20.01 °C)，最大温升=0.009 K，步数=6
```

对于包含共享 thermal basis 的几何族模型，输出的是重建后的最大节点温度/温升，而不是没有直观意义的 `||a||`。

为了兼容已经用较小 state box 训练出的旧模型，`PREDICTION["allow_extrapolation"]` 默认是 `True`。如果轨迹离开保存的训练盒，会继续计算但打印醒目警告：结果属于 NN 外推。新模型应通过扩大 `TRAINING["state_lower/state_upper"]` 避免长期依赖外推。

如果希望严格禁止外推：

```python
PREDICTION["allow_extrapolation"] = False
```

`times` 中的 `"inf"` 使用独立稳态求解，不是把有限时间设得特别大。

## 主要输出

```text
results/uwpt/model.npz
results/uwpt/model.config.json
results/uwpt/quadratic_joule_dataset.npz
# 数据较大时自动为 quadratic_joule_dataset.store/
results/uwpt/training.report.json
results/uwpt/predictions.json
results/uwpt/logs/
```

## 测试

核心新架构测试可运行：

```bash
python -m pytest -q \
  tests/test_quadratic_joule_tensor.py \
  tests/test_tensor_dataset_pod.py \
  tests/test_resumable_tensor_generator.py \
  tests/test_neural_training_control.py \
  tests/test_neural_training_smoke.py \
  tests/test_neural_integrators.py \
  tests/test_neural_model_persistence.py
```

完整回归：

```bash
python -m pytest -q
```
