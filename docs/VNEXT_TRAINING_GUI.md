# vNext 一键训练、缓存与 PyQt 控制界面

当前 vNext 训练入口统一为仓库根目录的 `run.py`。面向实验的配置全部集中在
`run.py::CONFIG`，数值模块不保存具体实验参数。

## 安装

```bash
pip install -e ".[dev,neural,gui]"
```

`neural` 提供 PyTorch，`gui` 提供 PyQt6 与 pyqtgraph。

## 一键启动

```bash
python run.py
```

默认 `CONFIG["RUN"]["mode"] = "gui"`，会打开训练控制窗口。训练本身不在 GUI
线程运行，而是由 `QProcess` 启动独立 Python worker；JSONL metrics 由独立
`QThread` 增量读取，因此 teacher 求解、CUDA 训练和磁盘 checkpoint 均不会阻塞
Qt 主线程。

也可以完全不启动 GUI：

```bash
python run.py --mode train
```

## 控制按钮

GUI 提供：

- **启动 / 续训**：如果兼容 checkpoint 存在，从保存的 batch 边界继续；
- **暂停**：在当前安全 batch 或 teacher chunk 完成后进入暂停状态；
- **恢复**：继续同一个 worker；
- **停止**：在安全边界保存 checkpoint 后退出 worker；再次点击“启动 / 续训”即可继续；
- **忽略模型 checkpoint 重新训练**：只重置神经训练，不删除昂贵 teacher 数据缓存。

模型 checkpoint 保存了模型、optimizer、best model、early-stop 状态、当前 epoch、
batch 调度、下一 batch 位置和 NumPy RNG 状态，因此停止后的恢复不是简单从上一
完整 epoch 重新开始。同一 GUI 会话中停止后再次启动时，已有训练曲线也会保留并
继续追加；选择“重新训练”时才清空当前曲线。

## Teacher 数据缓存

缓存目录由：

```python
CONFIG["CACHE"]["root"]
```

控制。默认策略：

```python
CONFIG["CACHE"]["policy"] = "reuse"
```

缓存是内容寻址的。以下内容参与缓存身份：

- `DATA.seed`；
- `SAMPLER` 参数空间；
- `TEACHER` MQS 数值配置；
- `TRUTH` baseline / SIE / volume / background truth 配置。

以下内容**不参与 teacher 缓存身份**：

- `DATA.count`；
- `DATA.workers`；
- `DATA.native_threads_per_worker`；
- CUDA/CPU device；
- 网络宽度、深度；
- batch size；
- learning rate；
- epoch 数；
- GUI 配置；
- bundle / log 输出路径。

因此把 `count` 从 64 增加到 256 时，只会补 index 64..255；修改网络或 GPU 参数
不会重新跑 correctness teacher。

缓存每个 deterministic sample 独立保存为 shard，并在 manifest 中记录 SHA-256。
启用 `verify_checksums` 时损坏 shard 会被当成缺失样本重新生成，而不是静默使用。

缓存策略：

```python
"CACHE": {
    "policy": "reuse",    # 默认：复用 + 补缺
    # "policy": "refresh", # 当前参数空间强制重建
    # "policy": "readonly",# 只读，缺任何 shard 就报错
}
```

## CPU 并行和防止过度占核

Teacher correctness solve 使用多进程：

```python
"DATA": {
    "workers": 8,
    "native_threads_per_worker": 1,
    "generation_chunk_size": 8,
}
```

当 `workers > 1` 时，通常建议 `native_threads_per_worker = 1`，避免 8 个 worker 各自
再开 8 个 MKL/OpenBLAS 线程造成 64 路过度订阅。`generation_chunk_size` 决定暂停/
停止命令在 teacher 阶段的响应粒度。

## CUDA / 精度

全局默认：

```python
"RUNTIME": {
    "device": "auto",
}
```

`auto` 的选择顺序是 CUDA -> MPS -> CPU。`PORT_TRAINING.device` 和
`SPATIAL_TRAINING.device` 默认写成 `inherit`，继承该设置。

Port 训练支持：

```python
"precision": "auto"
```

其中 CUDA/MPS 默认 float32，CPU 默认 float64。也可以显式使用 `float32` 或
`float64`。Artifact 保存/加载会保留实际模型 dtype。

## Checkpoint 兼容规则

Teacher 缓存和模型 checkpoint 是两套独立身份。

Teacher 参数空间不变但 `count` 增加时：

- teacher cache 继续复用；
- 模型 checkpoint 会因为训练数据集合变大而失效，重新训练模型。

修改网络结构、optimizer、batch size、precision、训练/验证划分等关键训练语义时：

- teacher cache 继续复用；
- 不兼容的模型 checkpoint 不会被错误载入。

训练处于 `running`/`stopped` 状态时可以从精确 batch 边界续训。如果一次训练已经
正常 `completed` 并选出了 best model，则该完成 checkpoint 视为终态，不自动把
best-state 与最后一个 optimizer-state 拼接起来继续训练。此时若要增加 epoch，使用
“忽略模型 checkpoint 重新训练”即可；昂贵的 teacher 数据仍从原缓存直接复用。

## 主要可配置区域

`run.py` 中依次包含：

1. `RUN`：默认 GUI / headless 模式；
2. `FILES`：日志、checkpoint、artifact、bundle；
3. `CACHE`：缓存目录和策略；
4. `DATA`：样本数、多进程和 native thread 数；
5. `SAMPLER`：几何、材料、频率和 tensor-electric 参数空间；
6. `TEACHER`：MQS correctness 配置；
7. `TRUTH`：port/spatial truth 分辨率；
8. `RUNTIME`：device 与 worker 环境变量；
9. `TRAINING`：resume 总开关；
10. `PORT_TRAINING`：port 网络、optimizer、batch/early-stop；
11. `SPATIAL_TRAINING`：spatial 网络、optimizer、归一化积分阶数；
12. `CONTROL`：心跳/metrics 落盘；
13. `GUI`：刷新率、窗口尺寸、曲线点数等。

## 输出

完成训练后默认产生：

```text
results/vnext/tensor_port.pt
results/vnext/tensor_spatial.pt
results/vnext/tensor_bundle/
results/vnext/training_summary.json
results/vnext/checkpoints/
results/vnext/teacher_cache/
results/vnext/logs/<session>/metrics.jsonl
```

推理仍可使用 tensor bundle：

```bash
sdfmpneo-vnext-tensor results/vnext/tensor_bundle request.json --device auto
```