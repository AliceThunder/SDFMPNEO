# 结构保持神经电热 ROM

这一实现只做一件事：用普通神经网络替代最难训练的解析响应网络，同时把已知物理结构保留下来。

核心方程：

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u)+f_T,
\]

其中

\[
q_{\theta,j}(a,g,u)=\zeta^T G_{\theta,j}(a,g)\zeta.
\]

神经网络只学习

\[
(a,g)\rightarrow \beta_\theta\rightarrow G_\theta.
\]

它不学习时间、不学习热扩散算子，也不学习电流的二次关系。

## 安装

```bash
python -m pip install -e '.[neural,dev]'
```

项目不使用 GitHub Actions。

## 使用方式

只有三个命令：

```bash
sdfmpneo-neural train --config examples/neural_electrothermal_rom/train.example.json
sdfmpneo-neural retrain --config examples/neural_electrothermal_rom/retrain.example.json
sdfmpneo-neural predict --config examples/neural_electrothermal_rom/predict.example.json
```

### 1. train

配置直接给出训练状态范围和 operating 范围：

```json
{
  "physical_config": "../configs/uwpt_research.json",
  "state_lower": [-1.0, -1.0],
  "state_upper": [1.0, 1.0],
  "operating_lower": [-1.0],
  "operating_upper": [1.0],
  "n_snapshots": 10000,
  "work_directory": "results/neural_rom",
  "pod_relative_tail_tolerance": 1e-4,
  "network": {
    "width": 256,
    "blocks": 4
  },
  "training": {
    "epochs": 500,
    "batch_size": 256,
    "learning_rate": 0.001
  }
}
```

流程就是：

```text
sample (a,g)
-> reduced EM multi-RHS solve
-> exact quadratic Joule tensor G(a,g)
-> POD
-> residual MLP learns (a,g) -> beta
-> save neural ROM
```

固定 `(a,g)` 时一次 multi-RHS EM solve 会得到整个 current quadratic tensor，因此不需要对每个电流工况分别生成标签。

训练阶段 MLP 不调用 EM solver。

### 2. retrain

如果 tensor dataset 已经生成，调整 POD rank、网络宽度、深度或 optimizer 时可以直接：

```bash
sdfmpneo-neural retrain --config examples/neural_electrothermal_rom/retrain.example.json
```

这一步不会重新生成 EM 标签。

### 3. predict

```bash
sdfmpneo-neural predict --config examples/neural_electrothermal_rom/predict.example.json
```

有限时间默认使用 `etd2_adaptive`。也可使用 `etd2` 或 `imex`。

`"inf"` 单独求稳态，不靠无限延长时间积分。

在线模型包含真实 thermal operator，因此预测时不需要重新运行 EM。

## 物理结构

保留的 hard physics 只有必要部分：

- 真实 thermal mass/stiffness：`M_r(g), K_r(g)`；
- 已知 deterministic thermal forcing：`f_T`；
- 电流二次型：`q_j = zeta^T G_j zeta`；
- ODE 初值和连续时间演化；
- 几何变化时重新组装真实 reduced thermal matrices。

网络只是普通 residual MLP，没有特殊解析神经元，也不靠复杂 loss 强行制造物理性。

## 数据量较大时

这部分只是内部性能实现，不改变使用流程：

- tensor 较大时自动用 memmap 存储；
- POD 较大时自动用 out-of-core truncated SVD；
- MLP 训练 epoch 只读取低维 POD coefficients，不反复读取完整 `G`。

如果不需要调整，配置里完全不用关心这些细节。

## 验证

不使用额外的 Gate/certification 体系。

开发时只做普通的三类检查：

1. `zeta^T G zeta` 与原 EM Joule heat 是否一致；
2. validation/test split 上 `G` 和 heat-source 误差；
3. 若需要，选若干典型工况与原 reduced electrothermal ODE 做 trajectory 对比。

核心测试：

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

## 最终结构

```text
(a,g)
  -> ordinary residual MLP
  -> beta
  -> POD reconstruction/contraction
  -> exact current-quadratic layer
  -> q_theta
  -> M_r(g) a_dot = -K_r(g) a + q_theta + f_T
  -> ETD2 / adaptive ETD2 / IMEX
  -> a(t)
```

这就是主路线，不再增加额外流程层。
