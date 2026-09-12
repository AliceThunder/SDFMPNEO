# 结构保持神经电热 ROM：实现使用说明

本目录的 LaTeX 文档定义理论；本页说明当前实现的可执行工作流。

## 安装

神经训练额外依赖 PyTorch：

```bash
python -m pip install -e '.[neural,dev]'
```

新路线使用独立命令，不经过旧解析响应网络训练器：

```bash
sdfmpneo-neural train   --config examples/neural_electrothermal_rom/train.example.json
sdfmpneo-neural retrain --config examples/neural_electrothermal_rom/retrain.example.json
sdfmpneo-neural predict --config examples/neural_electrothermal_rom/predict.example.json
sdfmpneo-neural audit   --config examples/neural_electrothermal_rom/audit.example.json
```

项目不使用 GitHub Actions；验证入口保留为本地 `pytest` 与 `audit`。

## 1. Train：只生成局部物理张量标签

`train` 从现有物理 JSON 构建 EM/thermal ROM，但 EM 只用于离线生成

\[
(a,g)\mapsto G(a,g).
\]

随后保存：

- `quadratic_joule.partial.npz`：可恢复标签生成检查点；
- `quadratic_joule_dataset.npz/.json`：冻结 train/validation/test 数据集；
- neural `.npz`：POD、归一化、MLP、训练域、物理签名和在线热算子；
- `training.report.json`：训练、采样和数据版本摘要。

网络训练阶段不会调用 EM solver，也不使用瞬态轨迹作为监督标签。

### 1.1 完整 state box，而不是 initial-state box

`state_lower/state_upper` 表示 **NN 允许看到的完整热状态域**。它必须覆盖目标初值、工况和几何下的可达状态以及稳态附近区域。

模型默认在 ETD/IMEX 的每个中间阶段检查该盒；轨迹一旦离开训练状态域立即报错。因此不要机械复制旧 `initial_lower/initial_upper`，除非已经证明整个目标轨迹都留在其中。

### 1.2 高维状态推荐 hybrid reachable sampling

默认 `sampling.strategy="box"` 使用整个 state/geometry box 的 LHS。对于约 198 维热状态，这通常样本效率很低。

推荐先声明一个足够保守的完整 state box，然后用：

```json
"sampling": {
  "strategy": "hybrid_reachable",
  "initial_lower": [-0.1, -0.1],
  "initial_upper": [0.1, 0.1],
  "box_fraction": 0.25,
  "trajectory_count": 32,
  "samples_per_trajectory": 16,
  "time_horizon": 1000.0,
  "time_min": 1e-6
}
```

其中真实 reduced physics 轨迹**只用于选择在哪里采样 $G$**，轨迹状态不会成为 NN 的监督 target。仍保留 `box_fraction` 的全盒样本以覆盖 off-manifold / restart 区域。

若任何真实采样轨迹离开声明的 state box，训练直接抛出 `StateDomainInsufficientError`，并提供观测到的 state envelope；实现不会通过 clip 偷偷隐藏训练域设计错误。

几何族训练时网络几何坐标固定为归一化 `[-1,1]^d`；模型文件保存 affine geometry chart、热材料参数和共享热基，在线精确重组 `M_r(g),K_r(g)`，不依赖 EM。

## 2. Retrain：调网络/POD 不再重建 EM

物理 tensor dataset 一旦生成，可以反复做：

```bash
sdfmpneo-neural retrain --config examples/neural_electrothermal_rom/retrain.example.json
```

`retrain` 只读取：

- 冻结 `quadratic_joule_dataset.npz`；
- 一个已经保存的 neural model 作为在线 thermal operator / physical signature 模板。

它不会构建 EM problem、不会重新生成标签。POD rank、MLP width/depth、optimizer、mixed precision 等实验都应该走这条路径。

## 3. Predict：自包含在线模型

`predict` 只加载 neural `.npz`。固定几何和当前 affine geometry family 都不需要物理 JSON 或 EM ROM。

连续时间模型始终是

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u).
\]

推荐有限时间方法为 `etd2_adaptive`：广义谱分解每个 geometry 只做一次，之后用 ETD1/ETD2 嵌入误差指标自动调整步长。热响应进入慢尾部后会自然使用大步长；`rtol/atol/max_step` 仍由用户显式控制。

固定步长 `etd2` 和鲁棒参考 `imex` 仍保留。`times` 中字符串 `"inf"` 调用独立稳态求解器，不把有限时间积分无限延长。

默认 `allow_extrapolation=false`：

- state 必须留在训练 state box；
- operating 必须在训练工况盒；
- geometry 必须在已认证的归一化 `[-1,1]^d` chart 内。

即使显式打开 state/operating extrapolation，embedded geometry family 仍不会允许超出保存的几何 chart。

## 4. Audit / Gate 1--7

`audit` 会重新构建真实物理模型，仅用于独立对照，并验证 neural 模型保存的物理签名。

当前自动执行：

1. **Gate 1**：`zeta^T G zeta` 与直接 EM-ROM Joule heat 数值恒等；
2. **Gate 2**：只用 train split 拟合 SVD，对 validation split 做 POD rank sweep，并报告 tensor/heat error；
3. **Gate 3**：真实 `G(a,g)` 的有限差分敏感度与 active-subspace spectrum；
4. **Gate 4**：真实物理 Jacobian 在 `M(g)` 能量范数下的 logarithmic norm；
5. **Gate 5**：冻结 test split 上 `G/q/F` 的 RMS、95/99 percentile 和 maximum；
6. **Gate 6**：配置指定的 neural ETD/IMEX 与真实 EM-ROM + Radau 轨迹对照，可同时重构节点温度；
7. **Gate 7**：使用与 Gate 6 相同 production integrator 的真实 wall-clock vector-field / trajectory benchmark。

Production readiness 是 fail-closed：没有 Gate 7 的明确时间预算和实际测量，结果不会判为 ready。

`reproducible_training` 与 `persistence_roundtrip` 也必须由正式实验明确填写，不能自动假定为真。

## 5. 在线结构

```text
(a, normalized geometry)
        -> ordinary residual MLP
        -> normalized POD coefficients
        -> direct POD/current contraction (不还原完整 G)
        -> exact current quadratic layer zeta^T G zeta
        -> q_theta
        -> exact M_r(g) a_dot = -K_r(g) a + q_theta
        -> generalized fixed/adaptive ETD2 / IMEX
```

网络不直接输入时间，也不直接输入电流；电流只进入 hard quadratic physics layer。

## 6. 本地测试

```bash
python -m pytest -q \
  tests/test_quadratic_joule_tensor.py \
  tests/test_tensor_dataset_pod.py \
  tests/test_resumable_tensor_generator.py \
  tests/test_neural_tensor_physical_layer.py \
  tests/test_neural_integrators.py \
  tests/test_neural_training_smoke.py \
  tests/test_neural_model_persistence.py \
  tests/test_neural_geometry_persistence.py \
  tests/test_neural_snapshot_sampling.py \
  tests/test_neural_gate_suite.py
```

完整仓库回归仍应运行：

```bash
python -m pytest -q
```

## 7. Production 切换原则

当前实现分支提供训练、无 EM retrain、自包含推理和 Gate audit 能力，但不会自动替换旧 production 默认入口。只有正式 UWPT 数据上的 Gate 1--7 全部满足**预先给定**的工程预算后，才允许切换默认模型。
