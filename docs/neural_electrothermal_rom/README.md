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
sdfmpneo-neural predict --config examples/neural_electrothermal_rom/predict.example.json
sdfmpneo-neural audit   --config examples/neural_electrothermal_rom/audit.example.json
```

项目不使用 GitHub Actions；验证入口保留为本地 `pytest` 与 `audit`。

## 1. Train

`train` 从现有物理 JSON 构建 EM/thermal ROM，但 EM 只用于离线生成

\[
(a,g)\mapsto G(a,g).
\]

随后保存：

- `quadratic_joule.partial.npz`：可恢复标签生成检查点；
- `quadratic_joule_dataset.npz/.json`：冻结 train/validation/test 数据集；
- neural `.npz`：POD、归一化、MLP、训练域、物理签名和在线热算子；
- `training.report.json`：训练与数据版本摘要。

网络训练阶段不会调用 EM solver，也不生成瞬态轨迹标签。

### 最重要的训练域约束

`state_lower/state_upper` 表示 **NN 允许看到的完整热状态域**，不是只表示初始条件范围。
必须覆盖目标初值、工况和几何下的可达状态以及稳态附近区域。模型默认在每个 ETD/IMEX 中间阶段检查该盒；轨迹一旦离开训练状态域立即报错。

因此不要机械地把旧 `initial_lower/initial_upper` 复制为 state box，除非已经证明目标轨迹始终留在其中。

几何族训练时网络几何坐标固定为归一化 `[-1,1]^d`；模型文件内保存 affine geometry chart、热材料参数和共享热基，在线精确重组 `M_r(g),K_r(g)`，不依赖 EM。

## 2. Predict

`predict` 只加载 neural `.npz`。对于固定几何和当前 affine geometry family 都不需要物理 JSON。

有限时间默认使用广义 ETD2：

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u).
\]

`times` 中字符串 `"inf"` 会调用独立稳态求解器，而不是把有限时间积分无限延长。

默认 `allow_extrapolation=false`：

- state 必须留在训练 state box；
- operating 必须在训练工况盒；
- geometry 必须在已认证的归一化 `[-1,1]^d` chart 内。

即使显式打开 state/operating extrapolation，embedded geometry family 仍不会允许超出已保存几何 chart。

## 3. Audit / Gate 1--7

`audit` 会重新构建真实物理模型，仅用于独立对照，并验证 neural 模型保存的物理签名。

当前自动执行：

1. **Gate 1**：`zeta^T G zeta` 与直接 EM-ROM Joule heat 数值恒等；
2. **Gate 2**：只用 train split 拟合 SVD，对 validation split 做 POD rank sweep，并报告 tensor/heat error；
3. **Gate 3**：真实 `G(a,g)` 的有限差分敏感度与 active-subspace spectrum；
4. **Gate 4**：真实物理 Jacobian 在 `M(g)` 能量范数下的 logarithmic norm；
5. **Gate 5**：冻结 test split 上 `G/q/F` 的 RMS、95/99 percentile 和 maximum；
6. **Gate 6**：neural ETD 与真实 EM-ROM + Radau 轨迹对照，可同时重构节点温度；
7. **Gate 7**：真实 wall-clock vector-field / trajectory benchmark。

Production readiness 是 fail-closed：没有 Gate 7 的明确时间预算和实际测量，结果不会被判为 ready。

`reproducible_training` 与 `persistence_roundtrip` 也必须由正式实验明确填写，不能自动假定为真。

## 4. 模型物理结构

在线链条固定为：

```text
(a, normalized geometry)
        -> ordinary residual MLP
        -> normalized POD coefficients
        -> POD decode in Frobenius-isometric svec coordinates
        -> exact current quadratic contraction zeta^T G zeta
        -> q_theta
        -> exact M_r(g) a_dot = -K_r(g) a + q_theta
        -> generalized ETD2 / IMEX
```

网络不直接输入时间，也不直接输入电流；电流只进入 hard quadratic physics layer。

## 5. 本地测试

新模块测试：

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
  tests/test_neural_gate_suite.py
```

完整仓库回归仍应运行：

```bash
python -m pytest -q
```

## 6. Production 切换原则

当前实现分支提供完整训练、推理和 audit 能力，但不会自动替换旧 production 默认入口。
只有正式 UWPT 数据上的 Gate 1--7 全部满足预先给定的工程预算后，才允许切换默认模型。
