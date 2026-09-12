# 结构保持神经电热 ROM：实现使用说明

本目录中的 LaTeX 文档定义冻结理论；本页说明当前实现工作流。新路线与旧解析响应网络完全分离，主模型为

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u),
\]

其中普通 residual MLP 只学习 `(a,g) -> POD coefficients`，电流只进入严格二次型物理层，`M_r(g),K_r(g)` 始终来自真实热降阶模型。

## 安装

```bash
python -m pip install -e '.[neural,dev]'
```

项目不使用 GitHub Actions。验证统一通过本地 `pytest`、开发 audit 和正式 certification 完成。

推荐工作流：

```bash
sdfmpneo-domain probe --config examples/neural_electrothermal_rom/domain_probe.example.json
sdfmpneo-domain train --config examples/neural_electrothermal_rom/train_from_domain.example.json
sdfmpneo-neural retrain --config examples/neural_electrothermal_rom/retrain.example.json
sdfmpneo-neural predict --config examples/neural_electrothermal_rom/predict.example.json
sdfmpneo-neural audit --config examples/neural_electrothermal_rom/audit.example.json
sdfmpneo-certify audit --config examples/neural_electrothermal_rom/certify.example.json
sdfmpneo-diagnose --config examples/neural_electrothermal_rom/diagnose.example.json
```

---

## 1. Domain probe：先确定完整 neural state domain

高维热 ROM 不推荐人工填写几十到上百维 `state_lower/state_upper`。`sdfmpneo-domain probe` 使用真实 reduced electrothermal vector field，在初值盒、完整 operating box、固定几何或归一化 geometry box `[-1,1]^d` 上运行真实 Radau 轨迹，并可从轨迹末端继续求稳态。

输出包含：

- initial + transient + successful steady-state 的观测 envelope；
- 每个 thermal coordinate 的安全 margin；
- `suggested_state_lower/upper`；
- steady-state 成功/失败统计；
- physical signature；
- geometry / operating domain；
- report 内容 SHA-256。

Domain report 是冻结工件。`sdfmpneo-domain train` 会检查：

1. report hash 未被修改；
2. physical signature 匹配；
3. thermal rank 匹配；
4. geometry domain 匹配；
5. operating domain 匹配。

report hash 会继续进入 dataset 和 model metadata，形成

```text
domain report -> frozen tensor dataset -> neural model
```

的 provenance 链。

Domain probe 是采样覆盖证据，不是数学可达集证明。正式 Gate 6 仍必须独立覆盖目标短时、长时和边界工况。

---

## 2. Train：生成局部物理 tensor 标签

离线标签是

\[
(a,g)\mapsto G(a,g),
\]

其中每个 thermal modal Joule source 满足

\[
q_j(a,g,u)=\zeta^T G_j(a,g)\zeta.
\]

固定 `(a,g)` 时使用一次 multi-RHS reduced EM solve 构造整个 current quadratic tensor，不需要逐电流采样。

训练输出包括：

- `quadratic_joule.partial.npz`：可恢复 snapshot 检查点；
- `quadratic_joule_dataset.npz/.json`：冻结 train/validation/test dataset；
- neural `.npz`：POD、MLP、训练域、物理签名和在线 thermal operators；
- training report：loss、配置、随机种子、precision、软件/硬件信息等。

冻结 dataset 同时保存 physical/ROM provenance：thermal backend/rank、EM reduced rank、频率、constitutive budget、current dimension、EM reduction certificate 摘要、geometry certificate 摘要和源码 revision。大数组保存 shape/dtype/hash，不复制到 manifest。

网络训练阶段**不调用 EM solver**，也不使用瞬态轨迹值作为监督 target。

### 2.1 完整 state box，不是 initial-state box

训练域是 NN 允许出现的完整 thermal state 域，必须覆盖目标初值、真实轨迹、restart/off-manifold 区域和稳态附近区域。

ETD/IMEX 在每个实际 stage 检查该盒。默认一旦离开训练域立即失败；不会 silently clip。

### 2.2 高维推荐 hybrid reachable sampling

在约 198 维 thermal coordinate 中对完整 box 纯 LHS 会浪费大量样本。正式训练推荐同时使用：

- 一部分 full-box samples；
- 一部分真实 reduced-physics trajectory states。

轨迹仅决定“在哪里生成 G 标签”，不是 trajectory supervision。

示例：

```json
"sampling": {
  "strategy": "hybrid_reachable",
  "initial_lower": [-0.1, -0.1],
  "initial_upper": [0.1, 0.1],
  "box_fraction": 0.25,
  "trajectory_count": 64,
  "samples_per_trajectory": 24,
  "time_horizon": 100000.0,
  "time_min": 1e-6
}
```

如果真实采样轨迹离开冻结 state box，抛出 `StateDomainInsufficientError`，应重新做更保守的 domain probe。

---

## 3. Retrain：调 POD/MLP 不再生成 EM 标签

冻结 tensor dataset 一旦生成，POD rank、MLP width/depth、optimizer、mixed precision 等实验走：

```bash
sdfmpneo-neural retrain --config examples/neural_electrothermal_rom/retrain.example.json
```

`retrain` 只读取 frozen dataset 和一个 neural model 作为在线 thermal operator 模板，不构建 EM，不生成新标签。

重新训练继承 dataset / physics provenance，但不会继承旧 certification；任何权重变化后必须重新 audit/certify。

---

## 4. Predict：自包含在线模型

```bash
sdfmpneo-neural predict --config examples/neural_electrothermal_rom/predict.example.json
```

在线 `.npz` 不需要物理 JSON 或 EM-ROM。固定几何和 affine geometry family 都保存足够的 thermal operator 信息来在线构造

\[
M_r(g),\quad K_r(g).
\]

有限时间默认推荐 `etd2_adaptive`：每个 geometry 做一次 generalized thermal eigendecomposition，随后用 ETD1/ETD2 嵌入误差指标调整步长。长时间进入慢尾部后可自动增大步长。

其他方法：

- `etd2`：固定步长二阶指数积分；
- `imex`：线性热扩散隐式、NN source 显式；
- `reference`：仅研究/验证用的通用参考积分器。

`"inf"` 使用独立 steady-state solver，不把有限时间积分无限延长。

默认 `allow_extrapolation=false`：state、operating 和 geometry 均须留在训练/认证域内。

### 4.1 批量固定步长 ETD2

共享 geometry 和目标时间的独立查询可以：

```python
batch = model.predict_batch_fixed_etd2(
    100.0,
    initial_states=A0,
    geometry=g,
    operating=U,
    max_step=1.0,
)
```

批量路径共享一次 generalized thermal spectrum；每个 ETD stage 对整批状态做一次 MLP forward，最终导数用 multi-RHS thermal solve。时间步本身仍严格串行，保持真实因果依赖。

---

## 5. 开发 Audit：Gate 1--7

```bash
sdfmpneo-neural audit --config examples/neural_electrothermal_rom/audit.example.json
```

开发 audit 重新加载真实物理模型，仅用于独立对照：

1. **Gate 1**：`zeta^T G zeta` 与直接 EM-ROM Joule heat 恒等；
2. **Gate 2**：train split 拟合 POD，validation split 做 rank sweep；
3. **Gate 3**：真实 `G(a,g)` 对 thermal state 的 sensitivity / active-subspace 分析；
4. **Gate 4**：真实 physical Jacobian 在 `M(g)` energy norm 下的 logarithmic norm；
5. **Gate 5**：冻结 test split 的 `G/q/F` RMS、95/99 percentile、maximum；
6. **Gate 6**：neural integrator 与真实 EM-ROM + Radau trajectory 对照；
7. **Gate 7**：production integrator 的真实 wall-clock benchmark。

Gate 3 两档：

- `randomized_screening`：低成本开发筛查，不能让 production readiness 通过；
- `full`：逐 thermal coordinate 中心差分，是正式 Gate 3 证据。

Gate 7 没有明确预算或真实测量时，readiness fail closed。

---

## 6. 正式 Certification：真实 retrain + roundtrip + Gate 1--7

```bash
sdfmpneo-certify audit --config examples/neural_electrothermal_rom/certify.example.json
```

正式认证比开发 audit 更严格，不能通过手填布尔值绕过证据。

认证过程会自动执行：

1. physical signature 校验；
2. dataset physical signature / dataset hash 校验；
3. training provenance completeness 检查；
4. **真实 frozen-dataset retrain reproduction**：
   - 从 train split 重新拟合同 rank POD；
   - 使用保存的 network/training config 和 seed 重新训练；
   - 在冻结 test split 比较 POD-decoded `G` tensor；
   - 在固定随机 operating samples 比较 quadratic heat；
   - 比较 best epoch / epochs completed；
5. **真实 model persistence roundtrip**：save -> load -> 比较 metadata、training domain、physical signature、heat source、vector field；
6. 完整 Gate 1--7；
7. Gate report 内容 hash。

因此“保存了 seed”只说明 provenance 完整，**不等于训练可复现**。只有实际 retrain reproduction 通过后，reproducibility Gate 才通过。

配置中可单独设置 reproduction/roundtrip tolerance，例如：

```json
"reproduction_device": "cpu",
"reproduction_packed_rtol": 1e-6,
"reproduction_packed_atol": 1e-7,
"reproduction_heat_rtol": 1e-6,
"reproduction_heat_atol": 1e-7,
"reproduction_operating_samples": 2,
"roundtrip_rtol": 1e-12,
"roundtrip_atol": 1e-12
```

`audited_model_output` 可以保存失败或成功的完整冻结证据；`certified_model_output` 只有全部 Gate 通过时才生成。

模型格式当前为 v3。v1/v2 可只读加载；v3 的 load -> save 会保留历史 training/dataset/domain metadata，并递归追加 certification evidence。

---

## 7. EM 输出诊断：按需计算阻抗/损耗

快速 `predict` 始终只运行 neural surrogate + exact thermal operators，不加载 EM。

需要阻抗、互感/电感、区域损耗或 EM residual 时：

```bash
sdfmpneo-diagnose --config examples/neural_electrothermal_rom/diagnose.example.json
```

流程：

1. neural model 给出 `a(t)` 或 steady state；
2. 检查 neural/physical signature；
3. 检查 state/geometry/operating 是否仍在 neural training domain；
4. 在该状态调用真实 reduced EM；
5. 输出 physical heat source、drive-RHS residual、端口阻抗/证书、区域损耗；
6. 同时比较 `q_theta` 与真实 EM heat source。

阻抗始终来自与 neural thermal state 一致的真实 reduced EM，不训练第二个独立输出网络。

---

## 8. 在线结构

```text
(a, normalized geometry)
        -> ordinary residual MLP
        -> normalized POD coefficients
        -> direct POD/current contraction
        -> exact quadratic-current physics layer
        -> q_theta
        -> exact M_r(g) a_dot = -K_r(g) a + q_theta
        -> generalized fixed/adaptive ETD2 / IMEX
```

网络不直接输入时间，也不直接输入 current；current 只进入 hard quadratic layer。

---

## 9. 本地测试

新路线的核心回归：

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
  tests/test_reachable_state_domain.py \
  tests/test_neural_batch_inference.py \
  tests/test_neural_provenance.py \
  tests/test_neural_certification.py \
  tests/test_neural_gate_suite.py
```

完整仓库回归：

```bash
python -m pytest -q
```

---

## 10. Production 切换原则

当前实现分支提供：

- state-domain probe；
- resumable tensor dataset generation；
- POD + ordinary residual MLP training；
- no-EM retrain；
- self-contained fixed/geometry inference；
- fixed/adaptive ETD2 / IMEX；
- batch inference；
- Gate 1--7；
- actual retrain reproducibility verification；
- model roundtrip verification；
- formal certification；
- on-demand true EM diagnostics。

但不会自动替换旧 production 默认入口。只有真实 UWPT 数据上的全部 Gate 满足**预先给定**工程预算，并由 `sdfmpneo-certify audit` 实际产出 certified model 后，才允许切换默认模型。
