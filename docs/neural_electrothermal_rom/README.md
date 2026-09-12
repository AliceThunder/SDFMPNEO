# 结构保持神经电热 ROM：实现使用说明

本目录中的 LaTeX 文档定义冻结理论；本页说明当前实现工作流。新路线与旧解析响应网络完全分离，主模型为

\[
M_r(g)\dot a=-K_r(g)a+q_\theta(a,g,u)+f_T,
\]

其中普通 residual MLP 只学习 `(a,g) -> POD coefficients`；电流只进入严格二次型物理层；`M_r(g),K_r(g)` 和已知 deterministic thermal forcing `f_T` 都是 hard physics，不由网络学习。对外 `heat_source` 始终表示纯 Joule `q_theta`，总动力学才使用 `q_theta + f_T`。

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

Domain report 是冻结工件。`sdfmpneo-domain train` 会检查 report hash、physical signature、thermal rank、geometry domain 和 operating domain。report hash 在标签生成前写入 frozen dataset，因此形成

```text
domain report -> frozen tensor dataset -> neural model
```

的 provenance 链，而且无需在训练结束后重写大型 tensor 数据。

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

固定 `(a,g)` 时使用一次 multi-RHS reduced EM solve 构造整个 current quadratic tensor，不需要逐电流采样。支持 direct-reduced loss operator 的 EM 后端会直接在 reduced coordinate 中计算 `G`，不恢复 full EM state。

训练输出包括：

- `quadratic_joule.partial.npz` + `.packed.npy`：可恢复 snapshot 工作文件；
- 小数据：`quadratic_joule_dataset.npz/.json`；
- 大数据：`quadratic_joule_dataset.store/`，其中 `outputs.npy` 为只读 memmap；
- neural `.npz`：POD、MLP、训练域、物理签名、hard thermal forcing 和在线 thermal operators；
- training report：loss、配置、随机种子、precision、软件/硬件信息等。

冻结 dataset 同时保存 physical/ROM provenance：thermal backend/rank、EM reduced rank、频率、constitutive budget、current dimension、EM reduction certificate 摘要、geometry certificate 摘要和源码 revision。大数组保存 shape/dtype/hash，不复制到 manifest。

网络训练阶段**不调用 EM solver**，也不使用瞬态轨迹值作为监督 target。

### 2.1 大数据默认 out-of-core

真实 UWPT 下 `thermal_rank x symmetric-current-size` 很宽，不能假设完整 `N x dim(G)` 能放进 RAM。当前默认行为：

- packed tensor 总量超过 **256 MiB** 时，自动冻结成 `.store` memmap；
- train split dense SVD 预计超过 **512 MiB** 时，自动使用 chunked `LinearOperator + svds`；
- tensor snapshot 从生成开始就逐行 `svec` 写盘，不先建立全量 4D `G` 数组；
- 神经训练前只对宽 tensor **分块读取一次**，得到小型 `N x K` POD coefficients；
- 后续每个 epoch 只训练 `(a,g)->beta`，不会反复读取宽 `G`；
- Gate 5 与正式 retrain reproduction 也按块读取。

可选配置：

```json
{
  "snapshot_disk_threshold_bytes": 268435456,
  "pod": {
    "exact_svd_max_bytes": 536870912,
    "out_of_core_max_rank": 256,
    "chunk_rows": 256
  }
}
```

若自动 POD rank 在 `out_of_core_max_rank` 内仍达不到 `pod_relative_tail_tolerance`，训练会 fail closed；应提高最大 rank、放宽表示误差预算，或明确指定 `pod_rank`，不能假装已满足压缩目标。

已完整生成并通过 hash/采样/physical-signature 校验的 frozen dataset 会被直接复用；重新运行训练不会重复最昂贵的 EM 标签阶段。

### 2.2 完整 state box，不是 initial-state box

训练域是 NN 允许出现的完整 thermal state 域，必须覆盖目标初值、真实轨迹、restart/off-manifold 区域和稳态附近区域。

ETD/IMEX 在每个实际 stage 检查该盒。默认一旦离开训练域立即失败；不会 silently clip。

### 2.3 高维推荐 hybrid reachable sampling

在约 198 维 thermal coordinate 中对完整 box 纯 LHS 会浪费大量样本。正式训练推荐同时使用一部分 full-box samples 和一部分真实 reduced-physics trajectory states。轨迹仅决定“在哪里生成 G 标签”，不是 trajectory supervision。

如果真实采样轨迹离开冻结 state box，抛出 `StateDomainInsufficientError`，应重新做更保守的 domain probe。

---

## 3. Retrain：调 POD/MLP 不再生成 EM 标签

冻结 tensor dataset 一旦生成，POD rank、MLP width/depth、optimizer、mixed precision 等实验走：

```bash
sdfmpneo-neural retrain --config examples/neural_electrothermal_rom/retrain.example.json
```

`retrain` 同时支持 `.npz` 与 `.store`，只读取 frozen dataset 和一个 neural model 作为在线 thermal operator 模板，不构建 EM，不生成新标签。重新训练保留 exact thermal operator 和 `f_T`，继承 dataset / physics provenance，但不会继承旧 certification；任何权重变化后必须重新 audit/certify。

---

## 4. Predict：自包含在线模型

```bash
sdfmpneo-neural predict --config examples/neural_electrothermal_rom/predict.example.json
```

在线 `.npz` 不需要物理 JSON 或 EM-ROM。固定几何和 affine geometry family 都保存足够的 thermal operator 信息来在线精确重组

\[
M_r(g),\quad K_r(g).
\]

变化几何不是对 `M,K` 做参数线性插值：模型保存 affine tetrahedral chart、材料参数和共享热基，查询 geometry 时重新组装 P1 thermal matrices 后投影。

有限时间默认推荐 `etd2_adaptive`：每个 geometry 的 generalized thermal eigendecomposition 使用模型级有界 LRU 缓存。固定 operating 的 quadratic POD contraction 也会预收缩并缓存，后续 ETD stage 只做 MLP + `r x K` contraction。

其他方法：`etd2`、`imex` 和研究验证用 `reference`。`"inf"` 使用独立 steady-state solver，不把有限时间积分无限延长。

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

批量路径共享 generalized thermal spectrum；每个 ETD stage 对整批状态做一次 MLP forward，最终导数用 multi-RHS thermal solve。时间步本身仍严格串行，保持真实因果依赖。

---

## 5. 开发 Audit：Gate 1--7

```bash
sdfmpneo-neural audit --config examples/neural_electrothermal_rom/audit.example.json
```

1. **Gate 1**：`zeta^T G zeta` 与直接 EM-ROM Joule heat 恒等；
2. **Gate 2**：train split 拟合 POD，validation split 做 rank sweep；
3. **Gate 3**：真实 `G(a,g)` 对 thermal state 的 sensitivity / active-subspace 分析；
4. **Gate 4**：真实 physical Jacobian 在 `M(g)` energy norm 下的 logarithmic norm；
5. **Gate 5**：冻结 test split 的 `G/q/F` RMS、95/99 percentile、maximum；
6. **Gate 6**：实际配置的 neural integrator 与真实 EM-ROM + Radau trajectory 对照；
7. **Gate 7**：同一 production integrator 的真实 wall-clock benchmark。

Gate 3 的 `randomized_screening` 只能开发筛查；production readiness 必须完成 `full`。Gate 4 只要求真实稳定性被分析，不强制系统必须 contraction。Gate 7 没有明确预算或真实测量时 fail closed。

---

## 6. 正式 Certification

```bash
sdfmpneo-certify audit --config examples/neural_electrothermal_rom/certify.example.json
```

正式认证不能通过手填布尔值绕过证据。它会自动执行 physical/dataset hash 校验、training provenance completeness、**真实 frozen-dataset retrain reproduction**、**真实 save/load numerical roundtrip**、完整 Gate 1--7 和 Gate report 内容 hash。

“保存了 seed”只说明 provenance 完整，**不等于训练可复现**。只有实际 retrain reproduction 通过后，reproducibility Gate 才通过。大数据 reproduction 按 chunk 比较 POD-decoded tensor 和 quadratic heat，不一次性展开完整 test tensor。

`audited_model_output` 可以保存失败或成功的冻结证据；`certified_model_output` 只有全部 Gate 通过时才生成。

模型格式当前为 v3。v1/v2 可只读加载；v3 额外持久化 POD 总中心化能量和 hard thermal forcing，load -> save 会保留历史 training/dataset/domain metadata 并递归追加 certification evidence。

---

## 7. EM 输出诊断：按需计算阻抗/损耗

快速 `predict` 始终只运行 neural surrogate + exact thermal operators，不加载 EM。

需要阻抗、互感/电感、区域损耗或 EM residual 时：

```bash
sdfmpneo-diagnose --config examples/neural_electrothermal_rom/diagnose.example.json
```

诊断会检查 neural/physical signature 和训练域，然后在神经预测出的 thermal state 上调用真实 reduced EM，输出 physical Joule heat、drive-RHS residual、端口阻抗/证书和区域损耗，并比较 `q_theta` 与真实 EM Joule heat。`f_T` 不参与这项 Joule 对照。

---

## 8. 在线结构

```text
(a, normalized geometry)
        -> ordinary residual MLP
        -> normalized POD coefficients
        -> operating-specific POD/current contraction
        -> exact quadratic-current physics layer
        -> q_theta
        -> + deterministic hard thermal forcing f_T
        -> exact M_r(g) a_dot = -K_r(g) a + q_theta + f_T
        -> generalized fixed/adaptive ETD2 / IMEX
```

网络不直接输入时间，也不直接输入 current；current 只进入 hard quadratic layer。

---

## 9. 本地测试

新路线核心回归：

```bash
python -m pytest -q \
  tests/test_quadratic_joule_tensor.py \
  tests/test_tensor_dataset_pod.py \
  tests/test_resumable_tensor_generator.py \
  tests/test_out_of_core_neural_tensor_pipeline.py \
  tests/test_neural_pod_persistence_and_operating_cache.py \
  tests/test_neural_tensor_physical_layer.py \
  tests/test_neural_integrators.py \
  tests/test_neural_training_smoke.py \
  tests/test_neural_model_persistence.py \
  tests/test_neural_geometry_persistence.py \
  tests/test_neural_hard_thermal_forcing.py \
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

当前实现分支提供 state-domain probe、resumable/out-of-core tensor dataset、POD + ordinary residual MLP、no-EM retrain、自包含 fixed/geometry inference、fixed/adaptive ETD2 / IMEX、batch inference、Gate 1--7、真实 retrain reproducibility verification、model roundtrip、formal certification 和按需真实 EM diagnostics。

但不会自动替换旧 production 默认入口。只有真实 UWPT 数据上的全部 Gate 满足**预先给定**工程预算，并由 `sdfmpneo-certify audit` 实际产出 certified model 后，才允许切换默认模型。
