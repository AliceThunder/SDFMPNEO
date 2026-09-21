# SDF-MPNEO — geometry → spatial Joule surrogate → geometry-local thermal ROM

当前生产主链已经不再使用跨 geometry 的共享 thermal state basis。正式流程是：

[
oxed{
g
ightarrow
widehat Z_{m field}(g), widehat D_{m vol}(g), widehat H_{m cell}(g)
ightarrow
Phi_{m online}(g),M_r(g),K_r(g)
ightarrow
	ext{current/circuit + thermal ODE}
ightarrow
T(t)
}
]

其中神经网络只代理 geometry-dependent electromagnetic / Joule quantities；thermal operator 仍由查询 geometry 的真实材料占据直接组装。

## 1. 快速运行

安装：

```bash
python -m pip install -e '.[gui,neural,dev]'
```

训练：

```bash
python run.py --mode train
```

纯控制台训练：

```bash
python run.py --mode train --headless
```

推理：

```bash
python run.py --mode predict
```

日常只需要修改 `run.py` 顶部配置。preflight、Physics Gate 和 completely-held-out final Go/No-Go 都已经接在同一个训练入口里；任何硬 Gate 失败都会 fail closed，不保存 production model。

## 2. 为什么取消全局 thermal atlas

此前 v14 仍要求一个经过 rigid transport 的共享 thermal state space 覆盖完整 geometry 域。实际日志已经显示该假设不成立：训练空间继续扩到约 2100 个 thermal states 后，training error 可以压到 5% 内，但未见 geometry 仍有约三成量级误差，而且 residual enrichment 本身需要数小时。

现在 thermal state 不再跨 geometry 学习或运输。网络输出与 thermal rank 完全解耦，查询 geometry 自己用真实 (M(g),K(g)) 构造一个很小的 thermal ROM。

## 3. 离线 Maxwell truth

固定 geometry：

[
A_{m em}(g)X(g)=B(g),
qquad
B=-iomega S,
qquad
Z_{m field}=-S^TX.
]

体耗散矩阵：

[
D_{m vol}=X^H H_sigma X,
qquad
P_{m vol}(c)=rac12 c^H D_{m vol}c.
]

生产 truth 继续使用：

- finite-cross-section stranded source；
- Silver–Müller open boundary；
- independent outward-power form；
- canonical local fine-minus-coarse self correction；
- 原始物理矩阵 true residual (le 10^{-9}) 的 Maxwell 认证；
- 合法、约 95k-edge 的 global problem 在 iterative 路径无法认证时，可使用受限 sparse-LU correctness fallback；更大的 local validation problem 不走该 fallback。

这些 EM/source 语义没有因为 thermal 架构切换而放宽。

### 3.1 Persistent certified Maxwell field cache

训练背景会配置：

```text
results/uwpt/unified.thermal_maxwell_fields.npz
```

缓存只保存完整 port fields (X(g))。每次命中都会重新用当前物理 (A(g),B(g)) 计算 true residual，只有继续满足当前 residual Gate 才会复用。

该 cache 使用独立的 physical preflight signature，因此改变 neural optimizer、final audit 或 online thermal ROM 参数不会让已经认证的 Maxwell fields 无谓失效。

## 4. Cellwise Joule tensor truth

对每个 background cell (k) 直接由 port fields 构造 Hermitian tensor：

[
[H_k]_{ij}
=
sigma_k
sum_e
W_{ek},
X_{ei}^*X_{ej}.
]

任意 complex port current (c) 下：

[
oxed{
q_k(c)=rac12 c^H H_k c
}
]

并且：

[
sum_k H_k=D_{m vol},
qquad
sum_k q_k(c)=rac12c^HD_{m vol}c.
]

local self correction 仍只改变对应 port 的 diagonal self term。其 corrected (Delta D_{pp}) 先放回该 port 的物理 line-heat support，再统一做 cellwise PSD projection 和 port-space congruence normalization。

最终 decoder 强制：

[
oxed{
H_ksucceq0quadorall k,
qquad
sum_k H_k=D_{m vol}.
}
]

所以对任意相位/幅值的 current：

[
q_k(c)ge0
]

且总 volume Joule power 与 (D_{m vol}) 精确闭合。

## 5. Neural surrogate

网络学习：

[
g
longmapsto
left(
Z_{m field},
D_{m vol},
H_1,ldots,H_{N_{m cell}}
ight).
]

但 MLP 并不直接输出约 (4N_{m cell}) 个 spatial coefficients。训练 split 先对完整 packed output 做 training-only POD，且对超宽 spatial output 使用 sample-Gram / dual POD：

[
YY^T
]

而不是对 (N_{m sample}	imes N_{m output}) 直接做 full SVD。

因此网络实际输出维数最多受 training sample rank 控制，不再随 thermal rank 增长。

decoder 分两层：

1. 对 (Z,D) 保持 reciprocity/passivity 结构；
2. 对每个 cell 做 PSD projection，然后用单个 port-space congruence 使所有 cell tensor 的和严格恢复为 (D_{m vol})。

`projection_correction` 会进入 held-out diagnostics；不能靠巨大 projection 掩盖网络误差。

## 6. Geometry-local online thermal ROM

对一个查询 geometry，先直接组装真实：

[
M(g),qquad K(g).
]

对 2-port 系统，完整 Hermitian current quadratic space有 (n_p^2=4) 个确定性 volume-source directions：

```text
e_tx
e_rx
e_tx + e_rx
e_tx + i e_rx
```

再加入：

- 2 个 physical wire-heat directions；
- 1 个 uniform initial-temperature direction。

默认 thermal time scales：

```python
[0.1, 1.0, 10.0]
```

对应 rational shifts：

[
sin{0, 10, 1, 0.1}.
]

每个 (K+sM) 只 factorize 一次，并对所有 RHS 做 block solve。uniform initial field 本身也直接加入 basis，因此 scalar uniform initial temperature 在 (t=0) 可以精确表示。

默认 2-port 情况下，未经线性相关剔除的上界只有：

[
1 + 4	imes(4+2+1)=29
]

个 directions；实际 rank 通常更低。它与之前上千维的跨 geometry atlas 没有关系。

同一 geometry 的 online thermal context 会缓存，因此重复时间查询不会重复构造 ROM。

## 7. Online thermal correctness Gate

新的 thermal Gate 不再检查“held-out geometry 能否被共享 atlas 表示”，而是直接检查当前 geometry-local ROM。

对独立 geometry 和真实 cell-Joule truth：

1. 构造 full sparse (M,K)；
2. 构造 online ROM；
3. 对完整 4 个 volume-current directions、2 个 wire directions和 uniform initial condition计算 full-cell transient；
4. 与 reduced transient 在 `0.1 / 1 / 10 / 100 s` 比较 mass-weighted field error；
5. 默认要求最大误差 (le5%)。

这一步直接验证 full-cell thermal PDE → online ROM，不依赖 neural surrogate 是否准确。

## 8. 训练顺序

`python run.py --mode train` 当前执行：

1. 构建固定 open-boundary background。
2. 执行 spatial EM truth preflight。
3. 生成 geometry → corrected `Z / D / cell-Joule` truth dataset。
4. 在独立 geometry 上执行 spatial-Joule / online-thermal Physics Gate。
5. 用 training split 拟合 dual POD + residual MLP。
6. 在 completely-held-out geometry 上重新求 corrected Maxwell truth。
7. 比较 surrogate 的 (Z,D,H_{m cell}) 与完整 current-space contractions。
8. 分别用 truth cell-Joule 和 predicted cell-Joule 构造 geometry-local thermal ROM，比较 current-controlled / circuit-controlled transient 与 steady state。
9. 检查 production ETD2 integrator。
10. 全部通过后才保存 model artifact。

没有任何跨 geometry thermal basis build/residual-enrichment 阶段。

## 9. 推理

`python run.py --mode predict` 对一个新 geometry：

1. 校验 geometry 是否在 production domain；
2. MLP 一次 forward 得到 packed spatial tensor POD coefficients；
3. hard decode 得到 (Z,D,H_{m cell})；
4. 第一次访问该 geometry 时组装真实 (M,K)，构造并缓存小型 online thermal ROM；
5. current-controlled 直接使用 prescribed current，或显式求解 voltage/circuit system；
6. 更新 (R_{m wire}(T)) 与 wire heat；
7. 用 reduced thermal ODE 求任意时间或 steady state。

在线 **没有 Maxwell solve**。第一次查询某个 geometry 会有少量 sparse thermal factorization；同一 geometry 后续查询复用缓存。

## 10. 关键配置

`run.py` 当前核心训练配置：

```python
TRAINING = {
    "seed": 17,
    "spatial_tensor_schema": "cellwise_joule_tensor_v1",
    "online_thermal_relative_tolerance": 5e-2,
    "online_thermal_conditioning_limit": 1e10,
    "thermal_time_scales": [0.1, 1.0, 10.0],
    "thermal_trajectory_times": [0.1, 1.0, 10.0, 100.0],
    "n_tensor_samples": 96,
    ...
}
```

旧的以下配置已经退出 production path：

```text
thermal_basis_schema
thermal_basis_design
basis_samples
basis_validation_samples
thermal_basis_max_rank
thermal_component_target_multiplier
```

修改 online thermal tolerances/time scales 不会使 spatial Maxwell truth dataset cache 失效；修改 geometry/material/source/self-correction 或 truth sample policy 会失效。

## 11. Artifact 与 cache

主要文件：

```text
results/uwpt/unified.cache.json
results/uwpt/unified.tensor_dataset.npz
results/uwpt/unified.thermal_maxwell_fields.npz
results/uwpt/model.tensor_training.pt
results/uwpt/model.geometry_thermal.npz
```

当前版本：

```text
MODEL FORMAT_VERSION = 52
RUNTIME CACHE_FORMAT = 55
spatial tensor schema = cellwise_joule_tensor_v1
online thermal schema = geometry_local_rational_krylov_v1
```

旧 geometry-aware thermal atlas artifact/cache 不再是 production dependency。

prediction release 仍要求：

- truth preflight certified；
- local self correction certified；
- spatial Physics Gate certified；
- completely-held-out final audit certified；
- production integrator passed；
- `certificate_level = frozen_held_out_numerical_validation`。

## 12. 测试

重点回归：

```bash
python -m pytest -q \
  tests/test_run_neural_user_defaults.py \
  tests/test_unified_spatial_joule_online_thermal.py \
  tests/test_unified_fast_global_maxwell.py \
  tests/test_unified_thermal.py \
  tests/test_unified_thermal_batch_failfast.py \
  tests/test_unified_thermal_source_partition.py \
  tests/test_uwpt_geometry.py
```

旧 thermal atlas 单元测试仍保留用于历史/底层算法回归，但 production runtime 已不再调用该 atlas。

训练和测试均从本地/普通 Python 入口运行，不依赖 CI workflow。
