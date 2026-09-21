# SDF-MPNEO — geometry → spatial Joule surrogate → geometry-local thermal ROM

当前 production 主链不再使用跨 geometry 的共享 thermal state atlas。

主流程：

    geometry
      -> neural Z_field(g), D_vol(g), H_cell(g)
      -> cellwise PSD + exact sum(H_cell) = D_vol
      -> query geometry 的真实 M(g), K(g)
      -> geometry-local rational-Krylov thermal ROM
      -> explicit current/circuit + wire resistance
      -> thermal ODE / steady state
      -> temperature

神经网络只代理 geometry-dependent electromagnetic / Joule quantities。thermal operator 始终由查询 geometry 的真实材料占据直接组装。

## 1. 运行

安装：

    python -m pip install -e '.[gui,neural,dev]'

训练：

    python run.py --mode train

纯控制台训练：

    python run.py --mode train --headless

推理：

    python run.py --mode predict

用户配置集中在 run.py 顶部。preflight、Physics Gate 和 completely-held-out final Go/No-Go 都在同一个训练入口中；任一硬 Gate 失败都会 fail closed，不保存 production artifact。

## 2. 为什么取消跨 geometry thermal atlas

此前 v14 要求一套经过 rigid transport 的共享 thermal state space 覆盖完整 geometry 域。实际运行已经出现明确反证：thermal rank 扩到约 2100 后，training error 可以压到 5% 内，但未见 geometry 仍保持约三成量级误差，而且 residual enrichment 本身需要数小时。

因此 production 不再训练、运输或扩张跨 geometry thermal states。网络输出与 thermal rank 完全解耦；每个查询 geometry 自己用真实 M(g), K(g) 构造小型 thermal ROM。

## 3. 离线 Maxwell truth

固定 geometry 后：

    A_em(g) X(g) = B(g)
    B(g) = -i omega S(g)
    Z_field(g) = -S(g)^T X(g)

体耗散矩阵：

    D_vol(g) = X(g)^H H_sigma(g) X(g)
    P_vol(c) = 0.5 * c^H D_vol c

production truth 继续使用以下物理语义，不因 thermal 架构切换而放宽：

- finite-cross-section stranded source；
- Silver–Müller open boundary；
- independent outward-power form；
- canonical local fine-minus-coarse self correction；
- 原始物理 Maxwell matrix 的 true relative residual 必须满足 1e-9 Gate；
- 少数合法、约 95k-edge global problem 在 iterative path 无法认证时，可以使用受限 sparse-LU correctness fallback；
- localized compatible-transverse self solve 对 <=100k-edge 的中等 local system 直接使用 bounded sparse solve；这覆盖实测会在 ILU/LGMRES 上完全停滞的约 64k-edge 情形，同时仍明确排除 118k/254k refined validation systems。

### 3.1 Persistent certified Maxwell field cache

训练背景使用：

    results/uwpt/unified.thermal_maxwell_fields.npz

缓存只保存完整 port fields X(g)。每次命中都会重新组装当前物理 A(g), B(g) 并重新计算 true residual，继续满足当前 residual Gate 才能复用。

该 cache 使用独立 physical preflight signature，因此改变 neural optimizer、final audit 或 online thermal ROM 参数不会让已经认证的 Maxwell fields 无谓失效。

## 4. Cellwise Joule tensor truth

每个 background cell k 都直接由 port fields 构造一个 Hermitian Joule tensor H_cell[k]。对任意 complex port current c：

    q_cell[k](c) = 0.5 * c^H H_cell[k] c

并要求：

    sum_k H_cell[k] = D_vol
    sum_k q_cell[k](c) = 0.5 * c^H D_vol c

### 4.1 Spatial local self correction

local self correction 仍只修改对应 port 的 diagonal self term，但 spatial truth 不再把 delta-D 粗略铺到 wire line support。

canonical local coarse/fine Maxwell solve 会保留 refinable spatial Joule cell field。该 integrated cell heat 通过 coil pose 保守映射回 parent grid，然后形成：

    H_self_defect[p,p] = 2 * (q_fine_mapped - q_coarse_mapped)

映射过程保持总功率，因此：

    sum_cells H_self_defect[p,p] = corrected delta D_pp

这使 local self correction 的空间位置与其真实 canonical local solution 一致，而不是被强行放在线圈中心线上。

### 4.2 Hard spatial physical decoder

coarse global cell tensors加上 spatial self defect 后，decoder 执行：

1. 每个 cell tensor 做 Hermitian PSD projection；
2. 使用一个 port-space congruence 使所有 cell tensors 的和严格恢复为 corrected D_vol；
3. 返回前再次硬检查 minimum cell eigenvalue 和 relative sum-to-D mismatch；失败直接抛错。

最终 invariant：

    H_cell[k] is PSD for every cell k
    sum_k H_cell[k] = D_vol

所以任意 current 的 cell heat 非负，并且总 volume Joule power 与 D_vol 精确闭合。

## 5. Neural surrogate

网络学习：

    geometry -> Z_field, D_vol, H_cell[0 ... N_cell-1]

MLP 不直接输出完整超宽 spatial vector。training split 先对 packed output 做 training-only POD；对宽输出使用 sample-Gram / dual POD，即先构造 sample-space Gram matrix Y Y^T，而不是直接对 N_sample × N_output 做 full SVD。

因此网络实际输出维数受 training sample rank 控制，不随 thermal rank 增长。

decoder 分两层：

- Z/D 保持 reciprocal / passive algebraic structure；
- spatial block执行 cellwise PSD + exact sum-to-D congruence。

projection_correction 会进入 held-out diagnostics，不能靠巨大 projection 掩盖网络误差。

## 6. Geometry-local online thermal ROM

对每个查询 geometry，直接组装真实 full thermal matrices M(g), K(g)。

2-port 系统的完整 Hermitian current quadratic source space有 4 个确定性 volume directions：

    e_tx
    e_rx
    e_tx + e_rx
    e_tx + i e_rx

再加入：

- 2 个 physical wire-heat directions；
- 1 个 uniform initial-temperature direction。

默认 thermal time scales：0.1 s、1 s、10 s，对应 rational shifts 0、10、1、0.1。

每个 K + s M 只 factorize 一次，并对所有 RHS 做 block solve。uniform initial field 本身也直接加入 basis，因此 scalar uniform initial temperature 在 t=0 可精确表示。

默认 2-port 情况下，未经线性相关剔除的方向上界为：

    1 + 4 * (4 volume + 2 wire + 1 initial) = 29

实际 rank 通常更低。它与此前上千维的跨 geometry atlas 无关。

同一 geometry 的 online thermal context 会缓存，所以多个时间点和后续同 geometry 查询不会重复构造 ROM。

## 7. Online thermal correctness Gate

新的 thermal Gate 不再问“held-out geometry 能不能被共享 atlas 表示”，而是直接验证当前 geometry-local ROM：

1. 在独立 geometry 上取得真实 corrected cell-Joule truth；
2. 组装 full sparse M, K；
3. 构造 online thermal ROM；
4. 对完整 4 个 volume-current directions、2 个 wire directions和 uniform initial condition计算 full-cell transient；
5. 在 0.1 / 1 / 10 / 100 s 比较 mass-weighted field error；
6. 默认要求最大 full-vs-ROM error 不超过 5%。

这一步直接验证 full thermal PDE → online ROM，与 neural surrogate 是否准确相互独立。

## 8. 训练顺序

python run.py --mode train 当前执行：

1. 构建固定 open-boundary background。
2. 执行 spatial EM truth preflight。
3. 生成 geometry → corrected Z / D / cell-Joule truth dataset。
4. 在独立 geometry 上执行 spatial-Joule / online-thermal Physics Gate。
5. 只用 neural training split 拟合 dual POD + residual MLP。
6. 在 completely-held-out geometry 上重新求 corrected Maxwell truth。
7. 比较 surrogate 的 Z、D、cell-Joule field、完整 current-space heat contractions和独立 physical outward loss。
8. 分别用 truth cell-Joule 与 predicted cell-Joule 构造 geometry-local thermal ROM，比较 current-controlled / circuit-controlled transient 与 steady state。
9. 检查 production ETD2 adaptive integrator。
10. 全部通过后才保存 model artifact。

production path 已没有跨 geometry thermal basis build / transport / residual-enrichment 阶段。

## 9. 推理

python run.py --mode predict 对一个新 geometry：

1. 校验 geometry 是否在 production domain；
2. MLP 一次 forward 得到 packed spatial tensor POD coefficients；
3. hard decode 得到 Z、D、H_cell；
4. 第一次访问该 geometry 时组装真实 M, K，构造并缓存小型 online thermal ROM；
5. current-controlled 直接使用 prescribed complex current，或显式求解 voltage/circuit system；
6. 更新 R_wire(T) 与 wire heat；
7. 推进 reduced thermal ODE 或求 stable steady state。

在线没有 Maxwell solve。第一次查询一个新 geometry 只有少量 sparse thermal factorizations；同一 geometry 后续查询直接复用缓存。

## 10. 核心配置

run.py 当前 production TRAINING 关键字段：

    spatial_tensor_schema = cellwise_joule_tensor_v1
    online_thermal_relative_tolerance = 5e-2
    online_thermal_conditioning_limit = 1e10
    thermal_time_scales = [0.1, 1.0, 10.0]
    thermal_trajectory_times = [0.1, 1.0, 10.0, 100.0]
    n_tensor_samples = 96

以下旧 atlas 配置已经退出 production path：

    thermal_basis_schema
    thermal_basis_design
    basis_samples
    basis_validation_samples
    thermal_basis_max_rank
    thermal_component_target_multiplier

修改 online thermal tolerance / time scales 不会使 spatial Maxwell truth dataset cache 失效；修改 geometry/material/source/self-correction 或 truth sample policy 会失效。

## 11. Artifact 与 cache

主要文件：

    results/uwpt/unified.cache.json
    results/uwpt/unified.tensor_dataset.npz
    results/uwpt/unified.thermal_maxwell_fields.npz
    results/uwpt/model.tensor_training.pt
    results/uwpt/model.geometry_thermal.npz

当前 production schema：

    MODEL FORMAT_VERSION = 53
    RUNTIME CACHE_FORMAT = 56
    spatial tensor = cellwise_joule_tensor_v1
    online thermal = geometry_local_rational_krylov_v1

旧 geometry-aware thermal atlas artifact/cache 不再是 production dependency。

prediction release 仍要求：

- truth preflight certified；
- local self correction certified；
- spatial Physics Gate certified；
- completely-held-out final audit certified；
- independent outward-power comparison passed；
- production integrator passed；
- certificate_level = frozen_held_out_numerical_validation。

## 12. 测试

重点本地回归：

    python -m pytest -q tests/test_run_neural_user_defaults.py tests/test_unified_spatial_joule_online_thermal.py tests/test_unified_self_correction.py tests/test_unified_self_correction_audit.py tests/test_unified_fast_global_maxwell.py tests/test_unified_thermal.py tests/test_unified_thermal_batch_failfast.py tests/test_unified_thermal_source_partition.py tests/test_uwpt_geometry.py

旧 thermal atlas 单元测试仍保留用于历史/底层算法回归，但 production runtime 已不再调用该 atlas。

训练与测试均从普通 Python / 本地入口运行。
