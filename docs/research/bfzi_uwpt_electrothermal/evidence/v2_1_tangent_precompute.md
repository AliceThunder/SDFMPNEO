# V2.1：BFZI 高精度海水阻抗灵敏度的离线降本（8 次方向求解）

日期：2026-11-12。研究分支 `research/bfzi-uwpt-electrothermal-surrogate`。承接 [高精度外推物理主线](v2_1_extrapolation_precision.md)。

## 核心目标及目前结论

前一阶段已经在独立 P03（约 20°）和 P07（约 27°）几何上，通过原 BFZI V150 的8模态海水热阻抗灵敏度，在高功率和 15s 非训练时间步得到 **0.0019%～0.0027% 的完整动态复阻抗误差**。这一模式每个新几何要额外求8个温度热模态的正、负扰动，即 **16 次原 BFZI 电磁方向求解**，而冷态复阻抗在创建空间锚时已经计算并缓存。

本轮直接复用缓存的原 BFZI 冷态海水复端口，将可选择的**单侧有限差分**加入 `sdfmpneo_bfzi/geom_tangent.py`。在相同8维物理热模态上，方向电磁求解从 **16 次减为8次**，原 BFZI Green、海水标量势投影、铜趋肤、三维温度与守恒ROM均不修改。现有双侧差分仍是默认高精度基线。单侧差分的真实 P03/P07 精度需与既有双侧留出结果比较后再决定是否作为默认方案。

## 实现及计算量

原高精度双侧差分：

```text
D_j Zsea = [Zsea(T0+h B_j)-Zsea(T0-h B_j)]/(2h)
original BFZI directional EM solves = 2 * nmode = 16
```

新的单侧差分直接使用冷态锚：

```text
Z0 = 已保存的 BFZI 冷态 Zsea(T0)
D_j Zsea = [Zsea(T0+h B_j)-Z0]/h
original BFZI directional EM solves = nmode = 8
```

本轮给 `SeaTangent.build` 增加 `method="central"|"forward"` 和正扰动 `amplitude` 参数。保留 BFZI 原始海水正交热基、同一个复杂2×2端口及模型几何/场数据，额外在NPZ中保存 `fd_method`、`fd_amplitude_K` 和 `em_directional_solves`，并兼容既有旧NPZ。原前向方案一阶截断误差随扰动幅度减小，双侧方案二阶；优先通过实际BFZI数据选择`h`，从而避免为减少求解数而牺牲最终动态复阻抗精度。

`examples/bfzi_param.py` 新增：

- `--sea-tangent-fd forward`：8模态只解8次原 BFZI EM；
- `--sea-tangent-amplitude .2`：该几何方向热模态的最大温度扰动 0.2 K；
- 几何元数据JSON额外记录方向求解数和真实灵敏度建造耗时。

新 `examples/bfzi_tangent_compare.py` 使用同一原 BFZI P03/P07 动态真值温度、精确铜阻抗和动态复端口比较两种差分，明确列出整个复端口热变化误差、单独海水阻抗热变化误差以及近似切线之间的差别。此脚本采用真实动态温度以隔离电磁切线误差；端到端测试继续运行既有 `examples/bfzi_geom_heat.py --sea-tangent ... --rom-rank 32`，对照完整三维自主预测，避免将前者误作自主推理精度。

## 已执行的最小验证

在本轮可用沙箱内运行：

```text
python -m pytest -q tests/test_geom_tangent.py tests/test_geom_tangent_fd.py
5 passed
```

其中新增3项测试覆盖：

1. 线性端口响应：8模态单侧8次原物理求解接口调用，双侧16次；输出导数与三维热场上的端口增量一致。
2. 非线性复阻抗响应：减小单侧热扰动后导数更接近准确的一阶导数，双侧差分具有更低截断误差。
3. 参数非法值、差分元数据归档和既有无元数据的切线文件兼容。

这些是在沙箱中以与原 BFZI 同接口的可控复端口算例完成的局部代码验证。**本轮未在 XEnergy 调用原 BFZI 电磁求解器，尚无 P03/P07 单侧差分真实精度和耗时数据。** 原有16次双侧模型的 P03/P07 真实留出结果保持不变。

## 下一次原 BFZI 直接验收命令

在已有 BFZI V150 计算环境中，以 P07 的原生圆形—圆角方形几何、27° yaw/pitch/roll、独立双封装为例，只需做一次原几何构建和8次方向求解：

```powershell
python -m examples.bfzi_param --bfzi-root D:/work/bfzi-et-v2.1/BFZI --sample M4 --alpha1 1 --core1-mm 0 --alpha2 .72 --core2-mm 23 --shift-mm 13 0 9 --yaw-deg 7 --pitch-deg 9 --tilt-deg 27 --box1-mm 216 --box2-mm 240 --sea-tangent-rank 8 --sea-tangent-fd forward --sea-tangent-amplitude .2 --out runs/bfzi_pose_p07_fd_anchor.npz

# 与已归档16次双侧求解的原始P07温度/Z真值同轨迹比较
python -m examples.bfzi_tangent_compare --anchor runs/bfzi_pose_p07_anchor.npz --reference runs/bfzi_pose_p07_tangent_anchor_sea_tangent.npz --candidate runs/bfzi_pose_p07_fd_anchor_sea_tangent.npz --teacher runs/bfzi_pose_p07_15s_holdout.npz --out runs/bfzi_pose_p07_fd_compare.json

# 完整三维自主端口/温度反馈、五材料守恒与32阶ROM：两条150V/15s留出时程
python -m examples.bfzi_geom_heat --anchor runs/bfzi_pose_p07_anchor.npz --teacher runs/bfzi_pose_p07_15s_holdout.npz --model runs/bfzi_geom_diverse.pt --rom-rank 32 --rom-file runs/bfzi_pose_p07_rom.npz --sea-tangent runs/bfzi_pose_p07_fd_anchor_sea_tangent.npz --tangent-weight 1 --out runs/bfzi_pose_p07_fd_heat.json
```

同一 `forward` 差分应再在 P03 20° 留出几何评估。对 `amplitude=.2` 与 `.05` 的新几何导数比较，可以选择能维持完整端口相对精度且受BFZI数值求解误差影响较小的扰动。真实直接验收后，才能确认是否把8次单侧方案升级为生产默认的高精度离线构造方法。

## 后续科研主线

- **优先完成真实原 BFZI 8次与16次方向求解精度、离线耗时、端到端时程的对照。** 如果8次满足高精度目标，直接节省50%的方向求解次数，保留原三维保守ROM。
- **跨几何物理切线学习。** 目前库内完整8模态切线只有P03和P07。两个几何不足以支撑可靠的跨 SE(3)、连续圆角、封装与介质泛化训练。下一步用已建立的原BFZI参数生成器，为 P01/P02/P04/P05/P06（训练）与 P03/P07（严格独立留出）生成一致的热模态—端口导数数据，再训练以40维显式几何特征及物理冷态端口为输入、预测低秩阻抗灵敏度残差的小网络。要处理不同几何热基的方向/顺序一致性，按原 BFZI 3D热源的物理模态对齐后监督，最终根据完整动态复阻抗、海水单独阻抗、三维J/q和预计算成本选模型。
- 扩展新几何下的真实高电压与强海水升温案例，继续约束铜电磁热反馈、五材料功率守恒和空间场误差。
