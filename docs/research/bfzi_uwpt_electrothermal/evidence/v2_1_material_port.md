# UMFC–BFZI V2.1：材料温度训练与完整电磁端口的物理组装

实算日期：2026-11-12。研究分支：`research/bfzi-uwpt-electrothermal-surrogate`。物理教师：独立原 `AliceThunder/BFZI` V150 分支，M4 复阻抗物理内核继续使用原 Green/resolvent 方法。完整计算和验证在 XEnergy 执行，源码、模型权重和 NPZ/JSON 证据均已提交，无 GitHub Actions。

## 1. 主任务和新增方法

在前一轮已实现的三通道分数阶 SOE+指数历史记忆、独立双端口电路反馈、三维海水/封装/铜热态、海水 J/q 空间场神经解码器基础上，本轮解决两个具体问题：

1. **温度精度**：原联合 20 阶 POD 基按全部温度场提取，导致不同材料及局部热点的输出精度不均。新增 `--basis-mode material`：分别为 384 个海水 FV、发射端 32 个封装 FV、接收端 32 个封装 FV 和两只铜温度建立空间子基；rank 20 时分别用 6+6+6+1+1 阶。增加 `--material-loss`（分组归一化温度重建误差）和 `--hotspot-loss`（空间最大绝对温度误差），与双网格动态阻抗监督共同训练。各物理材料的低维状态在同一网络内部预测，不增加网络数量。
2. **端口物理一致性**：原神经输出完整复 Z 和海水空间头分别训练，存在两套可同时使用的 Z 读出。新增 `sdfmpneo_bfzi/frac_port.py` 导出的物理端口模型，直接从**独立原 BFZI** 读取两只线圈的空气 Neumann 电感矩阵、真实导线长度/半径及铜电导率温度系数，并复用原 `BFZISources.conductor_internal_impedance` 的圆柱导体 Bessel 比值公式。仅需一次 BFZI checkout 导出小 NPZ，后续神经推理无 BFZI 依赖：

```text
sigma_Cu,p(T) = sigma_ref,p / (1 + alpha_p (T - Tref,p))
k_p = sqrt(j omega mu_p sigma_Cu,p)
Z_Cu,p(T) = length_p * k_p I0(k_p a_p) /
                      (2 pi a_p sigma_Cu,p I1(k_p a_p))
Ztotal(Tsea,Tcu) = Zsea_NN(Tsea) + j omega Lair
                   + diag(Z_Cu,1(Tcu1), Z_Cu,2(Tcu2))
I(t) = [Ztotal(t)+Zexternal]^-1 [V(t),0]^T
```

Bessel 比值采用与原 BFZI 相同的缩放修正和小变量近似；源/负载及串联补偿保持原 M4 双端口外部复阻抗。神经每次用**本次物理组装的总 Z**重新求解电流并更新下一时段的热历史。海水温度条件化 `Zsea_NN` 由已训练的 `SeaHead` 求得，海水空间 `J/q` 仍从同一温度与电流预测，`q` 按预测海水端口功率归一化，完成空间热源和复端口能量一致性。

纯神经复 Z 头仍作为训练和对照输出；添加 `--port runs/m4_frac_port.npz` 时开启物理组装闭环。

## 2. 相同 BFZI 数据下的分材料/热点训练结果

教师：原 BFZI V150 M4 三维海水 384 FV、两封装各 32 FV、两铜节点，30 s 和 15 s 两套共 60 条完整 600 s 真实时程；15 个训练历史、5 个验证历史、10 个完全留出历史在两套网格上相同（不同网格不算不同物理历史）。新模型 `runs/m4_frac_mat_hybrid.pt` 训练 2400 epoch，最佳第 2261 epoch，分数阶 `alpha=[0.3424637,0.5278218,0.7295387]`，指数时间常数约 `[630.67,260.63,138.61]s`，`material_loss=.3`、`hotspot_loss=.15`。

下表为 **10 条留出历史的逐轨迹最大空间绝对误差再平均**，不是单元平均温度：

| 温度/动态指标 | 原统一 POD+双网格训练 | 本轮分材料/热点训练 |
|---|---:|---:|
| 30 s 海水最高误差平均 | 0.031359 K | **0.030473 K** |
| 30 s 封装最高误差平均 | 0.419807 K | **0.343516 K** |
| 30 s 铜最高误差平均 | 1.301170 K | **1.300406 K** |
| 30 s 动态复 Z 变化相对 L2 平均 | **0.053228** | 0.057194 |
| 15 s 海水最高误差平均 | 0.039387 K | **0.033869 K** |
| 15 s 封装最高误差平均 | 0.673482 K | **0.549387 K** |
| 15 s 铜最高误差平均 | **1.491776 K** | 1.560504 K |
| 15 s 动态复 Z 变化相对 L2 平均 | **0.060629** | 0.067317 |

**结论：** 物理材料分组 POD+热点损失将 30 s/15 s 双封装平均最高误差分别减少 **18.17%/18.43%**，海水也改善；原独立神经复 Z 动态变化误差略增。故保留前一轮 `runs/m4_frac_multi_hybrid.pt` 作为动态复阻抗参考，另以本轮物理组装端口方式处理阻抗一致性，后续再改进同一模型的多目标训练。

## 3. 原 BFZI 铜/空气物理参数导出与严格对照

`examples/bfzi_frac_port.py` 只在导出物理常量阶段使用原 BFZI V150，原铜丝内部复阻抗与空气电感完整遵从原 `BFZIV148Adapter` 和 `BFZISources`。

100 V、海水三维 FV 和双封装完整直接法 `runs/m4_pkg3d_weak_100v.json/.npz` 的 11 时刻留出对照：

| 指标 | 实测 |
|---|---:|
| 使用原 BFZI 真实海水 Z、真实铜温度重组总复阻抗，最大绝对差 | **1.11022e-16 Ω** |
| 上述完整重组相对 L2 差 | **1.12715e-18** |
| 海水小型空间头 + 真实铜/海水温度 + 原 BFZI 空气与趋肤阻抗，总 Z 相对 L2 差 | **1.69801e-10** |
| 原独立神经总 Z（分材料训练）相对 L2 差 | 1.40842e-5 |
| **物理组装神经总 Z** 相对 L2 差 | **1.38506e-5** |
| 独立神经总 Z 的动态变化 `ΔZ` 相对 L2 差 | 6.43725% |
| **物理组装神经总 Z** 的动态变化 `ΔZ` 相对 L2 差 | **6.33046%** |
| 新神经闭环最大电路功率绝对收支差 | **2.84217e-14 W** |
| 新神经闭环 `qsea` 三维空间积分与预测 `Zsea` 的最大端口功率差 | **1.77636e-15 W** |

新模型相较原独立神经总 Z 动态误差有小幅改善，主要进展是**所有温度预测—海水复阻抗—空气电感—铜内部复阻抗—电路电流—空间海水热源**使用了同一物理一致的双端口公式。

## 4. 完整独立神经推理检查

以 100 V，通电 210 s、断电 210 s、再次启动，15 s 步共 41 个时刻实跑：
- 复 `Z11/Z12/Z22`，双端口 `I1/I2` 和效率；
- 海水温度 `[41,384]`、两个三维封装温度 `[41,2,32]`、两只铜温度 `[41,2]`；
- 复 `Jsea[41,384,3]`，非负空间 `qsea[41,384]`；
- 预测 `Zsea[41,2,2]`、铜内部复阻抗 `Zcu[41,2]`；
- 每个时间点 `Psea`、最大海水局部热源、**电路功率收支误差**均单独写入 JSON；
- 实测此次推理中，最大电路功率收支差 **4.26326e-14 W**，经过独立脚本复算的海水空间积分收支差 **1.77636e-15 W**；
- 19 项与本次修改相关的专项测试通过。

源：`sdfmpneo_bfzi/frac_port.py`、`frac_field.py`、`frac_memory.py`；主要运行入口：`examples/bfzi_frac_port.py`、`examples/bfzi_frac.py`；验证：`tests/test_bfzi_frac.py`。

## 5. 使用方式（无需训练即可直接运行）

```powershell
# 物理参数已经从原 BFZI 导出并入库。需要重新生成时，执行：
python -m examples.bfzi_frac_port --bfzi-root D:/work/bfzi-et-v2.1/BFZI --field runs/m4_frac_field.npz --model runs/m4_frac_mat_hybrid.pt --case runs/m4_pkg3d_weak_100v.json --port runs/m4_frac_port.npz --out runs/m4_frac_port_metrics.json

# 后续推理无需原 BFZI checkout：
python -m examples.bfzi_frac infer --model runs/m4_frac_mat_hybrid.pt --field runs/m4_frac_field.npz --port runs/m4_frac_port.npz --voltage 100 --heat 210 --cool 210 --step 15 --out runs/m4_frac_mat_infer.json

# 分材料温度/热点监督重新训练（两套教师均已入库）：
python -m examples.bfzi_frac train --data runs/m4_frac_teacher.npz runs/m4_frac_teacher_15s.npz --modes hybrid --basis-mode material --material-loss .3 --hotspot-loss .15 --baseline runs/m4_frac_multi_hybrid.pt --model runs/m4_frac_mat.pt --out runs/m4_frac_mat_metrics.json --epochs 2400
```

模型 `runs/m4_frac_mat_hybrid.pt`、端口常量 `runs/m4_frac_port.npz`、空间头 `runs/m4_frac_field.npz` 全部是独立可加载文件。测量记录在 `runs/m4_frac_mat_metrics.json/.npz`、`runs/m4_frac_port_metrics.json`、`runs/m4_frac_mat_infer.json/.npz`。

## 6. 下轮优先级

1. 在**保留材料分组及端口物理组装**的基础上，引入同时监督三维温度热点、铜温度、动态 `ΔZ` 的端到端物理约束，避免不同输出之间精度此消彼长；将 `Zsea` 与空间 J/q 读出纳入训练，而不只在后处理阶段组装。
2. 从原 BFZI V150 统一圆形—圆角方形连续几何/位姿生成更多**不同几何**的真实物理教师，不改变已经过 M4 真实回归的电磁内核；用物理几何参数条件化端口模型与网络，做完全独立的几何留出。
3. 扩展时间窗与高功率热历史，检验长记忆与分数阶指数谱在超过 600 s 后的动态精度；同时比较热 ROM 物理参考的时程速度与误差。

当前物理模型为 M4 固定形状的直接法/神经试点。研究主线保留用户要求的圆形和圆角方形连续统一几何，并继续推进跨几何代理。
