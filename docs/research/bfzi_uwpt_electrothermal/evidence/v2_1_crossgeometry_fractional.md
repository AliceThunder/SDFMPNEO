# UMFC–BFZI V2.1：跨圆形—圆角方形几何的分数阶神经代理实算

日期：2026-11-12。研究分支 `research/bfzi-uwpt-electrothermal-surrogate`。物理教师来自独立 `AliceThunder/BFZI` 原 V150 `bfzi-v150-turn-scan`，使用原生 `run_v150.build_data`、原 V148/V150 Green/resolvent 与 FV 投影、完整两端口空气电感和铜圆导线趋肤阻抗。全部教师、训练、留出测试均在 XEnergy 真实运行；22 项相关测试通过，未使用 GitHub Actions。

## 1. 主任务：同一个神经历史模型支持不同原 BFZI 双线圈几何

原 M4 是连续圆角方形家族的圆形极限（`alpha=1`、直线核心半长为零），M1/M2/M3 采用同一 BFZI 家族的圆角方形（`alpha=0.5`）。全部线圈、导线回路、空气互感、源和海水几何均从原 BFZI 仓库读取，未创建第二套几何定义。

| BFZI V150 原样本 | 两线圈 | 匝数 | 线圈中心距 | 海水 FV 单元 | 封装+铜热节点 | 总热态 |
|---|---|---|---|---:|---:|---:|
| M4 | 圆形极限 180/180 mm | 10/10 | 40 mm | 384 | 64+2 | 450 |
| M1 | 圆角方形 200/200 mm | 10/10 | 50 mm | 2250 | 64+2 | 2316 |
| M2 | 非对称圆角方形 200/240 mm | 10/13 | 100 mm | 2250 | 64+2 | 2316 |
| M3 | 圆角方形 240/240 mm | 13/13 | 100 mm | 2250 | 64+2 | 2316 |

原 BFZI 等温海水复阻抗回归相对误差为：M4 `4.4857e-18`、M1 `7.3858e-17`、M2 `2.1418e-16`、M3 `1.9569e-16`。

初版原 M4 神经网络将海水、封装和铜全部还原到固定 450 维网格。此方法无法直接在 M1/M2/M3 的 2316 维网格上运行，因此本轮新增 **`sdfmpneo_bfzi/geom_memory.py`**：共享分数阶记忆、几何条件化动态阻抗和网格无关的温度观测输出。在下一轮建设连续空间查询解码器前，此模型输出每个工况的海水平均/最高温升、双封装平均/最高温升、双铜温升和完整复双端口阻抗，共 14 个实数独立动态量。

## 2. 原 BFZI 真实动态教师与明确的独立几何留出

`examples/bfzi_frac_data.py` 新增 `--sample M1/M2/M3/M4`，仅调用原 BFZI 的 `run_v150.build_data`；增加 `--indices` 选择原 30 条启停程序，保留原样本 ID 以隔离训练和测试。存储海水/封装 FV 坐标及体积，并提取原始 BFZI 16 维线圈/电路/热几何描述：外半径、圆角核心半长、连续几何 α、匝数、铜丝半径、中心距离、空气电感 `Lair11/Lair12/Lair22`、双铜热容。

真实教师：`runs/m4_frac_teacher.npz`（M4 原 30 条）、`runs/m1_frac_teacher.npz`（M1 15 条）、`runs/m2_frac_teacher.npz`（M2 8 条）。额外 `runs/m2_frac_holdout.npz`（M2 3 条**未见启停时序**）、`runs/m3_frac_holdout.npz`（M3 2 条**完全未见的几何和时序**）。每条均为 600 s、30 s 时间步、21 个真实电磁热闭环状态。实际网络训练为 30 条、模型选择 8 条；最终独立评估为 M4 10 条、M1 5 条、M2 3 条、M3 2 条。

M3 的 240/240 mm、13/13 匝及中心距 100 mm 在训练中没有出现；在 M3 上评估时，向网络提供一次预计算的原 BFZI **冷态复阻抗**和物理几何/电路参数；动态过程完全由神经网络与便携物理端口自行演化。

## 3. 共享分数阶记忆与原 BFZI 物理热源

共享三路可学习记忆通过 14 个对数间隔的指数状态近似分数阶幂律核，另叠加三路可学习单指数记忆。网络的物理条件包括原 BFZI 16 维几何特征及已知冷态 2×2 互易复 Z 的六个独立实数分量。时间步可按实际 `dt` 递推，无须存储整个历史序列。

首次版本使用 `[Ptotal, |I1|²/2, |I2|²/2]`。M1/M4 对称训练集迁移至 M2 非对称尺寸/匝数时，出现高达 53.08% 的动态 Z 相对误差及 78.03 K 的封装热点平均最大误差。针对这一物理问题，修订为

```text
Z_Cu,p(T) = 原 BFZI 圆柱导线 Bessel 趋肤阻抗公式
Zsea = Ztotal - j omega Lair - diag(Z_Cu(T))
I = (Ztotal + Zexternal)^(-1) [V,0]^T
u = [ 0.5 Re(I^H Zsea I),
      0.5 Re(Z_Cu1)|I1|^2 / C_Cu1,
      0.5 Re(Z_Cu2)|I2|^2 / C_Cu2 ]
```

其中铜热容从原 BFZI 铜线圈长度、半径和材料体积热容得到。这样每条铜温度记忆直接学习单位热容的真实发热率，使不同匝数和尺寸能在同一物理尺度训练。增加 M2 的非对称真实教师后，在 M2 未见时序中的动态 Z 误差降至 **6.30%**、封装热点差降至 **1.831 K**。未见 M3 几何的对应误差为 **9.79%** 和 **0.900 K**。

## 4. 从纯神经 Z 到原 BFZI 铜阻抗物理组装

后续两版保留分数阶记忆，改为预测**海水温度引起的复阻抗增量 `Delta Zsea`**；铜阻抗变化采用原 BFZI 的精确公式，由预测铜温度计算：

```text
Z(t) = Zcold + Delta Zsea,NN(t)
       + diag( Zcu(Tcu,NN(t)) - Zcu(Tambient) )
```

这与 `Zsea(T)+j omega Lair+diag(Zcu(Tcu))` 等价，保证原 BFZI 冷态复阻抗初值准确，并在每次递推时用物理组装的 Z 重新解两端口电流。

再引入**完整复阻抗的训练一致性损失**，利用真实铜温度邻域的原 BFZI 趋肤阻抗导数，直接关联海水 `Delta Zsea` 的误差与两只铜温度误差：

```text
error(Delta Ztotal) =
  error(Delta Zsea)
  + diag( (dZcu/dT)|Ttrue * (Tcu_pred-Tcu_true) )

L_phys = mean( error(Delta Ztotal)^2 / scale(Delta Z)^2 )
```

训练时该局部 Jacobian 是求导稳定的物理约束，**实际推理使用完整原 BFZI Bessel 阻抗公式**。第五版再增加两只封装的平均/热点温度监督权重和海水平均/热点温度监督。

## 5. 真实留出误差及技术选择

各指标为**逐条留出动态历史的最大绝对温度差，再跨历史平均**；动态复阻抗相对 L2 以真实 `Z(t)-Z(0)` 为分母，不以较大的冷态电抗本底掩盖热反馈误差。

### 动态复阻抗变化相对 L2 误差，平均值

| 留出场景 | V1 对称场景训练 | V2 几何物理功率 | V3 海水Z+原BFZI铜组装 | V4 联合总Z物理损失 | V5 兼顾封装/海水热点 |
|---|---:|---:|---:|---:|---:|
| M4 圆形，10 历史 | **6.24%** | 10.11% | 14.81% | 4.52% | **4.50%** |
| M1 对称方形，5 历史 | **5.99%** | 9.00% | 12.75% | 4.53% | **4.35%** |
| M2 非对称，3 历史 | 53.08% | 6.30% | 8.52% | 7.15% | **5.17%** |
| M3 未见几何，2 历史 | — | 9.79% | 9.95% | 8.28% | **8.27%** |

V1 的 M2 失败源于训练样本均为两只几何相同的圆形或圆角方形线圈；M2 的 200/240 mm、10/13 匝、100 mm 间距和不同铜热容量缺乏训练覆盖。M2 加入非对称训练后，不同启停时序完全留出；M3 始终是独立新几何。

### 封装热点与铜温度留出误差，单位 K

| 重点留出指标 | V2（独立总Z） | V4（物理总Z） | V5（物理总Z+封装热点） |
|---|---:|---:|---:|
| M1 封装热点平均最大差 | **1.296** | 2.604 | 2.221 |
| M2 封装热点平均最大差 | **1.831** | 5.882 | 4.664 |
| M3 封装热点平均最大差 | **0.900** | 2.735 | 3.172 |
| M4 封装热点平均最大差 | **1.607** | 3.555 | 3.440 |
| M1 铜温度平均最大差 | 1.727 | 0.338 | **0.303** |
| M2 铜温度平均最大差 | 4.782 | **1.320** | 1.334 |
| M3 铜温度平均最大差 | 2.558 | **1.623** | 1.662 |
| M4 铜温度平均最大差 | 2.261 | 0.755 | **0.745** |

**研究判断与下一步：** V5 在四种几何上的动态复阻抗最稳健，且铜温度误差显著小于 V2；V2 对封装热点和海水温度有明显优势。对 M3 未见几何，增加封装热点损失后温差升高到 3.172 K，因此现阶段保留 V2 与 V5 两种真实性能基准，后续以物理热流守恒与坐标查询空间温度头改善包络表面热点，避免继续对同一个 14 量标量网络反复调损失系数。

V5 完全留出的 M3 动态推理记录：`runs/bfzi_geom_m3_thermal_infer.json/.npz`，21 个时刻，端口功率最大收支绝对差 **1.5987e-13 W**。同样条件下 V3/V4 也已实测端口功率收支约 `6.04e-14/1.03e-13 W`。误差来自浮点计算。训练仍使用 M1/M2/M4 原 BFZI 数据，不使用 M3 动态教师数据拟合模型。

## 6. 可执行入口

```powershell
# 原 BFZI 真实 M1 圆角方形双端口及三维电磁热数据
python -m examples.bfzi_frac_data --bfzi-root D:/work/bfzi-et-v2.1/BFZI --sample M1 --step 30 --indices 0 1 4 5 8 9 12 13 16 17 20 22 24 26 28 --out runs/m1_frac_teacher.npz

# 原 BFZI 真实 M2 非对称圆角方形数据（M3 另行留出）
python -m examples.bfzi_frac_data --bfzi-root D:/work/bfzi-et-v2.1/BFZI --sample M2 --step 30 --indices 0 1 4 8 9 12 16 17 --out runs/m2_frac_teacher.npz

# 共享 SOE 记忆：训练海水 Z 变化、精确 BFZI 铜阻抗、动态完整 Z 和封装热点
python -m examples.bfzi_geom_frac --train-files runs/m4_frac_teacher.npz runs/m1_frac_teacher.npz runs/m2_frac_teacher.npz runs/m2_frac_holdout.npz --m4-geom runs/m4_geom_probe.npz --test-geo runs/m3_frac_holdout.npz --port-sea --phys-loss .65 --pkg-loss .9 --sea-loss .15 --model runs/bfzi_geom_frac_v5.pt --out runs/bfzi_geom_frac_v5_metrics.json --epochs 1800

# 完全无需 BFZI 求解器的推理；参数从已保存的原 BFZI 冷态几何与复端口读取
python -m examples.bfzi_geom_frac_infer --model runs/bfzi_geom_frac_v5.pt --geometry runs/m3_frac_holdout.npz --case-index 1 --out runs/bfzi_geom_m3_thermal_infer.json
```

代码：`sdfmpneo_bfzi/geom_memory.py`、`examples/bfzi_frac_data.py`、`examples/bfzi_geom_frac.py`、`examples/bfzi_geom_frac_infer.py`；目标测试：`tests/test_bfzi_geom_frac.py`。留存旧版权重和训练记录以支持真实消融：`runs/bfzi_geom_frac.pt`、`v2.pt`、`v3.pt`、`v4.pt`、`v5.pt` 及对应 metrics。

## 7. 下一阶段主线

1. 从当前网格无关的 14 个物理观测量，发展**连续空间坐标查询温度/热源网络**：以原 BFZI 三维海水/双封装原始单元坐标、体积和材料标签训练 `T(x,t)`、`q(x,t)`，并使源项空间积分与端口损耗一致。优先解决 M3 未见几何下的封装热点温差。
2. 以原 BFZI 的温度相关铜损耗和 `port-weak` 海水热源，在共享分数阶记忆中加入**材料能量收支约束**，统一铜、封装和海水的动态记忆，避免单纯依赖损失权重权衡温度/复阻抗精度。
3. 保留圆形极限与圆角方形同一 BFZI 几何定义，在已有 M1–M4 双端口基础上扩展**连续圆角参数、相对位姿 SE(3)、封装尺寸与海水介质参数**，并使用完整独立几何留出测试。M1–M4 只是当前可运行的真实原样本集合，尚未覆盖任意姿态和封装几何域。
4. 使用原 BFZI 完整物理直接法和已经验证的物理 EM/热 ROM 对照，面向高速推理评估完整复阻抗、三维温度和局部 J/q、功率、热流守恒及动态效率。
