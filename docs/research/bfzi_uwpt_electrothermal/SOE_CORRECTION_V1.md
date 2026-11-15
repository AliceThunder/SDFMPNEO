# 分数阶 SOE 与 GeomFracET 训练一致性修复 V1

2026-11-15。仓库 `research/bfzi-uwpt-electrothermal-surrogate`。此次修改直接针对原 BFZI V150/S5 真实物理教师、分数阶 SOE、预训练时间约定和跨几何神经热代理的训练/推理一致性。真实计算在 XEnergy，BFZI 源码分支 `bfzi-v150-turn-scan`。

## 1. 分数阶 SOE 数学及数值实现

目标有限时间记忆核为 `k_alpha(t) ∝ t^(alpha-1)`，三个输入分别描述海水损耗以及两组铜损生热。选定对数均匀衰减率 `lambda_j`，当前有 14 项，范围为 `[1/2400, 1/12] s^-1`。每个状态是规范化指数卷积，满足：

```text
dh_j/dt = -lambda_j h_j + lambda_j u(t)
h_j(t+dt) = exp(-lambda_j*dt)*h_j(t)
          + (1-exp(-lambda_j*dt))*u_interval
```

该更新对区间内分段常数输入为精确解。分数阶型读出由下式生成：

```text
q_j = 1/2 (两端点), 1 (内部节点)
w_j(alpha) = q_j * lambda_j^(-alpha)
             / sum_l(q_l * lambda_l^(-alpha))
m_alpha = sum_j w_j(alpha)*h_j
```

权重指数的负号来自 h_j 中含有 `lambda_j` 因子；相应有效卷积核的指数幅度按 `lambda_j^(1-alpha)` 分布。改用对数梯形求积后，对归一化的同频带高阶积分参考，其 15–600 s 阶跃响应的最大相对误差为：

| α | 旧相等节点系数 | 新对数梯形系数 |
|---:|---:|---:|
| 0.240 | 3.924% | 0.215% |
| 0.360 | 5.315% | 0.330% |
| 0.536 | 7.561% | 0.567% |
| 0.719 | 9.913% | 0.874% |
| 0.903 | 12.050% | 1.238% |

这组指标检验**有限频带 SOE 离散求积**，不是对无穷频谱完整幂律核的绝对误差，也不等同于神经预测误差。新增 `soe_weights()`，保留老权重加载时 `legacy` 方案，新模型显式 `trap`。指数状态初始为零，非负源项产生非负历史记忆。

## 2. 物理时间定义与旧权重兼容

统一最新 V150/S5 教师与新代理：

- `[t_(k-1),t_k]` 时间区间的加热/历史输入采用 `V_(k-1)`、前一状态的阻抗和铜温。
- 到达 `t_k` 后的端口电流和即时功率采用 `V_k`。
- `geom_memory.py`、`geom_train.py`、`bfzi_geom_frac.py`、`frac_memory.py`、`bfzi_frac.py`、分数阶 ROM 训练/推理按此统一。
- 保存 `drive_rule="left"` 与 `soe_quad="trap"`，新训练和推理都读取相应配置。
- 原旧模型缺失这两项时，使用原 `drive_rule="right"` 和 `soe_quad="legacy"` 还原旧训练时的计算约定。GeomFracET 的较早 S5 检查点已经采用左端输入，沿用该约定。

旧 GeomFrac 预训练权重可以载入新 45 维 GeomFracET 后继续训练，不将旧模型未经重训的结果当作新时序和新求积方案的验收指标。

## 3. 网络结构及时间步长

GeomFracET 当前结构保持：

1. 三通道、14 对数率 SOE 和记忆附加指数状态，每个通道独立可学习分数阶 α 与额外指数时间常数。
2. 45 维物理几何/五热参数条件输入和共享 MLP，预测 6 个复阻抗实参数 + 8 个辅助温度摘要。
3. 32 维共享热状态神经头、空间热基重构和五材料物理能量校正。
4. 自主推理实时求双端口复电流以及海水/双铜损耗；无在线八方向切线求解和隐式热 ROM 求解。

将神经热状态积分细分成固定最大步长 `heat_dt=1.875 s`，独立于输出采样 `15/7.5/3.75/1.875 s`。同一恒定热输入下，SOE+热状态的跨步长组合误差均通过 `2e-6` 绝对容差检验。改变外部采样时间时，回路会在不同采样点重新计算输入和五材料校正，故**整个闭环仍有采样差异**，下面报告其实测值。

## 4. 训练/推理不一致的修复

原来的 `closed=True` 训练只使热状态自回归，仍始终使用物理教师热源。现增加可微分双端口源项 `_port_src`，使用预测的海水动态复阻抗和双铜温升恢复预测电路电流、海水损耗及铜损，反向梯度可传递至 SOE、神经阻抗和热状态头。

进一步增加 `_heat_balance`：按真实 FEM/FV 降阶算子的五材料容量与导热矩阵、海水+双铜功率，迭代求解同一 5×5 守恒校正并保持梯度。专项测试与原推理 `_balance` 的结果一致。新训练先进行教师监督，然后在 20–80 轮逐渐把功率源输入和闭环状态切换到预测值；**从第 80 轮起才能选择最佳自主模型**，阻止尚未启动真实闭环训练时就保留第 1 轮权重。

复查还发现 32 模态的归一化误差不能直接代替物理温度精度。在热方程相对残差和热模态损失之外，新增两个封装 64 节点及双铜的真实温度误差直接监督，避免对局部热点的错误模型选择。优化器使用 Adam、梯度裁剪阈值 3，并在训练与验证损失出现非有限值时停止。

本次热源校正训练采用原冷态附近的复铜阻抗线性切线构造可微分电路；线上推理保留原 BFZI 复铜阻抗函数。当前 60s 试验的铜温在 300K 左右，较高温完整时程后续需继续验收训练与在线热源之间的差别。

## 5. XEnergy 真实 S5 训练结果

数据直接使用已有原 BFZI V150/S5 生成的非冻结 T01–T03 三组训练几何、六条 60 s 时程，以及完全留出几何 T04 的两条 60s 时程。统一模型使用 45 维输入、32 维热状态。新训练显式使用 `--soe-quad trap --heat-dt 1.875 --epochs 240 --patience 70 --lr 0.0003 --closed-weight 0.15 --therm-weight 0.05`，使用已有 GeomFracET 热头权重作为初始化。

训练在第 107 轮取得最佳验证结果，第 177 轮提前停止。所选权重在 T04 上自主闭环推理：

| 留出时程 | 原模型动态 Z 相对 L2 | 本版动态 Z 相对 L2 | 本版铜温最大误差 | 本版封装最大误差 |
|---|---:|---:|---:|---:|
| T04，历史 1 | 15.59% | **8.48%** | 0.346 K | 1.314 K |
| T04，历史 2 | 15.40% | **6.34%** | 0.488 K | 1.611 K |

完整复阻抗相对 L2 为 `3.0234e-6 / 4.8511e-6`。五材料预算最大残差为 `1.912e-9 / 1.746e-9 W`。本版相对旧模型提升了动态阻抗跨几何预测，同时封装温度误差相较旧模型 `1.027 / 0.658 K` 增大：准确度权衡已在报告中保留，后续训练以完整 600s 和更强空间热层表达为重点。

T04 第一条历史 15s 采样、60s 物理时间的模型独立推理 `0.0472 s`；相同物理时程 BFZI S5 直接法耗时 `40.7323 s`，在线动态计算约 `864×`，不含新几何冷态锚构建和神经训练。

## 6. 跨时间网格验证和可复现命令

同一 T04 75V、30s 断电轨迹，对齐 15s 时刻的结果：

| 输出间隔 | 对 15s 网格的最大温度差 | 动态 Z 差 / 物理动态 Z 范数 |
|---|---:|---:|
| 7.5s | 0.1215 K | 2.975% |
| 3.75s | 0.1597 K | 3.873% |
| 1.875s | 0.1687 K | 4.062% |

上表反映整个自主闭环的电路热源采样与能量校正差异；每一条的五材料预算残差仍低于 `1e-8 W`。SOE/神经热状态内部更新的组合一致性已由独立测试通过。下一阶段如需进一步减小全闭环跨输出间隔差，可将回路和材料校正也统一按固定内部时钟计算。

```powershell
python -m pytest tests/test_bfzi_frac.py tests/test_bfzi_frac_rom.py tests/test_bfzi_geom_frac.py tests/test_bfzi_geom_neural.py tests/test_bfzi_geom_canon.py tests/test_bfzi_s5_bridge.py tests/test_bfzi_s5_teacher.py tests/test_bfzi_s5_buildset.py tests/test_bfzi_soe_consistency.py -q

python -m examples.bfzi_geom_multi --manifest runs/geomfrac_s5_pilot60_manifest.json --init-model runs/geomfrac_et_s5_pilot60.pt --soe-quad trap --heat-dt 1.875 --epochs 240 --patience 70 --lr .0003 --closed-weight .15 --therm-weight .05 --out runs/geomfrac_et_s5_soe.pt --report runs/geomfrac_et_s5_soe.json

python -m examples.bfzi_soe_review --anchor runs/bfzi_s5_T04_ultra_pilot60_anchor.npz --model runs/geomfrac_et_s5_soe.pt --teacher runs/bfzi_s5_T04_ultra_pilot60_drive1.npz --out runs/geomfrac_et_s5_soe_steps.json
```

本轮 XEnergy 与 SOE、GeomFracET、BFZI S5 适配有关的 **42 项测试通过**，包含新求积、可学习梯度、跨步长 SOE 状态组合、预测回路热源与五材料可微校正测试。最终模型 `runs/geomfrac_et_s5_soe.pt`、自主推理与跨步长指标均在仓库存档。

本轮结论为**网络结构与训练链路可运行、能够从真实多几何物理教师获得有效训练结果**；600s、更多热参数和冻结 U01–U12 仍需独立验收。
