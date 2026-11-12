# BFZI V2.1：跨几何三维温度、空间 J/q 与分数阶神经-热守恒闭环

日期：2026-11-12。原独立 BFZI V150 源代码仍为 AliceThunder/BFZI 的 bfzi-v150-turn-scan 分支；研究主分支为 research/bfzi-uwpt-electrothermal-surrogate。全部物理数据和速度记录均来自真实 XEnergy 运行。

## 目标和两条实际验证路线

此前 M1/M2/M4 训练、M3 全新几何留出的共享分数阶 SOE 记忆网络已经能够产生双端口复 Z 与三材料代表性温度，但不同 BFZI 几何对应 450 或 2316 个热状态，无法用一套固定网格直接输出空间场。M3 仍然没有参与神经模型时间序列训练。

**路线 A：连续 XYZ 坐标神经网络。** 新增 sdfmpneo_bfzi/geom_field.py、examples/bfzi_geom_field.py，输入真实原 BFZI 的海水/封装 FV 坐标、圆形-圆角方形统一线圈几何以及时序代理的八个热态观测值。空间形状训练使用 M1/M2/M4 原始 BFZI 教师温度，M3 全几何留出。输出非负温升，通过空间形状单调变换使每材料的 FV 加权平均温度与空间峰值精确对应分数阶时间网络。已保存 runs/bfzi_geom_field.pt、runs/bfzi_geom_field_metrics.json。

**路线 B：原 BFZI 冷态物理场锚定 + 分数阶电磁动态代理 + 保守三维热传导。** 新增 sdfmpneo_bfzi/geom_heat.py、examples/bfzi_geom_heat.py。为每个新 BFZI 几何仅进行一次冷态电磁与热算子预计算，归档空气自感互感、冷态复 Z、双端口单位电流三维海水 J、空间弱形式热源矩阵、海水/封装/铜 FV 热容量 C 及导热矩阵 H。此后任意电压与时间步都只需已训练 SOE 网络、两端口复电路求解、一个稀疏 FV 隐式热步，完全不调用动态 BFZI Green/GMRES 物理电磁求解。后续任何时刻都能输出三维 Tsea、双封装 Tpkg、两铜温度、三维复 Jsea、弱形式 qsea、端口 Z/I 与效率的计算所需量。

## 热-电反馈及严格功率关系

为充分利用原 BFZI 自身物理方法，海水阻抗由冷态值与学习的动态增量相加，铜自阻抗采用原 BFZI Bessel 圆导线趋肤公式：

    Zsea(t) = Zsea,cold + Delta_Zsea,SOE(t)
    Ztotal(t) = Zsea(t) + j omega Lair + diag(ZCu(TCu))
    I(t) = inv[Ztotal(t)+Zexternal] * [V(t), 0]^T

保留冷态原 BFZI 弱形式局部焦耳矩阵 W0(x)、单位电流场 J0,p(x) 和真实原海水 FV 体积 Vi，当前温度电导率变化采用 sigma/sigma0 = 1 + alpha_sea*(Tsea-Tamb)：

    q_raw(x,t) = max(0, sigma/sigma0 * 0.5 Re[I^H W0(x) I])
    Psea(t) = max(0, 0.5 Re[I^H Zsea(t) I])
    qsea(x,t) = Psea(t) * q_raw(x,t) / sum_i(Vi*q_raw_i)
    Jsea(x,t) = sigma/sigma0 * sum_p[J0,p(x)*Ip(t)]
    PCu,p(t) = 0.5 Re[ZCu,p(TCu,p)] * |Ip(t)|^2
    [diag(C)+dt*H]*theta_next =
        diag(C)*theta_prev + dt*[Vi*qsea_i, 0_pkg, PCu,1, PCu,2]^T

这样局部 q 必为非负，sum_i(Vi*qsea_i) 与同一个电路海水端口损耗 Psea 在数值精度内相等，三材料逐节点热方程满足离散能量恒等式。铜温度更新后重新计算趋肤复阻抗、电流及热源，随后把动态功率反馈到下一步可学习分数阶记忆。完整空间温度历史来自 BFZI 三维 FV 热传播，含封装滞后与铜升降温。

支持任意 XYZ 坐标温度查询：海水与封装对应材料的原 FV 中心插值，封装内返回封装温度；海水 J/q 仅在海水点有定义。

## 路线 A：完整独立 M3 几何的空间误差

只用 M1/M2/M4 时程训练空间形状神经网络，最多训练 1000 epoch。M3 两条未见过的物理工况，分别比较给解码器真实热摘要与给解码器自回归网络摘要：

| M3 两条时程的平均误差 | 真实热摘要输入 | 神经热摘要输入 |
|---|---:|---:|
| 海水最大温差 | 0.915 K | 0.828 K |
| 双封装最大温差 | 2.383 K | 3.769 K |
| 海水相对空间 L2 | 64.2% | 77.6% |
| 双封装相对空间 L2 | 28.3% | 68.1% |

本组实验已经揭示：普通坐标神经头能支持可变 FV 网格，并精确满足均值与峰值约束，但尚难恢复原 BFZI 的局部热源与跨材料导热空间结构。因此保留模型及真实留出结果作为研究基准，并采用路线 B 推进高精度三维主线。

## 路线 B：M3 原 BFZI 完整时程留出结果

M3 的 240/240 mm 圆角方形双线圈、13/13 匝、100 mm 中心距、2250 海水单元、两封装各32单元与两铜节点，其时间演化从未用于训练本次神经模型。在推理前单独计算并保存 M3 冷态 BFZI 场/热算子；动态热时程不需要 BFZI 电磁求解器。

| 原 M3 两条时程 | 40 V | 115 V |
|---|---:|---:|
| 原 BFZI 完整直接法时程 | 19.588 s | 24.769 s |
| 分数阶神经+保守 FV 时程 | 0.12394 s | 0.11825 s |
| 加速比 | 158.0x | 209.5x |
| 海水空间最大温差 | 0.00001477 K | 0.0007669 K |
| 双封装空间最大温差 | 0.00000334 K | 0.0001775 K |
| 铜温度最大误差 | 0.0000285 K | 0.001682 K |
| 动态复阻抗增量相对 L2 | 5.827% | 7.913% |
| FV 单节点热能量最大绝对残差 | 5.209e-13 W | 7.728e-11 W |
| 海水热源积分与海水端口功率最大差 | 3.553e-15 W | 5.684e-14 W |

时间仅统计两种方法相同 21 时刻的动态轨迹计算，几何冷态 Green/电感建立与网络训练均不计入。以上全部是实际直接法与神经-FV 轨迹重新运行，基于完整原始 M3 时程 NPZ 独立复核动态 Z 误差。

额外用真实 BFZI 温度和电流抽样重复求解原海水 J/q，验证由冷态场推导的温度场修正：高功率 115 V 时，冷态锚空间 J 的最大相对 L2 为 4.169e-4（0.0417%），弱形式 q 最大为 2.904e-5（0.00290%）。数据及原始采样时刻写在 runs/m3_bfzi_heat_metrics.json。

## 无 BFZI 安装的实际独立时程和空间点查询

仅加载 runs/m3_bfzi_anchor.npz 与 runs/bfzi_geom_frac_v5.pt，新运行 100 V、通电210 s、关断210 s、再通电、15 s 步长、总600 s：

- 41 时刻、2316 维三材料温度；2250x3 的复 J 与 2250 维 q；完整 Z/I 动态。
- 该新时程实测耗时 0.244 s；最大 FV 热收支残差 2.060e-11 W，q 积分与 Psea 差 3.553e-14 W。
- 使用 12 个预先保存的海水和双封装三维坐标做查询，结果含查询位置温度、材料类别；封装内部海水 J/q 输出 NaN。
- 9 项与新增三维模块相关的测试通过。

文件：runs/m3_bfzi_heat_user.json/.npz 和 runs/m3_bfzi_query.npy。

## 复现命令

首次对新 BFZI 几何做原始冷态场导出与完整真实动态对比：

    python -m examples.bfzi_geom_heat --bfzi-root D:/work/bfzi-et-v2.1/BFZI --sample M3 --anchor runs/m3_bfzi_anchor.npz --teacher runs/m3_frac_holdout.npz --model runs/bfzi_geom_frac_v5.pt --out runs/m3_bfzi_heat_metrics.json --bench-direct --field-check

后续直接使用预计算的原 BFZI 冷态锚，不需要 BFZI 软件，也无需任何已计算的动态教师时程：

    python -m examples.bfzi_geom_heat --standalone --anchor runs/m3_bfzi_anchor.npz --model runs/bfzi_geom_frac_v5.pt --voltage 100 --heat 210 --cool 210 --step 15 --horizon 600 --query runs/m3_bfzi_query.npy --query-time 300 --out runs/m3_bfzi_heat_user.json

可复算的三维场和电路数据：runs/m3_bfzi_heat_metrics.json/.npz、runs/m3_bfzi_heat_infer.json/.npz、runs/m3_bfzi_anchor.npz；纯坐标神经消融：runs/bfzi_geom_field.pt 与 runs/bfzi_geom_field_metrics.json。

## 下一主线

1. 继续推进原 BFZI 统一圆形—圆角方形的连续圆角参数、双线圈相对 SE(3) 位姿和封装尺寸变化。以冷态预计算可复用为边界，先引入新的几何点和独立直接法验证，再训练几何条件化。
2. 在准确的冷态原 BFZI 三维 J/q 锚上训练小型温度空间修正网络，以减少每个新几何的冷态场预计算负担，并保持 q 与端口复阻抗的功率关系。
3. 尝试用带材料热容量权重的空间热 ROM 替代每步 2316 节点的稀疏热求解，检验相同热预算、长时程/多次关断记忆与真实推理加速。
