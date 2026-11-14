# V2.1 BFZI 温变电磁亚网格积分与近源物理协方差

日期：2026-11-14；状态：已实现并通过 P03/P07 原 BFZI V150 温变电磁实算；完整600 s电磁热动态闭环待集成。

## 核心变化

真实 OBB 旋转封装海水湿FV的原 BFZI Green/GMRES和双端口方程保持不变。新增 BFZIV148Adapter(source_mask="subcell")，复用全部湿FV原有的积分点 q、体积权重 w、双端口空气源投影 C(q)。每个湿FV的 Green 计算采用积分平均温度及其平均电导率；局部近源质量积分则增加温度与源场在湿FV内的协方差。

物理公式（所有量均在对应湿FV上）：

Tbar_i = sum_q(w_q T_q) / sum_q(w_q)

sigma_i = sigma0 [1 + alpha (Tbar_i - T0)]

M_i = sum_q(w_q C_q^T C_q) - V_i Cbar_i^T Cbar_i

K_i = sum_q[w_q (sigma(T_q)-sigma_i) C_q^T C_q]

Z_new = Z_original(sigma_i, M_i) + (omega mu)^2 sum_i K_i

Wq_new(i) = Wq_original(i) + (omega mu)^2 K_i / V_i

Wq 是弱式焦耳热的双端口损耗密度，海水损耗为 0.5 sum_i(V_i I^H Wq_i I)。K_i 对单元内均温严格为零，保留原冷态解。计算同一物理温度场的亚网格积分别改善温变Green主项与近源矩项的一致性，未增加任何BFZI电磁节点、GMRES求解或者热方向电磁求解。

调用：em=BFZIV148Adapter(data,bf,heat_mode="port-weak",fv_faces="exact",source_mask="subcell")；em.solve(T_cell,T_copper,sea_temp=fn)，fn(points_xyz)返回物理绝对温度。仅按单元均温求解时 source_mask="subcell" 的结果与 source_mask="all" 一致。

## 原 BFZI V150 真实 P03/P07 572 vs 1312 海水FV检验

相同真实旋转封装位姿，固定293.15 K铜温。海水解析温度场为 T(x)=293.15+5 exp[-d(x,OBB)^2/(2 ell^2)]。以本身冷态为基准比较两个网格的六分量温变复阻抗增量。原列采用湿FV体积均温+全湿近源积分，新增列进一步计算近源温度协方差。

| 热场尺度 | P03 原 | P03 新 | P07 原 | P07 新 |
|---|---:|---:|---:|---:|
| 3 mm | 19.3068% | 8.6552% | 14.7153% | 7.0817% |
| 10 mm | 21.1774% | 11.5749% | 20.5195% | 11.3733% |
| 30 mm | 9.1222% | 2.4178% | 9.7927% | 3.0765% |
| 80 mm | 3.5573% | 1.3424% | 3.8885% | 1.8406% |

这是独立固定解析温度场的温变电磁阻抗增量跨网格相对差，并非完整高功率温变电磁热动态解误差，也非神经网络训练误差。

原 BFZI 弱式电损积分与原 BFZI 端口海水电损在全部新增 P03/P07 1312单元温场的相对误差低于 2.1e-15，验证了新增协方差项的端口功率一致性。相关局部测试 3 项通过，与原热层/间隙桥联合9项通过。

代码：sdfmpneo_bfzi/bfzi_v148_adapter.py；原湿FV对照脚本：examples/bfzi_obb_em_grid_probe.py；局部单测：tests/test_bfzi_subcell_temp.py；真实数据：runs/bfzi_obb_em_subcell_probe.json。

运行：

python -m examples.bfzi_obb_em_grid_probe --bfzi-root D:/work/bfzi-et-v2.1/BFZI --runs runs --poses P03 P07 --source-mask subcell --integrate-temp --out runs/bfzi_obb_em_subcell_probe.json

## 下一项关键工作

1. 将 BFZIV148Skin3D 八层、2 mm被动热层状态重建成原湿FV物理积分点 T(q,t)，并使 Green 所用的湿FV平均温度与守恒热网格的体积平均温度一致。协方差中先减去该湿FV的求积平均T，实现温度空间变化对近源积分的独立影响。
2. 真 P03/P07 在572/1312单元下重跑600s电磁热直接法，对照 legacy、all、subcell，比较动态阻抗增量、两个封装、铜、三维J/q与总热容量/功率守恒。已有5mm间隙热桥对动态阻抗改善很小，应保持可选方案。
3. 用更一致的原 BFZI 物理真值重新衡量6方向温变电磁切线+32阶五材料热ROM+40维几何分数阶神经代理的未见几何、功率、时间和介质外推；逐步减少新几何冷态场/热算子/电磁切线的离线构建成本。不要将特定网格代理的约0.01%误差当作绝对物理空间精度。

此阶段按实测验证保留原高精度物理基线默认，subcell 作为新的可选物理积分模式。
