# BFZI V2.1 物理直接法参考基线锁定

日期：2026-11-14。依据 [主线A–E闭合计划](BFZI_MAINLINE_CLOSURE_V1.md)，任务A已有的真实P03/P07四条600s时程满足所有数值验收阈值，现固定**可用物理参考基线**，供B五材料ROM、C神经代理和D跨几何泛化复用。

## 固定参考模型

- 原 BFZI V150 Green/resolvent、双端口复阻抗与原生湿FV。
- 原双旋转封装真实OBB排水，封装不相交，原湿FV公共面精确湿面积。
- 2mm八层海水热网络、铜与封装三维温度、原湿FV容量严格守恒。
- `source_mask="subcell"`：真实湿FV高斯点温变源场协方差、端口阻抗与弱式焦耳热。
- `skin_heat="source"`：真实高斯点温变局部J/q，按原弱式功率逐FV保守分配到体水和八层海水。
- P03/P07两几何，572/1312原湿FV，600s电压启停时程，四条物理直接法归档。

## 验收记录

| 指标 | P03 | P07 | 闭合阈值 |
|---|---:|---:|---:|
| 动态ΔZ跨网格相对L2 | 1.1116% | 0.2106% | ≤2% |
| 封装最大跨网格温差 | 0.2770 K | 1.1290 K | ≤1.5 K |
| 铜最大跨网格温差 | 0.0816 K | 0.6620 K | ≤1 K |
| 海水体积均温最大跨网格差 | 0.000534 K | 0.003001 K | ≤0.05 K |
| 端口海水电损/弱式电损最大相对差 | 5.15e−16 | 5.77e−16 | ≤1e−10 |
| 最大热节点收支残差 | 5.08e−7 W | 1.98e−6 W | ≤1e−4 W |

**A数值验收：通过。** 该结论针对已定义的两级原湿FV空间一致性与真实600s时程，不扩大为对所有几何/网格的数学收敛声明。

## 参考文件

- `runs/bfzi_obb_skin_source.json`
- `runs/bfzi_pose_p03_skin_source_subcell_medium.npz`
- `runs/bfzi_pose_p03_skin_source_subcell_extra.npz`
- `runs/bfzi_pose_p07_skin_source_subcell_medium.npz`
- `runs/bfzi_pose_p07_skin_source_subcell_extra.npz`
- [局部源热沉积完整报告](evidence/v2_1_source_resolved_skin_heat.md)
- [高阶J/q重建报告](evidence/v2_1_near_source_Jq_reconstruction.md)

## 后续边界

局部面片热源在不同原湿FV划分下仍有空间误差，已明确量化为封装温差P03 0.277K、P07 1.129K。该残差低于本轮A的验收门槛。继续保留 `volume` 与 `source` 两种热源模式供消融；不以更细网格、更复杂热网络或反复调整源归一阻碍B的推进。后续发现可重复的能量不守恒或物理定义错误时，才重新打开A。

**下一执行任务：B，32阶五材料热ROM整合最新物理基线，完成P03/P07独立电压/启停留出、动态ΔZ和温度精度验证。**
