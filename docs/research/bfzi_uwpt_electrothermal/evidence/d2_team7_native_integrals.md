# D2 TEAM-7 Elmer 原生体积分收敛（2026-11-10）

**执行方案：** [BFZI-PDR V1.1](../BFZI_PDR_V1_1_NUMERICAL_CONVERGENCE.md)  
**状态：** TEAM-7 数值精度 OPEN，D2 OPEN，G1 FAIL，神经代理教师冻结。

## 本轮独立物理积分

沿用已完成的 TEAM-7 50/200 Hz 三档空间 h、三档外域 R，保持 CSC 原始几何、材料、真实体线圈源、外边界及相同 BDF2 求解设置。新增 Elmer 原生 `SaveScalars`，读取 `MagnetoDynamicsCalcFields` 输出的：

- `res: electromagnetic field energy`（J）
- `res: eddy current power`（W）

该结果由 Elmer 内部物理场积分产生，避免了 VTU 节点插值与 SaveLine 点曲线的后处理影响。**未使用任何实验数据。**

先以每个频率的第一个 BDF2 时间步进行数值通路验证。50 Hz `Δt=1.25 ms`，200 Hz `Δt=0.3125 ms`；均采用官方闭合体线圈源。10 个独立 ElmerSolver 实算均 **exit=0**，全部生成 `phys.dat` 和 `phys.dat.names`。第一步仅为时间瞬态启动阶段，本表不代表四周期终点或周期稳定后的收敛。

## 实际原生积分值

| 频率 | 网格 | 电磁场能量 (J) | 涡流功率 (W) |
|---|---|---:|---:|
| 50 Hz | h15 | 0.4390936345 | 9.345060027 |
| 50 Hz | h10 / R0.20 | 0.4574253965 | 9.152455840 |
| 50 Hz | h075 | 0.4652717391 | 8.962002012 |
| 50 Hz | R0.15 | 0.4540297076 | 8.774195376 |
| 50 Hz | R0.30 | 0.4603030877 | 9.504175698 |
| 200 Hz | h15 | 0.4355955265 | 20.45027592 |
| 200 Hz | h10 / R0.20 | 0.4539051022 | 20.46788466 |
| 200 Hz | h075 | 0.4618224193 | 20.10709613 |
| 200 Hz | R0.15 | 0.4506085003 | 19.77121565 |
| 200 Hz | R0.30 | 0.4566855716 | 21.13529658 |

相邻网格的相对差以第二档为归一化参考：

| 频率 | 扫描 | 电磁场能量相对差 | 涡流功率相对差 |
|---|---|---:|---:|
| 50 Hz | h15 → h10 | 4.00760% | 2.10440% |
| 50 Hz | h10 → h075 | 1.68640% | 2.12513% |
| 200 Hz | h15 → h10 | 4.03379% | 0.08603% |
| 200 Hz | h10 → h075 | 1.71436% | 1.79433% |
| 50 Hz | R0.15 → R0.20 | 0.74235% | 4.13288% |
| 50 Hz | R0.20 → R0.30 | 0.62517% | 3.70069% |
| 200 Hz | R0.15 → R0.20 | 0.72628% | 3.40372% |
| 200 Hz | R0.20 → R0.30 | 0.60884% | 3.15781% |

### 结论

两种频率的电磁场能量在 h 细化中表现出相邻相对差缩小（约 4% → 1.7%），证明原生积分量能够提供与 Bz 点值、VTU 采样不同的收敛证据。涡流功率 h/R 的相对差仍达到数个百分点，200 Hz 的 h 第一档功率差仅 0.086% 也不能独立认定收敛。

本轮对原生结果的实际导出方式进行了验证：在同一个 h10、50 Hz 的单步 SIF 中加入 `Solver 6 / SaveScalars`，输出电磁场能量 0.4574253964612 J 和涡流功率 9.152455840128 W。增加 `Save Elemental Fields` 至 `ResultOutput` 的另一项测试仍只在 VTU 中导出 `point_data`，并未得到独立单元积分场，因此没有将这些 VTU 数据误标为原始单元旋度场。

### 复现实验

```powershell
python -m pytest tests/test_bfzi_d2_team7_integrals.py -q
python -m examples.bfzi_d2_team7_integrals --root D:\\work\\BFZI-D2-verify\\runs\\team7_hr --out docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_native_integrals.json
```

上述命令读取 10 份已实算 `phys.dat` 并重算数值收敛指标。若需要重新求解，在命令中增加 `--solve --elmer D:\\work\\ElmerFEM\\ElmerFEM-nogui-nompi-Windows-AMD64`。使用 `--steps 64` 可开展四周期终点积分测试，须与 `--solve` 配合。真实 MSH、ElmerGrid 网格、单步 SIF、求解日志和场数据保存在 `D:\\work\\BFZI-D2-verify\\runs\\team7_hr`。机器可读结果为 [d2_team7_native_integrals.json](d2_team7_native_integrals.json)。

**下一阶段精度关口：** 完成同源 50/200 Hz 四周期终点的原生磁能量/耗散量 h/R 扫描，继续建立原始 Whitney 旋度 B 单元数据与体积分的对照，采用事先冻结的数值误差预算判定收敛，并补齐物理一致性。**D2 继续 OPEN。**
