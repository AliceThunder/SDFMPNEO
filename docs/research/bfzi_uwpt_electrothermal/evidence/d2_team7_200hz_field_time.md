# TEAM-7 200 Hz：四周期同相位空间场时间收敛与后处理机制

BFZI-PDR V1.1，2026-11-11。数据由 XEnergy 官方 Elmer 26.2 在 TEAM-7 `h10` 真正三维体网格上完成的 BDF2 64/128/256 步计算导出，与 [原生功率/磁能量时间证据](d2_team7_200hz_time_20261111.md) 的三组求解完全相同。固定 R=0.20m、官方三分量体线圈源 `cos(2*3.14*200*t)`、材料和边界，仅改变 Δt。无实验拟合。

## 新增代码与独立同物理时间场比较

`examples/bfzi_d2_team7_field_time.py` 验证三份 `case_ene200s{steps}.sif` 的输入物理一致，从 `resene200s{steps}/f.dat` 读取完整 Elmer 原生 SaveLine 数据，检查 4×101 测线、27 列字段、坐标一致性与时步索引。对四周期中最后一个周期 `t=0.0153125...0.0200s` 的同一 16 个时刻，比较 A1/A2 空气磁场、上/下铝板表面 J 的时间波形。所有数值均保留原始输出，没有在时间轴上平移场。

| 固定测线观测量 | 64→128 L² | 128→256 L² | 观测 p |
|---|---:|---:|---:|
| A1 Bz | 0.33829675% | 0.08712189% | 1.956350 |
| A1 Bxyz | 0.21199890% | 0.05469454% | 1.954650 |
| A2 Bz | 0.39567447% | 0.10207013% | 1.953565 |
| 上表面 Jy | 5.47216454% | 3.03662372% | 0.834750 |
| 下表面 Jy | 8.30050240% | 4.29963255% | 0.938554 |
| 上表面 Jxyz | 7.41319577% | 3.77793467% | 0.957518 |
| 下表面 Jxyz | 14.19378680% | 6.57373305% | 1.103612 |

这些是真实 Elmer 元素场经 SaveLine 固定位置采样的数值差异，不是 raw Whitney curl B 的积分点结果。B 场接近二阶，J 场约一阶。此前原生涡流功率完整波形约一阶，与该独立 J 结果一致；完整磁能量波形约二阶，也与 B 收敛一致。

## 源码机制与验证边界

官方 ElmerCSC `fem/src/modules/MagnetoDynamics/CalcFields.F90`（`devel`，读取 blob `cc738e71dcb0e45a69863c5ffb17e88be05272c1`）第 1153–1157 行显示：调用 `GetLocalSolution(PSOL,...,Tstep=-1)`，当 `pSolver % TimeOrder==1` 时使用 `(SOL-PSOL)/dt`；第 1594–1602 行用于瞬态 `E=-dA/dt`，第 2018–2032 行用于 Joule 功率。另一方面 `WhitneyAVSolver.F90` 第 1668–1675 行在自身另一处导数求值显式区分 BDF1/BDF2。此源码事实与实际测得的 B/J 收敛阶、涡流功率半步相位延迟诊断相互吻合。**仍需确认 XEnergy Elmer 26.2 二进制的对应源版本以及 `TimeOrder` 的运行时取值，才能作为最终修复依据。**

一项额外尝试在真实 `f.dat` 上以导出的节点化 `A{e}` 的二阶差分直接修正 `Current Density E`：得到上表面 Jxyz 的跨步长差 35.0251%→18.3862%，下表面 73.4665%→47.0197%，均明显差于原始元素场 7.4132%→3.7779% 与 14.1938%→6.5737%。**否决该后处理替换**：导出的节点化 `A{e}` 与恢复后的 J 缺少严格统一的 Whitney 积分点定义，且不能直接重建导体标量势的贡献。没有修改实际 Elmer 功率、场或教师数据。

新增 `tests/test_bfzi_d2_whitney.py` 的实际自研 TEAM-7 3D 边元求解对照：原始有向 Whitney 边自由度计算的 curl B，与 `solve_closed_ports` 内部独立构造的单元 B，在金属/真空两个求解中均一致；制造解的方向和规范不变性也继续通过。该测试建立自研求解器的原始场恢复可信度，尚未完成 Elmer 原始边自由度同源交叉提取。

## 执行与后续

```powershell
python -m pytest tests/test_bfzi_d2_team7_field_time.py tests/test_bfzi_d2_team7_time.py tests/test_bfzi_d2_whitney.py -q
python -m examples.bfzi_d2_team7_field_time --root D:/work/BFZI-D2-verify/runs/team7_hr --out docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_200hz_field_time.json
```

实际完成：**8 passed**。全量真实结果：[JSON](d2_team7_200hz_field_time.json)；源文件、SIF、日志与 `f.dat` 均保留在 XEnergy `D:/work/BFZI-D2-verify/runs/team7_hr/h10`，不改动 `D:/research`。

下一项依然是：核验实际 Elmer CalcFields 时间导数实现，并基于原始 Whitney 边自由度和导体标量势在积分点构造 BDF2 一致的 J/E/q；随后将经过独立验证的时间预算用于真实三维 h/R 空间与外域收敛分析。当前 **TEAM-7 数值精度 OPEN、D2 OPEN、旧 G1 FAIL、教师冻结**。
