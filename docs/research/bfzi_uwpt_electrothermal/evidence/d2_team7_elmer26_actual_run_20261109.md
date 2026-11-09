# BFZI-PDR P1-ELMER-3D0｜Elmer 官方 TEAM-7 首次真实三维涡流场实验

**执行日期：2026-11-09** · **机器：XEnergy（Windows）** · **结论：3D H(curl) 求解链路 SMOKE PASS；已确认 `transient.sif` 在该运行中并未真正施加 50Hz 周期源，TEAM-7 独立 AC 基准 OPEN；D2 OPEN。**

## 1. 事实来源与完整可追溯输入

- **官方独立软件案例（并非本项目仿造案例）**：CSC 官方仓库 `ElmerCSC/elmer-elmag`，`TEAM7/`；精确 upstream 提交 [843037c](https://github.com/ElmerCSC/elmer-elmag/tree/843037c374ba61cf61d02d17a2c5fc11e0d33de3/TEAM7)。
- **软件**：CSC/FUNET Windows AMD64 `ElmerGrid 26.2` 与 `ElmerSolver 26.2`，无 GUI / 无 MPI；`D:\work\ElmerFEM\ElmerFEM-nogui-nompi-Windows-AMD64\bin`。
- **本机试验目录**：`D:\work\BFZI-D2-verify\runs\bfzi_pdr_team7`（原版官方 `steady.sif`、`transient.sif`、`TEAM7` 网格、日志、`res/*.vtu` 和 `f.dat`）；`D:\work\BFZI-D2-verify\runs\bfzi_pdr_team7_4period`（只修改总时步数的隔离复算、摘要 `team7_run_summary.json`）。
- **官方原生 mesh.header 实读**：23,128 网格节点，130,707 个四面体、14,556 个边界三角形。官方物理算例为偏心开孔铝板和物理线圈（不是 BFZI）；`CoilSolver → WhitneyAVSolver → MagnetoDynamicsCalcFields → ResultOutputSolver`。
- **官方参考数据**：同一目录 `TEAM7_A1B1.csv` 和 `TEAM7_ProblemDescription.pdf`；当前直接 CSV 对照使用 A1–B1 `y=0.072 m,z=0.034 m` 线、17 个 CSV 采样点。

## 2. 实际执行，而非 EXE 启动或合成夹具

在上述两个独立实验目录中（会话级设 `ELMER_HOME` 与 `PATH`）：

```powershell
# 官方示例的绝对原始版本，不更改线圈、材料、网格、频率：
& 'D:\work\ElmerFEM\ElmerFEM-nogui-nompi-Windows-AMD64\bin\ElmerSolver.exe' steady.sif
& 'D:\work\ElmerFEM\ElmerFEM-nogui-nompi-Windows-AMD64\bin\ElmerSolver.exe' transient.sif

# 单独复制原始 transient.sif 后，只更改：
# TimeStep Intervals(1) = 16  →  32
# Output Intervals(1) = 1    →  4
# dt=0.0025 s, 50 Hz, material/mesh/source/coil direction 均保持不变
& 'D:\work\ElmerFEM\ElmerFEM-nogui-nompi-Windows-AMD64\bin\ElmerSolver.exe' transient_4period.sif
```

| 实际运算 | 退出码 / 时间 | 输出与空间场 | 官方 SIF 自带参考范数校验 |
|---|---|---|---|
| 官方 steady 3D | `0` / 14.22 s | `steady_t0001.vtu` 7.72 MB、`fs.dat`；VTU `max|B_component|=0.02155486 T`、感应 `J=0`（恒定源，符合此工况定义） | CoilSolver 相对 `4.88e-7` PASS；WhitneyAVSolver **0.3324774% FAIL** |
| 官方 `transient.sif` 标称 50Hz、16步（实际恒定励磁） | `0` / 126.02 s | `transient_t0001..0016.vtu` 16 份；`f.dat` 6,464行；第16步 VTU `max|B_component|=0.02078765 T`、`max|J_component|=82031.43 A/m²`；数据有限且非零 | CoilSolver `4.88e-7` PASS；WhitneyAVSolver **15.73784% FAIL** |
| 恒定励磁延长至 32步，dt不变 | `0` / 250.06 s | 32时步探针，`f.dat` 12,928行，4条官方探针线 | WhitneyAVSolver **16.88488% FAIL**（SIF 内部参考范数来自官方初始配置，并非直接实验场误差） |

所有求解都显示 `MAIN: *** Elmer Solver: ALL DONE ***`，VTU 使用真实 `pyvista.read()` 读取并检查了 `magnetic flux density e`、`current density e` 三分量，均为有限非零值（稳态 J=0 除外）。`steady` 并不代表时变涡流物理验证；非零涡流证据来自 `transient.sif` 的**恒定源接通磁扩散过程**，不是可认证的 50Hz 周期稳态。

## 3. 独立揭示的尚未解决误差（不得强行标 PASS）

对 `res/f.dat` 内的 **A1–B1 同一条线**，固定每周期同一相位的第 8、16、24、32 步，按 101 个相同 x 点的三分量 B 的相对 L² 范数计算：

| 周期末采样 | B 同相位相邻周期相对 L² 差 |
|---|---:|
| 8 → 16 | **8.54450%** |
| 16 → 24 | **2.26414%** |
| 24 → 32 | **0.62136%** |

该序列显著下降，但**四周期后仍约 0.62%，不能宣称周期稳态已足够高精度**。官方原版 16 步的最后时步受初始瞬态影响。

直接将该线第 16 步 `B_z` 插值到官方 `TEAM7_A1B1.csv` 的 17 个有效 x 点，得到 **53.1227%** 的原始相对 L² 差；第 32 步约 **56.7017%**。**这些原始差值尚未对齐实测参考的正确激励相位（含“线圈电流为零/峰值”的时刻定义）、峰值/实值口径、参考样本语义、网格与时间积分误差**，不能直接解释为本项目 FEM 的相对物理误差；更不允许为凑数修改源幅度、材料或数据容差。相邻周期收敛和官方自带参考范数检查是**不同指标**，不能混淆。

说明：VTU 使用二进制 VTK 数据块，简单 `xml.etree.ElementTree.parse()` 会失败；本实验已经实际通过安装的 `pyvista.read()` 读取磁通密度、电流密度及所有网格点数据。`f.dat` 与 `fs.dat` 为真实 SaveLine 原生数值输出。

## 3a. 关键纠正：`transient.sif` 的 50Hz 指令未实际调制电流

在延长为 32 步的原官方 `transient.sif` 输出中，逐帧真实读取 `res/transient_t0001..0032.vtu` 的**线圈源** `coilcurrent e` 和 `coilcurrent`，发现二者的最大范数在 32 个时步上逐位不变（分别约 `1,067,046.40119 A/m²` 与 `1,067,046.48273 A/m²`）。同时感应 `current density e` 的最大矢量范数从第 1 步约 `2,280,654 A/m²` 单调衰减到第 32 步约 `8,390 A/m²`，B 场趋向静态，而不是随期望余弦正反激励。**上述“同相位周期差 8.54%→2.26%→0.621%”实际描述了 DC 接通后暂态衰减，不能用于证明 50Hz 周期稳态收敛！** 直接与 `TEAM7_A1B1.csv` 比较的 `53.1%/56.7%` 也首先是**错误施加周期源/状态不对应**，不应被误称为参考测量误差或 FEM 离散误差。

根因路径：官方 `transient.sif` 的 `CoilSolver` 在 `Exec Solver="before all"` 只产生固定 `CoilCurrent e`；其 `WhitneyAVSolver` 开启 `Use Elemental CoilCurrent=True` 直接使用静态线圈电流场，而 `Body Force` 所写的 `Current Density Amplitude=cos(...)` 未在输出中体现为有效磁场源调制。官方仓库另附 `TEAM7.sif`，使用 `Current Density 1/2/3 = Variable "time", "CoilCurrent e 1/2/3"` 并显式 `Real LUA "cos(2*3.14*50*tx[0])*tx[1]"`，且没有直接打开 `Use Elemental CoilCurrent`；本轮已安排在隔离目录 `D:\\work\\BFZI-D2-verify\\runs\\bfzi_pdr_team7_ac` **原样运行此官方 SIF 并逐时步核验有效 B/J 激励**，其运行结论另待实测，不得事先写 PASS。

## 4. 本轮关口与后续严格顺序

- **P1-ELMER-3D0 最小软件链路：恒源磁扩散 SMOKE PASS**，官方 130,707 四面体的三维磁场及恒流接通暂态感应涡流均真实运行，非零 `B,J` 已读出；**有效 50Hz 正弦励磁尚未由该运行证实**。此 PASS **仅表示可执行 H(curl) 成功，不表示官方实验精度通过**。
- **TEAM-7 标准数值关口：OPEN**。优先明确 CSV 测量采样的物理相位/时刻、对照 Elmer `B_z` 的正确选择并继续消除初始瞬态（必要时提高时间离散阶次/减小 `dt`），然后在正确可比条件下做三档独立 h 和三档 R 或两类边界处理；追加 200Hz。每次保留原始不合格差异，不事后改门槛。
- **D2：OPEN**。尚未执行同一真实 BFZI 双独立斜置 SE(3) 共形 OCC/ElmerGrid，也没有两套与本项目严格相同的闭合源/最终 Whitney RHS 和完整复 `2×2 Z` 的独立对照；D3/D4 OPEN，旧 G1 FAIL，神经教师冻结。
- **工作方式**：沙箱优先；只有必须完整 Elmer Windows 求解时批量使用 XEnergy 的 `D:\work`；不更改用户 `D:\research`、不使用 GitHub Actions。大型原生网格/VTU 保留于 XEnergy `D:\work`，仓库提交此轻量可复现证据。

本报告在**锁定的 [BFZI-PDR V1.0](../BFZI_PDR_EXECUTION_PLAN.md)** 框架下只更新可核验实算证据，不修改上位物理契约与预先确定的 D0–D6 验收路线。
