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

## 3b. 正确的官方 `TEAM7.sif` 已在 XEnergy 实际运行：真正 50Hz AC 成功

发现上述 DC 接通问题后，从**同一 CSC 官方仓库提交**原样复制另一份 `TEAM7.sif`（区别于 `transient.sif`），其 `Body Force 1` 三个源分量均使用 `Variable "time", "CoilCurrent e 1/2/3"` 与 `Real LUA "cos(2*3.14*50*tx[0])*tx[1]"`，并由 `WhitneyAVSolver` 消费。**不再把只含静态 `CoilCurrent e` 的模式错当周期源。**

独立实际运行目录：`D:\\work\\BFZI-D2-verify\\runs\\bfzi_pdr_team7_ac`，运行原样 SIF（不改变几何、线圈、网格、材料、频率和求解容差）：

```powershell
& 'D:\work\ElmerFEM\ElmerFEM-nogui-nompi-Windows-AMD64\bin\ElmerSolver.exe' TEAM7.sif
```

**结果**：ElmerSolver 26.2 退出码 **0**，耗时 **150.61 秒**，16 步、`dt=0.0025s`，官方原始 130,707 个四面体，生成实际 16 份 VTU 与 6,464 行 `res/f.dat`。真实用 `pyvista.read` 读取第16步复场（瞬态为实场）：`max(|B_component|)=0.02006847 T`、`max(|J_component|)=1,565,294.80 A/m²`。原版内置参考范数：CoilSolver `4.877344e-7`（PASS），WhitneyAV `0.003324258`（**0.3324258% FAIL**）；保留原始警告，不能假称官方内置 PASS。

**50Hz 激励真实性实证**：在固定 A1–B1 第一个 x 点 `x=0`、`y=0.072m,z=0.034m`，每四步相隔半个周期，`B_z` 依次为：

| 时步 | t(s) | 实测 `Bz` (T) |
|---|---:|---:|
| 4 | 0.010 | `+0.0006051084` |
| 8 | 0.020 | `-0.0005816920` |
| 12 | 0.030 | `+0.0005936934` |
| 16 | 0.040 | `-0.0005868447` |

存在明确的周期反向，不再是 DC 接通的单调缓变。两个同相位周期末（第 8→16 步）的 **A1–B1 101 点三分量 B 相对 L² 差 = 1.039409%**；表示仍有一定时间初值/离散误差，不可直接宣告周期稳态精度。

**参考测点初步对照**：从 Elmer 原生 `res/f.dat` 读取 `B_z`，插值到同官方 `TEAM7_A1B1.csv` 的 **17 个有效 x 点**（全程未调整电流幅值、材料或后验截正），第 **16 步**原始相对 L² 差 **3.423216%**，最大绝对差 **`3.988858e-4 T`**；第8步差 **4.351277%**。第 4/12 步因磁场符号相反，相对差约 200%，说明未经相位匹配的逐点比较无物理意义。该 CSV 的时间/相位、实验测量及 Bz 场定义仍须严格核对；当前只是同一官方案例的定量对照，**并未满足事先冻结的高精度与 h/R 证据门槛**。

**结论修正**：`transient.sif` 的 16/32 步数据只可称 DC 接通磁扩散实证；`TEAM7.sif` 才是本轮**真正 50Hz 周期受迫三维涡流 SMOKE PASS**。TEAM-7 正式高精度验证、200Hz、分辨率 h/外域 R/时间步长分轴收敛与本项目 BFZI 两口同源电磁场比较仍 **OPEN**。

## 3c. 200Hz 真实 AC 独立频率工况已执行

在 XEnergy 隔离目录 `D:\\work\\BFZI-D2-verify\\runs\\bfzi_pdr_team7_200hz`，以**同一 CSC 官方 `TEAM7.sif`** 为物理原型，仅做两处确定性参数变更：

- 线圈实际三分量电流的 `cos(2*3.14*50*tx[0])` 改为 `cos(2*3.14*200*tx[0])`；
- `dt=0.0025s` 改为 `dt=0.000625s`，仍每周期 8 个离散时步、共 16 步/两周期；原 mesh、材料、线圈实际电流、容差、模型和场探针均保持不变。

ElmerSolver 26.2 实际完成，**退出码 0、143.51 s**，真实最终 VTU 分量最大绝对值 `|B|_component=0.01989988908 T`，`|J|_component=3,451,491.21 A/m²`，均有限且非零。

| 200Hz 时步 | 时间 | A1–B1 x=0 处 `Bz` |
|---|---:|---:|
| 4 | 0.0025s | +0.0004941278 T |
| 8 | 0.0050s | -0.0004717827 T |
| 12 | 0.0075s | +0.0004808157 T |
| 16 | 0.0100s | -0.0004739625 T |

两周期末 A1–B1 101点三分量 `B` **同相位相对 L² 差=0.3449784%**；有真实 200Hz 正负交替且低于本轮 50Hz 相同 8→16 步的 1.0394%。官方 `TEAM7.sif` 尾部仍硬编码了**50Hz 对应的 Solver2 Reference Norm=2.15104858e-06**；200Hz 仿真读到 Solver2 Norm=2.09552265e-06，相对此旧值差 `2.581342%`。**这个跨频率参考范数不应直接作为 200Hz PASS/FAIL 的实验误差**，也不得通过改写它或放宽门槛来“通过”。尚缺官方 200Hz 实验时刻/实测参考、频率一致的物理场对照、h/R/时间误差预算。

**状态：Elmer 50Hz + 200Hz 真实 AC 三维场/涡流 SMOKE PASS，TEAM-7 物理精度 OPEN，D2 OPEN。**

## 4. 本轮关口与后续严格顺序

- **P1-ELMER-3D0 最小软件链路：恒源磁扩散 SMOKE PASS**，官方 130,707 四面体的三维磁场及恒流接通暂态感应涡流均真实运行，非零 `B,J` 已读出；**有效 50Hz 正弦励磁尚未由该运行证实**。此 PASS **仅表示可执行 H(curl) 成功，不表示官方实验精度通过**。
- **TEAM-7 标准数值关口：OPEN**。目前已分别实际运行正确的 50Hz/200Hz AC 源，但尚未获得可比的 200Hz 测量场验收，也未完成 h/R/时间误差预算。下一步先明确官方 CSV 实测相位/时刻、导出正确 Bz 与 J 参考，验证周期稳态/时间离散，再按预先冻结的指标做三档独立 h 与三档 R 或两类边界处理。每次保留原始不合格差异，不事后改门槛。
- **D2：OPEN**。尚未执行同一真实 BFZI 双独立斜置 SE(3) 共形 OCC/ElmerGrid，也没有两套与本项目严格相同的闭合源/最终 Whitney RHS 和完整复 `2×2 Z` 的独立对照；D3/D4 OPEN，旧 G1 FAIL，神经教师冻结。
- **工作方式**：沙箱优先；只有必须完整 Elmer Windows 求解时批量使用 XEnergy 的 `D:\work`；不更改用户 `D:\research`、不使用 GitHub Actions。大型原生网格/VTU 保留于 XEnergy `D:\work`，仓库提交此轻量可复现证据。

本报告在**锁定的 [BFZI-PDR V1.0](../BFZI_PDR_EXECUTION_PLAN.md)** 框架下只更新可核验实算证据，不修改上位物理契约与预先确定的 D0–D6 验收路线。
