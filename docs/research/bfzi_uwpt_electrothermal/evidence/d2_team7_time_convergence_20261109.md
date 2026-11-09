# BFZI-PDR P1｜TEAM-7 真实 50Hz 时间步长与周期稳态收敛证据

**2026-11-09，XEnergy Windows，ElmerSolver 26.2，真实官方物理网格 130707 四面体。**
**结论：四周期周期稳态已相对稳定，BDF1 时间离散尚未收敛；TEAM-7 精度 OPEN，D2 OPEN。**

## 1. 同一物理问题与预先固定的比较量

- CSC 官方仓库 `ElmerCSC/elmer-elmag` [commit 843037c](https://github.com/ElmerCSC/elmer-elmag/tree/843037c374ba61cf61d02d17a2c5fc11e0d33de3/TEAM7) 中 **`TEAM7.sif`**，其三个 `Current Density` 分量由显式 `cos(2*3.14*50*t)` 驱动。**不使用**已发现会将线圈 `CoilCurrent e` 固定的旧 `transient.sif` 作为交流证据。
- 全部试验固定偏心开孔铝板、`sigma=3.526e7 S/m`、`mu_r=1`、官方线圈/真实分布、峰值 `2742 A.turn`、原始网格、真实边界、求解容差、BDF **阶数 1**、频率 **50Hz**。只分别调整**时间步长**和**总积分周期数**；输出频率参数（`Output Intervals=8`）只改变 VTU 写文件次数，`SaveLine` 仍保存每一时步的 101 个采样。
- 官方测点是 `A1–B1`：`(x, y=0.072m, z=0.034m)`，x从0至0.288m。用 `res/f.dat` 的 `20:23` 列三分量 B，每时步 **101 个探针点**，比较周期内**相同相位** B 的相对 L²；用 `22` 列 Bz 线性插值至**17个测量点**比较实验 Bz。
- `TEAM7_A1B1.csv` 为 CSC 50Hz `omega*t=0°` 17 点；另建 [50/200Hz 固定参考 CSV（国际 TEAM-7 公开测量，单位已换算成 SI）](team7_a1b1_measured_bz_0deg.csv)。官方规定电流**在 ωt=0° 最大**，所以采用完整整数周期末比较，不允许为凑实验曲线挑选其他时步或调整电流幅值。
- `err_period = ||B(t)-B(t-T)||₂ / ||B(t)||₂`（A1–B1 上101点×3分量），`err_dt = ||B_Δt(t)-B_Δt/2(t)||₂ / ||B_Δt/2(t)||₂`；`err_measure = ||Bz_pred-Bz_measured||₂ / ||Bz_measured||₂`（17点）。三个指标性质**各不相同**。

## 2. 全部已实跑时间工况（非估计）

| 案例及 XEnergy `D:\\work\\BFZI-D2-verify\\runs` 下目录 | 时间步长(s) | 总时步 | 周期数 | 实际耗时(s) | 最后一周期 B 相对 L² 差 | 17点测量 Bz 相对 L² 差 |
|---|---:|---:|---:|---:|---:|---:|
| `bfzi_pdr_team7_ac` | 0.0025 | 16 | 2 | 150.61 | 1.039409% | 3.423216% |
| `bfzi_pdr_team7_ac_4period` | 0.0025 | 32 | 4 | 309.22 | **0.0483176%** | **3.233607%** |
| `bfzi_pdr_team7_ac_dt2` | 0.00125 | 32 | 2 | 269.53 | 0.725430% | 4.466747% |
| `bfzi_pdr_team7_ac_dt2_4period` | 0.00125 | 64 | 4 | 559.36 | **0.0251768%** | **4.317445%** |

**全部 ElmerSolver 实际退出码为 0**，并显示 `MAIN: *** Elmer Solver: ALL DONE ***`，四周期半步长耗时实际为 559.36 秒，日志和原生场留在各自独立目录下的 `TEAM7_case_run.log` / `res/f.dat` / `res/*.vtu`，数据未经二次拟合。Elmer 官方 SIF 内置 `Solver 2 :: Reference Norm` 是固定值，未随时间步、周期或 200Hz 同步改变，其 FAIL 不等同实验 Bz 的场误差；警告保留。

**关键同一时刻比对**：

| 比较 | 物理时刻 | 整条测线上三分量 B 的相对 L² 差 |
|---|---:|---:|
| 2周期：0.0025s/16步 与 0.00125s/32步 | 0.04 s | **1.143364%** |
| 4周期：0.0025s/32步 与 0.00125s/64步 | 0.08 s | **1.265094%** |
| 粗步长同周期数：16步与32步的相同初始段 | 0.04 s | **0**（确认重新运行确定性） |

粗时间步长四周期第24→32步相差0.0483%；半时间步长四周期第48→64步相差0.0252%；二者**都已达到约千分之一以下的周期重复量级**。与此同时同一时刻两时间步长仍相差 **1.265%**，这就是本轮主数值结论：**周期稳态差足够小，无法证明时间离散精度已达到要求。**

实测 Bz 的17点 L² 差随减小步长从 **3.234% 上升至 4.317%**。这不是对新的时间步长作“负面结论”，它说明实验差包含**时间离散、空间网格、人工外边界、线圈源和实验参考不确定性**的叠加，粗步长有数值抵消误差的可能。不得将对实验误差下降当作必然的数学时间收敛，更不得用实验表值反向改电流幅度。

## 3. 200Hz 同频真实实验参考补充

原先只用了 CSC `TEAM7_A1B1.csv` 的 50Hz 列，现在根据 TEAM Problem 7 公开 A1–B1、`ωt=0°` 的 200Hz 17 点 Bz 建立同一 [SI 参考 CSV](team7_a1b1_measured_bz_0deg.csv)，计算命令见后文。200Hz 正确源的两周期 `dt=0.000625 s` 初步实验 Bz 相对 L² 差 **6.241396%**，最大绝对差 **0.000535506 T**；同相位 B 线周期差 **0.344978%**。这只是首次定量对照，200Hz 独立时间/空间/外域收敛仍 OPEN。

数据参考：国际 [TEAM Problem 7 物理规范](https://www.compumag.org/jsite/images/stories/TEAM/problem7.pdf)（相量、几何、频率和测点定义）、[CSC 官方 Elmer 示例](https://github.com/ElmerCSC/elmer-elmag/tree/843037c374ba61cf61d02d17a2c5fc11e0d33de3/TEAM7)（50Hz CSV）、[TRAFOLO TEAM-7 实验数据校正及独立 FEM 对照](https://docs.trafolo.eu/Validation/TEAM%20Problem%207_%20Eddy%20Currents%20in%20a%20Plate%20Compared%20with%20Measurements/)（公开指出部分 Jy 与200Hz的90°数据存在测量表问题）。当前使用的 **Bz 0°**，不以其被质疑的200Hz 90° Bz 曲线作通过指标。

## 4. 已实现并在 XEnergy 真实验证的统一评价入口

新增源码 [`examples/bfzi_d2_team7_elmer_eval.py`](../../../../examples/bfzi_d2_team7_elmer_eval.py)，直接读取官方 Elmer `res/f.dat`，只有**完整整数周期、相同源相位**才允许和固定表比较，同时输出周期重复差；已经在 XEnergy 本机对 **5 个真正已求解场文件**实跑并生成 `team7_eval.json`，均正常退出。Windows 独立工作树在提交 `8ff0e1ae` 验证。

```powershell
cd D:\work\BFZI-D2-verify
python -m examples.bfzi_d2_team7_elmer_eval --data runs/bfzi_pdr_team7_ac_4period/res/f.dat --freq 50 --dt .0025
python -m examples.bfzi_d2_team7_elmer_eval --data runs/bfzi_pdr_team7_ac_dt2_4period/res/f.dat --freq 50 --dt .00125
python -m examples.bfzi_d2_team7_elmer_eval --data runs/bfzi_pdr_team7_200hz/res/f.dat --freq 200 --dt .000625
```

## 4a. 第三档真实时间步长与 BDF1 收敛阶（新增）

2026-11-09 随后在 XEnergy `D:\\work\\BFZI-D2-verify\\runs\\bfzi_pdr_team7_ac_dt4_4period` 实际完成 **128 步、dt=0.000625s、50Hz、四周期、相同官方原始网格/材料/振幅和实际三分量 cos 激励、BDF Order=1**。原始 `TEAM7_case.sif`、`TEAM7_case_run.log`、`res/f.dat`、`res/*.vtu` 与结构化摘要 `time_compare.json` 均保存在该目录。**ElmerSolver 退出码0，耗时1000.86s**；原 SIF 自带 WhitneyAV Solver2 参考范数 `2.14591661e-6` 与其硬编码的 `2.15104858e-6` 相差 `0.23858%`，仍被内置固定阈值标为 FAIL，该范数不是实验测点场误差。

| 同一 t=0.08s / 四周期 | BDF1 dt=0.0025s，32步 | dt=0.00125s，64步 | dt=0.000625s，128步 |
|---|---:|---:|---:|
| 实算时间（s） | 309.22 | 559.36 | **1000.86** |
| 最后一周期同相位 B 相对L²差 | 0.048318% | 0.025177% | **0.022301%** |
| 17点官方 50Hz 实测 Bz 相对L²差 | 3.23361% | 4.31744% | **5.07220%** |
| A1–B1 101点×三分量 B 的L²范数 | 0.08259536 | 0.08263916 | 0.08266364 |

**相邻时间步长的场误差严格在相同时间/源相位比较：**

- `||B_2.5ms-B_1.25ms||₂ = 0.00104546294`，相对较细阶 `1.265094%`；
- `||B_1.25ms-B_0.625ms||₂ = 0.000558908215`，相对较细阶 `0.676123%`；
- **实测收敛阶 `p=log₂(0.00104546294/0.000558908215)=0.90345864`**，吻合 BDF1 预期的一阶精度量级，说明时域细化方向合理，但最细两档场差仍约 **0.676%**，不能宣称达到高精度；
- 实测磁场 Bz 误差三档 **3.234%→4.317%→5.072%**，表明部分粗时间误差与其他来源抵消，**不能**以粗网格实验误差更小证明真实精度更高。

**本轮结论**：第三档已证明 BDF1 的时间收敛阶，**时间轴仍 OPEN**。以 BDF1 再减半的成本迅速增加；因此下一项在同一物理源/网格/持续时间上验证 **BDF2（dt=1.25ms，四周期）** 是否能以合理成本降低时间误差，再决定 h/R 扫描统一采用哪种时间积分条件。BDF2 已在另一个独立目录 `D:\\work\\BFZI-D2-verify\\runs\\bfzi_pdr_team7_ac_bdf2_4period` 启动，**其结果在实际完成前不记 PASS**。

## 5. 判断与下一项最短执行路线

- **50/200Hz 官方 Elmer 3D AC SMOKE PASS**，真实三维 `B,J` 和时变源已核实；当前时间轴的**两个步长+两个持续时间**形成了可信的不收敛证据。**时间高精度 OPEN**：第三档 `dt=0.000625s` 四周期已实际运行，观测 BDF1 阶数 **0.90346**，最细相邻两档 B 差 **0.6761%**；接下来用同一网格、源、四周期的 BDF2 二阶时间积分直接对照，不能跳过物理源和场等价性。
- 空间 h/R 的正式扫描下一步：保留同一真实 TEAM-7 线圈/材料和**已控制的**时间积分条件，分别变更至少三档空间网格、三档有限外域半径或两种有效边界处理；必须完整说明实际 Gmsh/ElmerGrid 标签及线圈体流、单位电流和测点一致性。
- 当前尚无本项目 A–φ 与成熟软件的**同一物理激励/同一相量/同一空间网格**的独立场与 2×2 复 Z 对照；**D2 OPEN、D3/D4 OPEN、G1 FAIL、正式神经教师冻结**。

文档属于 [BFZI-PDR V1.0](../BFZI_PDR_EXECUTION_PLAN.md) 的 P1 实算证据，保留失败误差，不改物理规范、不放宽阈值、不使用 GitHub Actions。
