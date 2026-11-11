> **2026-11-11 D2 200 Hz 功率波形数值机制推进：** 对已完成真实 Elmer 64/128/256 步第四周期原生功率，在同一物理时刻比较 16 个样本：全波形 L² 差 9.65674%→4.51009%，观测 p=1.08194；磁能量全波形 p=2.08945。末端功率单点 p=−1.82243 的非单调不代表全波形无收敛。功率二倍频拟合相邻相位差 0.210874/0.102043 rad，接近一阶后向时间差分半步延迟预期 0.19625/0.098125 rad；半步前移的**假设模型**波形 p≈1.86550，仅作机制诊断，不修改原始值或训练标签。已提交真实 JSON 与正式[200 Hz 时间证据](evidence/d2_team7_200hz_time_20261111.md)，XEnergy 相关 5 tests passed。下一项核验 Elmer 原始边自由度与功率恢复的时间导数，继续空间 H(curl)/h/R；D2 OPEN，旧 G1 FAIL，教师冻结。

> **2026-11-11 D2 200 Hz Δt 三档真实原生积分完成：** 64/128/256 步四周期均完成并保存原生能量/涡流功率。终点磁能量 p=2.04697，256 步对 Richardson 外推相对差 0.002215%；终点涡流功率 p=−1.82243、相邻差从 0.58121% 增至 2.09874%；第四周期平均涡流功率 p=1.50577，256 步外推剩余差 0.52903%。单独保留瞬时与周期平均时间预算，不以其中一个量替代另一个。[完整证据](evidence/d2_team7_200hz_time_20261111.md) 与 [JSON](evidence/d2_team7_200hz_time.json)。独立 3D 原始 Whitney curl B/磁能量提取已实现，尚需和真实 Elmer 边自由度同源交叉。TEAM-7 数值精度 OPEN、D2 OPEN、旧 G1 FAIL、教师冻结。

> **2026-11-11 BFZI-PDR V1.1 D2 四周期原生积分完成：** TEAM-7 官方同源 50/200 Hz 三档 h + 三档 R 共 10 个 Elmer BDF2 四周期 64 步积分求解均退出 0，[双频 h/R 数值结果](evidence/d2_team7_native_integrals_64.md) 与 [原始 JSON](evidence/d2_team7_native_integrals_64.json) 已生成。50/200 Hz 磁能量 h 相邻差分别为 3.86194%→2.90520% 与 4.00746%→2.98629%；涡流功率 R 相邻差分别为 4.93586%→4.36012% 与 3.45848%→2.97690%。第三至第四周期名义相位终点差保留源相位偏移的影响。原生积分和 DG 单元节点重建的差异已独立记录。数值精度 TEAM-7 OPEN、D2 OPEN，旧 G1 FAIL、神经代理教师冻结。

> **2026-11-10 D2 V1.1 原生物理积分验证：** 已对官方 TEAM-7 50/200 Hz 的三档 h、三档 R 完成 10 组真实 Elmer 首时间步 `SaveScalars` 电磁场能量及涡流功率求解（10/10 exit 0）。能量 h 相邻差约 4.0%→1.7%；涡流功率 h/R 仍有数个百分点差异；全四周期末端积分及原始 Whitney 旋度 B 精度仍 OPEN。见 [真实积分证据](evidence/d2_team7_native_integrals.md) 和 [JSON](evidence/d2_team7_native_integrals.json)。D2 OPEN、G1 FAIL、教师冻结。

> **2026-11-10 BFZI-PDR V1.1 D2 3D body field evidence:** actual 50/200 Hz four-cycle Elmer VTUs sampled on identical fixed interior 3D regions, no experiments. 50 Hz air-gap B L2 h 7.9479%/7.1604%, upper-air 8.2709%/5.9261%; 200 Hz plate-bulk J h 19.2427%/17.3987%. RMS-only criteria can hide vector field differences. Same-mesh one-step CSC source representation check confirms identical B and plate J but changed reported coil-body J variable. [Full 3D evidence](evidence/d2_team7_3d_regions.md) ([JSON](evidence/d2_team7_3d_regions.json)). Original elemental curl and true volume-energy convergence still OPEN; D2 OPEN, G1 FAIL, teacher frozen.

# BFZI-PDR V1.1：数值收敛优先的 D2 验收规范

**生效日期：2026-11-10** · **适用分支：** `research/bfzi-uwpt-electrothermal-surrogate` · **关口：D2 OPEN**

本文件是 [BFZI-PDR 执行计划](BFZI_PDR_EXECUTION_PLAN.md) 的现行 D2 验收修订，覆盖旧方案关于 TEAM-7 必须拟合实测 Bz/Jy 曲线才能通过的执行条件。 [物理直接法 V1.0 锁定规范](PHYSICAL_DIRECT_METHOD_LOCKED.md) 的几何、Maxwell–MQS、源、材料、端口及物理目标均维持原样。原始测量与历史比较结果保留为证据，不继续驱动求解器调参。

## 一、D2 判定顺序

1. **数值收敛主关口。** 固定连续问题与物理观测量，分别改变时间步长 Δt、体网格 h、外域 R、离散阶数和场恢复方式。每次改变一轴，保存实际求解场并报告逐轴差异及参考不确定性。最终判定依据数值解趋于同一极限的可靠证据，绝不以某一档与实验值接近作为精度证明。
2. **物理一致性。** 检查源电流守恒、Maxwell–MQS 弱式及边界/界面条件、可定义的功率与耗散收支、可重复性及端口互易。单个离散代数恒等式不能独立证明连续物理场准确。
3. **独立参考交叉检验。** 在完全相同材料、几何、源项、相量、边界、频率和观测定义下，将独立方法（例如自研 A–φ 和 Elmer）之间的解与误差预算进行比较。源不同的两种算法计算值不能用于互相认证。
4. **实验比较作为外部验证。** 在数值验收后再单独报告 TEAM-7 实验 Bz/Jy 拟合差及实验/材料/几何不确定度。实验拟合结果不参与 D2 数值 PASS/FAIL 判定，也不用于选择网格、外域或场恢复算法。

**核心门槛：** 观测量必须先有可追溯的数值极限及可解释的不确定性。当前已观察到约 8%–20% 的相邻空气域 Bz 场差，所以 TEAM-7 场精度和 D2 均 **OPEN**。原 G1 **FAIL**，神经代理教师保持冻结。

## 二、数值收敛的独立观测量

以相同物理位置、频率、终点相位的两个场 (u_i,u_j) 为例：

\[
d_{ij}(u)=\frac{\|u_i-u_j\|_2}{\|u_j\|_2},\qquad
m_i(B_z)=\frac1L\int_\Gamma B_{z,i}\,ds,\qquad
r_i(B_z)=\sqrt{\frac1L\int_\Gamma B_{z,i}^2\,ds}.
\]

同时报告 `|m_i-m_j|/r_j`、`|r_i-r_j|/r_j`，防止有符号线积分的抵消掩盖局部不收敛。端点/近表面/材料界面的 B/J 量须明确所属区域与场重建定义，不以一条线积分替代完整空间场收敛。

每次独立扫描要包含：
- **h：** 至少三档真正三维体网格；固定 R/连续方程/源/观测方式。关注全域可比体场的 L²、能量相关量以及多个固定探针/局部平均量。
- **R：** 至少三档外域或两类独立有效开域处理，固定固体源/导电体几何与离散，记录空气域网格变化造成的剩余误差；改变外盒距离不等于单独测得外边界误差。
- **Δt：** 在同一网格/外域用至少两档时间步长估计参考时间误差；200 Hz 另行核实，不把原 50 Hz 时间预算自动转移。
- **场恢复与方法：** 在同一离散解上单独检查边元旋度 B、原生节点重建、采样误差；同一瞬态线输出来源与索引必须匹配，平滑后的表观拟合不作为真值。
- **物理：** 完整体场、Maxwell–MQS、端口/功率及场路关系；在 BFZI D3/D4 上延续完整复数 Z 和 q 验收。

比较两相邻网格的相对差仅代表**尚未消失的离散差**；要声明 PASS，必须获得足够的网格层次与独立的误差估计，使剩余不确定性小于案例卡预先冻结的精度预算，并且物理一致性及同源参考交叉验证通过。不得用当前失败数据反向调整阈值以获取 PASS。

## 三、现有可复核证据（不使用实验拟合）

正式入口：

```powershell
python -m pytest tests/test_bfzi_d2_team7_conv.py -q
python -m examples.bfzi_d2_team7_conv --root D:\work\BFZI-D2-verify\runs\team7_hr --json docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_numerical_conv.json --md docs/research/bfzi_uwpt_electrothermal/evidence/d2_team7_numerical_conv.md
```

- [数值收敛结果与观测量表](evidence/d2_team7_numerical_conv.md) 和 [完整 JSON](evidence/d2_team7_numerical_conv.json)：两种频率的真实四周期 BDF2 最后一步 101 点固定测线，对 h/R/空气网格/局部测线分段单轴比较，**不读取实验数据**。
- [原始网格/求解证据](evidence/d2_team7_hr_mesh_and_runs_20261110.md)：保留此前 h/R 和实测拟合研究的历史记录；原始高体积 MSH、Elmer mesh 与场文件存放 XEnergy `D:\work\BFZI-D2-verify\runs\team7_hr`。
- **关键观察：** 50 Hz h15→h10 的 A1 Bz 点 L² 差约 11.64%，而有符号线平均变化仅占参考 RMS 的约 0.30%；h10→h075 相应约为 10.89% 与 3.52%。两者并列保留，证明单一低误差积分量不构成空间场收敛证据。

## 四、下一轮唯一数值任务

**从真实 H(curl) 三维体场出发，优先改善场恢复与空气/线圈界面的体网格离散。** 在相同物理源、相同外域及材料下，获取可比较的空间体场误差、磁能量或区域均方观测量，逐一检验边元阶数/旋度 B 恢复方法是否提高收敛；再回头检验三档 R 与更严格的 200 Hz 时间预算。避免继续重复纯测线分段加密或为了实验拟合而调参。完成 D2 数值与物理关口后，按既有计划进入 BFZI 两端口同源比较。

**状态：TEAM-7 完整数值精度 OPEN，D2 OPEN，旧 G1 FAIL，神经代理教师冻结。**

