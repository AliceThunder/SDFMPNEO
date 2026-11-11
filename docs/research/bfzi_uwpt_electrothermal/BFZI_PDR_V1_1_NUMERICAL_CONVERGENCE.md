> **2026-11-11 D2 六档空气间隙 h + 三档外域 R 四周期实算完成：** 官方 TEAM-7 同源 200 Hz、BDF2 64 步，新增 gap05（217759 单元）、gap04（310784 单元）、rgap15（144082 单元）、rgap30（267145 单元）四组真实 ElmerSolver 均 64/64、exit 0；所有新增网格的线圈 3600、铝板 27570 四面体与实体基准完全一致。三维空气间隙 B 的 h 相邻 L² 差依次 **7.2861→5.7092→3.9063→2.3601→1.9155%**，A1 Bz **11.2145→9.2239→7.7902→4.0690→3.4593%**，A2 Bz **13.8108→11.3662→10.2002→5.0628→3.0832%**。固定 6 mm 间隙的 R 0.15→0.20→0.30 m 三档，B 间隙相邻差 **2.8926→2.6271%**，上方空气 B **7.3291→6.2915%**；原生磁能量 R 相邻差 0.7011→0.6222%。磁能量 h 最后一档相邻差回升，J/涡流功率空间变化未形成稳定极限。真实[六档 h 与三档 R 证据](evidence/d2_team7_airgap_h.md)、[h JSON](evidence/d2_team7_gap_200hz.json)、[R JSON](evidence/d2_team7_rgap_200hz.json)、[3D JSON](evidence/d2_team7_gap_volume.json) 均已提交，主线下一项集中外域剩余误差与原始 Whitney curl-B 场验收；TEAM-7 数值精度 OPEN、D2 OPEN、G1 FAIL、教师冻结。

> **2026-11-11 D2 真实 200 Hz 三维空气间隙 h 收敛四档完成：** TEAM-7 官方同源 64 步四周期 BDF2 新增 `gap12/gap08/gap06`（132909/145876/177022 四面体），三档实际 ElmerSolver 全部 exit 0，原 h10 加新三档共四级。所有网格线圈 3600/3600、铝板 27570/27570 四面体精确相同，只加密空气间隙，R=0.20m。gap06 于 03:34:18 结束，原生磁能量 0.526432759591 J、功率 14.8052596085 W。h10→gap12→gap08→gap06 的相邻磁能量差 0.2561%→0.1696%→0.0777%，三维空气间隙 B L² 差 7.2861%→5.7092%→3.9063%，A1 Bz 测线差 11.2145%→9.2239%→7.7902%；功率差 1.5451%→0.8182%→0.4997%。上方空气 B 与铝板 J 未完全单调，D2 OPEN。真实 [证据](evidence/d2_team7_airgap_h.md)、[原生积分及测线 JSON](evidence/d2_team7_gap_200hz.json)、[三维区域 JSON](evidence/d2_team7_gap_volume.json)，复现代码 `examples/bfzi_d2_team7_gap.py`。下一主线继续三维空气 h 与外域 R 收敛。旧 G1 FAIL，教师冻结。

> **2026-11-11 D2 二阶 H(curl) Elmer 真正首步试算：** 同一 TEAM-7 真实 h10 129305 四面体、200Hz 原三分量源与 R=0.2m，仅将 WhitneyAVSolver/CalcFields 改为 `Quadratic Approximation=True`、Piola，实际启动两组一步 ElmerSolver：原预条件 `none` 与 `ILU0` 的 BiCGStabL 残差均显著增长，两项被终止，**无可用二阶原生 B/J/能量**。独立[实际日志与 SIF 变更证据](evidence/d2_team7_hcurl2_smoke.md) 已入库。继续保持已完成的一阶三维场结果，优先空气/线圈 H(curl) 场精度与 R 轴；二阶方案需解决规范/线性求解器后再进入四周期比较。原始 Whitney BDF2 电场/Joule 核及真实同终点 h/R/Δt 预算已完成；XEnergy 相关 6 passed。TEAM-7 数值精度 OPEN、D2 OPEN、旧 G1 FAIL、教师冻结。

> **2026-11-11 D2 同终点数值误差预算与 BDF2 原始 Whitney 电场：** 已新增 `sdfmpneo_bfzi/direct3d_whitney.py::bdf2_joule()`，在原始有向边自由度和导电域 P1 势上求 BDF2 瞬态 E/q/Joule，制造解检查二阶时间精度与时变规范不变性。实际 Elmer 200Hz TEAM-7 h10 时间序列增加终点字段，以相同 t=0.02s 比较三档 h/R 与 64→128 Δt：A1 Bz 时间差 0.2190%、h 差 12.7059–13.5350%、R 差 12.2332–14.2841%；A2 Bz 时间差 0.2588%、h 差 19.8957–23.0022%、R 差 11.3899–12.0894%。J 的空间与时间差同量级，应单独保留时间预算。见[真实同终点证据](evidence/d2_team7_200hz_budget.md)、[JSON](evidence/d2_team7_200hz_budget.json)和 `examples/bfzi_d2_team7_budget.py`；XEnergy 相关 6 passed。已核对编译日期前 Elmer 原始 CalcFields 源码的瞬时 E 一阶差分路径，实际二进制运行分支和 Elmer 原始边/势历史仍待获取。下一项重点是三维空气/界面 H(curl) h/R 原始场精度，教师冻结、TEAM-7 与 D2 OPEN、G1 FAIL。

> **2026-11-11 D2 200 Hz 原生 B/J 场时间误差已实测：** 固定 TEAM-7 h10、同一网格/外域/源，64/128/256 步完整 Elmer SaveLine 第四周期同 16 个物理时刻（每条 101 点）比较。A1/A2 Bz 观测 p=1.95635/1.95357，A1 Bxyz p=1.95465；上/下铝板 Jy p=0.83475/0.93855，完整 Jxyz p=0.95752/1.10361。说明当前 200 Hz B 时间误差接近二阶，而 J/功率后处理约一阶。[真实空间场时间证据](evidence/d2_team7_200hz_field_time.md)、[JSON](evidence/d2_team7_200hz_field_time.json)、[新代码](../../../examples/bfzi_d2_team7_field_time.py)。Elmer 官方源码 CalcFields.F90 瞬态电场路径使用相邻时间差分，和数值相位诊断吻合；节点化 A 直接进行后验 BDF2 电流校正反而恶化，不采用。自研实际 TEAM-7 3D H(curl) 解的原始 Whitney curl B 与内部独立单元 B 一致。XEnergy 专项 8 passed。仍待基于 Elmer 原始边自由度/标量势的同源积分点修复，再做 h/R；TEAM-7 数值精度 OPEN、D2 OPEN、旧 G1 FAIL、教师冻结。

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

