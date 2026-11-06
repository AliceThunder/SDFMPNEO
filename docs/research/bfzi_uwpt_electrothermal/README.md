# BFZI 水下 WPT 电磁热代理：研究目录

> **2026-11-06 D2 最新入口（优先于下方旧路线与论文拟合描述）：** 已实际提交独立三维 A–φ 边元求解器和 TEAM-7 官方几何初始受迫涡流案例，而不是继续复现 Shafiei/Wang 论文表格。请从 [三维核心实现与物理局限](D2_APHI_CORE_IMPLEMENTATION.md)、[官方 TEAM-7 几何及粗网格结果](D2_TEAM7_OFFICIAL_CASE.md)、[两独立盒 Gmsh 三维入口](../../../examples/bfzi_d2_aphi.py) 进入。当前只有基础测试和极粗 TEAM-7 运算通过，真实 Gmsh 官方外域、网格收敛及测量数据对照仍缺失：**D2 未通过、G1 FAIL、禁止教师训练**。理论方法衔接见 [D1→D2](METHOD_TRANSFER_D1_TO_D2.md)。
>


## 当前首要任务：物理直接法 V1.0 锁定（2026-11-06）

**从此优先阅读：** [物理直接法锁定规范](PHYSICAL_DIRECT_METHOD_LOCKED.md)、[论文证据与 D0–D6 验收表](PHYSICAL_DIRECT_METHOD_EVIDENCE_AND_GATES.md)、[TeX 第15节](../../../theory/sections/15_physical_direct_reference_lock.tex)。

用户提供的 2026-11-05 UWPT 综述把普通物理模型（B）、前向代理（A）与在线辨识（C）严格区分。当前只推进 B 类：先复现圆形共轴分层海水成熟半解析解，再建立经独立 benchmark、网格与外域验证的三维 A–φ 参考，并在真实 BFZI 双独立 SE(3) 盒上验证复端口阻抗、空间涡流、海水焦耳热及相位/功率；BEM 与 Green–Galerkin 为待独立评价的快速候选。

**历史五模态 P1-C/G1 求积修补全部挂起，尚未物理认证。** 旧 P1-C 日志及大段试错记录保留供追溯，但不再代表当前下一步。禁止利用“112 passed”或同族两积分规则接近来宣称物理高精度；正式神经教师仍关闭。本轮只更新理论文档，不改求解代码，不使用 GitHub Actions。

---



**当前进度（2026-11-04）请查看 [📊 BFZI 总体研发路线图、封装专项进度和验收看板](PROJECT_PROGRESS.md)。** 物理原型、207 模态参考空间、数据/ROM、神经实验和基础电热时间推进已经进入代码实现阶段；**高精度空间参考解尚未通过收敛验收**。下文按创建日期保留早期理论与代码推进历史，当时“尚未修改代码／尚未运行测试”等表述仅适用于相应历史时间点，不能代替最新看板。

## 当前首要任务：G1 理论体系与数值方法复核（2026-11-04）

**先读 [G1 理论与数值实现系统复查](G1_THEORY_NUMERICS_REVIEW.md)**：逐项区分已经从 Maxwell--MQS 推导成立的弱式/互易/被动定理，以及 **尚未认证** 的无限海水外边界、curl 字典完备性、真实 Coulomb L、体积 R/K、铜海水全耦合及训练教师。该文件附有对照真实数值分析文献的出处和按 P0–P5 排序的后续验收关口。

正式多文件理论请读 [G1 文献、奇异积分与重新验收 TeX](../../../theory/sections/12_g1_numerical_foundations_literature.tex)，已由 [theory/main.tex](../../../theory/main.tex) 纳入主稿并在 [references.bib](../../../theory/references.bib) 增补 9 篇已核对 DOI 的文献。2022 年 Passarotto--Pitassi--Specogna 的涡流体积分基础论文对电感双积分的奇异自项以及内外积分分离造成的不稳定有直接分析，优先于继续添加采样规则。**这一轮是理论/文献复核，未宣称已经新完成代码求解或物理收敛；G1 仍 FAIL。** 以现有 70 passed 本机实测和未运行的后续 QMC 实验为真实进度边界。

## P0 历史过程：锁定物理与闭合 BFZI 源（2026-11-04）

第一步已形成 [P0 物理问题／源项验收契约](P0_PHYSICS_SOURCE_CONTRACT.md)，正式 LaTeX 见 [第 13 节](../../../theory/sections/13_p0_physical_source_contract.tex)。它冻结圆形—圆角方形连续线圈、两独立绝缘封装、完整定向回流、均匀线性非磁性海水、MQS 峰值相量与人工外球假设；**当前只批准 source-only 复核，不批准新的海水 R/K/L 精度声明或正式神经教师。**

P0 最新本机 **79 passed**，生产 `LineSource` 4 阶解析直段＋圆角 Gauss 在两端口各 **84 个**盒面／棱／角海水探针上，C/B 相对误差为 **0.00728%/0.0427%**、边角 B 相对误差 **0.0748%**，均低于先定 0.1% 门槛；闭合源 `divC_scaled` 则为 1.06e-4，12 阶收敛至 8.86e-11。为从数学结构上解决“有限 Gauss 点线元分布不严格闭合”的问题，本轮新增**按真实 BFZI 曲线逐点连弦的闭合折线**：所有线段均用解析场核，`div C=0` 严格望远镜相消，但仍需弦段几何收敛。原 BFZI `geometry.py` 补上角部切向不反转的充分条件；这些新增模块待本机验证。完整线元节点仍保留给空气互感、导体长度与铜热。运行：

```powershell
python run.py --mode bfzi-p0-source --with-tests
```

**P0 日志**自动提交至 [`runs/p0/`](runs/p0/)，不覆盖此前 [G1 日志](runs/latest_summary.md)；79 passed 的面／棱／角及 div C 收敛已经实测，新增独立闭合折线与非退化几何测试待本机验证。完成 P0 的明确适用范围验收后，才进入 P1 真实 Coulomb L 算子。

## 已按五阶段路线进入 P1：原 Coulomb 1/r 磁能真值（2026-11-04）

**P0 限定演示基准已在本机 83 passed 实测通过：** 双端口各 84 点，4 阶源 C/B 误差为 7.28e-5 / 4.27e-4；独立严格闭合折线源的 16/32/64 面板 B 误差为 3.90e-4 / 9.76e-5 / 2.44e-5，`div C` 接近机器零，验证了独立几何收敛与闭合恒等式。有限线径自碰撞与全参数域不属于该限定结论。

**P1 第一个可证伪关口已写成程序与 [完整推导／验收合同](P1_COULOMB_ANALYTIC_GATE.md)**：真实均匀球中的三个严格无散度零法向电流 `W_i=e_i×x`，其**原始 1/r Coulomb** 三模态磁能矩阵有解析真值 `16π μ0 R^7/315 I3`。直接检验现有原核 `assemble_coulomb_split` 的 x/y 分别细化、局部截断分区，不运行 BFZI 229 模态，也不换软化核。

```powershell
python run.py --mode bfzi-p1-coulomb --with-tests
```

**P1 的旧分区算法在解析球上 FAIL，新一阶 Taylor 算法已在本机 87 passed 的非线性解析球上 PASS（相对磁能误差 0.01596%，先定目标 0.1%）。** 以下保留旧算法失败的数据与最新求积修正的独立比较： 85 passed 只证明程序运行；原 `assemble_coulomb_split` 的 `base/x_only/y_only/both` 磁能矩阵相对解析真值误差为 **2.121%/1.556%/1.146%/0.6145%**，同时加密仍未达到原定 **0.1%**；改变局部分区半径后 L11 差 **0.4230%**。已实现精确局部球矩 `8πa²/7 W(x)` 与误差分解，进而实现保留真实 `1/r` 的全域常值＋**一阶 Taylor 奇异减除**：需要独立可信的几何标量势 `SΩ` 和几何一阶矩 `MΩ=∮|y-x| n_y dS`。为确保不只是把线性涡流的余项消成零，新增**非线性且无散度**的三模式 `W=r²(e×x)` 和解析矩阵真值 `16πμ0R¹¹/693 I₃`。独立短程球积分同样 480×756 节点下，该非线性磁能误差由仅减常值约 **0.245%** 降为一阶减除约 **0.016%**，但不是用户本机 GitHub 新日志。**真实两盒 `SΩ/MΩ` 尚未认证；P1 新代码待本机复验，P1-A/G1 仍 FAIL。** 命令及自动日志路径不变，正式 TeX 见 [第 14 节](../../../theory/sections/14_p1_exact_coulomb_sphere.tex)。禁止 GitHub Actions。

## P1 最新第二关：双封装真实海水几何矩（2026-11-04）

球形 Taylor 新算法已在本机 `0863a41` 的 **87 passed** 中实测：线性环流误差 3.58e-16、非线性无散度环流原核磁能误差 **1.596e-4**，已通过 0.1% 的该项解析球关口；旧 split 和仅减常值依然不合格。

新代码 `sdfmpneo_bfzi/coulomb_geometry_moments.py` 严格计算**外球减去两只各自独立任意刚体旋转的长方体绝缘封装**的 Coulomb 标量几何势 `SΩ` 与一阶矩 `MΩ`；所有盒面均用 `∬face(1/r)` 和 `∬face r` 的**精确解析四角原函数**，不使用任何软化、uniform-ball 电流云、球几何势代替双盒或固定世界网格。新 `bfzi_p1_geometry.py` 同步提供实际近面、棱、角 2.5mm 海水探针的独立分片面 Gauss 验证，**提前固定 1e-8 的几何矩相对误差目标**，并增加独立三维盒 Gauss 与 SE(3) 协变测试。

以下仍用同一命令，把 `p1_geom_box/p1_geom_wet/p1_geom_gate` 并入自动更新的 [P1 日志](runs/p1/latest_summary.md)：

```powershell
python run.py --mode bfzi-p1-coulomb --with-tests
```

**新双盒几何关尚待本机复验；真实双盒原核 L 的电流余项、自/邻积分及无限海水外域仍未认证，G1=FAIL。** 以 [P1 验收理论](P1_COULOMB_ANALYTIC_GATE.md) 和 [项目进度看板](PROJECT_PROGRESS.md) 为准。

## P1 真实双盒合法电流的自／互项解析基准（2026-11-05 UTC）

最新用户本机 **90 passed**：[P1 原始日志](runs/p1/latest_full.log)，代码 `585bf12`。外球减去两只任意 SE(3) 旋转盒的**真实海水域**，盒面、棱、角附近的 (S_\Omega)、(M_\Omega) 的盒体贡献已通过独立分片曲面积分验收：最大相对差分别 **5.51e-16** 与 **1.63e-15**，远小于预先确定的 1e-8。非线性解析球涡流的一阶 Taylor 磁能误差仍为 **1.596e-4**，通过其 0.1% 目标；旧 split 算法仍不合格。

本轮新增 `examples/bfzi_p1_wet_vortices.py`：在**两只真实 BFZI 绝缘盒旁的海水中**分别构造一根不与盒壁接触的光滑紧支涡流 (W_p=[1-r_p^2/a_p^2]_+^2(e_p\times(x-c_p)))，它在真实双盒海水域严格无散度、零法向。两个原 Coulomb (1/r) 自项以及**非零**互项都可独立精确推导，从而直接测试同一物理域的 (L_{11},L_{22},L_{12})；不是 uniform-ball 导线电流替代模型。新增 `p1_wet_L`／`p1_wet_gate`，预定整体、自项、互项误差各小于 0.1%，针对性测试已加入自动本机运行。

仍使用：

```powershell
python run.py --mode bfzi-p1-coulomb --with-tests
```

**新双电流基准待本机运行；即使通过，也只认证实海水域中的两维紧支测试子空间，不证明旧 229 模态、贴壁电流、外球趋于无穷海水或完整 G1。** 数学公式及下一关口见 [P1 契约](P1_COULOMB_ANALYTIC_GATE.md) 与 [正式 TeX 第 14 节](../../../theory/sections/14_p1_exact_coulomb_sphere.tex)。不使用 GitHub Actions。

## P1-C：接通真实全海水 curl 基的原核 Taylor 磁算子（2026-11-05 UTC）

用户本机最新 [P1 原始日志](runs/p1/latest_full.log)：**92 passed**。实际双盒海水中，两只严格合法、与盒壁保持间隙的光滑紧支涡流，其解析 (L_{11}/L_{22}) 自项最大相对误差 **0.02382213%**、非零 (L_{12}) 互项相对误差约 **1.37751e-8%**，整体矩阵误差 **0.02382024%**，均达到预定 0.1% 门槛。旧 split 的原核球误差仍有 0.6145%，**不能混淆新旧算法通过范围**。

本轮已实现 `sea_basis_differential.evaluate_nonguided_curl_jacobian`，对原有 BFZI 海水 `SeaCurrentBasis` 的真实**外球包络、两只独立 SE(3) 旋转绝缘盒距离包络、原 Gaussian/Wendland 窗口**给出物理 Hessian，从而解析计算非源引导 `curl` 电流的 `dW/dx`。新 `green_coulomb.assemble_basis_coulomb_taylor` 使用真实两盒外海水域的精确 `SΩ/MΩ`、未软化 (1/r) 的一阶 Taylor 残差，不另造等效体积球云。

新 `bfzi_p1_full_sea_taylor.py` 固定旧 27 维基中的 5 个真实模式，输出 `p1_full_sea` 与 `p1_full_sea_change`：同一电流、同一物理域、分别加密 x/y、矩阵原偏斜和特征值。**这是尚待本机运行的算子数值诊断，尚无独立五维连续真值；不会仅凭两阶差值宣称全域 G1 通过。** 源引导模式需另外验证端口场的空间 Jacobian，暂显式拒绝。

使用原命令：

```powershell
python run.py --mode bfzi-p1-coulomb --with-tests
```

P0 闭合 BFZI 源、R/K、热代理和神经模型均未修改，不使用 GitHub Actions。

## P1-C 首轮运行错误修复与单独日志入口（2026-11-05 UTC）

最新用户本机代码 `983ad60` 的 [原 P1 日志](runs/p1/latest_full.log)显示 **94 passed，但 P1-C 主实验在导入时 SyntaxError，运行状态 FAIL**。不是磁能 L 的误差实测。已经修复 `bfzi_p1_full_sea_taylor.py` 两个 `_relative()` f-string 漏闭合括号，pytest 现直接导入该主实验模块。另发现原 `x_only` 规则 `x1=(4,6,10)` 与 `y0` 完全相同，会触发原核 Taylor 程序拒绝共址积分节点；现在采用互不重复的 `x0=(3,5,8)`、`y0=(4,6,10)`、`x1=(5,7,12)`、`y1=(6,8,14)`，不修改物理电流或真核。

**新的独立 P1-C 运行入口**：

```powershell
git pull --ff-only
python run.py --mode bfzi-p1-full-sea --with-tests
```

将自动把最新摘要和原始日志推送到 [`runs/p1_full/`](runs/p1_full/)，不覆盖现有 [P1-A/B 解析参考日志](runs/p1/latest_summary.md)。**这次代码修正还未取得用户本机完整 P1-C 实测；仍须分析实际五维 curl L 的 x/y 规则差、正定性与盒壁数值问题，不能仅因 pytest PASS 就认定 G1。** 禁止 GitHub Actions。

## 阅读顺序

1. [P1 原 Coulomb L 真值／P0 限定交接](P1_COULOMB_ANALYTIC_GATE.md)：当前阶段，三个有解析答案的无散度球形电流。
2. [P0 物理与源项锁定契约](P0_PHYSICS_SOURCE_CONTRACT.md)：已完成限定物理基准，保留全参数域约束与历史数值过程。
3. [G1 理论与数值实现系统复查](G1_THEORY_NUMERICS_REVIEW.md)：当前优先级，严禁把离散自洽当作连续真值。
4. [项目当前进度总图与看板](PROJECT_PROGRESS.md)：物理模块、**封装**、数值验收、神经代理、热演化、下一关口和本地测试证据；**当前状态以此为准**。
5. [新理论主文档](../../../theory/main.tex)：多文件 LaTeX 主入口。
6. [文献分析](literature_review.md)：已讨论的电动机、IPT/WPT、UWPT 和谐振变换器文献及限制。
7. [方案分析](architecture_proposal.md)：BFZI 几何、电磁反作用、损耗、热动力、谐振电路的耦合选择。
8. [验证路线](validation_roadmap.md)：研究阶段、基准、开放问题。

## 几何权威来源

[0UnifiedModelingUWPT](https://github.com/AliceThunder/0UnifiedModelingUWPT) 的 sections/02_geometry_model.tex、03_package_seawater_impedance.tex 和 04_geometry_induced_fast_computation.tex；最新版 BFZI 是支撑函数构造的圆形—圆角方形**连续导体路径**，不是旧双阶数超椭圆。跨匝仿射结构是理论关键；封装与海水边界保持独立的物理区域定义。

本理论强调同一电磁场产生端口复阻抗和材料分区损耗，以守恒热动力和谐振电路组成双向反馈。允许 FEM 教师、局部求积；目标是不依赖固定世界网格的在线推理。每项创新与性能都尚待实证，不得照搬其他论文的误差或加速倍数。

**禁止 GitHub Actions。** 今后优先完善核心物理与模型实现，不新增不必要的审计脚本。

## 新一轮算法推导（2026-10-31）

- [文献启发 → 数学算法](derivation_to_algorithm.md)：解释从 Xie、Zhu、Yuan 等工作到 BFZI 保结构海水涡流降阶、神经子空间和多导电材料 Schur 补的推导依据，以及已经证明和仍未解决的部分。
- [海水低维电流与功率证明](../../../theory/sections/03a_solenoidal_eddy_galerkin.tex)：在明确的线性 MQS / 绝缘盒体 / 合规电流空间假设下，证明端口互易、被动以及同源海水损耗积分。
- [可训练的保结构几何子空间](../../../theory/sections/04a_geometry_conditioned_operator.tex)：学习子空间选择，保留物理 Gram 矩阵与分布热源一致性。
- [铜、盒壁与海水联合 Schur 补](../../../theory/sections/05a_coupled_conductive_schur.tex)：避免独立阻抗补丁漏掉交叉耦合或重复计损。

**特别注意：** 该推导并没有解决任意几何下连续散度自由电流基的高效构造、任意磁芯或高速运动；也没有获得实验精度与加速性能。最新理论仅属于数学条件明确的研究方案，现有运行代码未被修改。

## 第三轮推进：合法电流基、保正定快速装配与误差界（2026-10-31）

- [专题 Markdown：从数学到可实施算法](geometry_admissible_space_and_fast_assembly.md)：curl/Piola 物理电流图谱、BFZI 模式耦合、正权磁能 Gram、计算成本和回退。
- [03b：六自由度双封装的合规电流图谱](../../../theory/sections/03b_geometry_admissible_basis.tex)：近盒、间隙、远场及拓扑模式，并给出保持散度和绝缘界面条件的构造。
- [04b：正权核矩阵与 BFZI 在线装配](../../../theory/sections/04b_positive_kernel_assembly.tex)：R 和 L 的 PSD Gram、双盒相对姿态的谱特征、K 的互易积分以及成本。
- [04c：能量范数残差界及在线流程](../../../theory/sections/04c_residual_online_algorithm.tex)：同一高精度有限字典中的电流误差证明及快速推理/参考回退分工。

本轮补充了参数化 MQS ROM 与保被动降阶的先行文献，以明确已有数学工具与本研究创新之间的界限。**没有修改代码，也未进行实验或编译。**

## 理论封版并进入代码（2026-10-31）

- [首版理论封版和实现契约](../../../theory/sections/10_frozen_mvp_contract.tex)：冻结 BFZI-MQS-v1 的线圈/封装/海水物理域、端口与损耗定义、REFERENCE/FAST 以及最小验收。
- [代码启动清单](CODE_ENTRY.md)：具体 Python 子包、文件顺序、接口、直接求积到保结构 ROM 的实现路线。不再等待所有高级介质理论完成后才写代码。

首版优先海水场—复阻抗—空间耗散的真实闭合。当前代码仍是旧 vNext 实现，新 BFZI 子包按该契约逐步接入。
