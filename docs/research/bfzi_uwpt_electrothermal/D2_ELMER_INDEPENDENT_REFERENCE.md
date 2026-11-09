# D2 独立成熟参考：Elmer 官方 TEAM-7（2026-11-06）

> **2026-11-09 真实成熟 Elmer 参考首次场求解已运行：** [TEAM-7 实算证据](evidence/d2_team7_elmer26_actual_run_20261109.md) 提供 CSC 官方 mesh/SIF 的完整来源提交、50Hz 原始 16 步及单改时长的 32 步日志、非零 `B/J` 场范围、A1–B1 周期差与未通过的官方参考范数检查。此事实**覆盖本文历史“尚未真正运行 ElmerSolver”**；但仅为软件 H(curl) 涡流链路 **SMOKE PASS**，TEAM7 公开场测量/相位与 h/R/时间收敛**仍 OPEN**，更没有 BFZI 两端口物理同源比较。优先核对实验采样相位、延长周期、频率/时间离散，完成 D2 标准参考，再构建真正同源 BFZI 场。
>

> **2026-11-09 现行优先级：** 按 [BFZI-PDR V1.0](BFZI_PDR_EXECUTION_PLAN.md) 执行。XEnergy 已有经真实启动验证的 `ElmerGrid` 与 `ElmerSolver` **26.2**（安装于 `D:\\work\\ElmerFEM\\ElmerFEM-nogui-nompi-Windows-AMD64\\bin`）。**此刻不要继续只完善“最终边元 RHS 导出接口”**：先以真实官方 Elmer 三维涡流算例取得非零 B/J 和求解日志，继而使用真实 TEAM-7 非轴对称几何/线圈/50 与 200Hz/公开探针值建立独立参考；标准基准通过后才将 BFZI 独立斜置 OCC/材料网格及真实两端口相同闭合源接入，最后对照完整复 Z、空间 E/B/J/q 与 h/R/源收敛。历史“没有安装 Elmer”描述保留作为原日期记录，**已被本条更新**。Elmer 启动不等于 Maxwell 解、TEAM-7 或 D2 PASS。
>

## 2026-11-09：Elmer WhitneyAV 最终边元载荷同一性桥接

较单纯的 CoilSolver 单元 `J_{s,p}` 读取，本次增加对*真正进入磁场解算的两个独立单位端口源泛函* `L_{p,e}` 的逐边核对。使用 [compare_final_elmer_whitney_source](../../../sdfmpneo_bfzi/direct3d_elmer_interchange.py) 和 CLI `--elmer-final-edge-load FILE`：读取 Elmer 本机独立导出的 **四列** `ElmerNodeFrom ElmerNodeTo RHS_Port1_A RHS_Port2_A`，每条全局边恰好一条记录（零载荷边也必须提供）；通过两个端点的**真实世界坐标**映射到原始 Gmsh mesh 节点，再按全局有向边翻转符号，对两个单位端口分别与本项目精确体积分载荷 `L^{volume}` 做 L∞ 绝对/相对比较。实现同时拒绝缺边、重复边、未知节点和非有限值；结果作为独立字段写入 JSON，失配拒绝同源求解。项目新增两组反向边与错误 RHS 测试、基于真实 BFZI 三材料 MSH2 夹具的完整交接主程序合成文本集成测试。**不是**声称 Elmer 程序已运行。

注意：Elmer 官方 `mgdyn_steady_coils` 示例分别配置 `CoilSolver`、`Use Elemental CoilCurrent`、`Fix Input Current Density`；ElmerModelsManual 将 CoilSolver 定义为依据**真实闭合体线圈的传导电流**。官方算例**不会**直接生成这里约定的逐边最终 RHS 四列表，必须从经过源修正的独立 `WhitneyAVSolver` 最终源组装环节额外实现导出，而且要区分原始源载荷与边界提升、规范及矩阵项。即使最终有限元 `L` 一致，只能证明**同一离散弱源**；真正的零半径细丝对有限截面体源的物理极限、实际铜损耗及终端口阻抗归一化、人工外域与 h/R/源加密仍需单独通过。此时**仍无真正生成的铜封装闭合体域与 Elmer 两源数值**。D2 OPEN、G1 FAIL、教师冻结，禁止 GitHub Actions。



> **2026-11-09 体源空间热二次型：** 两实单位体源及导出的复四点电场 `E_p(T,Q)` 允许构造 `Q_{pr}(T,Q)=sigma(T)/2 E_p(T,Q)^*·E_r(T,Q)`；新的 NPZ 中以 `q_port_cross_W_m3[T,Q,2,2]` 保存，包括 `p!=r` 的干涉项，任意 `I` 以 `q=I^H Q I` 恢复。仅有每个单位端口 `q_basis` **不能**恢复任意相位热场。该空间二次型对本项目给定有限盒/同源体电流近似精确，但独立 Elmer、铜源和无限外域尚未验收。D2 OPEN。
>

## 2026-11-09 新的可运行交接路径与源不等价硬门槛

已新增独立于原细丝 CLI 的可执行 [Gmsh/ElmerGrid → BFZI 体源 A–φ 交接程序](../../../examples/bfzi_d2_elmer_handoff.py)：输入已经真实生成的三材料 OCC MSH2、严格哈希源合同、ElmerGrid 实际转换目录，先核对全部三材料物理体域、外边界、四面体 ID 和独立 SE(3) 几何一致性（可由 `--run-elmergrid` 在工作站直接调用 ElmerGrid）；只有显式提供两个独立真实单位体电流的七列外部文件时，才在**本项目自身**的 A–φ 中做相同加载、真空差分、完整复 2×2 Z、I10/I01/I11/I1minus1/I1plusi 功率，并可按实际四面体导出 E/B/J/q 的复数 NPZ。其本质是源/场交接的生产主程序，**并不调用 ElmerSolver**，不能把其结果当独立 Elmer 高精度参考。

上游 [Elmer 官方两个闭合线圈示例](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/tests/mgdyn_steady_coils/case.sif) 用 `CoilSolver`、`Use Elemental CoilCurrent=True` 和 `Fix Input Current density=True`；模型手册也说明 CoilSolver 是在三维物理导体域上求一个体电流场，**不是**当前零截面闭合细丝泛函。重要：`Fix Input Current Density` 可以改变输入源场使其兼容磁矢势规范；本项目不允许直接比较未修正的 CoilSolver 源与独立 Elmer 最终实际用的源，也不允许不经证明直接投影或转化源并声称“同源”。当前仅三个物理域 sea/tx dry/rx dry，尚无真实铜闭合体域和两套实际 Elmer 电流，因而独立参考最大障碍仍未解除。

进入高精度验收之前必须完成：①真实两个独立 SE(3) OCC + `ElmerGrid` 本地运行及测试；②构建真正封闭铜源体或 Elmer 中与细丝完全一致的自定义 RHS（严格对应同一端口回流/相量/源分辨率）；③导出或独立核对 Elmer **最终**边元源的同一性，两个各 1 A 单位端口、离散连续性和铜/海水损耗定义；④核对独立两求解器的空间 E/B/J/q、完整复 2×2 Z、端口功率恒等式，再各自进行 h/R/线源/有限铜横截面收敛。不能以同源体源的**本项目自身**运行替代独立 ElmerSolve 的证据。**D2 OPEN；G1 FAIL；教师冻结；禁止 GitHub Actions。**



## 2026-11-09：真实 ElmerGrid 单元/材料/外边界映射及外加源连续性硬门槛

[新增主代码](../../../sdfmpneo_bfzi/direct3d_elmer_interchange.py) 按 Elmer 官方文本格式读取 `mesh.nodes`（原生节点 ID/坐标）、`mesh.elements`（体材料编号、四节点 tetra=504）、`mesh.names`（ElmerGrid 重编号的物理组）、`mesh.boundary`（外边界 triangle=303）。先将 Elmer 节点坐标与**本项目同一真实 MSH2 网格**做有限公差的唯一双射，再按每个四面体**排序后节点集合**恢复原 Gmsh 单元下标，不允许按数组顺序碰巧对应。读取 `mesh.names` 后逐四面体检查 sea/tx dry/rx dry 原三材料体标签；读取实际 Elmer 外边界组 ID 后逐三角面核对它与原四面体立方体外表面完全一致，防止误把内部干湿界面当作外部总 A 边界。随后可读取显式文本 `ElmerElementID J1x J1y J1z J2x J2y J2z`（A/m²、实单位端口基）并重排为 `(2,Ntet,3)`。**这张七列表必须由独立 Elmer 后处理明确导出，ElmerGrid 并不会自动生成。** 官方 [ElmerSolver 网格格式手册](https://ftp.csc.fi/index/files/Historical/pub/index/elmer/doc/ElmerSolverManual.pdf) 描述节点/元素/边界文件字段。

更根本的电磁源约束：闭合稳态外加电流不仅要有合法的边元体积分，还必须对离散全部 P1 梯度满足 `L_p · grad(phi_h) = 0`。当前 [主 A–φ 端口求解入口](../../../sdfmpneo_bfzi/direct3d_galerkin.py) 在求解前拒绝归一化弱连续性缺陷 >1e-8 的体源；`source_nodal_conservation(mesh,L)` 返回各端口的残差。独立单四面体 NumPy 实算闭合线源 2.22e-16、任意常体源在初版全局 L1 归一化下为 6.67e-1；进一步修正为局部活动节点归一化后该算例为 1.0（仍 FAIL），避免 h 加密增加无关边使残差被稀释；说明**随意逐单元填入 CoilSolver J 即使维度与单位都对，也不一定是闭合线圈源**。严格离散守恒为必要但不充分条件，Elmer 原来实际使用的高阶/节点场和本项目当前**分片常量**表示并不自动等价，必须通过同源边元 RHS、1A 截面通量、连续性及 h 加密共同证明。

本地另运行独立 NumPy 合成数据（并非 ElmerGrid 调用）：8 节点、6 四面体、三个重编号材料组、12 外边界三角面均匹配，删去一外面后严格失败；已新增仓库源码测试 `tests/test_bfzi_d2_elmer_interchange.py` 及非守恒源测试。**当前工作环境无 Gmsh、Elmer、整仓库本地源树；正式新测试尚未执行。** 待工作站运行 `python -m pytest tests/test_bfzi_d2_elmer_interchange.py tests/test_bfzi_d2_galerkin.py -q`，然后实际 OCC 两独立 SE(3) → ElmerGrid 14 2 → 物理标签与闭合源同一性 → 同外域双端口求解及 h/R/源独立收敛。**D2 OPEN，G1 FAIL，教师冻结，禁止 GitHub Actions。**



## 2026-11-09：已落地的 Elmer 体源弱载荷桥接（不是独立参考 PASS）

新增 `elemental_current_edge_load(mesh, element_current_density)` 和 `solve_elemental_current_ports(mesh, frequency_hz, element_current_density, gauge='tree')`，复用现有树—余树 A–φ 材料及真空两组同源场求解。数组严格为 `(n_ports, n_tetra, 3)`；当前端口双对偶读取仅接收**实数空间单位电流基**，复相位通过已求解端口列线性组合，复数空间源须另行定义含相量约定的真实端子电压泛函，不能直接由未共轭 `L@A` 冒充阻抗，每项为目标有限元网格上该四面体的**分片常量**单位端口外加体电流密度 `J_s`，单位 `A/m²`（峰值相量）。对四面体 `T` 的局部 Whitney 边 `e=(i,j)`，装配精确积分：

\[
L_{ep}^{\mathrm{volume}}\big|_T
=\int_T \boldsymbol J_{s,p,T}\cdot\boldsymbol w_e\,dV
=\frac{|T|}{4}\,s_{T,e}\,
\boldsymbol J_{s,p,T}\cdot
(\nabla\lambda_j-\nabla\lambda_i).
\]

`s_{T,e}` 是四面体局部边到**全局有向边**的符号。同一个 `L` 既加载右端又读取广义材料增量端口响应（相同外域/真空差分），但它**不是**凭公式自动恢复铜端子电压、铜损耗或零半径细丝真解。已在当前隔离 NumPy 中实际核对两组 J 的**单四面体**精确公式等于四点 Whitney 积分，且交换顶点存储顺序后全局载荷一致；项目对应测试已写入，**完整仓库 pytest 未执行**。

要使本桥接真正可与 Elmer 比较，下一次必须：**先**用 Elmer CoilSolver 在拥有相同实际闭合铜体积的共形网格上独立求出两个单位端口的 `J_{s,1},J_{s,2}`；**逐个单元 ID/顶点坐标建立 ElmerGrid → 原始 Gmsh 四面体的映射**，不能把同长度数组按文件顺序直接复制；核对 1A 截面通量、闭合回流、各单元法向跳跃/连续性、两个端口独立方向、频率与 `exp(+iωt)`，并确保 Elmer WhitneyAVSolver 和本项目加载**完全相同的**分片体源。只有上述门槛通过后，才能把 `L^{volume}`、两组空间 B/E/J/q、完整复 2×2 增量 Z 与 h/R/体源离散收敛作为独立对照。当前**仍是三个物理材料域（sea/tx dry/rx dry）**，还未生成包含**真实铜闭合体**的 Elmer CoilSolver 网格，也未获得 Elmer 的两套源 J，故以上 API 是数学接口而非已完成的物理源同一性。

与此独立，正式物理 MSH2 导入现在验证 `<mesh>.ports.json` 的 SHA256 **以及由当前场景重新计算的全部有向细丝源合同**，不得用 `--mesh --export-port-manifest` 改写证据。该验证只适用于**现有理想细丝路径**，不能被误读为体源与细丝源等价。上游 [Elmer 两线圈 CoilSolver/WhitneyAV 示例](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/tests/mgdyn_steady_coils/case.sif) 表明可以取用元素线圈电流，但其元素电流表示和方向、正常化须实际导出核验，不得按名称推定。**D2 OPEN、G1 FAIL、教师冻结、禁止 GitHub Actions。**



## 2026-11-08 补充：MSH2 Surface 101 的强制网格关口与尚未解决的源等价

旧 MSH2 材料验证仅根据 3D 物理标签 1/2/3、包围盒与体积检查，未核实 Elmer 读取时用于施加 `n × A_total = 0` 的物理 Surface 101。现已在 [主网格导入器](../../../sdfmpneo_bfzi/direct3d_mesh.py) 中要求：101 的所有 2D 三角形恰好组成四面体实际**整个外表面**，任何缺少、重复、错误标到盒壁或海水/绝缘分界面都拒绝；逐三角面验证其确在外立方体六面之一。正式 `--generate-msh --mesh-only` 和 `--mesh` 均走这一关口，旧单标签仅在显式 legacy 诊断选项下绕过，不能用于 Elmer reference。相关回归见 [专项测试](../../../tests/test_bfzi_d2_physical_material_mesh.py)。目前只实际运行过一个独立的 NumPy 六四面体示例（1 合法＋4 非法 101 组合通过）；**尚未安装或执行**本分支完整 pytest、Gmsh、ElmerGrid、ElmerSolver。

**源等价的数学硬门槛仍未通过：** 本求解器的单口理想闭合线电流对应每条有向边元激励
\[
L_{ep}^{\mathrm{line}}=\oint_{\Gamma_p}\boldsymbol w_e\cdot d\boldsymbol\ell.
\]
Elmer 上游 `CoilSolver → WhitneyAVSolver` 的线圈电流是三维导线域内的 `\boldsymbol J_{s,p}(\boldsymbol x)`，其对应弱式
\[
L_{ep}^{\mathrm{vol}}=\int_{\Omega_{\mathrm{coil}}}
\boldsymbol J_{s,p}\cdot\boldsymbol w_e\,dV.
\]
若未证明源电流的连续物理定义相同，就算电流均为 1 A，**也不能比较得到的空间 E/B/J/q 与完整复 Z 并声称某求解器误差**。对独立 Elmer 仅有两条合规路径：(A) 真实实现与本项目相同、世界坐标/方向/闭合路径/相量一致的分布细丝边元 RHS，并独立验证边方向和端口读数；(B) 双方改为**同一个明确的闭合有限截面体电流密度**，保持电荷连续性、边界与材料分区、两单位端口及铜/海水损耗口径一致，分别求解。仅在 Elmer SIF 启用 `Use Elemental CoilCurrent` 或赋标称铜线径**不能**自动满足 (A) 或 (B)。Elmer 官方 [两线圈示例](https://github.com/ElmerCSC/elmerfem/blob/devel/fem/tests/mgdyn_steady_coils/case.sif) 可以作为体源加载方法参考，**不是 BFZI 同源验证结果**。

工作站执行优先级：①运行 `python -m pytest tests/test_bfzi_d2_physical_material_mesh.py tests/test_bfzi_d2_port_manifest.py -q`；②真实双独立斜置 SE(3) OCC 建网，再 `ElmerGrid 14 2 <mesh.msh> -autoclean -names -out <directory>` 核实标签和外边界；③选定 (A) 或 (B) 并实算**等价源弱载荷/端口取向与两单位源**；④才核对同源、同材料、同外域、同频率和相量下的完整 2×2 复阻抗及 B/E/J/q，并做独立 h/R/源离散收敛。**D2 OPEN，G1 FAIL，教师冻结，不使用 GitHub Actions。**



## 2026-11-08 新增：独立 Elmer 与 BFZI 共用的严格闭合双端口源合同

独立参考不能只共享网格和材料，**还必须共享实际闭合馈回流、电流方向、源截面、频率与相量约定**。原来只有三材料物理标签；并没有相同的 Elmer 线圈源，因此不能直接比较两个求解器的复阻抗。

已提交 [双端口物理源合同主代码](../../../sdfmpneo_bfzi/direct3d_port_manifest.py)，实际使用项目原有的 polygonal_port_vertices 生成**世界坐标有向闭合路径、完整回流、单位峰值电流、铜线径/电导率、两盒独立 SE(3) 姿态、圆角弦段数、外域 R 与 SHA256 内容哈希**。不是为 Elmer 重新绘制一套路径。相量采用 exp(+iωt)；本项目对每个单位线源的定义是：

\[
\boldsymbol J_{\mathrm s,p}(\boldsymbol x)
=I_p\oint_{\Gamma_p}\delta^{(3)}(\boldsymbol x-\boldsymbol r)
\,\mathrm d\boldsymbol\ell,\qquad
L_{e,p}=\oint_{\Gamma_p}\boldsymbol w_e\cdot\mathrm d\boldsymbol\ell.
\]

其中 L 是 Whitney 有向边元的源加载泛函；闭合源端口读取使用同一个有向泛函。实现中的弧线以实际弦段近似，**物理回流严格闭合**。

无需安装 Gmsh，即可单独导出当前两个 BFZI 端口的物理合同：

    python -m examples.bfzi_d2_aphi --case bfzi --ports-only --export-port-manifest runs/bfzi_ports.json --frequency 200000 --panels-per-corner 16

各自独立位姿可附加选项：

    --tx-rotation-vector 0.10 0.00 0.15 --tx-shift 0 0 -0.01 --rx-rotation-vector 0.35 0.10 0.20 --rx-shift 0.02 0.00 0.01

正式三材料 Gmsh 建网入口若使用 --generate-msh runs/bfzi_pose.msh，则在 OCC 网格成功生成且三材料 MSH2 读回验证之后，**自动同时导出** runs/bfzi_pose.ports.json，并把源合同哈希写入网格元数据。也可手动指定 --export-port-manifest PATH。由此确保 Gmsh、BFZI 主求解器及未来 Elmer 对照引用同一份闭合导线定义。

**严格区分：该 JSON 不是已经在 Elmer CoilSolver 实现的数学等价线源。** 项目目前采用零截面分布线电流；Elmer 上游 TEAM-7 CoilSolver 使用物理体线圈并通过截面电流约束求解。简单地用标称线径创建铜圆柱并赋均匀电流，并不自动等价于当前理想闭合线源。若双方源项不同，其 E/B/J/q/Z 差异首先是**物理模型差异**，不能当成两个 FEM 求解器的精度误差。下一正式关口必须在两个求解器中实现同一有限截面闭合体电流，或者为 Elmer 明确实现等价的理想闭合线源泛函；然后才允许量化同网格/同相量/同外域的求解误差。

保留锁定顺序：①三材料网格、源回流与外边界合同；②ElmerGrid 转换并核对 mesh.names；③基于上游成熟 TEAM-7 先核对独立 Elmer 方程/励磁；④实现相同物理源后比较 E/B/J/q/2×2 Z；⑤分别完成 h/R/源弦段与铜趋肤层收敛。D2 OPEN、旧 G1 FAIL、教师冻结，不使用 GitHub Actions。



> **2026-11-08 材料标签读取证据与仍缺项：** 本项目 BFZI FEM 现**严格使用** Gmsh 的 volume 1=sea、2=tx dry、3=rx dry 元素标签赋 σ，并检查每个四面体是否处于其真实 SE(3) 材料区域，以及对应几何体积；新 [Gmsh OCC 主代码](../../../sdfmpneo_bfzi/direct3d_gmsh_physical.py) 在 mesh-only 返回前强制读回并验证。实际写出/重读结构化三材料 MSH 2.2 后，完整 1350 单元双闭合端口的 `ΔZ/E/B/Joule` 与原始内存网格**逐项完全相同**，证据在 [JSON](evidence/d2_three_material_msh2_roundtrip_20261108.json)。对于旧 `Physical Volume("all",1)`，除非明确 `--legacy-single-tag-msh`，否则拒绝；且这种旧网格**不能直接给独立 Elmer 分海水/两盒材料**。当前 Gmsh API/Elmer 无法在本容器启动，OCC 任意斜置 SE(3) 真实建网、Elmer 等价铜闭合源、空间场与复阻抗独立对照均 OPEN。不得把三材料文件往返当成 Elmer PASS。
>


## 2026-11-08：真实独立 SE(3) BFZI 三材料共形网格入口（Gmsh/Elmer 待实跑）

旧的 Gmsh .geo 入口把整个计算域标成单一 Physical Volume("all",1)，
本项目 Python 端再依据元素中心重分海水/绝缘材料。**这种单标签网格不能直接让独立 Elmer 正确分配海水、发射端盒、接收端盒材料，因而不满足正式独立参考的输入要求。**

新增 [Python Gmsh OCC 物理分区主代码](../../../sdfmpneo_bfzi/direct3d_gmsh_physical.py)。
它不靠布尔后实体编号猜测，而直接读取官方 occ.fragment 返回的原实体至结果实体映射，分别赋值：

| Gmsh 物理组 | 意义 | 用途 |
|---|---|---|
| 3D Volume 1 | 导电海水 | 对应海水 sigma |
| 3D Volume 2 | 发射封装绝缘体 | sigma=0 |
| 3D Volume 3 | 接收封装绝缘体 | sigma=0 |
| 2D Surface 101 | 有限立方体六个真正外边界 | 总 A 切向零截断，不包括干/湿内部界面 |

两只封装的**独立真实三维旋转/平移**进入 OpenCASCADE 实体，
各自世界坐标包围盒只控制周围的局部单元尺寸，不改变实体。
输出 MSH2.2 ASCII，可分别由本项目 A–φ 网格读取器和
ElmerGrid 接入。生成材料网格时优先使用新的 --generate-msh，
不是旧的 --write-geo 单标签模式。

下列命令适用于装有 Gmsh Python API 与 Elmer 的工作站；
**本容器没有 Gmsh/Elmer 可执行程序，下列两项尚未实跑**：

    git pull --ff-only

    # 仅产生三材料物理共形网格，不立刻执行大型复数 Maxwell 求解
    python -m examples.bfzi_d2_aphi --case bfzi --generate-msh runs/bfzi_rx_oblique.msh --mesh-only --rx-rotation-vector 0.35 0.10 0.20 --outer-radius 0.80 --mesh-size 0.07 --near-mesh-size 0.008 --near-padding 0.025 --output runs/bfzi_rx_oblique_mesh_meta.json

    # 独立转换为 Elmer 原生网格，实际编号请查看 mesh.names
    ElmerGrid 14 2 runs/bfzi_rx_oblique.msh -autoclean -names -out runs/bfzi_rx_oblique_elmer
    Get-Content runs/bfzi_rx_oblique_elmer/mesh.names

    # 在有足够内存和计算资源的工作站才运行本项目真实双端口
    python -m examples.bfzi_d2_aphi --case bfzi --mesh runs/bfzi_rx_oblique.msh --rx-rotation-vector 0.35 0.10 0.20 --outer-radius 0.80 --frequency 200000 --panels-per-corner 16 --gauge tree --output runs/bfzi_rx_oblique_aphi.json --summary-only

**严格边界：** 三个材料体/外边界标签虽可让 Elmer 明确材料区域，
但**还没有**与本项目闭合线源泛函等价的 Elmer 两端口源实现。
本项目在绝缘盒内施加理想零截面闭合线源，
该网格不含 Elmer CoilSolver 所需的真实三维铜导线体积、
独立端口电流与连续体电流约束。正式独立 Elmer 必须增加
一致物理源及 200kHz 相量约定，或先采用上游 TEAM-7 已成熟的
CoilSolver + WhitneyAVSolver 源—场整体案例独立核对。
ElmerGrid 可能重新编号组标识，SIF 中的 Target Bodies/
Target Boundaries 只能采用转换后的 mesh.names 实际编号，
不能直接假定原 Gmsh 的 1/2/3/101 不变。

参考：
[Gmsh 官方 OCC fragment 实体结果映射](https://gmsh.info/doc/texinfo/#gmsh/model/occ/fragment)；
[ElmerGrid mesh.names 重编号问题](https://github.com/ElmerCSC/elmerfem/issues/437)。
未使用 GitHub Actions。


**定位：为 BFZI 三维 A–φ 直接法提供真正独立、成熟的 H(curl) 求解器对照。不是继续匹配 Shafiei/Wang 论文表格，也不是将 TEAM-7 当作 BFZI 几何。**

## 可信的外部实现

已直接检查 ElmerCSC 的公开电磁验证仓库 [ElmerCSC/elmer-elmag/TEAM7](https://github.com/ElmerCSC/elmer-elmag/tree/main/TEAM7) 中的 `README.md`、`TEAM7.sif`、`steady.sif`、`gmsh/TEAM7.geo` 和 `TEAM7_A1B1.csv`。研究时固定上游参考提交：

```text
ElmerCSC/elmer-elmag @ 843037c374ba61cf61d02d17a2c5fc11e0d33de3
```

其方法为 **CoilSolver → WhitneyAVSolver → MagnetoDynamicsCalcFields**：
- CoilSolver 先在独立真实线圈体域求符合总电流约束的电流分布，不是直接套二维等效环；
- WhitneyAVSolver 解含导电铝板与空气、真实偏心孔的三维边元磁动力学方程；
- MagnetoDynamicsCalcFields 输出铝板涡流与磁场测点；适用于检验独立的三维电磁有限元算法。

参考 [Elmer Tutorial 31](https://ftp.csc.fi/pub/files/index/elmer/doc/ElmerTutorials.pdf) 和 [官方电磁验证仓库](https://github.com/ElmerCSC/elmer-elmag)。这套系统满足“成熟独立软件验证”的要求，比用旧 G1 的相近求积规则互相比对更可信。

## 不能直接把其数字当作我们的真值

| 项 | 官方 Elmer TEAM7 | 当前 SDFMPNEO D2 |
|---|---|---|
| 电流源 | 分配在完整 25×100 mm 线圈体积的 CoilSolver，`Desired Coil Current = -2742` | 在该截面做加权多条**闭合赛道细丝** Gauss 积分；显式 `--current-sign -1` 可与 Elmer 保持同向，但体积分布仍非完全一致 |
| 3D 电磁变量 | WhitneyAVSolver，含导体/空气；规范与离散由成熟工具实现 | 独立 Nédélec 边元 + P1 phi + Coulomb 乘子，入射＋反作用场分解 |
| 外域 | 官方 Elmer `TEAM7/gmsh/TEAM7.geo` 使用 `x,y∈[-.2,.494] m`、`z∈[-.2,.394] m` | 默认 COMPUMAG 图 2 的大外盒；生成 `--outer-box elmer` 可匹配小盒 |
| 边界 | `TEAM7.sif` 中 `A {e}=0`，是总边元矢势切向零 | 新增 `--outer-bc total_zero`，边元有向线积分按源场投影；旧 `secondary_zero` 不是同一有限域问题 |
| 时间 | `TEAM7.sif` 为 **50Hz 瞬态**，每步 2.5 ms，16 步共 40 ms | `50/200 Hz` 单频峰值复相量 `e^{+iωt}`；需要从 Elmer 稳定周期的时间序列提取 cos/sin 基波，不能直接对比某一时刻与 `|B|` |
| 观测 | Elmer 保存线 A1–B1、A2–B2、A3–B3、A4–B4 | 使用相同的四组空间测点、物理导电侧 Jy 和 **B_total**；注意 Elmer 测点采样线不是单个接近表面单元平均 J |

**重要：** `TEAM7_A1B1.csv` 位于 Elmer 仓库，但其文件本身没有充分描述每列数据的单位、工况、时间/相位与是否为实测。没有这些信息，不将它擅自写成 COMPUMAG 官方实验真值，也不直接判精度。

此外，COMPUMAG 原始 Problem 7 大外域的磁边界条件与 Elmer 小盒的总 A 边界不是完全相同的连续问题。**只允许同一几何、源、外域、材料、频率和边界条件下作精度对照**。

## 独立成熟求解器运行流程（本机，禁止 GitHub Actions）

在装有 Gmsh、ElmerGrid 和 ElmerSolver 的工作站上：

```powershell
git clone https://github.com/ElmerCSC/elmer-elmag.git
cd elmer-elmag
git checkout 843037c374ba61cf61d02d17a2c5fc11e0d33de3
cd TEAM7

# 仓库已有 TEAM7/ 格式网格和 .sif；检查其对应所选 Elmer 版本
ElmerSolver TEAM7.sif
```

如需从源几何重新产生网格，读取 `TEAM7/gmsh/TEAM7.geo` 的真实三个物理体域标签 Coil=1、Plate=2、Air=3，`gmsh -3` 后用 `ElmerGrid 14 2` 转换，不能把本项目只有一个统一物理体积标签的 `team7_geo()` 误当作适用于 Elmer `TEAM7.sif` 的完整三域物理网格。

SDFMPNEO 的相同**有限外盒/总 A 边界/电流方向**初始求解：

```powershell
# 在 research/bfzi-uwpt-electrothermal-surrogate 分支根目录
python -m examples.bfzi_d2_team7 --write-geo team7_elmer_domain.geo --outer-box elmer
gmsh team7_elmer_domain.geo -3 -format msh2 -o team7_elmer_domain.msh
python -m examples.bfzi_d2_team7 --mesh team7_elmer_domain.msh --frequency 50 --outer-bc total_zero --outer-box elmer --current-sign -1
```

两者仍有**体源 vs 细丝安匝积分**的系统性差异，必须在源截面 Gauss 和 Elmer coil 源分布之间做单独匹配，再从后续周期提取 50Hz 复基波。不同时间节点、源方向或者外边界下产生的差异不是某一算法的连续精度误差。

## 本轮 D2 新发现和强制物理纠正

初版 `direct3d_aphi.py` 的 `delta_z` 由
`iω∫Ω A_src,p·J_e,q dV` 构成。`Herm(delta_z)` 由相同离散场和弱式可与体积 Joule 功率高度接近，这是**有限域弱式代数结果**，不是源线圈的感生端口电压证书。

为避免这一错误已新增：

- `solve_aphi(...,outer_bc='total_zero')`：真正沿 Nédélec 有向边投影非齐次 `n×A_react=-n×A_src`；常向量势（纯梯度）三维极限数值检验通过。
- `direct3d_port.closed_source_port_impedance()`：在实际闭合馈回流/赛道形线源上，从有限元反作用势提取 `ΔZ_port,pq=iω∮Γ_p A_react,q·dl`。支持两端口真实路径、多股权重、独立加密。真实感生端口与重叠积分在有限域间有需要核算的边界项（理论见 [第 17 节](../../../theory/sections/17_d2_aphi_reference_and_ports.tex)）。
- BFZI、TEAM-7 CLI 现在**分别**输出路径电压、体积重叠矩阵、海水/铝板总场 Joule 功率和相应差异；没有用平均值、后验阈值或强制对称化来“修正”误差。

在局部三维小网格测试中，两个不同的闭合端口来源出现预期的非零互阻，但原始线场的离散互易性和感生端口功率尚有不小误差。**这属于尚未收敛的参考求解器，而非已经验收。** 需要网格/外域/闭合路径积分与独立 Elmer 真值交叉细化。

## 下一步关口

1. 在本机有 Gmsh/Elmer 时先跑**正式共形 TEAM-7 网格**，同条件核对 Elmer 瞬态基波与 SDFMPNEO 复相量；同时进行源截面积分与路径数值积分细化。
2. 最少三档边元空间 h，三档外域 R（或另一可靠透明外域方法），逐项记录两单位端口的复阻抗及所有 E/B/J/q、真实线源感生电压与海水 Joule 功率。单一粗网格不能称高保真参考。
3. 之后把相同方程用于真实 BFZI 圆形/圆角方形连续线圈、双独立 SE(3) 封装和闭合馈回流；额外核算有限铜截面/邻近效应。旧 G1 FAIL，神经教师继续冻结。

**本轮当前执行环境没有可调用的 Gmsh/Elmer 求解器，因此没有伪称已完成成熟软件的全网格比对。**
