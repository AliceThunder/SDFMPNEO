# D2 独立成熟参考：Elmer 官方 TEAM-7（2026-11-06）

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
