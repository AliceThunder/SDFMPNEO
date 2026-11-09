# D2 独立成熟参考：Elmer 官方 TEAM-7（2026-11-06）

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
