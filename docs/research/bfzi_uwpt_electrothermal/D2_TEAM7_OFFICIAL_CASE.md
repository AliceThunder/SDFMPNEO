# D2｜COMPUMAG TEAM-7 官方三维涡流 benchmark：几何、源与初始实算

> **2026-11-06 后续实算与修正（以下内容为当前优先证据）：** 端口积分现已改为\`direct3d_port.closed_source_port_impedance_exact\`：将每条真实闭合源弦段与四面体精确裁剪、在每段上解析积分最低阶 Nédélec 切向场。并引入\`medium_vacuum_port_increment\`，对于\`total_zero\`外边界先扣除同一有限域真空镜像电感，不把纯真空边界反作用误标成海水/铝板感抗。旧表的端口重叠与功率闭合并非真实感生端口证据。新源码和真实运行结果见下文，**D2 仍 OPEN**。
>
## 新增：材料共形分层加密后的直接端口与 Joule 数值（不作 PASS）

为在不依赖尚未安装的 Gmsh/Elmer 软件时先检查三维离散核心，引入\`graded_team7_smoke_mesh(level=1..4)\`。其导电铝板、偏心方孔与真实厚度固定；对每个区间分别细分，\`z=0..19 mm\` 方向分成 \`level\` 层，金属和孔边界恰好落在网格面。材料真实体积在四级网格中均为 \`0.001420668 m³\`，不是通过重新缩放或平移金属面积获得数值变化。

固定\`50 Hz\`、\`+2742 AT\`、线圈截面\`2×2\` Gauss、每个圆角\`5\` 弦段、\`secondary_zero\`、外盒\`x,y∈[-0.15,0.44]m, z∈[-0.08,0.25]m\`，独立隔离环境实算：

| 分层级别 | 全部四面体 | 铝板四面体 | 精确闭合源线积分端口功率 (W) | 金属总场 Joule (W) | 两者比值 |
|---:|---:|---:|---:|---:|---:|
| 1 | 750 | 48 | 8.220679 | 21.544659 | 0.3816 |
| 2 | 2304 | 384 | 7.388486 | 13.162529 | 0.5613 |
| 3 | 5082 | 1296 | 7.312475 | 10.899082 | 0.6709 |
| 4 | 9408 | 3072 | 7.084795 | 9.802378 | 0.7228 |

这证明前一轮端口采样高低阶不是全部差异的根源；实际三维材料离散细化显著改变体积损耗。同一外盒有限边界的反作用场能量通量与路径/体积有限元误差仍未消除，**不能仅通过精确积分或一条网格序列就宣称物理功率守恒**。50 Hz 和 200 Hz 的真实材料趋肤深度、源截面积分、外域边界差异须分别处理。上表不是 Elmer 或实验测量结果，也不是在官方大外盒上得到的精度判断。

\`total_zero\` 在真空时反作用势可以非零：在该粗 TEAM-7 小盒下得到真空镜像项约 \`-22.6271 i Ω\`（50 Hz、同一细丝源离散），它是人工总场边界引入的电感偏移，不能算成金属感抗。代码明确输出：
- \`total_medium_reaction_z\`：导电时相对无限真空入射源的反作用；
- \`finite_domain_vacuum_image_z\`：无导电时、相同外域的边界镜像；
- \`material_increment_closed_port_z = total_medium_reaction_z - finite_domain_vacuum_image_z\`：材料相对同一有限外盒真空的增量；
- \`volume_dissipation_W_for_1A\`：导电区\`0.5 σ |E_total|²\`积分；
- \`source_voltage_vs_metal_joule_relative_gap\`：独立源路径与导电功率仍未闭合的差异。

用下面命令重复同一四档网格测试（无任何 GitHub Actions）：

\`\`\`powershell
git pull --ff-only
python -m pytest tests/test_bfzi_d2_aphi.py tests/test_bfzi_d2_port.py tests/test_bfzi_d2_mesh_probe.py tests/test_bfzi_d2_team7.py -q

python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 1 --frequency 50 --corner-panels 5
python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 2 --frequency 50 --corner-panels 5
python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 3 --frequency 50 --corner-panels 5
python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 4 --frequency 50 --corner-panels 5
\`\`\`

真实无限海水与 TEAM-7 标准验收的下一关仍是：**同一物理源、共形 Gmsh 官方尺寸细网格、外域条件与 Elmer WhitneyAVSolver 同工况对照**；一旦其中端口与体积能量出现不一致，就先按物理边界能量通量和源/场离散处理，禁止后验拟合论文表格。



> **2026-11-06 最新修订：** 下面粗网格表的 50/200 Hz 数值属于旧 `secondary_zero` 有限域和**体积重叠算子**，只代表代数内部计算，不是物理闭合端口认证。已新增三维真实线圈路径感生电压 `direct3d_port.py`、边元总场边界选项 `outer_bc='total_zero'`、独立 [Elmer TEAM7 源/网格/时域/边界对照契约](D2_ELMER_INDEPENDENT_REFERENCE.md)。真实端口电压、体积 Joule、互易、外域/网格还没有独立收敛；不可把下面的 `Pport=Pvolume` 旧描述视为本轮最终判断。独立参考正式理论见 [第 17 节](../../../theory/sections/17_d2_aphi_reference_and_ports.tex)。**D2 仍 OPEN。**
>


**2026-11-06｜状态：官方物理输入已转为代码，粗网格 50/200 Hz 已在隔离 Python 环境实跑。此处仅表示 D2 程序通路形成，**不是**测量精度或外域收敛验收。**

## 权威来源与物理输入

原始规范：<https://www.compumag.org/wp/wp-content/uploads/2018/06/problem7.pdf>（COMPUMAG TEAM Workshop Problem 7，图 1–4、表 1–3）。外部对照：[COMSOL 官方标准实例](https://doc.comsol.com/6.4/doc/com.comsol.help.models.acdc.multiturn_coil_asymmetric_conductor/multiturn_coil_asymmetric_conductor.html)；[Ansys 标准例](https://ansyshelp.ansys.com/public/Views/Secured/corp/v251/en/lsdyna_vm/Hlp_V_VM_L_EMAG010.html)。**本项目只研究官方基准的成熟三维场求解方法，不让 BFZI 线圈假装成 TEAM-7 激励。**

| 官方定义 | 本次代码使用 |
|---|---|
| 铝板 | `0≤x,y≤294 mm, 0≤z≤19 mm`，导电率 `3.526×10^7 S/m`，`μr=1` |
| 偏心贯穿方孔 | `18≤x≤126 mm, 18≤y≤126 mm`，沿 `z=0..19 mm` 贯通；孔内为不导电空气 |
| 激励 | 矩形赛道形多匝线圈，`2742 A·turn`、`f=50/200 Hz`；外周矩形 `x=94..294 mm`、`y=0..200 mm`；内周缩进 25 mm，外角 `R50`、内角 `R25`；`z=49..149 mm`（底部离铝板上面 30 mm） |
| 原始外域 | 图 2 外框 `x,y∈[-1353,1647] mm`、`z∈[-300,449] mm`，外边界原规范采用 `n·B_total=0`（必须与当前有限域反作用 A 的边界区别对待） |
| 磁场探针 | `x=0,18,…,288 mm`，`(y,z)=(72,34)` 和 `(144,34) mm`，测 `B_z` |
| 涡流探针 | 同一组 `x`，`y=72 mm`，`z=19 mm`（上表面）与 `z=0 mm`（下表面），测 `J_y`，只能取铝板单侧迹 |
| 相量方向 | `e^{+iωt}`，`ωt=0°` 对应 `Re`，`ωt=90°` 对应 `−Im`；B 的官方表格单位 **Gauss**，`1 T=10^4 Gauss` |

**注意：** 原始 PDF 表 2、3 的“数值栏”是供求解器计算后填写的空表，并**不提供实测曲线值**；必须另行获得 Fujiwara & Nakata (1990) 或可授权的独立 benchmark 测量数据，不得把该空表变成虚构的验证数字。

## 核心数值实现

`sdfmpneo_bfzi/direct3d_team7.py`：

1. `team7_geo()`：根据官方图 1/图 2 生成真实偏心贯穿方孔铝板、空气外域的 Gmsh OpenCASCADE `BooleanDifference/BooleanFragments` 共形四面体几何。默认使用官方外边界范围，局部网格尺寸在铝板附近细化。需要本机安装并实际运行 `gmsh`，**当前隔离环境未安装，故尚无 Gmsh 官方尺度网格成功记录**。
2. `team7_mesh()`：从 MSH2.2 读取四面体并按物理几何将铝板赋予 `σ=3.526e7`，孔、线圈空间和外空气 `σ=0`。
3. `racetrack_vertices()` 与 `team7_stranded_source()`：在 **25 mm × 100 mm** 截面上做二维 Gauss，展开为权重合计 `2742` 的分布式真实闭合赛道形丝源，每条使用解析直线势与 Biot–Savart。是对作者均匀绞线/匝密度的**离散近似**，必须独立细化截面求积 `n_width,n_height` 和弯角弦段 `n_corner`；不含铜近邻效应，不能视为实物单根 BFZI 绕线。
4. `official_probe_lines()`：忠实构造 4 条官方测量线；`direct3d_probe.sample_cell_fields` 采样 `B_total=B_inc+B_react` 和金属侧 `J_y`。最低阶 Nédélec 的单元 `B` 常值、`J` 单元均值只是初阶诊断，必须提高网格精度后再与测量对照。

`examples/bfzi_d2_team7.py` 生成官方几何、执行当前 3D A–φ 并导出完整 `17×4` 复场测点及平均功率；`tests/test_bfzi_d2_team7.py` 只检验此主求解代码通路，不设置后验“接近论文”的通过阈值。

## 本轮运行的有限意义

为首先确保代码真的解出了**带偏心孔的三维受迫涡流**，`coarse_team7_tetra_mesh()` 先生成包含实际铝板/方孔界面的非均匀结构四面体，只具有 **216 顶点、750 个四面体，其中 48 个铝板导电单元**，截断域约 `[-.15,.44]² ×[-.08,.25] m`（**小于官方外域**）。

在 1 A 驱动、`2742 A·turn` 且使用 `2×2` 截面 Gauss、每个圆角 `5` 等弦分段的输入下，隔离 Python 环境实测：

| 频率 | 海水/铝板增量端口 `Re ΔZ` (Ω) | `Im ΔZ` (Ω) | 铝板平均功率 (W) |
|---|---:|---:|---:|
| 50 Hz | 43.0893 | −54.0223 | 21.5447 |
| 200 Hz | 337.7857 | −262.6262 | 168.8929 |

利用三维空间场和源/端口积分独立组装的耗散二次型在当前离散上彼此吻合；实际四条测点线均成功采样，`J_y` 的金属表面选择为导电侧。

**禁止将这些幅值当作实测参考或“精度接近”证据。** 截断域、源截面离散、近壁 48 单元、边元一阶、人工外场条件严重限制精度。官方说明外域 `n·B_total=0`；当前 `direct3d_aphi.solve_aphi` 使用 `n×A_react=0`，**不是相同的外边界条件**。必须完成边界选型（或场外推/透明耦合）及外域独立收敛，再按同一物理条件做定量比较。

## 重现命令

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_d2_aphi.py tests/test_bfzi_d2_mesh_probe.py tests/test_bfzi_d2_team7.py -q

# 1. 无 Gmsh 也可运行，但仅用于 48 单元初步通路
python -m examples.bfzi_d2_team7 --coarse-smoke --frequency 50 --corner-panels 5
python -m examples.bfzi_d2_team7 --coarse-smoke --frequency 200 --corner-panels 5

# 2. 真正几何共形网格与官方外域（要求本机另装 Gmsh）
python -m examples.bfzi_d2_team7 --write-geo team7.geo
gmsh team7.geo -3 -format msh2 -o team7.msh
python -m examples.bfzi_d2_team7 --mesh team7.msh --frequency 50
python -m examples.bfzi_d2_team7 --mesh team7.msh --frequency 200
```

下一步**不是**继续拟合 Shafiei/Wang 曲线，而是：先运行 Gmsh 实际 TEAM7 网格，按外域边界条件及源截面求积分开收敛，再以 Fujiwara–Nakata 的实测 `B_z/J_y` 验证实际三维 FEM；之后直接迁移相同求解器到连续 BFZI + 双 OBB。**D2=OPEN，G1=FAIL，教师冻结。**
