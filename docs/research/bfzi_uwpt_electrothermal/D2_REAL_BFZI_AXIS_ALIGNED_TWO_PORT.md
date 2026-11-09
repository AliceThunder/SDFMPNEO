# D2｜真实 BFZI 两闭合端口：无需 Gmsh 的轴对齐共形三维基线

## 2026-11-09 实际双端口求解结果（本段覆盖下面仅有网格几何数据的旧状态）

**已在隔离 Python/SciPy 环境真实求解三维 A–φ 混合边元方程与同域真空差分。** 使用从仓库 `direct3d_aphi`、`direct3d_edge_source`、`direct3d_galerkin` 提取的**同一弱式及导线解析几何公式**在本地重新组装，采用稀疏直接解，并不是另造工程/旧 G1。不过隔离环境没有完整仓库依赖，**尚未在原 GitHub 整树运行完整 pytest 或 Gmsh/Elmer，不能将本地镜像复算等同于正式仓库全通过**。全部数值已写入 [真实双端口 JSON 证据](evidence/d2_real_bfzi_two_port_20261109.json)。

物理工况：真实圆角方形 BFZI 两匝连续螺旋，两个独立闭合馈回流，分别位于 z=±0.13 m 的两个 0.20×0.20×0.03m 绝缘封装内；海水 5 S/m、200 kHz、峰值相量 `exp(+iωt)`；理想闭合线源，不含有限铜集肤/邻近电流。结构化材料界面严格共形，内层核心半宽 `0.24m`、近壁 padding `0.08m`、近壁步长 `0.08m`、远场目标步长 `0.30m`、盒内分层 1。下述每个工况都是 6318 个四面体、1400 顶点，两端口都进行了实际复数矩阵求解。

### 外域 R 敏感性：仍未收敛

保持每段真实圆角 **4 条弦**与所有封装近场网格坐标切面完全一致，改变外部立方体半宽：

| R (m) | Re ΔZ₁₁ (mΩ) | Re ΔZ₁₂ (mΩ) | Im ΔZ₁₁ (mΩ) |
|---:|---:|---:|---:|
| 0.32 | 5.93736 | 1.02607 | −0.18565 |
| 0.40 | 6.72123 | 1.45942 | −0.29645 |
| 0.52 | 7.34428 | 1.93779 | −0.47495 |

相邻两档 2×2 复 `ΔZ` 的 Frobenius 相对改变量（按新档幅值归一）约 **13.14%/10.72%**。**尚无 R 收敛证据**。虽然内核心的三个方向网格坐标精确固定，但外部网格壳层随 R 延展、质量及离散误差也变化，因此外域边界截断误差与远场网格误差仍混合；不得把该差别全解释为“海水影响范围”。

### 真实连续圆角源弦段加密：观察到受限离散趋稳

固定 R=0.52 m、**完全相同** 6318 四面体、同一物理源、封装、海水和频率，只改变各解析圆角的折线弦段数：

| 每圆角弦段 | Re ΔZ₁₁ (mΩ) | Re ΔZ₁₂ (mΩ) | 相邻完整复矩阵相对变化 |
|---:|---:|---:|---:|
| 2 | 7.07861 | 1.85238 | — |
| 4 | 7.34428 | 1.93779 | 3.678% |
| 8 | 7.41484 | 1.96018 | 0.966% |
| 16 | 7.43289 | 1.96587 | 0.246% |

源几何这一项在**固定粗网格空间**上表现为逐次趋稳，但仅是中心线折线误差，不能证明有限元场或外域已收敛。双端口互易差约 `10^-16 Ω`；`I=(1,0),(0,1),(1,1),(1,-1),(1,i)` 五组峰值相量下，一致边元端口功率与体积 Joule 功率差约 `10^-16 W`，均属于离散代数恒等式，**不是连续物理场精度认证**。

### 海水空间 E/B/q：原始边元 vs 解析源修正差别仍巨大

在同一 R=0.52m、圆角16弦、6318四面体解的三个实际 3D 海水测点，取电流 `(1,i) A`，分别计算原始总场 FEM 和解析闭合源真空场＋同一个四面体的数值材料反作用。两个空间场都按各自实际复数 E 构建 `q=0.5σ|E|²`：

| 测点 xyz (m) | 原始 FEM q (W/m³) | 解析源修正 q (W/m³) |
|---|---:|---:|
| (0,0,0) | 0.00298777 | 0.00015891 |
| (0.15,0,0) | 0.02041878 | 0.03030978 |
| (0,0,0.16) | 0.03833727 | 0.00855489 |

同时实际求出了两个端口各自的复 E/B 矢量；[仓库结构化空间场强度及 Joule 证据](evidence/d2_real_bfzi_spatial_probes_20261109.json)。这组“解析源修正”是**事后主次场重构诊断**，尚不等同于 `solve_closed_primary_secondary` 的解析源参与 PDE 解，也不能选作无条件真值。大差异再次证明端口—Joule 离散代数一致性不能替代物理空间 E/J/q 验证，热教师必须继续冻结。

### 下一物理阻塞：稀疏直接线性求解器的规模限制

在 `R=0.52m`、固定核心、`padding=0.08m` 下将湿侧网格目标步长从 0.08m 缩至 0.06m，材料共形四面体由 6318 增至 **12960**，复数 A–φ–规范混合系统有 **18044** 个未知量、约 **596152** 个稀疏非零元。当前隔离环境的 SciPy `spsolve` 未在运行时间内完成；尝试常规 ILU+GMRES 时，原始 ILU 分解出现精确奇异，使用正则化 ILU 预条件也**未在限制时间内完成残差达标的迭代**。没有将这些未收敛值写为任何物理结果，也没有把失败的迭代法提交为“加速”。

**工程判断：不能继续盲目加密张量网格并无限增大 `spsolve`，必须先换成合适的、经过独立验算的三维 H(curl) 辅助空间预条件及混合鞍点块求解，或先使用成熟 Elmer/MFEM/hypre 参考运行正式材料 h 收敛。** 标准 [hypre AMS 辅助空间 Maxwell 求解器](https://github.com/hypre-space/hypre/blob/master/src/docs/usr-manual/solvers-ams.rst)适合 Nédélec H(curl) 子问题，但整个复数 A–φ–规范系统还需要相应块策略，不能简单声称普通 ILU 能处理。高质量 Gmsh 真三维局部加密可避免张量积网格外域爆炸，却不能单独解决线性鞍点规模瓶颈。

本轮已经将 [真实 BFZI 扫描脚本](../../../examples/bfzi_d2_convergence.py) 修正：固定 `--structured-inner-core-radius 0.24`；默认 padding 0.08m，`h` 真正改变湿侧法向单元数；源弦段 2/4/8/16。大型 `h=0.06/0.04m` 工况需要足够的工作站求解资源，**不要解不完就把结果外推成通过**。当前仍 `D2 OPEN`、`G1 FAIL`、神经教师冻结、禁止 GitHub Actions。



**2026-11-08｜状态：代码已接通，网格体积已在隔离 Python 环境实算；新的完整仓库双端口 PDE 入口尚未在本环境实跑。D2 OPEN。**

## 研究目的和边界

本任务不是 TEAM-7 论文数值复刻，也不是用旧 G1 的五模态直接法修修补补。目标是让**真实连续 BFZI 圆角方形/圆形螺旋、明确闭合馈回流、两套独立绝缘盒**真正进入一套三维 Maxwell–MQS A–φ 求解器，得到双端口增量复阻抗及空间 B/E/J/q；再分别验证 h、R、闭合源弦段数和铜有限截面/外域误差。

原来 [Gmsh 双独立 SE(3) 盒共形网格](../../../sdfmpneo_bfzi/direct3d_mesh.py) 已实现几何构造，但当前隔离 Python 环境没有 Gmsh/Elmer，因此**真实 BFZI 两端口尚没有可靠的本机端到端电磁数值运行结果**。这轮增加一个**受限但确为同一 BFZI 物理模型的可运行路径**：只要两只盒体各自的姿态为世界坐标轴的符号置换（包括分别正交旋转 90°、180°），使用直接生成的非均匀长方体分割四面体，且材料界面完整落在单元面上，不需要外部 Gmsh。

对任意真正倾斜的 SE(3) 位姿，此路径**明确拒绝**，绝不把旋转盒套上阶梯体素或以近似包围盒代替真实材料。任意 SE(3) 高保真必须采用现有 Gmsh OpenCASCADE `BooleanFragments` 几何接口。

## 2026-11-08：明确修复近壁海水 h，增加三类真正双端口独立扫描

原先仅调整 `--structured-box-layers` 会把盒体内部的分段加密，却**不能保证绝缘盒外导电海水紧贴材料界面的单元变细**。这不符合 D2 对海水空间 `E/J/q` 的网格收敛要求。

现已在真实 `axis_aligned_bfzi_tetra_mesh` 主代码增加 `near_step_m`，CLI 对应 `--structured-near-step`：它在两只盒体的每个世界坐标投影 `[a-pad,b+pad]` 内控制坐标步长，明确覆盖每一侧的**湿侧海水**，而不是只增加干侧盒体层数。若 `near_padding=0` 同时指定 `near_step` 会主动报错，防止误把仅干侧细化当作海水收敛。

在隔离环境中对相同材料共形网格生成算法实际运行的**几何与规模**如下，盒体尺寸和位置绝未改变：

| 外域半边长 R (m) | 近壁步长 (m) | 全域四面体 | 两盒绝缘体积 (m³) | 外部立方体积 (m³) |
|---:|---:|---:|---:|---:|
| 0.52 | 未独立设置 | 3,234 | 0.002400 | 1.124864 |
| 0.52 | 0.08 | 5,346 | 0.002400 | 1.124864 |
| 0.52 | 0.02 | 29,160 | 0.002400 | 1.124864 |
| 0.52 | 0.01 | 124,200 | 0.002400 | 1.124864 |
| 0.80 | 0.08 | 9,438 | 0.002400 | 4.096000 |

[真实网格尺寸与体积结构化证据](evidence/d2_bfzi_wet_wall_mesh_20261108.json)。这些是**网格几何计算**，不是电磁解的 `Z/E/B/J/q` 精度；张量积网格的三个方向切割会横贯整个计算域，0.01–0.02m 的全域四面体增长过快。后续任意 SE(3) 的高分辨率物理界面、开放海水与铜集肤/邻近效应仍必须使用真正三维局部共形/自适应 Gmsh/成熟 FEM，而不能扩大此受限网格代替。

### 直接运行真实 BFZI 三个独立 h/R/源弦段数扫描

新增 [`examples/bfzi_d2_convergence.py`](../../../examples/bfzi_d2_convergence.py)，不重写物理方程，逐工况调用**原来同一个** `examples.bfzi_d2_aphi` 三维真实闭合双端口求解入口。只改变一项数值维度，固定 BFZI 连续螺旋、真实馈回流、海水电导率、各自绝缘封装和频率，输出完整单工况 JSON 与一个简短 `sweep.json`：

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_d2_mesh_probe.py tests/test_bfzi_d2_structured_ports.py -q

# h：R=0.52m，闭合源弦段数4，近壁坐标步长0.12/0.08/0.06m
python -m examples.bfzi_d2_convergence --vary h --out runs/d2_h

# R：同一源与 h，外域半边长0.32/0.40/0.52m
python -m examples.bfzi_d2_convergence --vary R --out runs/d2_R

# source：同一外域与 h，BFZI 原连续圆角弦段数2/4/8
python -m examples.bfzi_d2_convergence --vary source --out runs/d2_source
```

可加 `--probe X Y Z` 和 `--solve-analytic-primary` 扩展到空间 B/E/J/q 和解析源直接参与的**导电材料 PDE**。`--summary-only` 仅让终端显示 2×2 复阻抗、互易差和各电流相位的端口与场功率差，`--output` 仍保存完整报告，不应为缩短日志丢失原始数据。

必须强调：这些脚本**当前仅完成代码接入，未在本次隔离环境真正求解完整双端口扫描**。相邻网格输出差异小不等于绝对误差小；真实无限海水高精度还需匹配 Elmer 独立参考、全物理源和外域，以及固定位置空间 B/E/J/q。**D2 仍 OPEN，G1 FAIL，神经教师冻结。**

## 已实现的主计算路径

- 新增 [`direct3d_structured_bfzi.py`](../../../sdfmpneo_bfzi/direct3d_structured_bfzi.py) 的 `axis_aligned_bfzi_tetra_mesh(scene,...)`：从 `BfziScene` **真实两只盒体姿态/尺寸**生成各坐标材料切面，保留盒内/盒外完全共形的六四面体长方体分割；允许独立选择外部 `far_step_m`、局部材料面间填充 `box_layers` 和近壁 `near_padding_m`。仅用于受限的轴对齐 h/R 初步计算。
- 更新 [真实双端口运行入口](../../../examples/bfzi_d2_aphi.py)：`--structured-axis-aligned` 时生成该实际物理网格；`--mesh` 时读取真正 Gmsh 旋转盒网格。两者之后使用**完全相同** `solve_closed_ports`（有向 Nédélec 边元、导电域 φ、同一闭合线源泛函加载与端口读出、同域真空相减），并可加 `--solve-analytic-primary`，使真实源的解析入射势参与材料 PDE，而不是仅事后替换 E。
- 加入 `--outer-radius`：用 `dataclasses.replace(scene,...)` 保留两套 BFZI 线圈、馈回流、盒体和电导率**完全不变**，独立调整有限外域立方体半边长，才可以在同一源情况下检查 R 外边界截断误差。
- 原有 `--probe X Y Z` 可查询两单位闭合端口和 `I=(1,0),(0,1),(1,±1),(1,i)` 的复 B/E/J 与平均 `q=0.5σ|E|²`。但这些量必须分别通过网格/源/外域收敛，不能被内部 Galerkin 功率恒等式替代验收。

## 这轮已实际计算的几何证据

采用 [已有 BFZI demo 真实几何](../../../examples/bfzi_mqs_demo.py)：每盒尺寸 0.20×0.20×0.03 m，盒体中心分别为 z=±0.13m，R=0.52m，海水 σ=5S/m；保持 `far_step_m=0.30`、`near_padding_m=0.02` 及两只盒体姿态不变，在隔离 Python 环境对**同一结构网格坐标生成算法**实算：

| 每盒材料间隔层数 | 全部四面体 | 空间节点 | 两只盒实际非导电体积 | 有限外立方体体积 |
|---:|---:|---:|---:|---:|
| 1 | 3,234 | 768 | 0.0024000000 m³ | 1.124864 m³ |
| 2 | 4,992 | 1,134 | 0.0024000000 m³ | 1.124864 m³ |
| 3 | 7,290 | 1,600 | 0.0024000000 m³ | 1.124864 m³ |

两个物理盒体的每个材料面均为四面体共享面，盒内 σ=0、海水 σ=5。上述结果**只核实网格的真实几何与体积**，并非已解出两端口复电磁量。尚未在当前环境将上述新网格链接到完整仓库的数值库并执行双端口 Galerkin，所以本轮不报告编造的 (Z,E,B,J,q) 数值。

## 立即可运行的主算法入口

在 Windows 工作站从项目根目录：

```powershell
git pull --ff-only
python -m pytest tests/test_bfzi_d2_mesh_probe.py tests/test_bfzi_d2_galerkin.py tests/test_bfzi_d2_primary_secondary.py -q

# 首先用粗但材料共形的真实 BFZI 两端口：无需 Gmsh
python -m examples.bfzi_d2_aphi --case bfzi --structured-axis-aligned --structured-far-step 0.30 --structured-padding 0.020 --structured-box-layers 1 --frequency 200000 --panels-per-corner 4 --solve-analytic-primary --probe 0 0 0 --output runs/d2_bfzi_axis_l1.json

# 独立增加材料/附近海水 h，不改动 BFZI 真实源/材料
python -m examples.bfzi_d2_aphi --case bfzi --structured-axis-aligned --structured-far-step 0.30 --structured-padding 0.020 --structured-box-layers 2 --frequency 200000 --panels-per-corner 4 --solve-analytic-primary --probe 0 0 0 --output runs/d2_bfzi_axis_l2.json

# 固定 local h 及闭合线源，独立改变外域半径 R
python -m examples.bfzi_d2_aphi --case bfzi --structured-axis-aligned --outer-radius 0.80 --structured-far-step 0.30 --structured-padding 0.020 --structured-box-layers 2 --frequency 200000 --panels-per-corner 4 --solve-analytic-primary --probe 0 0 0 --output runs/d2_bfzi_axis_R080.json
```

上述新入口尚不能宣称 200kHz 两端口数值高精度。以海水 σ=5S/m 计算 MQS 皮肤深度约 **0.503m**；而铜 σ≈5.8×10⁷S/m 的皮肤深度仅约 **0.148mm**。当前源仍是理想闭合细丝，**不含铜内集肤/邻近电流的三维有限铜域求解**；不得误用海水皮肤深度充当铜网格依据。

## 任意 SE(3) 的正式共形网格路径（仍需 Gmsh/Elmer 实跑）

```powershell
python -m examples.bfzi_d2_aphi --case bfzi --write-geo runs/bfzi_pose.geo --outer-radius 0.80 --mesh-size 0.07 --near-mesh-size 0.008 --near-padding 0.025
gmsh runs/bfzi_pose.geo -3 -format msh2 -o runs/bfzi_pose.msh
python -m examples.bfzi_d2_aphi --case bfzi --mesh runs/bfzi_pose.msh --legacy-single-tag-msh --outer-radius 0.80 --frequency 200000 --solve-analytic-primary --probe 0 0 0 --output runs/d2_bfzi_pose.json
```

Gmsh 独立生成任意两只合法 SE(3) OBB 的真实几何分域，近壁局部尺寸场仅控制网格 h 而不代替几何。正式关口仍为：三档独立材料/海水 h，三档外域 R，至少两档真实源弦段或连续源精度、空间 E/B/J/q 与 2×2 增量复阻抗全部收敛；再与独立 Elmer/TEAM-7 基准及真实 BFZI 共形场求解器核对。现阶段 **D2 OPEN、G1 FAIL、神经教师冻结，未使用 GitHub Actions**。
