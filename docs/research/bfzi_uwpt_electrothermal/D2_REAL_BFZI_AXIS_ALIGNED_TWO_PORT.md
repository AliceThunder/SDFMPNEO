# D2｜真实 BFZI 两闭合端口：无需 Gmsh 的轴对齐共形三维基线

**2026-11-08｜状态：代码已接通，网格体积已在隔离 Python 环境实算；新的完整仓库双端口 PDE 入口尚未在本环境实跑。D2 OPEN。**

## 研究目的和边界

本任务不是 TEAM-7 论文数值复刻，也不是用旧 G1 的五模态直接法修修补补。目标是让**真实连续 BFZI 圆角方形/圆形螺旋、明确闭合馈回流、两套独立绝缘盒**真正进入一套三维 Maxwell–MQS A–φ 求解器，得到双端口增量复阻抗及空间 B/E/J/q；再分别验证 h、R、闭合源弦段数和铜有限截面/外域误差。

原来 [Gmsh 双独立 SE(3) 盒共形网格](../../../sdfmpneo_bfzi/direct3d_mesh.py) 已实现几何构造，但当前隔离 Python 环境没有 Gmsh/Elmer，因此**真实 BFZI 两端口尚没有可靠的本机端到端电磁数值运行结果**。这轮增加一个**受限但确为同一 BFZI 物理模型的可运行路径**：只要两只盒体各自的姿态为世界坐标轴的符号置换（包括分别正交旋转 90°、180°），使用直接生成的非均匀长方体分割四面体，且材料界面完整落在单元面上，不需要外部 Gmsh。

对任意真正倾斜的 SE(3) 位姿，此路径**明确拒绝**，绝不把旋转盒套上阶梯体素或以近似包围盒代替真实材料。任意 SE(3) 高保真必须采用现有 Gmsh OpenCASCADE `BooleanFragments` 几何接口。

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
python -m examples.bfzi_d2_aphi --case bfzi --mesh runs/bfzi_pose.msh --outer-radius 0.80 --frequency 200000 --solve-analytic-primary --probe 0 0 0 --output runs/d2_bfzi_pose.json
```

Gmsh 独立生成任意两只合法 SE(3) OBB 的真实几何分域，近壁局部尺寸场仅控制网格 h 而不代替几何。正式关口仍为：三档独立材料/海水 h，三档外域 R，至少两档真实源弦段或连续源精度、空间 E/B/J/q 与 2×2 增量复阻抗全部收敛；再与独立 Elmer/TEAM-7 基准及真实 BFZI 共形场求解器核对。现阶段 **D2 OPEN、G1 FAIL、神经教师冻结，未使用 GitHub Actions**。
