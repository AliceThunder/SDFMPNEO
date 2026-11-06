# D2｜COMPUMAG TEAM-7 官方三维涡流 benchmark：几何、源与初始实算

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
