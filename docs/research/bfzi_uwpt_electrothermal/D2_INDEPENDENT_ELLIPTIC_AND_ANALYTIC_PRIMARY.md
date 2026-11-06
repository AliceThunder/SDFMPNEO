# D2：独立椭圆柱 A–φ 解析极限与真实解析入射场求解
> **2026-11-06 实测测试失败后完成的关键空间探针纠正：** 在加入解析源 PDE 的真空闭合方环专项测试时，`B` 已正确，但旧探针将“解析线源 + Whitney FEM 反作用”混合总场直接假定为单元仿射，导致真空 `E` 测点误差最大约 `1.38e-4 V/m`（本例两点）。这与 Maxwell PDE 数值积分或椭圆柱结果不同，是**空间查询后处理的具体实现错误**。现在在 [direct3d_probe.py](../../../sdfmpneo_bfzi/direct3d_probe.py) 中新增 `incident_potentials` 与 `frequency_hz`：对四面体 Gauss 的 `E_total` 先加回 `+iω A_source_exact(x_q)` 提取真正可仿射插值的有限元反作用电场，仅对该部分进行 Whitney 重构，然后在实际测点减去 `iω A_source_exact(x)`。正式 [analytic-primary 解](../../../sdfmpneo_bfzi/direct3d_primary_secondary.py) 自动传入真实闭合端口解析源。**本地 16 项合并 pytest 已实际通过（5.33 秒）**，含先失败后修复的真空源 `E/B`、有介质 `J/q`、两端口、非圆形 `phi`、椭圆柱网格极限及此前独立 Galerkin 基础；远端完整仓库测试尚无工作站日志。正式物理推导已同步 [第17节](../../../theory/sections/17_d2_aphi_reference_and_ports.tex)。仍必须通过材料反作用场的 h/R、趋肤与独立 Elmer/TEAM-7 收敛后才能确认高精度教师。
>


**2026-11-06｜方法验证已实算；正式 D2 仍为 OPEN。**

本工作**不是**为了重现 Shafiei/Wang 的某个图、表或曲线。这一步通过 Maxwell 方程建立**本项目自己的、独立可解析的三维导体物理基准**，验证导体电荷守恒、非零标量电势、空间 `E/J/q` 和真实 Joule 功率的 FEM 离散；再把已知真空线圈源直接带入 BFZI/TEAM-7 的材料 PDE，而非只在事后替换低阶全场结果。

## 1. 完整的低感应强度解析解：非圆形椭圆截面

令导电椭圆柱截面为 `(x/a)^2+(y/b)^2 ≤ 1`，高度为 `H`，材料 `sigma>0`、`mu=mu0`，受无限空间给定的均匀峰值轴向磁场 `B0 z_hat exp(+iωt)` 激励。已知真空入射势和导体边界条件：

```text
A0 = B0/2 (-y, x, 0)
E  = -i*omega*A0 - grad(phi)
div(sigma*E) = 0               (in conductor)
n . E = 0                      (on elliptical sides and z end faces)
```

忽略导电体自身造成的二次磁场，条件是无量纲感应强度 `alpha = omega*mu0*sigma*max(a,b)^2 << 1`。在这一严格极限，`phi` **不为零**：

```text
phi0 = -i*omega*B0*(a^2-b^2)/(2*(a^2+b^2)) * x*y
E0 = i*omega*B0/(a^2+b^2) * (a^2*y, -b^2*x, 0)
J0 = sigma*E0
q0 = 0.5*sigma*|E0|^2
P0 = sigma*omega^2*|B0|^2*pi*H*a^3*b^3/[8*(a^2+b^2)]
```

只有圆形特例 `a=b` 时，才退化为 `phi0=0` 与熟悉的环向感应电场。导体截面一旦椭圆化，省略 `grad(phi)` 即违反 `n·J=0`；这不是某种可凭经验忽略的附加“修正项”。

与有限长度圆柱涡流分离变量分析方法的相容性可参阅 Zhilichev (2018), *Analytical Solutions of Eddy-Current Problems in a Finite Length Cylinder*, DOI 10.7716/aem.v7i4.719：<https://www.aemjournal.org/index.php/AEM/article/view/719>。本项目以上极限公式在代码中独立定义，并不拟合该论文表格。

## 2. 本轮独立三维 FEM 实算

新文件：
- [椭圆柱共形四面体与真实三维解析基准](../../../sdfmpneo_bfzi/direct3d_elliptic_limit.py)
- [物理 E、phi 与 h 收敛专项](../../../tests/test_bfzi_d2_elliptic_limit.py)
- [可直接生成五档实验结果的命令行](../../../examples/bfzi_d2_elliptic_limit.py)

参数固定：`a=0.12 m`、`b=0.07 m`、`H=0.10 m`、`sigma=4 S/m`、`f=1000 Hz`、`B0=0.001 T`；`alpha=4.5479137080e-4`。导体整个体积用最低阶 Nédélec + 导体 P1 `phi`，四面体与材料边界共形为**内接椭圆多边形**；反作用矢势在有限柱面上的切向分量取零。本算例没有独立外空气域，在 `alpha<<1` 时是合适的极限算法检验，不能据此证明任意高导电/高频空间场正确。

解析真实椭圆柱平均功率 `P0 = 1.904408726789681e-4 W`；二阶四面体求积对二次型能量积分精确。分辨率变化得到：

| 截面边数 / 径向层 / 轴向层 | 四面体数 | 全场 E 相对体积 L2 误差 | FEM Joule 相对 P0 偏差 |
|---|---:|---:|---:|
| 8 / 1 / 2 | 48 | 29.2659% | −11.7999% |
| 16 / 2 / 2 | 288 | 15.0057% | −2.8845% |
| 24 / 3 / 3 | 1,080 | 10.3454% | −1.2152% |
| 32 / 4 / 4 | 2,688 | 7.8798% | −0.6647% |
| 48 / 6 / 4 | 6,336 | 5.2893% | −0.2915% |

这一序列**没有借助端口自洽等式**，而是真正把计算出的空间复电场/体积损耗与独立解析式比较。网格加密时 E 和 Joule 误差持续改善；椭圆柱仅以多边形逼近，其几何积分偏差也随截面边数减小，因此这里不伪称已独立分离出所有连续 PDE 误差。

对于本例，若强行取 `phi=0`，即只使用 `E=-iωA0`，相对解析椭圆电场的体积 L2 误差约 **56.55%**；这就是必须保留三维 A–φ 的明确物理证据。

同一隔离环境内，将新椭圆基准四项测试与旧本地三维核心/源/主次场测试合并，实跑 **13 passed**；本次远端仓库新增测试和主程序的全量组合尚未在用户工作站取得日志，不以本地原型替代完整正式 CI/验收（本项目禁止 GitHub Actions）。

## 3. 把解析源真正带入 Maxwell 材料方程

上一轮已建立 `direct3d_galerkin.solve_closed_ports`：用相同的闭合电流路径边元载荷 `L` 进行加载和端口读取，有严格离散互易与功率闭合。这是**正确的离散结构检验**，但粗网格对真实解析线源的 `B/E` 可能有几十个百分点误差。

此前还使用过`E_corrected=-iω A_analytic+(E_h,medium-E_h,vacuum)` 的**后处理替换**。它使已知真空源的解析部分准确，却没有重新计算材料反作用对新源场的响应，因此**并不自动满足被修正源下的 Maxwell 导电弱式**。不允许以它的 `q` 直接训练高精度热代理。

新增 [直接解析源主次场求解器](../../../sdfmpneo_bfzi/direct3d_primary_secondary.py) 与 [方法专项](../../../tests/test_bfzi_d2_primary_secondary.py)：

```text
A_total = A_source_exact + A_reaction
curl(mu0^-1 curl A_reaction) +
       sigma*(i*omega*A_reaction + grad(phi))
         = -i*omega*sigma*A_source_exact
div[sigma*(i*omega*(A_source_exact+A_reaction)+grad(phi))]=0
E_total = -i*omega*(A_source_exact+A_reaction)-grad(phi)
B_total = B_source_BiotSavart + curl A_reaction
q = 0.5*sigma*abs(E_total)^2
```

**此时真实 `A_source_exact` 在 FEM 方程装配时参与材料激励**，而不只是用完求解器后替换 `A` 的值。支持两条真实闭合 BFZI 端口、多股加权环、任意位姿外部场；其结果包含材料体积 Joule、从真实源闭合路径读得的反作用复阻抗、有限盒体积重叠阻抗。后两者受有限边界与离散误差影响，不强制宣布一致。

已将物理分账对照作为选项接入 [BFZI 三维入口](../../../examples/bfzi_d2_aphi.py) 和 [TEAM-7 入口](../../../examples/bfzi_d2_team7.py)：`--solve-analytic-primary` 会新增直接求解的物理 `E/Joule`，并以该解的完整 `B/E/J/q` 回答空间探针，不再把事后源场替换称为重新满足材料方程的解。

固定 50 Hz、TEAM-7 带孔铝板、小外盒、2×2 匝截面积分下，三个**尚未网格收敛**的算法分账在同一固定几何/源近似下：

| 全部四面体 | Galerkin 总场离散 Joule (W) | 解析源直接驱动材料 PDE 的 Joule (W) | 事后替换已求总场源的 Joule (W) |
|---:|---:|---:|---:|
| 750 | 6.9325 | 21.5447 | 46.9938 |
| 2,304 | 6.8338 | 13.1625 | 23.5577 |
| 5,082 | 7.0913 | 10.8991 | 17.4466 |

这是一个**尚未收敛、不同离散源处理会造成很大误差**的真实警示，不是三份可任选其一的“高精度解”，更不能用“其中两列端口功率相等”来证明绝对精度。真正的下一个任务是以物理一致的解析入射场方法做 h/R/源路径和截面加密，独立 Elmer/TEAM-7 B/J 比对，最后再用于任意 SE(3) 双封装海水物理直接参考。

## 4. 直接复现命令

```powershell
git pull --ff-only

python -m pytest tests/test_bfzi_d2_elliptic_limit.py tests/test_bfzi_d2_primary_secondary.py tests/test_bfzi_d2_fields.py -q

# 5 档独立解析 PDE 物理极限；保存小型数值证据
python -m examples.bfzi_d2_elliptic_limit --output runs/d2_elliptic.json

# TEAM-7 独立解析源材料 PDE，与 Galerkin 结果同时报告（只用于粗网格方法对照）
python -m examples.bfzi_d2_team7 --coarse-smoke --refinement-level 1 --frequency 50 --corner-panels 5 --solve-analytic-primary --summary-only

# 真实双封装 BFZI（须工作站先有 Gmsh 共形网格）
python -m examples.bfzi_d2_aphi --case bfzi --write-geo bfzi_d2.geo --mesh-size 0.025
gmsh bfzi_d2.geo -3 -format msh2 -o bfzi_d2.msh
python -m examples.bfzi_d2_aphi --case bfzi --mesh bfzi_d2.msh --solve-analytic-primary --probe 0.2 0 0 --output runs/bfzi_d2_compare.json
```

**当前验收：D1 限定方法已研究；D2 A–φ 独立椭圆柱低感应强度物理极限获得实证。TEAM-7 高导电材料场、空间 J/B、外域与真实 BFZI 任意位姿的高保真仍 OPEN；G1 FAIL，正式电磁热神经教师冻结。禁止 GitHub Actions。**
