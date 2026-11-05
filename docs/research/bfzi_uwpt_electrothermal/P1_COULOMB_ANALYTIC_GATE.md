# P0 限定验收冻结 → P1 原 Coulomb 磁能解析基准

**2026-11-05 UTC；P0 限定源项 PASS；P1 球形 Taylor 基准 PASS、真实双盒几何矩 PASS、两紧支海水涡流原核自／互项 PASS（92 passed）；P1-C 正式海水 curl 基的全域原核 L 已接入但待本机复验，完整 G1 仍 FAIL。旧 split 仍 FAIL。** 本文件是第一阶段已取得的实际证据及第二阶段第一道可证伪验收，不是整个 BFZI 海水代理的完成声明。

## 1. P0 的正式阶段结论（根据本机 83 passed）

[最新 P0 原始日志](runs/p0/latest_full.log) / [摘要](runs/p0/latest_summary.md)：代码 `21daadf`，**83 passed**。同一 BFZI 圆角方形连续双端口和每端口 84 个海水近面、近棱、近角、远场探针：

- 生产源为**直边和全部回流解析有限直线 C/B，真实圆角 Gauss**，其 4 阶源 C 误差 **7.28e-5**、B 误差 **4.27e-4**、棱角 B 误差 **7.48e-4**、最坏点峰值归一 B 误差 **4.66e-4**，均低于已预先固定的 0.1% 相对门槛。
- 12 阶曲角源相对 B 误差约 **1.28e-10**；原 4 阶源的归一化 `div C` 数值残差约 **1.06e-4**，随曲角求积加密收敛到 12 阶 **8.86e-11**。**点线元有限和本身并非严格闭合的分布电流。**
- 独立构造的**闭合折线解析源**对任意折线面板数都有 `div C` 机器精度残差，16/32/64 段曲角的 C 误差为 **4.664e-4 / 1.167e-4 / 2.917e-5**，B 误差为 **3.901e-4 / 9.760e-5 / 2.440e-5**，显示接近 **二阶的真实几何收敛**。这验证了源项的数值与闭合理论一致，不代表有限截面铜线邻近效应已建模。
- 线圈 XY 与 Z 封装保守裕量分别约 **23.4 mm、14.2 mm**，回流距盒壁保守裕量 **6.2 mm**。因为盒为凸集，回流多段折线在端点全在盒内部时可判定穿盒与否；**尚未建立有限线径的全路径自接触/回流与螺旋碰撞证书**。
- `geometry.py` 已拒绝可令原 BFZI 圆角有向切线反转的参数；圆形连续特例仍保留。已实测这批新几何检查的 83 项测试，但它不是跨参数域的线碰撞证明。

**冻结的 P0 适用范围：** 两只互不相交的非磁性实心绝缘长方体；在封装内部、按 BFZI 真实连续线圈＋显式回流构成的**指定理想闭合单位线源**；静止均匀线性非磁性海水；峰值相量 `exp(+jωt)`；源外场 `C/B` 由未软化的真空核定义。P0 是**给定示例和上述理想源物理问题的阶段性数值 PASS**，不是对任意线径、任意材料、任意姿态的统一误差证书。人工外球是有限域近似，不代表无穷海水。未进入 P1 前停止随意变更源、回流、端口符号和相量约定。

## 2. P1 首先验证的严格数学物理问题

考虑**半径 R 的均匀导体球** \(\Omega=B_R(0)\)。先把真实 BFZI 双盒及其复杂角/面边界**有意排除在本关之外**，以隔离 Coulomb 核和六维双重积分是否正确。选取三个明确的真实无散度、零法向电流场：

\[
\mathbf W_i(\mathbf x)=\mathbf e_i\times\mathbf x,\quad i=1,2,3,
\qquad
\nabla\cdot\mathbf W_i=0,\quad
\mathbf n\cdot\mathbf W_i|_{\partial B_R}=0.
\]

它们有满足零迹条件的旋度势：

\[
\mathbf F_i(\mathbf x)=\tfrac12(R^2-|\mathbf x|^2)\mathbf e_i,
\qquad \nabla\times\mathbf F_i=\mathbf W_i,\quad \mathbf F_i|_{\partial B_R}=0 .
\]

定义**原始、未软化、无球云近似**的真实 Newton–Coulomb 原核：

\[
L_{ij}=\frac{\mu_0}{4\pi}\iint_{B_R\times B_R}
\frac{\mathbf W_i(\mathbf x)\cdot\mathbf W_j(\mathbf y)}
{|\mathbf x-\mathbf y|}\,d\mathbf y\,d\mathbf x .
\]

每一分量是线性函数。利用 \(\Delta_{\mathbf x}|\mathbf x-\mathbf y|^{-1}=-4\pi\delta(\mathbf x-\mathbf y)\)，以及内外球的 \(l=1\) 多极匹配，可以**不做任何数值六维积分**得到：

\[
\mathbf U_j(\mathbf x)=\int_{B_R}\frac{\mathbf W_j(\mathbf y)}{|\mathbf x-\mathbf y|}\,d\mathbf y
=\left(\frac{2\pi}{3}R^2-\frac{2\pi}{5}|\mathbf x|^2\right)\mathbf W_j(\mathbf x), \quad |\mathbf x|<R.
\]

由各向同性和体积精确矩
\[
\int_{B_R}|\mathbf W_i|^2\,dV=\frac{8\pi R^5}{15},\qquad
\int_{B_R}|\mathbf x|^2|\mathbf W_i|^2\,dV=\frac{8\pi R^7}{21},
\]
获得可逐分量独立核算的**精确三模态 Gram 矩阵**
\[
\boxed{L_{ij}^{\rm exact}=\delta_{ij}\frac{16\pi\mu_0 R^7}{315}.}
\]

注意这里使用**未经归一化**的多项式基 \(W_i\)；因此矩阵数值含有模式归一化尺度，**不是以 H 为单位的实体 BFZI 端口电感**。该球场有严格的 Maxwell 约束、非零真实磁能、已知对角/非对角项，且源与海水边界不包含任何 BFZI 复杂度。

## 3. 代码与独立 P1 日志

- `examples/bfzi_p1_coulomb.py` 实现球 Gauss 体积规则、上述无散度三模式、解析内场 `vortex_green_potential`、解析矩阵 `exact_vortex_green`，并**直接调用现有** `green_coulomb.assemble_coulomb_split`（未修改主 Coulomb L 的求积核）。
- 使用 `base/x_only/y_only/both` 四组**固定同一物理球和模式**的分轴细化；另外 `near_radius_over_R=0.25/0.55` 固定 x/y 节点对比物理核分区半径。输出 `p1_exact`、`p1_ball`、`p1_partition`；**未知结果不能事先说已通过**。
- `tests/test_bfzi_p1_coulomb.py` 仅检查解析球势的独立体积矩等式及一个小的原核拆分计算，避免堆砌审核代码。

本机运行（在当前 BFZI 分支）：

```powershell
git pull --ff-only
python run.py --mode bfzi-p1-coulomb --with-tests
```

运行摘要和详细日志将单独自动版本化到 `runs/p1/latest_summary.md` 与 `runs/p1/latest_full.log`，**不会覆盖 P0 或 G1 旧证据，也不会启用 GitHub Actions**。

## 4. P1 首轮解析磁能实验：已确认原算法不满足精度

用户本机 [P1 原始日志](runs/p1/latest_full.log)，代码 `ba16d6a`，**85 passed**（只代表运行和针对性程序单测）；固定球半径 1m，理论原核三模态对角磁能均为 `2.00525295768e-07`（未经电流归一化的模式积分量，不能当成端口亨利）。

| y/x 求积规则 | x 节点 | y 节点 | L Frobenius 真值相对误差 | 最大对角误差 |
|---|---:|---:|---:|---:|
| base | 120 | 240 | **0.02120667** | 0.0322072 |
| x_only | 480 | 240 | **0.01556394** | 0.02141865 |
| y_only | 120 | 756 | **0.01146122** | 0.0126181 |
| both | 480 | 756 | **0.006144858** | 0.008792633 |

固定积分规则改变分区球半径至 0.25/0.55 R，`L11` 变化相对解析真值的差为 **0.004230034**。矩阵非对角/原矩阵反对称部分接近机器零，**证明对称性完全不能作为该积分正确性的证书**。预先指定的 0.1% 球基准：整体、对角、分区稳定性均 FAIL（只有非对角偏差 PASS）。**本轮严禁把 85 passed 当成算子收敛成功。**

## 5. 从理论直接拆出 y-远区积分误差，而非继续试采样数

在已有 `assemble_coulomb_split` 中，局部球半径是 `a(x)=min(a_max,0.6 dist(x,∂Ω))`；越靠近导电球边界，`a(x)` 越趋于零。尽管完整连续恒等式成立，这使内部离散的 `y` 积分必须处理越来越狭窄的 `(1-χ_a)/r` 过渡层，低阶全局球 Gauss 节点难以均匀分辨。

这个球基准可以**不用局部数值求积**准确分解现有算法。现有 C² 径向窗口的精确一阶矩是

\[
\int_0^1 t\,\chi(t)\,dt=\frac27.
\]

因 `W_i(y)=W_i(x)+e_i×(y-x)` 是线性的，完整局部球全部包含在导体内，奇对称项积分严格为零，因此

\[
\int_{B_{a(x)}(x)}\frac{\chi(|y-x|/a)}{|y-x|}\,W_i(y)\,dy
=\frac{8\pi a(x)^2}{7} W_i(x).
\]

另一方面，`vortex_green_potential(x,R)` 已提供**整个 y 积分的解析真值**，因此 `x` 节点上的完整真实电感可独立直接积分。新 `p1_decomp` 指标将报告：

- `x_exact_rel`：仅 x 节点对解析内势的外积分误差；
- `far_y_rel`：在完全准确的局部球积分已代入后，仅 y-远区规则对解析 x-内势的相对偏差；
- `near_local_l11_rel`：原程序局部积分与上述解析局部球对 L11 的相对差；
- `global_const_sub_rel`：以下独立未软化奇异减除方法的矩阵相对真值误差。

这会用明确的解析参照把误差归因到 x、y-远区、局部球三项，而不是换一个更大的 6D 点数希望偶然收敛。

## 6. 真 Coulomb 核的全域常量奇异减除

不软化原始核，对于连续、局部 Lipschitz 的模式 `W`，恒等式

\[
\int_\Omega\frac{W(y)}{|x-y|}\,dy
= W(x)\underbrace{\int_\Omega\frac{dy}{|x-y|}}_{S_\Omega(x)}
+\int_\Omega\frac{W(y)-W(x)}{|x-y|}\,dy
\]

精确成立。后一个核在 `y→x` 时有界，虽然可能有方向相关极限，**不是简单光滑到可以忽略节点重合**。新实现 `green_coulomb.assemble_coulomb_subtracted` 接受**独立可信的** `S_Ω(x)`，对剩余实际 `1/r` 残差作积分；要求两套积分节点不重合，绝不加入 Plummer、球云或人为平滑的自项。

对于这次解析球，`S_ball(x)=2π(R²-|x|²/3)` 有独立精确答案，可真正验证该方法。对**真实 BFZI 两盒外海水域绝不能直接套用球公式**。一个后续可实施、仍然保持真实几何的路径是由散度定理获得

\[
S_\Omega(x)=\frac12\int_{\partial\Omega}
\frac{n_y\cdot(y-x)}{|y-x|}\,dS_y,
\]

也就是仅含外球和两个旋转盒真实表面的几何标量势；但其近壁表面项本身也需独立精度验证，**目前尚未实现并通过**。此方案仅在 `S_Ω` 已经独立正确时才能提升实际 BFZI 的原核积分，不能把球上有效的公式当成双盒已通过。

本轮新增的 `p1_decomp` 和 `p1_sub_gate` **尚待本机实测**；0.1% 的既定阈值不变。除对现有 Coulomb 模块增加这个明确的、可复用的数学恒等式外，没有修改 BFZI 源、R/K、电流基、外球边界或神经教师。

## 7. 恢复后的理论与主代码推进：一阶 Taylor 奇异减除＋非线性解析涡流

**恢复节点：** 旧 `assemble_coulomb_split` 的解析球磁能误差为 0.6145%（既定目标 0.1%，FAIL）；之前已提交的全域常值减除，用独立短程数值核算的相同 480×756 球 Gauss 节点得到约 **0.1497%**，有改善但仍未过门槛。这不是新的本机 GitHub P1 日志，也不能把本地短程计算当成用户的完整 `--with-tests`。

因为仅将 `W(y)-W(x)` 相减后，残差除以 `|x-y|` 虽有界，仍具有方向相关极限，对非线性曲率大的模式并不平滑。对可微且梯度有意义的 `W`，进一步精确分解：

\[
\begin{aligned}
U_W(x)&=\int_\Omega\frac{W(y)}{|x-y|}\,dy\\
&=W(x)S_\Omega(x)+\nabla W(x)\,M_\Omega(x)
 +\int_\Omega\frac{W(y)-W(x)-\nabla W(x)(y-x)}{|x-y|}\,dy,
\end{aligned}
\]

其中 (S_\Omega(x)=\int_\Omega |x-y|^{-1}dy)，新引入的**实际几何一阶矩**为

\[
M_\Omega(x)=\int_\Omega\frac{y-x}{|x-y|}\,dy
=\int_{\partial\Omega}|x-y|\,n_y\,dS_y.
\]

右端边界式来自散度定理，不假设球形、不修改 (1/r) 核。若 `W∈C²`，积分余项在 `y→x` 时为 `O(|y-x|)`，比原 (1/r) 核及仅减常值的有限方向极限更规整。它是**严格数学等价**，但只有 `SΩ` 和 `MΩ` 均独立可信时才是真正的无偏数值方案。

球内第二个**非线性、无散度、零法向、零迹旋度势**基准：

\[
W_i^{(3)}(x)=|x|^2(e_i\times x),\qquad
F_i^{(3)}(x)=\tfrac14(R^4-|x|^4)e_i,
\]
\[
U_i^{(3)}(x)=\left(\frac{\pi R^4}{3}
-\frac{\pi|x|^4}{7}\right)(e_i\times x),
\quad
L_{ij}^{(3)}=\frac{16\pi\mu_0R^{11}}{693}\,\delta_{ij}.
\]

这些真值可从 Poisson 方程 `ΔU=-4πW`、球外 `l=1` 衰减匹配和精确球体积矩直接推导，**不会因为当前数值算子改变而改变**。该电流的 Taylor 余项**不为零**，因此检验的是实际核余项求积，而不是只让线性场项望远镜消去。

对于此球基准：

\[
S_B(x)=2\pi\left(R^2-\frac{|x|^2}{3}\right),\qquad
M_B(x)=\frac{4\pi}{15}(|x|^2-5R^2)x.
\]

已把 `assemble_coulomb_subtracted` 扩展为可选 `gradient` 和 `first_moment` 配对参数；没有更换 BFZI 源、原 1/r 核、R/K、紧支电流字典或人工外球。新增 `p1_taylor` 输出**线性解析基准、非线性球场的仅减常值与一阶减除误差**。在独立短程数值计算（固定 480×756 球规则）中，非线性基准仅减常值的矩阵相对误差约 **2.453e-3**，一阶减除约 **1.596e-4**，解析外积分自身误差约 **5.91e-6**；这些数字须再由新 GitHub 日志复验。

**明确的使用边界：** 真实双旋转盒海水域的 `SΩ,MΩ` 可以由真正的外球和盒壁几何边界积分定义，但近壁数值精度尚未核实；**绝不拿球的 (S_B,M_B) 代替真实双盒域**，也不据此恢复神经教师。

## 8. P1 第二步：真实双绝缘盒海水域的几何势与一阶矩（新代码待本机运行）

球形解析基准的一阶减除已经在本机 87 passed、非线性源相对磁能误差 **1.59619e-4**（0.01596%）中达到原定 0.1% 门槛；旧分区算法仍然 FAIL，不能因新方法通过单个球基准就宣布实际 BFZI 双盒原核 L 已经收敛。

本轮不再使用球的 `S_B/M_B` 近似真实海水域；已为**实际的两只 SE(3) 旋转长方体绝缘封装和外球**实现严格几何积分。有限海水域

\[
\Omega=B_R(c_0)\setminus(\overline B_1\cup\overline B_2)
\]

的真 Coulomb 几何矩始终按积分域线性拆分：

\[
S_\Omega=S_{B_R}-S_{B_1}-S_{B_2},\qquad
M_\Omega=M_{B_R}-M_{B_1}-M_{B_2}.
\]

对于位于外球内部的 \(x\)，球的两项仍可解析：
\[
S_{B_R}(x)=2\pi\left(R^2-\frac{|x-c_0|^2}{3}\right),
\quad
M_{B_R}(x)=\frac{4\pi}{15}\left(|x-c_0|^2-5R^2\right)(x-c_0).
\]

**矩形盒无需有限元或体积高斯点。** 对任一面的局部正交坐标 \((u,v)\) 和有符号面间距 \(w\)，令 \(d=\sqrt{u^2+v^2+w^2}\)、\(z=|w|\)。两个完全解析的矩形面原函数为

\[
F_{-1}(u,v;w)=u\,\operatorname{asinh}\frac{v}{\sqrt{u^2+w^2}}
+v\,\operatorname{asinh}\frac{u}{\sqrt{v^2+w^2}}
-z\,\operatorname{atan2}(uv,zd),
\]

\[
F_{+1}(u,v;w)=\frac{uvd}{3}
+\frac{u(u^2+3w^2)}{6}\operatorname{asinh}\frac{v}{\sqrt{u^2+w^2}}
+\frac{v(v^2+3w^2)}{6}\operatorname{asinh}\frac{u}{\sqrt{v^2+w^2}}
-\frac{z^3}{3}\operatorname{atan2}(uv,zd).
\]

在面矩形的四个角点作符号交替差，直接精确得到
\[
I_{-1}=\iint_{\mathrm{face}}\frac{du\,dv}{d},\qquad
I_{+1}=\iint_{\mathrm{face}}d\,du\,dv.
\]
依据真实边界散度定理（注意内盒是**从海水域减去**）：
\[
S_{\mathrm{box}}(x)=\frac12\sum_f
[\mathbf n_f\cdot(\mathbf y_f-x)] I_{-1,f},\quad
M_{\mathrm{box}}(x)=\sum_f\mathbf n_f I_{+1,f}.
\]
实际计算先在每只盒的局部坐标完成，再用该盒的刚体旋转矩阵把向量矩转回世界坐标，支持两只盒**独立任意合法姿态和尺寸**。这两个积分量与电流基、源、频率无关，**没有任何 1/r 软化或矩形表面网格求解**。

主代码：`sdfmpneo_bfzi/coulomb_geometry_moments.py` 的 `box_coulomb_moments`、`ball_coulomb_moments`、`sea_coulomb_moments`。

**独立验收，不用解析面原函数自我测试：** 新 `examples/bfzi_p1_geometry.py` 对每端口真实封装的**盒面中心附近／棱附近／角附近**分别选取 2.5mm 海水探针，使用另一种实现的**自适应分片正权二维高斯曲面积分**逐面比较标量和向量矩。这套独立算法不调用解析矩形原函数；预先将每种矩在各个近壁探针上的最大相对差目标固定为 **<1e-8**。同时测试解析盒面公式与独立三维盒体 Gauss 积分的一致性，以及 SE(3) 刚体协变。日志为 `p1_geom_box`、`p1_geom_wet`、`p1_geom_gate`，与现有 P1 原球矩阵日志共同保存在 `runs/p1/`。

**新关口的意义与限制：** 这使真实双盒域所需的 `S_\Omega,M_\Omega` 成为精确几何算子而不是未经认证的拟合或体积软化；但它**不提供真实双盒合规连续电流模式的 Taylor 余项求积真值**，也不完成其自/邻 Coulomb L 磁能、人工外球的无限域外推。新增代码和阈值目前**待用户本机运行**；在实际日志验证前不可宣布几何矩 PASS，G1 继续 FAIL。

## 9. P1 实测双封装几何 PASS；启动真实海水内两紧支电流的自/互 L（待本机验证）

**最新用户本机证据：** 2026-11-05 UTC，代码 `585bf12`，[P1 原始日志](runs/p1/latest_full.log)，**90 passed**。对两只真实绝缘盒附近的面／棱／角、每盒 3 个湿区探针，独立分片二维曲面 Gauss 对照已确认解析盒几何矩，预先确定的 \(10^{-8}\) 阈值全部通过：

| 几何矩 | 所有盒壁探针的最大相对误差 | 门槛 |
|---|---:|---:|
| \(S_B\) 的解析盒体势 vs 独立分片二维面 Gauss | \(5.509265\times10^{-16}\) | \(10^{-8}\) PASS |
| \(M_B\) 的解析盒体一阶矩 vs 独立分片二维面 Gauss | \(1.63309\times10^{-15}\) | \(10^{-8}\) PASS |

真实湿区 \(S_\Omega=S_{\rm outer}-S_1-S_2\) 与 \(M_\Omega\) 用精确几何加减构造；此处 PASS 只针对几何矩，不是电流磁能 \(L\) 的通过声明。球域中的非线性涡流一阶 Taylor \(L\) 误差仍为 \(1.59619\times10^{-4}\)（原定 \(10^{-3}\) 门槛 PASS），**旧 `assemble_coulomb_split` 的 `both` 误差仍为 0.6145%，不能据此洗白旧算法**。

### 9.1 双盒真实海水域内两个真正合法的紧支电流

为了给原核 \(L_{11},L_{22},L_{12}\) 提供均有**独立解析真值**的下一关口，而不退回到随机调节求积点，在两只**实际** BFZI 绝缘封装面对的海水中，各放一个不接触盒壁的半径 \(a\) 球形测试支撑（**它不是等效球云导线，不用于替换连续 BFZI 端口源或 Coulomb 核**）。每球中心 \(c_p\) 随该封装独立 SE(3) 位姿更新；使用外侧球内的光滑涡流

\[
W_p(x)=\Big[1-\frac{|x-c_p|^2}{a^2}\Big]_+^2
(e_p\times(x-c_p)).
\]

该场在球外严格为零，在支撑边界自身及一阶导数连续为零，且

\[
\nabla\cdot W_p=0,\qquad
n_{\rm box}\cdot W_p|_{\partial B_1\cup\partial B_2}=0,\qquad
n_{\rm outer}\cdot W_p|_{\partial B_R}=0.
\]

可用
\[
F_p(x)=e_p\int_{|x-c_p|}^a
s\big(1-s^2/a^2\big)^2 ds
\]
（球外取零）构造其零迹旋度势：\(W_p=\nabla\times F_p\)。因此两个模式构成真正双封装海水域的**合法连续电流**，不是球域有错误边界法向的模式。现有默认例中每支撑球半径 \(a=45\,\mathrm{mm}\)，离盒壁最近 \(2.5\,\mathrm{mm}\)，两个支撑不重叠。真实磁能双积分只需在电流非零的两支撑球上积累，因为其它海水处 \(W=0\)。

### 9.2 两个自项和非零互项的解析真值

对任一 \(r=|x-c_p|<a\)，原始 \(\int W(y)/|x-y|\,dy\) 精确为

\[
U_p(x)=4\pi a^2
\left(\frac1{18}-\frac{r^2}{10a^2}
+\frac{r^4}{14a^4}-\frac{r^6}{54a^6}\right)
(e_p\times(x-c_p)).
\]

球外的同一物理电流只有严格的偶极项：

\[
U_p(x)=D_p\frac{e_p\times(x-c_p)}{|x-c_p|^3},
\qquad D_p=\frac{32\pi a_p^5}{945}.
\]

代入未经软化的原 Coulomb 磁能双线性型，得到

\[
\boxed{L_{pp}^{\rm exact}=
\frac{2048\pi\mu_0 a_p^7}{1216215}}
\]

以及两球不相交时**非零**的互项（令 \(n=(c_1-c_2)/|c_1-c_2|\)）：

\[
\boxed{L_{12}^{\rm exact}=
\frac{\mu_0D_1D_2}{4\pi |c_1-c_2|^3}
\left[3(e_1\cdot n)(e_2\cdot n)-e_1\cdot e_2\right].}
\]

因此该基准能同时检验真实双域中的 \(L_{11},L_{22},L_{12}\) ，并排除「矩阵对称但整体错」的假通过。这里的模式未做安培端口归一化，所有结果只是相应合规测试模式的原 Coulomb 能量双线性型。

新代码 `examples/bfzi_p1_wet_vortices.py` 直接用已实现的**原核** `assemble_coulomb_subtracted`，严格固定 \(W_p\)，分别对 `base` 和 `both` 使用相互独立正权 \(x/y\) 体积规则。这里减除使用**两只测试电流支撑球之并集**的精确 \(S,M\)，而不是完整海水域的几何矩：因为测试电流在支撑外严格为零，算子的原双积分本来就在支撑并集；此做法不会错误地把球几何矩用于全海水积分。该关测试整个物理域内一个**严格合法且具解析真值的两维紧支子空间**，**不检验遍布海水、贴近盒壁的 229 模态原核积分**。

`p1_wet_L` 将报告真值和数值 \(L_{11},L_{12}\)、对角最大误差、非零互项相对误差及整体 Frobenius 相对误差。**首次运行前冻结各项 0.1% 门槛**，`p1_wet_gate` 分别报告 `self_pass`、`mutual_pass`、`matrix_pass`。理论自项在外积分径向是 14 次多项式，因此细规则选取径向 Gauss **8 节点**（可准确积分到 15 次），不是根据日志事后任意调点。内积分仍需单独验证，不能把外规则精确误当成整体保证。

新增程序和针对性测试已提交，**尚无用户本机该双电流自/互项运行日志**。即使获得 PASS，下一关仍须使用真实带盒壁效应的合规 curl 字典，完成全海水域边缘积分和基收敛；不能提前宣布 G1 成功。

## 10. P1-B 已实测通过；P1-C 接通现有海水 curl 字典的真实原核 L

### 10.1 最新本机 P1-B 验收（代码 `79c9eed`）

用户本机 [P1 原始日志](runs/p1/latest_full.log) 为 **92 passed**；前一轮两只实海水紧支涡流的两个对角自项与非零互项已满足先定 0.1% 真值目标：

| 真实 (1/r) 项 | 原积分规则 | 独立细积分规则 |
|---|---:|---:|
| 最大自项相对误差 | 0.5535874% | **0.02382213%** |
| 非零互项相对误差 | 7.712203% | **1.37751e-8%** |
| Frobenius 矩阵相对误差 | 0.5620203% | **0.02382024%** |

`p1_wet_gate` 明确为 `self_pass=True, mutual_pass=True, matrix_pass=True`。因此**在真正双盒海水域中，两个不接触盒壁、满足 Maxwell 电流守恒和全部边界零法向条件的紧支涡流所对应的原 Coulomb L，自项和非零互项已通过解析真值验证**。它不覆盖旧 split 算子、全海水流场、任意电流字典、频率响应或端口 K/R。

### 10.2 从紧支基准接到真实 BFZI 海水电流

为避免把两团测试涡流的通过扩大解释为全海水磁算子准确，现已将一阶 Taylor 奇异减除正式接入已有的 `SeaCurrentBasis` 中**非源引导的物理 curl 模式**。对每个真实模式

[
F_m(x)=g_m(x)a_m,quad
W_m(x)=\nabla g_m(x)\times a_m,
quad
\partial_k W_m(x)=
\big(\partial_k\nabla g_m(x)\big)\times a_m.
]

(g_m) 保留该模式的原高斯／紧支窗口、真实两只独立旋转绝缘盒的距离包络、人工外球的边界包络，因而这是**与旧求解器相同的实际物理电流模式**，没有为此引入新的试验窗口或导线几何。

`sea_basis_differential.evaluate_nonguided_curl_jacobian` 利用链式法则和显式 Hessian，对真实旋转盒外距离、外球包络、每个原模式窗口求出 (\nabla W_m)；在盒棱附近距离函数只有分片光滑，导数按各光滑分片定义，必须另外控制近棱角数值误差。

`green_coulomb.assemble_basis_coulomb_taylor` 将完整真实海水域解析
(S_\Omega,M_\Omega)（不是 P1-B 的两支撑球矩）与上述真实电流及 Jacobian 同时交给未软化的 `assemble_coulomb_subtracted`，原核和连续物理弱式都不变。当前**明确拒绝源引导** `curl(g C_{port})` 模式：其 (C_{port})、(\nabla\times C_{port}) 的空间导数还没有独立实现与验证，不能以任意有限差分伪造精确 Taylor 项。

### 10.3 下一轮验证的准确边界

新增 `examples/bfzi_p1_full_sea_taylor.py` 从旧 27 维基固定抽取 5 个模式（两端口真实盒面模式＋一个原始全局模式），在**外球减去两只真实盒的完整海水体积**上构造独立正权 (x/y) 规则，输出 `base/x_only/y_only/both` 的完整 (L) 矩阵范数、迹、最小特征值、原未对称双积分偏斜以及分轴变化。所选模式、几何、材料、原核完全固定；它不改动旧 27 维或 229 维基，不参与现有训练。

新 `p1_full_sea`／`p1_full_sea_change` **只是首次真实海水 curl 模式上的求积收敛诊断**：由于尚无该五维实海水 (L) 的独立连续真值，**不能从两条规则比较便宣称通过 0.1% 或全 G1 PASS**。有独立证据后才能判断是否进入近盒棱角高阶求积、源引导的解析 Jacobian、其后 R/K 与电流空间收敛。实验和专项测试**待本机运行**，不会修改经过验证的 P0 BFZI 连续线源。

## 11. P1-C 首次运行阻断诊断与修复（2026-11-05 UTC）

用户本机 [最近一次 P1 运行日志](runs/p1/latest_full.log) 对应 `983ad60`：**94 passed in 44.00s，但主程序随后 FAIL**；`bfzi_p1_full_sea_taylor.py` 的 `p1_full_sea_change` 两个 f-string 中缺少关闭 `_relative(...)` 的右括号，Python 在导入该模块时即报 `SyntaxError`。因此本次**根本没有执行任何 P1-C 全湿区磁能装配**，既不能据此判断原核 Taylor 收敛，也不能将 94 passed 作为全域物理 PASS。

修复时另外找到一个会导致下一次执行失败的数学约束问题：原先 `x_only` 的 `x1=(4,6,10)` 恰等于 `y0=(4,6,10)`。虽然未软化原核的一阶 Taylor 余项在 `y\to x` 时趋近零，对**直接写成原始求积代码的残差**仍要避免计算同一积分点的未定义分式，当前 `assemble_coulomb_subtracted` 明确拒绝 `|x-y|<1e-14`。现将四套积分轴规则固定为：

```text
x0=(3,5,8),   y0=(4,6,10)
x1=(5,7,12),  y1=(6,8,14)
base=(x0,y0), x_only=(x1,y0), y_only=(x0,y1), both=(x1,y1)
```

这些是**同一真实海水和同一电流的独立正权 (x/y) 积分坐标**；角度中点规则不会重合，不改变源、核或磁算子，只消除无定义的共址点评价。已补充：专项 pytest **直接导入**完整 P1-C 实验模块，提前发现任何语法错误；用户传入同一 x/y 张量规则时立即抛出明确异常。

另外开设独立 P1-C 入口，以减少重复运行已验证的球形和两紧支电流案例：

```powershell
python run.py --mode bfzi-p1-full-sea --with-tests
```

该入口自动将日志提交到 `docs/research/bfzi_uwpt_electrothermal/runs/p1_full/`，不覆盖 `runs/p1/` 的 P1-A／P1-B 解析真值证据。本轮**修复代码尚未经过用户本机重新执行**；必须先读取 `runs/p1_full/latest_summary.md` 和 `latest_full.log`，取得固定五模式的 `p1_full_sea`／`p1_full_sea_change` 数据，再分析 (x) 外积分、(y) 内积分、盒面近壁及谱误差。**该五模式连续磁矩阵暂无独立精确值；即使运行成功，也不能仅凭两次规则比较或 `skew=0` 宣布 G1 PASS。**

## 12. P1 的逐级停止规则

1. **P1-A 原核解析球基准：** 提前固定**解析球专用**数值目标：最终 `both` 规则原核矩阵 Frobenius 相对误差、最大对角相对误差和离对角矩阵相对误差均 **<0.001（0.1%）**；相同基准 `x/y` 节点下两个物理上等价的局部分区半径对应的 `L11` 差值，除以解析 `L11`，也必须 **<0.001**。这些只是第一道算子基准门槛，还应人工检查 `base/x_only/y_only/both` 向解析真值的稳定收敛。程序输出 `p1_gate`；任何一项不满足都记录 FAIL，不通过临时改阈值伪装成功。**旧 split 已实测 FAIL**；新 Taylor 原核在用户本机 87 passed 的非线性球参考误差 1.596e-4 已 PASS，真实两盒几何矩 90 passed/近壁独立误差约 1e-15 也 PASS。**两紧支双盒涡流的自/互 L 已在本机 92 passed 验收；后续真实全海水 curl 字典的磁算子仍未认证。**
2. **P1-B 真实 BFZI 双盒域：** 球基准通过后，固定一个真实的有限海水域、固定小数目的 \(\mathbf W=\nabla\times\mathbf F\)，独立比较对角、近邻及远区原核 1/r 积分，不替换为 uniform-ball 或 Plummer。需按 Passarotto–Pitassi–Specogna 2022 的原核自项/奇异提取思路评估现有公式，不仅对比两种未认证规则。
3. **P1-C 独立几何边界与外域：** 球子域成功不代表盒边角的裁剪积分或无限海水尾流已正确；两者分别保留后续关口。
4. 若球算子都不能稳定收敛，不开展 229 模态、神经教师或因反作用而改变电源路径的数值尝试；优先修复数学上的奇异项积分。

当前 **P0 限定基准 PASS；P1-A 球解析真值 PASS；双盒几何矩 PASS；P1-B 双紧支海水自/互原核 L PASS；旧 split FAIL；P1-C 最近一次 94 passed 但主执行 SyntaxError，现已修复并补独立节点规则和专用入口，等待用户本机新的五模式全湿区 L 结果；完整 G1=FAIL**。
