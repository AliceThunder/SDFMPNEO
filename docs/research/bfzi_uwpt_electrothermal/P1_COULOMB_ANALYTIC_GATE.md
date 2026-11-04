# P0 限定验收冻结 → P1 原 Coulomb 磁能解析基准

**2026-11-04；G1 仍 FAIL。** 本文件是第一阶段已取得的实际证据及第二阶段第一道可证伪验收，不是整个 BFZI 海水代理的完成声明。

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

## 4. P1 的逐级停止规则

1. **P1-A 原核解析球基准：** 提前固定**解析球专用**数值目标：最终 `both` 规则原核矩阵 Frobenius 相对误差、最大对角相对误差和离对角矩阵相对误差均 **<0.001（0.1%）**；相同基准 `x/y` 节点下两个物理上等价的局部分区半径对应的 `L11` 差值，除以解析 `L11`，也必须 **<0.001**。这些只是第一道算子基准门槛，还应人工检查 `base/x_only/y_only/both` 向解析真值的稳定收敛。程序输出 `p1_gate`；任何一项不满足都记录 FAIL，不通过临时改阈值伪装成功。当前**未运行，不能宣布 PASS**。
2. **P1-B 真实 BFZI 双盒域：** 球基准通过后，固定一个真实的有限海水域、固定小数目的 \(\mathbf W=\nabla\times\mathbf F\)，独立比较对角、近邻及远区原核 1/r 积分，不替换为 uniform-ball 或 Plummer。需按 Passarotto–Pitassi–Specogna 2022 的原核自项/奇异提取思路评估现有公式，不仅对比两种未认证规则。
3. **P1-C 独立几何边界与外域：** 球子域成功不代表盒边角的裁剪积分或无限海水尾流已正确；两者分别保留后续关口。
4. 若球算子都不能稳定收敛，不开展 229 模态、神经教师或因反作用而改变电源路径的数值尝试；优先修复数学上的奇异项积分。

当前 **P0=限定基准 PASS；P1=已开始，解析基准等待实测；G1=FAIL**。
