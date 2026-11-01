# BFZI 任意位姿海水电流基与快速装配：数学到算法的执行契约

**版本：2026-10-31，理论方案。尚未验证运行精度和加速性能。**

这份文档对应三个新理论分节：[03b 合规电流空间](../../../theory/sections/03b_geometry_admissible_basis.tex)、[04b 正权核装配](../../../theory/sections/04b_positive_kernel_assembly.tex)、[04c 误差界与在线算法](../../../theory/sections/04c_residual_online_algorithm.tex)。

## 1. 合规电流基从何而来

旧方案只写了抽象要求 \(\mathbf W\in H(\mathrm{div};\Omega_s)\)，本轮补充两种具体构造：

- **边界消失矢量势的 curl 模式**：在局部封装曲面图谱上取 \(\mathbf V_m=d(\mathbf x)\chi_m(\mathbf x)\mathbf a_m(\mathbf x)\)，令 \(\mathbf W_m=\nabla\times\mathbf V_m\)。其中 \(d\) 为本图谱界面函数，\(\chi_m\) 有紧支撑。由 div curl = 0 与界面切向迹可得 \(\nabla\cdot\mathbf W_m=0,\ \mathbf n\cdot\mathbf W_m=0\)。注意壁面电流**可以有非零切向值**。
- **局部 Piola 模式**：对局部几何变换 \(F\)，取 \(\mathbf W=(DF/\det DF)\widehat{\mathbf W}\circ F^{-1}\)，保留散度与法向通量。纯刚体转动退化为 \(\mathbf W(\mathbf x)=R\widehat{\mathbf W}(R^T(\mathbf x-\mathbf c))\)。

针对两侧封装，在局部图谱中分为近盒 1、近盒 2、狭小间隙、远场、可能的拓扑环流模式。不能假设全体参数范围存在同一个失真始终有界的参考空间映射，也不能让随盒体移动的局部电流模式穿入另一盒体。

**现有数学不等于我们的新发明：** curl 保散度与 Piola 映射是成熟的数学工具。我们的工作是为 BFZI 线圈与双封装 UWPT 物理空间选出可复用、可加速、能处理近间隙的具体模式族。

## 2. 快速装配怎样保持被动和能量一致？

**电阻矩阵**使用同一空间电流基上的正权积分：
\[
\widehat R_{ij}=\sum_{\nu=1}^{Q_R}
\frac{w_\nu}{\sigma_s(\mathbf x_\nu)}
\mathbf W_i(\mathbf x_\nu)\cdot\mathbf W_j(\mathbf x_\nu),
\quad w_\nu>0.
\]
因此 \(\widehat R=D_R^TD_R\succeq0\)，若采样列满秩则正定。用**相同节点**计算海水焦耳热，得到离散一致的总功率；不能把求积一致误报成连续积分无误差。

**磁矩阵**使用同一个自由空间 Coulomb 核的正权谱积分：
\[
\widehat L_{ij}=\mu_0\sum_{\nu=1}^{Q_L}
\frac{a_\nu}{|\mathbf k_\nu|^2}
\operatorname{Re}\bigl[\widehat{\mathbf W}_i(\mathbf k_\nu)^*
\cdot\widehat{\mathbf W}_j(\mathbf k_\nu)\bigr],\quad a_\nu>0.
\]
整个矩阵由相同谱特征组成 Gram，因此半正定。如果铜线、导电盒壁、海水都在同一个 Schur 系统中，**所有磁矩阵块必须使用共同的核/特征组装**，否则合并后的系统未必保被动。

**BFZI 耦合矩阵**可通过互易重排为
\[
K_{mp}=\int_{\Gamma_p}\mathbf A_m^{\rm mode}(\mathbf r)\cdot d\mathbf l,\qquad
\mathbf A_m^{\rm mode}(\mathbf r)=
\mu_0\int_{\Omega_s}G_0(\mathbf r-\mathbf x)\mathbf W_m(\mathbf x)\,dV.
\]
这样可以在 BFZI 原始八线段组、跨匝仿射位置/线元的基础上，加速沿匝号的线积分；对模式矢量势的展开需要**新的余项界**，不能直接复制原 Neumann 核的常数。

**限制：** 单纯加速 BFZI 线源求和不自动消除海水积分；对每个新几何都从头生成大字典和磁核，整体仍可能非常慢。需要缓存、局部模式复用和矩阵收缩，后续用真实耗时检验。

## 3. 在线计算的最小流程

1. 输入真实 BFZI 连续线圈、两侧封装、海水参数、六自由度位姿，检查正间隙及制造约束。
2. 生成近盒 / 间隙 / 远场的 curl 或 Piola 合规模式，并处理复杂海水拓扑需要的全局环流。
3. 网络只输出子空间选择 \(\mathbf S_\theta\)，令 \(\mathbf W_\theta=\mathbf W_0\mathbf S_\theta\)；用同一物理积分生成 \(R_\theta,L_\theta,K_\theta\)。
4. 求解低阶涡流系统或（含铜导体 / 导电盒壁）同源耦合 Schur 补，不独立叠加阻抗补丁。
5. 从同一电流状态解码局部空间损耗与端口复阻抗，向热降阶系统投影相同的热源。
6. 场景不满足图谱覆盖、模态秩退化、间隙过小或局部误差超限时，增广图谱或回退高精度参考模型。

理论复杂度在未计几何构建、场查询时，对稠密字典 \(M\)、空间和谱节点 \(Q_R,Q_L\)、选中维数 \(r\)，矩阵收缩与小系统求解典型开销为
\(O((Q_R+Q_L)Mr+(Q_R+Q_L)r^2+r^3)\)。
这**不是总推理复杂度**，源项积分、字典生成和空间场评价还需另算。

## 4. 物理误差如何控制？

对**同一个高精度有限维字典**下的 \(A=R+j\omega L\) 和真实解 \(Az^*=f\)，令残差 \(r=f-Az_r\)。当 \(R\succ0\)：

\[
\boxed{\|z^*-z_r\|_R\le\|r\|_{R^{-1}}.}
\]

由此能控制同一有限维状态的电阻能量范数，进一步推导端口电压和总海水损耗误差。但是它**不自动保证最大热点损耗或最高温度**；近界面空间误差仍需独立参考验证。在线计算完整残差可能昂贵，可以用于抽样教师或低成本近似指标，不必强制每次运行完整认证。

## 5. 与新文献的关系

- Chen, Henneron, Clénet, [Comparison of parametric model order reduction methods to solve magneto-quasistatic and electro-quasistatic problems](https://doi.org/10.1016/j.finel.2025.104444), *Finite Elements in Analysis and Design* 252 (2025):104444：已有参数化 MQS/EQS ROM 与子空间插值研究。
- Kerler-Back, Reis, Stykel, [Regularization and passivity-preserving model reduction of quasilinear magneto-quasistatic coupled problems](https://arxiv.org/abs/2411.14226), arXiv (2024)：已有磁准静态耦合系统的保被动降阶研究。

所以新颖性需要来自**BFZI 结构化核、双封装几何图谱、物理同源快速矩阵装配，以及端口—空间损耗—热反馈的一体化结果**，而不是宣称 Piola、curl 或保被动降阶为首次发明。

## 6. 下一次应完成的最小实验

限定为两侧非导电封装、均匀静止海水、一个频率、合法静态任意位姿。首先实现一个能同时输出涡流电流、端口阻抗、材料热源的参考低阶系统，验证约束与 FEM 场误差，再比较是否真的有可重复的在线加速。**现阶段只完善理论，没有编写或运行实际模型代码，也没有做 LaTeX 编译。**
