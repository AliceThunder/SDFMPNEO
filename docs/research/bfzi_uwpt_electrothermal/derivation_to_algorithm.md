# 从文献启发到 BFZI 新算法：首轮可证明的理论结果

> **2026-10-31，推导稿 v1。** 对应 \`theory/sections/03a_solenoidal_eddy_galerkin.tex\`、\`04a_geometry_conditioned_operator.tex\`、\`05a_coupled_conductive_schur.tex\`。这是研究方案的数学收敛，不代表代码已实现或性能已被验证。

## 1. 文献启发为什么没有直接变成网络堆叠

| 来源 | 已有技术得到的启发 | 在 BFZI-UWPT 中实际发生的改造 | 仍须证明 |
|---|---|---|---|
| Xie et al., [Sensors 2026](https://doi.org/10.3390/s26051712) | 将电磁解析先验提供给神经网络 | BFZI 匝号仿射结构形成端口场源；网络改为选择/压缩受约束海水电流子空间，而非只回归互感和损耗 | 场精度、数据效率、跨安装盒泛化 |
| Zhu et al., [IEEE TPEL 2026](https://doi.org/10.1109/TPEL.2025.3636298) | 空间损耗 + 参数热 ROM 支持快速全场重构 | 用海水同源焦耳热投影到几何条件化连续热基，热态显式演化 | 跨 BFZI 形状、界面热传递、长时精度 |
| Li et al., [IEEE TPEL 2026](https://doi.org/10.1109/TPEL.2025.3640679) | 时间序列信息帮助预测错位工况下的温度 | 温度预测采用守恒 ODE 和连续场读出，不直接依赖固定时间窗口回归 | 时域精度和效率是否优于 LSTM |
| Yuan et al., [Energies 2024](https://doi.org/10.3390/en17174507) | 真实 WPT 必须考虑温度到阻抗及工作点的反馈 | 同源端口矩阵进入补偿电路，电流再决定分域热源和下一热状态 | 闭环稳定性、失谐与效率的真实性 |
| Dammak et al., [Technisches Messen 2026](https://doi.org/10.1515/teme-2026-0106) | 不能只用平均误差，需验证解析缩放及物理规律 | 直接把互易、被动与积分功率一致性做进有限维结构；另验证全空间场误差 | 数值积分误差和真实教师偏差 |

**关键界限：** BFZI 的连续几何、源项单匝号聚合、Neumann 双匝号快速求和是原 BFZI 论文的已有贡献。新贡献候选是将其用于**受约束低阶涡流场状态、统一电磁端口—材料耗散—热动力和可学习解空间**，并证明结构与算法性能。

## 2. 第一条可证明结果：海水端口阻抗不是单独训练出来的

### 前提

均匀磁导率、线性海水，磁准静态、静态几何、非导电封装绝缘界面；真实海水电导率严格正且有界。使用峰值相量 \(e^{j\omega t}\)。

选择满足

\[
\nabla\cdot \mathbf W_m=0,\qquad
\mathbf W_m\cdot\mathbf n|_{\partial\Omega_s^{\rm ins}}=0
\]

的连续海水电流基（多连通海水域还要包含全局环流模式）。

定义 \(\mathbf J_s(\mathbf x)=\mathbf W(\mathbf x)\mathbf z\)，令

\[
\begin{aligned}
R_{mn}&=\int_{\Omega_s}\sigma_s^{-1}\mathbf W_m\cdot\mathbf W_n\,dV,\\
L_{mn}&=\mu_0\int_{\Omega_s}\mathbf W_m\cdot\mathcal G_0\mathbf W_n\,dV,\\
K_{mp}&=\mu_0\int_{\Omega_s}\mathbf W_m\cdot\mathbf C_p^{\rm BFZI}\,dV.
\end{aligned}
\]

从 \( \mathbf J=\sigma_s(-j\omega\mathbf A-\nabla\phi)\) 的无散度弱式得

\[
\boxed{
(\mathbf R+j\omega\mathbf L)\mathbf z=-j\omega\mathbf K\mathbf I.
}
\]

端口感应电压与海水电流的互感互易关系给出

\[
\boxed{
\mathbf Z_{\rm sw}=\omega^2\mathbf K^{T}
(\mathbf R+j\omega\mathbf L)^{-1}\mathbf K .
}
\]

对于真实的对称 \(\mathbf R\succ0,\mathbf L\succeq0,\mathbf K\)，直接得到：

\[
\mathbf Z_{\rm sw}=\mathbf Z_{\rm sw}^{T},\qquad
\operatorname{Herm}(\mathbf Z_{\rm sw})\succeq0.
\]

并且**同一个电流状态的总海水焦耳热与端口海水耗散功率相等**：

\[
\boxed{
\frac12\mathbf I^H\operatorname{Herm}(\mathbf Z_{\rm sw})\mathbf I
=\frac12\mathbf z^H\mathbf R\mathbf z
=\int_{\Omega_s}\frac{|\mathbf W(\mathbf x)\mathbf z|^2}{2\sigma_s(\mathbf x)}\,dV.
}
\]

这三条是\`03a_solenoidal_eddy_galerkin.tex\`中有条件的数学证明；**不是对真实三维场精度或快速度的承诺**。

## 3. 第二条结果：神经网络只选择受约束的电流子空间

构造一个包含局部/全局海水环流的连续电流字典 \(\mathbf W_0\)，先用真实物理积分获得字典矩阵 \(\mathbf R_0,\mathbf L_0,\mathbf K_0\)。

训练网络在给定 BFZI 形状、盒体、位姿、海水参数和温度时输出实矩阵 \(\mathbf S_\theta\)，作为电流字典的子空间选择。

\[
\mathbf W_\theta=\mathbf W_0\mathbf S_\theta
\]

然后**不让网络独立预测阻抗和海水损耗**，而是进行严格的低阶矩阵收缩：

\[
\begin{aligned}
\mathbf R_\theta &= \mathbf S_\theta^T\mathbf R_0\mathbf S_\theta,\\
\mathbf L_\theta &= \mathbf S_\theta^T\mathbf L_0\mathbf S_\theta,\\
\mathbf K_\theta &= \mathbf S_\theta^T\mathbf K_0.
\end{aligned}
\]

求解 \((\mathbf R_\theta+j\omega\mathbf L_\theta)\mathbf z_\theta=-j\omega\mathbf K_\theta\mathbf I\)，用 \(\mathbf W_\theta\mathbf z_\theta\) 同时生成海水电流、阻抗及局部焦耳热。

这种构造**无论网络参数为何，都保留第一条证明的结构条件**（前提是子空间满列秩、物理积分一致）。

但是：给任意安装盒和任意姿态构造足够丰富、严格无散度、零法向且开销低的 \(\mathbf W_0\) 本身仍是研究难题；如果在线还必须执行巨大空间积分，那么小网络不意味着整个系统快速。因此需同时开发 BFZI 源项压缩、字典核缓存和参数化矩阵快速组装。

## 4. 第三条结果：铜损、盒壁、海水不能当成相互独立的阻抗补丁

前期方案按“空气项 + 铜损修正 + 盒壁修正 + 海水修正”组织代码职责是有用的，但**在导体、金属盒壁和海水互相磁耦合时，独立算完再简单相加通常不准确**。

改用所有导电区域的统一场：

\[
\mathbf J(\mathbf x)
=\mathbf U(\mathbf x)\mathbf I
+\mathbf W(\mathbf x)\mathbf z
\]

其中 \(\mathbf U\) 是满足端口电流约束的载流提升场，\(\mathbf W\) 是零端口环流校正模式。把电阻与磁场的双线性型联合组装成

\[
\begin{bmatrix}
\mathbf V\\0
\end{bmatrix}
=
\begin{bmatrix}
\mathbf A_{00}&\mathbf A_{01}\\
\mathbf A_{10}&\mathbf A_{11}
\end{bmatrix}
\begin{bmatrix}
\mathbf I\\\mathbf z
\end{bmatrix},
\qquad
\mathbf A_{ab}=\mathbf R_{ab}+j\omega\mathbf L_{ab}.
\]

因此完整的导电部分端口阻抗为

\[
\boxed{
\mathbf Z_{\rm cond,total}=
\mathbf A_{00}
-\mathbf A_{01}\mathbf A_{11}^{-1}\mathbf A_{10}.
}
\]

在正确的端口约定、线性被动介质和电流空间下：

\[
\frac12\mathbf I^H\operatorname{Herm}(\mathbf Z_{\rm cond,total})\mathbf I
=\sum_i\int_{\Omega_i^{\rm cond}}\frac{|\mathbf J_i|^2}{2\sigma_i}\,dV.
\]

这让有限铜线、海水和导电盒壁之间的交叉耦合进入**同一个**场问题，且各材料热源不重复。理想细导线 + 无耗散提升场的特殊情形退化回第一条海水公式。

**磁芯非线性、磁滞、位移电流、运动感应和频散仍需进一步扩展状态及方程**，不能用上述 \(\mu_0\) 块系统直接代表一切介质。

## 5. BFZI 在新算法中到底节约什么？

针对每个 BFZI 直边/角部线段组和 Gauss 节点，原论文成立：

\[
\mathbf r_k=\mathbf r_0+k\Delta\mathbf r,\quad
\mathbf h_k=\mathbf h_0+k\Delta\mathbf h.
\]

所以在构造海水电流基与线圈源的耦合矩阵 \(\mathbf K_0\) 时，

\[
K_{mp}\propto
\int_{\Omega_s}\mathbf W_m(\mathbf x)\cdot
\sum_{b,\mu,k}
\frac{\mathbf h_{b,\mu,k}}{|\mathbf x-\mathbf r_{b,\mu,k}|}\,dV.
\]

内层 \(k\) 求和可以继承原 BFZI 的匝号精确约简/有界分块方案，避开重复逐匝求核。

**不应宣称**它消除了外层海水积分，也不应宣称任意场景下的计算量天然不随匝数变化。我们需单独测量源项组装、字典构造、矩阵收缩、低维求解和空间场查询。

## 6. 热层怎样接入，而不重新独立训练一个损耗网络？

对热降阶测试基 \(\varphi_a(\mathbf x)\)，直接定义

\[
[\mathbf Q_a]_{mn}
=\int_{\Omega_s}\frac{\varphi_a(\mathbf x)
\mathbf W_m\cdot\mathbf W_n}{\sigma_s(\mathbf x)}\,dV
\]

则热动力系统的源输入是

\[
b_a=\frac12\mathbf z^H\mathbf Q_a\mathbf z .
\]

如果存在总热量通道 \(\varphi_0=1\)，其 \(\mathbf Q_0=\mathbf R\)，从而局部投影与总损耗保持同源。有限导体/盒壁扩展使用统一 \(\mathbf J=\mathbf U\mathbf I+\mathbf W\mathbf z\) 的相应材料分区形式。

这只是\*\*电磁热源的接口\*\*。热容量、界面热阻、海水热输运以及长时间非线性求解仍由热动力系统解决，不能认为完成上述投影就已经训练出高精度热代理。

## 7. 目前已获得的结论与尚未攻克的工作

**在明确的线性 MQS 假设和一致矩阵组装条件下，可以推导：**

1. 低阶海水复阻抗对称且被动。
2. 海水端口耗散与同源降阶涡流场焦耳热相等。
3. 神经网络仅选择一个合规电流子空间时，上述结构仍成立。
4. 对所有导电材料构造联合 Schur 补，可以保留完整端口与分区欧姆损耗的功率一致性。

**仍然没有证明或实现：**

1. BFZI 任意形状、盒体和六自由度姿态下的高效合规电流字典构造；
2. 几何变化后低阶字典如何以很少样本保持准确；
3. 字典级 \(\mathbf R_0,\mathbf L_0,\mathbf K_0\) 如何避免昂贵在线重组；
4. 磁芯/频散/高速运动等非线性跨材料扩展；
5. 低阶磁场电流误差如何可靠传递为热点温升误差；
6. 模型整体速度是否优于常规 FEM / BEM / 经典 ROM / 普通网络。

**下一轮最值得实现的最小物理原型：** 限定 BFZI 两线圈、两个非导电封装盒、均匀海水、静止姿态与单一正弦频率，先获得 \(\mathbf W,\mathbf R,\mathbf L,\mathbf K\) 的小型参考模型，真实验证本文内部能量恒等式与 FEM 端口/场误差；随后研究如何把合规电流字典变成高效几何条件化可训练空间。不要直接训练全耦合大模型。
