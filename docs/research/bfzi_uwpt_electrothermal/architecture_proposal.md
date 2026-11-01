# BFZI-UWPT 电磁—热—电路联合代理：新理论架构（方案稿）

> 2026-10-31，**新理论**。不继承 SDFMPNEO 旧版双阶数超椭圆作为线圈定义；旧理论在本研究分支的 \`theory/\` 将整体替换。以下定义含“可核实物理基底”和“待实现/待验证的创新”，不能混称已有能力。

## 0. 来源与准确性边界

最新版线圈权威文档：[0UnifiedModelingUWPT/sections/02_geometry_model.tex](https://github.com/AliceThunder/0UnifiedModelingUWPT/blob/main/sections/02_geometry_model.tex)；
海水物理：[03_package_seawater_impedance.tex](https://github.com/AliceThunder/0UnifiedModelingUWPT/blob/main/sections/03_package_seawater_impedance.tex)；
几何快速积分：[04_geometry_induced_fast_computation.tex](https://github.com/AliceThunder/0UnifiedModelingUWPT/blob/main/sections/04_geometry_induced_fast_computation.tex)。
该项目的“统一建模”**仅指圆形与圆角方形连续螺旋的几何统一**，封装空腔是一项独立的海水域修正；不得扩大表述。

### BFZI 的原始几何量

令 \(\eta=k+\tau/(2\pi)\)、\(\beta=s/(2\pi)\)，有
\[
A=A_0-s\eta,\qquad \varrho=\varrho_0-\alpha s\eta,\qquad
\chi=A_0-\varrho_0-(1-\alpha)s\eta,\quad A=\chi+\varrho.
\]
\(\sigma_w\in\{+1,-1\}\) 为绕制方向。四个直边组 \(j=0,\ldots,3\) 和四个圆角组 \(m=0,\ldots,3\) 由各自原文公式建立，不能用闭合等距同心环替代连续导体。圆形特例：\(\varrho_0=A_0,\alpha=1\)，直边长度退化为零，但沿每匝绕行的径向收缩及切向偏移不消失。

对于固定线段组 \(b\) 与局部求积参数 \(\zeta\)：
\[
\mathbf r_{b,k}(\zeta)=\mathbf r_{b,0}(\zeta)+k\Delta\mathbf r_b(\zeta),\quad
\mathbf g_{b,k}(\zeta)=\mathbf g_{b,0}(\zeta)+k\Delta\mathbf g_b(\zeta).
\]
这是几何诱导电磁积分加速的入口。令第 \(p\) 个装配件位姿为 \((R_p,c_p)\in SE(3)\)，三维线圈中心线为 \(R_p\mathbf r+c_p\)，有向线元为 \(R_p\mathbf g\)。

**几何合法性**：要求 \(A,\varrho\) 正、\(\chi\ge0\)、角部导数不退化、相邻导体有净间隙、端口与完整馈/回流路径明确、导体与盒壁无非法穿透；不满足条件时拒绝/标记超出模型域。

## 1. 场景、材料和任意位姿

用两个装配件 \(\mathcal A_p=(\Gamma_p,\mathcal P_p,R_p,c_p)\) 表示线圈、导体截面/绝缘、灌封、磁芯、屏蔽、盒壁等；封装几何 **独立于** 线圈包络变化。每个部件给材料分区的隐式位置函数或解析边界，查询可映射到随体局部坐标 \(x_p=R_p^\top(x-c_p)\)。海水 \(\Omega_s(t)=\mathbb R^3\setminus\bigcup_p\overline{\Omega_{\rm pack,p}(t)}\)；固体热域和海水域均可分段，以真实材料界面传热/电磁耦合。**不能**因原论文非导电封装假设而将金属盒壁也赋零法向导电电流。

材料参数 \(\sigma(x,T,\omega),\mu(x,T,H,\omega),\epsilon(x,T,\omega),\rho,c,k\) 可参数化、频散、分段；首版以磁准静态、线性各向同性、非导电非磁性封装和均匀海水为基线。复杂各向异性、强非线性和快速运动是明确的后续扩展，不是无条件保证。

## 2. 电磁基底：从 BFZI 解析结构到同源场/阻抗/损耗

### 2.1 主螺旋线元基底

给端口 \(p\) 的单位峰值电流激励
\[
\mathbf C_p(x)=\frac1{4\pi}\int_{\Gamma_p}\frac{d\ell'}{|x-r'|}.
\]
在 BFZI 主螺旋使用同组求积节点、跨匝仿射结构做误差受控聚合；非重复馈线、回流段独立积分。空气主项为 GMD/有限截面规则处理自项的 Neumann 积分 \(j\omega L^{\rm air}\)，不是简单点导线自感。

### 2.2 海水及封装反作用

现有论文基准中，非导电封装从海水域排除：
\(\Omega_s=\Omega_B\setminus\Omega_{\rm nd}\)，海水场满足
\[
J_s=\sigma_s(-j\omega A-\nabla\phi),\quad \nabla\cdot J_s=0,\quad n\cdot J_s=0
\]
（最后一个边界式仅用于绝缘界面）。其离散自洽响应形式
\((I+\mathcal KP_\phi)X_p=C_p\)，端口 \(Z_{\rm sw}\) 来自同一场响应。

**创新候选**：用物理预条件基底 \(X_p^{(0)}\) + 可学习受约束残差 \(\delta X_{\theta,p}\) 近似自洽响应，**先解同一空间场状态，再读取阻抗/电流/损耗**。网络应学习界面、材料和非局部散射，不应分别黑箱回归不相容的 \(R,L,M,Q(x)\)。训练中保留连续性投影或结构化散度自由表示；边界条件与近奇异局部项独立处理。保留可回退的高精度参考路径。

### 2.3 有限截面导体和损耗归属

从 BFZI 的几何中心线及导体截面建立完整闭合端口电流（包括馈电/回流）；导体集肤/邻近、盒壁、磁芯/屏蔽、海水等采用**互不重叠**的材料域损耗，防止 \(\operatorname{Re}Z_{\rm sw}\) 与铜损重复。单一相量峰值约定：
\[
Q_{\rm ohm}(x)=\tfrac12 \operatorname{Re}(J\cdot E^*)\ \ge0,\qquad
P_{\rm EM,int}=\tfrac12 I^{\mathrm H}\operatorname{Herm}(Z_{\rm EM})I
=\sum_{\Omega_i}\int_{\Omega_i} Q_i\,dV.
\]
分布式磁滞/磁芯损耗应采用受证实的材料模型/等效源并一致地计入端口耗散。互易线性场中 \(Z=Z^\top\)；被动性由 \(\operatorname{Herm}(Z)=(Z+Z^{\mathrm H})/2\succeq0\) 定义，不能以逐元素电阻非负替代。多频谱的耗散按各谐波求和。

## 3. 谐振电路：基波和多谐波渐进扩展

在每个热时间点，给定电源、负载、SS/SP/LCL/LLC/CLLC 等具体拓扑与频率，先用端口响应与器件/补偿元件组合生成 \(\mathcal Z_{\rm circuit}\)，再解实际端口电流 \(I\)，用其驱动损耗。谐振拓扑的模型必须包含电容、电阻、整流和器件损耗的归属；若以**基波近似**首发，明确并记录谐波误差，不可用单频模型宣称覆盖所有开关器件温升。

## 4. 热层：随体固体状态 + 世界坐标海水

固体满足
\[
\rho c\,\partial_t T=\nabla\cdot(k\nabla T)+Q_{\rm EM},
\]
海水满足
\[
\rho_sc_s(\partial_t T_s+v_s\cdot\nabla T_s)=\nabla\cdot(k_s\nabla T_s)+Q_s.
\]
界面温度/热通量连续，或设置有物理依据的热接触阻；封装外壁可用经确认适用的对流换热、辐射条件。对于两侧相对运动，热状态在各自**随体坐标**保持，海水流场在**世界坐标**表示；如果位姿变化速度较高，必须考虑运动电动势、流场改变和移动界面热输运，不能只旋转静态结果。

### 热 ROM 候选

\[
T_i(x,t)=T_{\rm ref}(t)+\sum_{\ell=1}^{r_i}a_{i,\ell}(t)\Phi_{i,\ell}(R_i^\top(x-c_i);\ g_i),
\]
\[
M(g,a)\dot a+K(g,a,q,v_s)a=B(g)P_{\rm loss}+\mathcal R_\theta(g,a,I).
\]
\(M\succ0\)，\(K\) 满足被动耗散；\(\mathcal R_\theta\) 为受能量约束的热闭合修正。**时间由显式 ODE/积分器演化，不把查询时间直接回归成一个无物理约束的标量输入**。选用可训练连续空间基，避免重新依赖固定世界 FEM 网格。海水热尾迹可用另一个低秩/Green 表示；必须单独验证温度近场和长期能量损失。

## 5. 完整双向反馈

\[
(g,q,\text{material}(T),\omega)\xrightarrow{\ \mathcal E_\theta\ }
(Z,\text{unit-port fields})
\xrightarrow{\ \mathcal C\ } I
\xrightarrow{\ \text{power-consistent loss}\ }Q
\xrightarrow{\ \mathcal H_\theta\ } T(t+\Delta t).
\]
强耦合可在热步内固定点/Newton 迭代；弱耦合可采用误差控制步进。需要比较温度反馈前后 \(Z,I,P_{\rm loss},T\) 的差异，避免只实现“热单向驱动”却称为双向。

## 6. 神经模块与非黑箱设计

研究方案只允许两个主要学习模块：

1. **场响应闭合网络**：输入 BFZI 源项压缩表示、封装边界材料特征、位姿、频率与当前热态；输出复场散射/近场校正的低维表示；同源解码到端口阻抗和空间损耗。
2. **热闭合与几何条件化基**：输入热状态、几何、界面和热源；输出小维修正、必要时学习连续坐标条件化基/局部 Green 核。

绝不另行训练互不一致的独立端口、海水损耗与温度黑箱拼接。已有解析项保持解析，学习的是复杂界面、有限截面、非局部反作用、非线性和降阶截断误差。

**适用条件**：推理可基于连续几何/积分和少量状态，无固定世界网格；但非均匀材料仍可能需要局部体积/边界离散。不能将模型误写成“所有介质完全免数值积分”。

## 7. 优先消融/失效模式

- BFZI 匝号聚合 vs 逐匝积分：核误差和吞吐，分远/近奇异区。
- 只有端口回归 vs 同源场/端口/损耗：能量、互易和热点误差。
- 固定热基 vs 几何条件化热基：跨形状/封装尺寸误差。
- 无海水导电反作用 vs 有：不同导电率、距离、位姿。
- 热单向 vs 双向：温度相关损耗反馈及谐振漂移。
- 弱/强耦合：热长时稳定性与开关运行误差。
- 海水静止 vs 有流：热边界和旋转装配影响。

## 8. 决策与非目标

- **本阶段**：完成理论替换、对齐 BFZI、收集文献、建立复现与判据；不改现有 SDFMPNEO 训练/推理代码。
- **下一阶段**：先实现由同一物理场产生的 \(Z,Q\) 接口和导体/海水能量核对，再引入热 ROM，最后建立电路闭环和残差网络。
- **非目标**：首版无限广材料参数外推、湍流全耦合 CFD、随意大速度运动中的全波电磁、模型未经训练就对所有封装器件有效。
