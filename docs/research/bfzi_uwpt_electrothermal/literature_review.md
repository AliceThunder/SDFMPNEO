# 电磁热联合代理相关文献与技术迁移分析

**整理日期：2026-10-31；状态：研究资料，不是成果声明。**

## 阅读规则

- “电磁—热联合预测”≠“双向耦合”；“监测点温度预测”≠“完整三维温度场”；“电磁端口代理”≠“空间电磁场代理”。
- 文献中的加速倍数与精度均依赖其参考方法、几何域、硬件及评价指标；不可直接移植为 SDFMPNEO 的性能目标。
- 下表给出前期对话中讨论的全部主线，优先级基于**BFZI 连续线圈、非导电封装空腔、海水导电反作用、全空间损耗与热时演化**的相关程度。
- 可核实论文入口以 DOI 为主，暂未核验的细节明确标记；本文件不是严格的系统性文献综述。

## 主文献清单

| 编号 | 领域、论文（入口） | 物理/学习类型 | 本项目取用与限制 |
|---|---|---|---|
| L01 | Wang et al., [Dynamic electromagnetic-thermal mapping in motor digital twins: A real-time multiphysics framework](https://doi.org/10.1016/j.dte.2025.100082), *Digital Engineering*, 8:100082 (2026) | POD+NN、电磁热场数字孪生 | **取**场状态降维与实验校准；**不取**固定电机 FEM 网格模态作为跨 BFZI 几何统一基 |
| L02 | Li, Wu & Hua, [Surrogate-Model-Assisted Electromagnetic-Thermal Optimization of PMSMs Considering PM Demagnetization](https://doi.org/10.1109/ECCE-Europe62795.2025.11238504), ECCE Europe 2025 | 设计优化的磁热代理 | **取**多物理代理优化流程；不是 UWPT 分布海水损耗基准 |
| L03 | Wan et al., [Physics-informed thermal-electromagnetic deep operator network for coupled field modeling of dual three-phase permanent magnet synchronous motor](https://doi.org/10.1016/j.neucom.2026.135445), *Neurocomputing* (2026) | 物理信息神经算子 | **取**物理残差约束与场映射；全文仍需确认耦合方向与在线推理口径 |
| L04 | Dammak et al., [Validation of a neural electromagnetic-thermal model for planar PCB coils in wireless power transfer using classical physics models](https://doi.org/10.1515/teme-2026-0106), *tm – Technisches Messen* (2026) | PCB WPT 联合预测的物理一致性验证 | **重点取**跨几何缩放规律、独立解析参照和单因素扫描；不视为任意 3D 时变全场模型 |
| L05 | Zhu et al., [Efficient Thermal Evaluation of IPT Pads Using Field-Aware Loss Approximation and Parametric Subspace Projection](https://doi.org/10.1109/TPEL.2025.3636298), *IEEE TPEL*, 41(5):7720–7731 (2026) | 分布损耗近似 + 参数化热 ROM | **重点取**全三维热场降阶、参数化换热；改造成随几何变化的连续基函数 |
| L06 | Li et al., [Dynamic Temperature Prediction of the Magnetic Coupler in Wireless Charging Systems](https://doi.org/10.1109/TPEL.2025.3640679), *IEEE TPEL* (2026) | LSTM/注意力、多测点瞬态温度 | **取**错位、持续充电、环境变化的时间数据设计；不取黑箱时间截断和仅测点输出 |
| L07 | Tiemann et al., [Model Order Reduction for Thermal Analysis of Wireless Power Transfer Systems Considering Radiation](https://doi.org/10.1109/COMPUMAG55718.2022.9827513), COMPUMAG 2022 | 含辐射换热的热降阶 | **取**边界热辐射、物理热降阶；不是完整神经电磁热耦合 |
| L08 | Xie, Zhu & Zhang, [Research on Transmission Characteristics of Magnetic Couplers for Underwater Wireless Power Transfer Based on Prior Knowledge Input Neural Network](https://doi.org/10.3390/s26051712), *Sensors*, 26(5):1712 (2026) | 镜像法先验 + NN，UWPT 互感和损耗 | **高度相关**：取解析基底与残差学习；需扩展到封装边界、空间场和热动力 |
| L09 | Yuan et al., [Efficiency Optimization of LCL-Resonant Wireless Power Transfer Systems via Bidirectional Electromagnetic–Thermal Coupling Field Dynamics](https://doi.org/10.3390/en17174507), *Energies*, 17(17):4507 (2024) | FEM 的电磁热双向耦合与谐振电路 | **重点取**温度→电阻→电路工作点→新损耗的闭环基准；论文并非神经代理 |
| L10 | García-Izquierdo et al., [Rapid Machine Learning–Driven Modeling for Large-Scale Validation and Optimization of Control Variables in Wireless Power Transfer Systems](https://doi.org/10.3390/make8070218), *MAKE* (2026) | WPT 端口参数快速预测 + 电路 | **取**将预测阻抗接入补偿/控制参数求解；未建立空间温度场 |
| L11 | Chen et al., [An Efficient Electrothermal Decoupling Method for Power MOSFETs in Three-Phase LLC Converters Based on Iterative Multi-Artificial Neural Networks](https://doi.org/10.1109/PEAS66638.2025.11403458), IEEE PEAS 2025 | 功率 MOSFET 器件电热代理 | **取**电路/热反馈多模型分工、散热几何扫描；不等同于线圈磁场/海水热源代理 |
| L12 | Dini et al., [Real-time electro-thermal modelling and predictive control design of resonant power converter in full electric vehicle applications](https://doi.org/10.1049/pel2.12527), *IET Power Electronics*, 16:2045–2064 (2023) | CLLC 等谐振电路的电热实时模型 | **重点取**快电气/慢热时间尺度分离与控制闭环；非三维场神经代理 |
| L13 | Di Barba et al., [Deep Learning Surrogate Models for Nonlinear Magneto-Thermal Analysis of TEAM Problem 36](https://doi.org/10.3390/en19133013), *Energies*, 19(13):3013 (2026) | 感应加热 CNN 非线性瞬态温度 | **取**材料强非线性和早期瞬态误差诊断；径向温度线输出非通用三维场 |
| L14 | [Machine Learning-Based Reduced Order Modeling of Nonlinear and Multiphysics Magnetic Devices](https://doi.org/10.1109/TMAG.2025.3626055), *IEEE Transactions on Magnetics* (2026) | POD+Gaussian-process 等降阶方法 | **取**非线性磁材局部模型；材料和几何可泛化范围需进一步查全文 |
| L15 | [Multi-Physics Field Coupling Modeling and Thermal Distribution Analysis of Underwater Rotary Magnetic Coupling Mechanism](https://doi.org/10.1109/TMAG.2025.3545775), *IEEE Transactions on Magnetics* (2025) | UWPT 磁热流多物理参考 | **取**海水流动及封装热边界的物理基准；非神经代理 |
| L16 | [Forced Cooling Thermal Model for 1-MHz, 5-kW LLC Converter Planar Transformer](https://doi.org/10.1109/APEC51134.2026.11516811), APEC 2026 | 平面磁器件热网络 | **取**高频磁器件节点热阻与实验标定；非连续场代理 |

**待追踪，不计作完整已核实正式发表论文：** Dammak, Cai & Kanoun, *WPTNet: A Dual-Branch CNN–MLP Model for Joint Electromagnetic–Thermal Prediction of Planar WPT Coils*, IEEE MetroXRAINE 2026 accepted；目前依据 L04 参考文献中的接收信息，尚需正式 DOI 和全文。相关工作不能作为本项目新颖性或性能比较的唯一证据。

## 主题化技术结论

### 1. 连续几何与位姿

多数传统 POD 方法在固定拓扑/网格下非常有效，但不能直接解决 BFZI 圆形—圆角方形连续过渡、不同匝数、封装尺寸与六自由度相对姿态。应把几何依赖放到解析源项、随体空间基、界面几何以及数据覆盖中，避免把几何变化转化为每个工况的重网格化。

### 2. 水下电磁与海水损耗

L08 的“物理先验 + 学习修正”比对复阻抗直接回归更适合本研究，但必须保留 **0UnifiedModelingUWPT** 中封装空腔修改海水导电区域、海水电流连续性和非局部反作用的数学机制。**端口复阻抗的实部不能无条件代表铜损或总系统损耗**。

### 3. 热场而非测点时间序列

L05 给出参数化全三维热场降阶经验，L06 给出错位和运行历程的时间建模经验。推荐用**物理守恒的连续空间降阶动力系统**组合两者：热态依赖几何和材料、神经闭合只负责难以解析的部分，而不是将 LSTM 当成完全通用的 PDE 求解器。

### 4. 真正双向耦合

L09 说明 WPT 温升可改变线圈电阻与电路工作点。项目需要显式闭环：

\`T → 材料 σ,μ → EM 场和阻抗 → 补偿电路 I → 分布损耗 → 热源 → T\`。

不能将“电磁损耗驱动一次热计算”误报为双向电磁热耦合。

### 5. 物理正确性与学习目标

优先保持电流连续性、被动性、互易性（只适用于线性互易条件）、端口/场功率一致、热容量正定及耗散性。学习空间非局部响应或低阶状态的**修正量**，而不是训练彼此独立的阻抗/温度/损耗黑箱。

## 论文学术定位（候选，不作已验证创新声明）

1. BFZI **匝号仿射结构**支撑的几何诱导场核/神经算子，实现跨形状与姿态复用。
2. 海水电流连续约束下的**非局部导电反作用代理**，复阻抗、场与体积损耗来自同一状态。
3. 封装组件**随体坐标**和海水**世界坐标**耦合的瞬态电磁热求解。
4. **空间损耗守恒的热 ROM**与谐振电路反馈闭环。

**这些贡献需通过基准对比、消融和独立实验决定是否成立。**

## 核对状态

已通过出版社摘要/页面在本轮交叉核实的主线包括 L01、L04、L05、L08、L09、L11、L12、L13。其他 DOI 属于先前文献查找整理，后续投稿前应逐一查全文确认具体页码、作者、方法及数据；**不应直接将本综述当正式论文参考文献列表**。
