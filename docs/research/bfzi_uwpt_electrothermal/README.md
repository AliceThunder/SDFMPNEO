# BFZI 水下 WPT 电磁热代理：研究目录

**研究状态：2026-10-31 新理论重构。** 本研究分支已整体替换 theory/ 的旧理论目录；旧理论仅保留在 main 等历史分支和 Git 提交记录中。当前没有变更 SDFMPNEO 的运行代码，新的理论架构仍是待验证研究方案。

## 阅读顺序

1. [新理论主文档](../../../theory/main.tex)：多文件 LaTeX 主入口。
2. [文献分析](literature_review.md)：已讨论的电动机、IPT/WPT、UWPT 和谐振变换器文献及限制。
3. [方案分析](architecture_proposal.md)：BFZI 几何、电磁反作用、损耗、热动力、谐振电路的耦合选择。
4. [验证路线](validation_roadmap.md)：研究阶段、基准、开放问题。

## 几何权威来源

[0UnifiedModelingUWPT](https://github.com/AliceThunder/0UnifiedModelingUWPT) 的 sections/02_geometry_model.tex、03_package_seawater_impedance.tex 和 04_geometry_induced_fast_computation.tex；最新版 BFZI 是支撑函数构造的圆形—圆角方形**连续导体路径**，不是旧双阶数超椭圆。跨匝仿射结构是理论关键；封装与海水边界保持独立的物理区域定义。

本理论强调同一电磁场产生端口复阻抗和材料分区损耗，以守恒热动力和谐振电路组成双向反馈。允许 FEM 教师、局部求积；目标是不依赖固定世界网格的在线推理。每项创新与性能都尚待实证，不得照搬其他论文的误差或加速倍数。

**禁止 GitHub Actions。** 今后优先完善核心物理与模型实现，不新增不必要的审计脚本。

## 新一轮算法推导（2026-10-31）

- [文献启发 → 数学算法](derivation_to_algorithm.md)：解释从 Xie、Zhu、Yuan 等工作到 BFZI 保结构海水涡流降阶、神经子空间和多导电材料 Schur 补的推导依据，以及已经证明和仍未解决的部分。
- [海水低维电流与功率证明](../../../theory/sections/03a_solenoidal_eddy_galerkin.tex)：在明确的线性 MQS / 绝缘盒体 / 合规电流空间假设下，证明端口互易、被动以及同源海水损耗积分。
- [可训练的保结构几何子空间](../../../theory/sections/04a_geometry_conditioned_operator.tex)：学习子空间选择，保留物理 Gram 矩阵与分布热源一致性。
- [铜、盒壁与海水联合 Schur 补](../../../theory/sections/05a_coupled_conductive_schur.tex)：避免独立阻抗补丁漏掉交叉耦合或重复计损。

**特别注意：** 该推导并没有解决任意几何下连续散度自由电流基的高效构造、任意磁芯或高速运动；也没有获得实验精度与加速性能。最新理论仅属于数学条件明确的研究方案，现有运行代码未被修改。

## 第三轮推进：合法电流基、保正定快速装配与误差界（2026-10-31）

- [专题 Markdown：从数学到可实施算法](geometry_admissible_space_and_fast_assembly.md)：curl/Piola 物理电流图谱、BFZI 模式耦合、正权磁能 Gram、计算成本和回退。
- [03b：六自由度双封装的合规电流图谱](../../../theory/sections/03b_geometry_admissible_basis.tex)：近盒、间隙、远场及拓扑模式，并给出保持散度和绝缘界面条件的构造。
- [04b：正权核矩阵与 BFZI 在线装配](../../../theory/sections/04b_positive_kernel_assembly.tex)：R 和 L 的 PSD Gram、双盒相对姿态的谱特征、K 的互易积分以及成本。
- [04c：能量范数残差界及在线流程](../../../theory/sections/04c_residual_online_algorithm.tex)：同一高精度有限字典中的电流误差证明及快速推理/参考回退分工。

本轮补充了参数化 MQS ROM 与保被动降阶的先行文献，以明确已有数学工具与本研究创新之间的界限。**没有修改代码，也未进行实验或编译。**
