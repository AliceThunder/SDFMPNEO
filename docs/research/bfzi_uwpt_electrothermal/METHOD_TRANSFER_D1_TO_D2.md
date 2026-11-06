# D1 研究口径纠正：学习成熟方法，服务 BFZI 物理直接求解（2026-11-06）

> **用户澄清后的最高优先级：研究 Shafiei / Wang 是为了提取成熟电磁计算方法、构建自己的 BFZI 直接法，并非逐项复制论文表格。论文曲线/实验参数不是本项目必须匹配的验收目标。** 本文修订旧 D1 的执行解释；不更改《物理直接法锁定规范》中的几何、方程、端口、D0→D1→D2 先后关系、独立参考和停止规则。已有论文数据对照留作历史、可选交叉检查，**不因其差 4%–5% 卡住三维求解器开发，也不为拟合论文改物理参数**。

## 一、从论文提取方法，而非提取“应当相同”的数值

| 研究来源 | 真正学习的方法 | 应用于 BFZI 的方式 | 不可直接移植的前提 |
|---|---|---|---|
| Shafiei et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3539925 | 从 Maxwell/Helmholtz 建立复场；按材料界面连续和源跳跃解方程；使用衰减/出射分支稳定传播；由场推出损耗及端口互阻 | 统一时间相量、复扩散尺度、材料边界、线圈的双端口复耦合与空间能量；将共轴半解析解保留为具备严格适用条件的 **局部检验特例** | 圆环同轴、横向无限平层、轴对称 Bessel 分离；不适用任意两盒 SE(3) |
| Wang et al., Measurement 2025, DOI 10.1016/j.measurement.2025.117224 | 把复自感/复互感与完整场—路网络联系，统一欧姆、相位、端口功率归属 | 从高保真场解抽取 **2×2 复端口阻抗**，独立构造负载/补偿电路，检验功率守恒 | 论文线圈的等效半径、共轴及电路拓扑，不能用来替代真实 BFZI 连接/铜几何 |
| TEAM Problem 7 | 使用非轴对称、外部实测数据的三维涡流问题校准 **独立场求解器** | 验证 H(curl) 边元、真实非轴对称受迫 J/B、网格/外域收敛 | TEAM-7 铝板不是 BFZI 两绝缘盒，不能以它单独替代 BFZI 验证 |

来源：[Shafiei 机构记录](https://nur.nu.edu.kz/items/e6fd0d12-a70b-4006-8c58-8bedf424396f)、[Wang 期刊摘要](https://www.sciencedirect.com/science/article/pii/S0263224125005834)、[TEAM-7 COMSOL 标准例](https://doc.comsol.com/6.4/doc/com.comsol.help.models.acdc.multiturn_coil_asymmetric_conductor/multiturn_coil_asymmetric_conductor.html)。

### Shafiei 形式可用在哪里？

三层复谱域传播 $u_j(k)=\sqrt{k^2+i\omega\mu_0\sigma_j-\omega^2\mu_0\epsilon_j}$ 及界面 Green 算子，提供 **平面分层/共轴的可信简化模型和材料极限**。它指导三维全局边值问题中的界面条件、外向衰减与复材料项，但 **不能直接把圆环 Bessel 积分换掉三维几何积分**。D1 库 `layered_helmholtz.py` / `layered_grid.py` 保留为受限参考模块；不再为对齐 Shafiei 表 2 的若干数值点投入主研发时间。

### Wang 的真正接口

全局统一峰值相量 $e^{+i\omega t}$；线性端口先得到
$\mathbf V=\mathbf Z_{EM}(\omega,G,M)\mathbf I$，
$\mathbf Z_{EM}=\mathbf Z_{EM}^{T}$（线性互易）及
$\operatorname{Herm}(\mathbf Z_{EM})\succeq0$（无源）。
海水增量阻抗 $\Delta\mathbf Z_{sw}$ 不含铜自身损耗、补偿网络和外部负载。
定义 $\Delta\mathbf L=\Delta\mathbf Z_{sw}/(i\omega)$，对于正实损耗，复感的虚部为负；这里的符号不可随拟合调整。电路层在场计算**之后**用实际拓扑的 KCL/KVL 组成。特别检查两个单位端口和 $I=(1,\pm1),(1,\pm i)$，互阻可为负而总功率必须非负。

## 二、D0→D1→D2 的更新解释：以 BFZI 为最终交付

**D0｜锁定物理定义**：使用现有 `BFZICoil` 生成圆形／圆角方形单根连续绕线及闭合馈回流，两套独立合法 $SE(3)$ 绝缘封装、无限导电海水、$\mu=\mu_0$、$e^{+i\omega t}$。将 MQS 相对位移电流 $\omega\epsilon/\sigma$、装置尺寸与传播尺度、皮肤深度、有限铜截面与几何非自交作输入域条件。限定输入域核验，不声称所有参数已证明。

**D1｜提取并验明半解析方法**：冻结 Shafiei 的三层 DtN/Helmholtz 传播、边界连续/源面跳跃、复场与互阻/损耗；检查无导电极限、各层同质极限、交换双端口的互易性、$k$ 积分收敛与总场 Joule 功率。**D1 的通过标准是这些方程、边界与稳健计算的有界适用性得到证据**；某篇论文表中电阻或峰值 E 的逐数对齐不是必要条件，论文数据仅作可选参考。若场源和材料/观察口径不同，禁止将数值差异视为通用算法失败。D1 方法审查完成后，停止“追论文表格”。

**D2｜独立通用三维 MQS A–φ 参考**：主算法必须以真实三维保结构有限元解为目标；允许体网格，不要求把 Shafiei 圆形共轴公式泛化成不存在的任意姿态闭式解。以 TEAM-7 检验非对称涡流求解器，再验证固定源、均匀海水/无海水、单绝缘盒、双独立封装和任意合法姿态。先独立参考，再考虑 D5 BEM/FEM–BEM/Green 候选。**D2 尚未完成或宣称 PASS**。

## 三、D2 的可执行数值核心——不是占位“审查”脚手架

**基础边值方程**（导电域 $Ω_c$、非导电域 $Ω_i$、整体有限参考域 $Q_R$）：

```text
curl(mu^-1 curl A) + sigma (i*omega*A + grad(phi)) = Jsrc     on Q_R
div[sigma(i*omega*A + grad(phi))] = 0                       on Ωc
B = curl A, E = -i*omega*A - grad(phi), Jsea = sigma*E     on Ωc
n · Jsea = 0                                                on insulating box walls
```

对 $\bm A$ 使用 $H(\mathrm{curl})$ Nédélec **边元**，$\phi$ 使用每个连通导体上的连续 Lagrange 标量元；保证 $\nabla\phi$ 与边元空间相容。全域场离散不应靠逐元素节点向量加罚项“自动正确”；必须显式处理 $\bm A$ 的梯度零空间（如 Coulomb 约束/辅助乘子/适配的树-余树规范）及 $\phi$ 每个连通导电分量的常数零模。固定物理源时，先对外侧 J/B 和复端口两列求响应，随后再纳入有限铜域的内部阻抗/邻近效应；线源场不能用来定义零半径导线自感。

**弱式核心**（测试场 $v\in H(\mathrm{curl};Q_R)$、$\psi\in H^1(\Omega_c)$，复共轭测试记 $\bar{v},\bar\psi$）：

```text
∫_Q μ^-1 curl(A)·curl(conj(v)) dV
  + ∫_Ωc σ (iω A + grad φ)·conj(v) dV
       = ∫_Q Jsrc·conj(v) dV
∫_Ωc σ (iω A + grad φ)·grad(conj ψ) dV = 0
```

另加一致的规范条件及不计入主电磁能量的零空间固定；若源在绝缘区，须先证明离散施加的电流路径闭合。自然绝缘壁零法向电流由第二弱式产生，不能对整个**远域截断球**将 $n\cdot J=0$ 误认为无限海水真值。首个可运行 D2 工况应先用数值成熟 FEM 工具构建 TEAM-7，**先得到一次可求解、有可查看的场数据**，再做三档网格和三档外半径（或两类外域边界）收敛；不是先搭数十层“验收/审计”框架。

**必须从同一物理解读的输出**：单位端口分别求 $\bm A_p,\phi_p$，再由源路径上的感生电压/磁链计算 $Z_{pq}$；海水中输出 $E_p,B_p,J_p$ 与
$q(\bm x;I)=\frac12\sigma\big|\sum_p I_p E_p(\bm x)\big|^2$，
并与 $\frac12 I^H\operatorname{Herm}(Z_{sw})I$ 的端口增量功率一致。总 $Z_{EM}$ 的铜/介质/电路归属要互不重计。不得只看 Z 或磁场的一个幅值。

**软件实施选择**：优先使用可获得且具有成熟 H(curl)/涡流能力的有限元工具；未确定本机软件及版本之前不假称已经写出可运行、验证过的 3D FEM。参考图、试验数据和网格/外域证据均需实际获得。严禁 GitHub Actions。

## 四、历史内容的地位

原 `D1_SHAFIEI_SPECTRAL_REPRODUCTION.md`、第 16 节和论文表 2 复现脚本作为**有限几何验证及文献阅读记录**保留，不继续以拟合论文数字为主线；它们不得阻止 D2 的真实通用 3D 求解器。旧 P1-C/G1 仍 FAIL、正式教师训练仍冻结。
