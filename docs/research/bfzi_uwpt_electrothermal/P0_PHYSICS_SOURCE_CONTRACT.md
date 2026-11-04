# P0｜BFZI 物理问题与闭合源项锁定

**日期：2026-11-04** · 分支 `research/bfzi-uwpt-electrothermal-surrogate` · **P0 尚未通过全部数值验收；G1 仍 FAIL。**

此文是 [G1 五阶段重核](G1_THEORY_NUMERICS_REVIEW.md) 的**第一阶段实际交付**。这轮没有调节 R/K/L 的体积积分，没有扩充海水电流基，也没有宣称外球等于无限域。下一步只能在这里的源和物理域契约满足之后推进 P1。

## 1. 本阶段固定的 *物理问题*

- **线圈**：每端口为一根 BFZI 支撑函数定义的圆角方形—圆形连续绕制中心线；参数为 \(N,A_0,\rho_0,s,\alpha,\sigma_w\)。只使用 `BFZICoil`，**不使用旧超椭圆线圈**。
- **端口**：每条线圈由外端沿连续螺旋到内端，再经用户明确给出的封装内折线回流到外端，形成**同一个有向闭合回路** \(\Gamma_p\)。两端口独立，单位电流方向遵循上述绕行顺序。
- **材料**：各盒为理想非导电、非磁性实心封装；盒内连续导线由预先规定的单位回路源代替，海水 \(\sigma_s>0\) 为均匀、线性、各向同性；整个 MQS 场先取 \(\mu=\mu_0\)。铜线有限截面趋肤损耗可以另用近似计算，但并未联合求解线圈邻近效应、导体内磁场重分布或介质极化。生产 MQS-v1 源项现已显式拒绝 `relative_permeability ≠ 1` 的导线，防止将磁性导线内部 μ 与真空外部 Green 场错误拼接。
- **位姿**：两盒分别以右手刚体映射 \(\mathbf x=R_i\hat{\mathbf x}+\mathbf c_i\) 给定，保持正间隙、互不穿透，且线圈与回流路径均在自身封装内部。
- **时变**：静止物体与端口单频峰值相量 \(e^{+\mathrm j\omega t}\)，不含运动感应、电晕、全波辐射或介电损耗。
- **空间**：真实目标是 \(\Omega_\infty=\mathbb R^3\setminus(\bar B_1\cup\bar B_2)\) 中的无限导电海水。现有代码的 \(\Omega_R=B_R\setminus(\bar B_1\cup\bar B_2)\) 是**人工绝缘外球边界**下的另一个问题；它只允许作为尚待收敛的有限域近似，不能用于认证无限海水。

## 2. 方程、符号及尺度

在导电海水中（没有外加体源）：

\[
\nabla\times\mathbf H=\mathbf J_s,\quad
\nabla\times\mathbf E=-\mathrm j\omega\mathbf B,\quad
\mathbf B=\mu_0\mathbf H,\quad
\mathbf J_s=\sigma_s\mathbf E,\quad
\nabla\cdot\mathbf J_s=0.
\]

绝缘盒壁满足 \(\mathbf n\cdot\mathbf J_s=0\)。源回路由

\[
\mathbf C_p(\mathbf x)=\frac{1}{4\pi}\oint_{\Gamma_p}
 \frac{\mathrm d\boldsymbol\ell'}{|\mathbf x-\mathbf r'|},\quad
\mathbf A_{\rm src,p}=\mu_0\mathbf C_p,\quad
\mathbf B_{\rm src,p}=\mu_0\nabla\times\mathbf C_p
\]

定义，因而在完整海水域内采用的积分核是**未软化的理想导线真空 Green 核**。理想线源假设只在封装外评价，不允许在导线中心线求解奇异场。

P0 必须区分两个无量纲指标：

1. \(\omega\epsilon_s/\sigma_s\)：导电海水内部的**位移电流/导电电流比**，应显著小于 1。当前求解器**没有海水介电常数输入**，因此不能宣称验证过全部真实海水的频散介电响应。例如，仅在假设 \(\epsilon_r=80\)、\(f=200\ {\rm kHz}\)、\(\sigma_s=5\ {\rm S/m}\) 下，该比值约 \(1.78\times10^{-4}\)；**80 是例示假设，不是材料实测或程序参数**。
2. \(k_0D=\omega D/c_0\)，以及实际介电波长与装置特征长度的比。上述示例 \(D=R_{\rm outer}=0.52\ {\rm m}\) 时，真空 \(k_0D\approx2.18\times10^{-3}\)；不代表全场材料波数的证明。海水穿透深度 \(\delta=\sqrt{2/(\omega\mu_0\sigma_s)}\approx0.503\ {\rm m}\)；它影响外域/感应反作用，**不等于 MQS 无效条件**。

经典依据：Ammari, Buffa & Nédélec, *A Justification of Eddy Currents Model for the Maxwell Equations*, SIAM J. Appl. Math. 60 (2000), https://doi.org/10.1137/S0036139998348979。该论文强调涡流近似的阶次和源兼容性/拓扑条件，不能仅依赖低频口号。

## 3. 此轮发现的源项离散问题

### 3.1 连续闭合曲线与离散点线元不能混为一谈

代码 `build_line_source` 对整条路径使用带权 Gauss 点线元：

\[
\mathbf C_{p,h}(\mathbf x)
=\frac{1}{4\pi}\sum_k
 \frac{\mathbf h_k}{|\mathbf x-\mathbf r_k|},\quad
\mathbf B_{p,h}(\mathbf x)
=\frac{\mu_0}{4\pi}\sum_k
 \frac{\mathbf h_k\times(\mathbf x-\mathbf r_k)}
 {|\mathbf x-\mathbf r_k|^3}.
\]

对这个**同源离散势**，\(\mathbf B_{p,h}=\mu_0\nabla\times\mathbf C_{p,h}\) 恒等成立，且 \(\sum_k\mathbf h_k\) 很小可以说明某种全局位移闭合；但有限点线元集合作为三维分布一般**不是严格无散度闭合线电流**，单靠矩积分闭合不能证明它等价于连续 \(\Gamma_p\) 对场的贡献。特别在盒壁附近，点线元计算误差可能进入 R/K，并与后续体积分误差混淆。

### 3.2 使用不依赖当前点线元实现的解析直线参考

已新增 `sdfmpneo_bfzi/source_reference.py`。对于真实有向直线段 \(\mathbf a\to\mathbf b\)，定义 \(L=|\mathbf b-\mathbf a|\)、\(\hat{\mathbf t}=(\mathbf b-\mathbf a)/L\)、\(r_1=|\mathbf x-\mathbf a|\)、\(r_2=|\mathbf x-\mathbf b|\)、\(\mathbf R_1=\mathbf x-\mathbf a\)、\(\mathbf R_2=\mathbf x-\mathbf b\)：

\[
\begin{aligned}
\mathbf C_{\rm seg}(\mathbf x)
&=\frac{\hat{\mathbf t}}{4\pi}
\log\frac{r_1+r_2+L}{r_1+r_2-L},\\
\mathbf B_{\rm seg}(\mathbf x)
&=\frac{\mu_0}{4\pi}
\frac{(\mathbf R_1\times\mathbf R_2)(r_1+r_2)}
{r_1r_2(r_1r_2+\mathbf R_1\cdot\mathbf R_2)}.
\end{aligned}
\]

它们是**有限直线源的解析解**，无源场点数收敛问题。螺旋直边与全部折线回流均用此式；BFZI 的实际弯曲角部保留 `corner_order=16/24/32` 的独立逐段 Gauss 求积，构成 P0 源场参考。支持圆形特例（其“直边”几何长度为零）、两盒各自 SE(3)、完整源回流，**没有引入新的线圈形状或其他核正则化**。

### 3.3 目前生产源不自动切换

`LineSource` 仍然采用原来带权 Gauss 点线元，防止未经实测就无意改变空气电感/热节点的模型身份。本轮参考只是独立对照器和本机源阶次实验，不作为连续 Maxwell 的完全真值。

## 4. 入口、日志与可证伪的 P0 验收

本机执行：

```powershell
git pull --ff-only
python run.py --mode bfzi-p0-source --with-tests
```

日志自动提交到 `runs/p0/latest_summary.md` 与 `runs/p0/latest_full.log`，不覆盖已有 G1 原始对照，**不使用 GitHub Actions**。相较于 G1 全求解，该模式**只计算源场**，不会执行海水 R/K/L、电热或神经模型。

`p0_physics` 记录频率、电导率、穿透深度和两项条件尺度（在假定介电常数下）；`p0_reference` 比较曲线角段 24→32 阶以确认参考自己已细化；`p0_source` 比较生产源 `order=3/4/6/8/12` 的 C、B 和完整回路点线元闭合残差。固定世界坐标的探针包括盒壁外 2.5mm/25mm 处及远场，并保持全在海水中。

建议的**事先标准**（先观测是否达到，不调阈值让旧结果通过）：

| 门槛 | 要求 | 当前状态 |
|---|---|---|
| P0-a 连续几何与回流 | 侧边→角部→下一匝严格连续；世界坐标两盒闭合路径的起止点相合，内外端方向一致；无导线穿越盒壁 | 公式和构造已核对；有限半径导线的全域自碰撞仍未认证 |
| P0-b 解析直线源场 | 有限直导线解析 C/B 与独立 64 节点 Gauss 高阶求积相符；合法位姿刚体协变 | 已新增针对性测试，待本机执行 |
| P0-c 曲线源高阶参考 | 角部参考 `corner_order=24→32` 在盒壁/远场探针上的 C、B 变化均 **小于 \(10^{-4}\)** | 待本机测量 |
| P0-d 当前生产源精度 | Gauss 点线元 `order=12` 对独立参考的全探针相对 C、B 误差 **均小于 \(10^{-3}\)**；再判断默认 `order=3/4` 是否允许用于高精度教师 | 待本机测量 |
| P0-e Maxwell–MQS 模型适用域 | 频率／σ／μ／ε／尺寸／静止线圈和封装均满足已规定假设；超范围（导电屏蔽、磁芯、有限导体邻近场）不得作为当前教师 | 限定基线已记录；任意材料和全波不可宣称通过 |
| P0-f 外边界 | 明确人工有限绝缘球 \(\Omega_R\) 与真实无限 \(\Omega_\infty\) 是不同的连续问题 | 理论声明已锁定；**一致无限外域数值解仍属 P4，未通过** |

如 P0-d 不通过，先升级**源项解析直边与回流的实际计算路径/角段求积阶次**，不要用修改海水体积分 Gauss/Sobol 节点掩盖源误差。

## 5. 与后续阶段的接口

只有上述源问题冻结后，才用同一 \(\mathbf C_p,\mathbf B_p\)、统一端口极性和同一闭合物理路径进入 P1（**真实无软化 Coulomb L**）。P1 不得在比较 L 积分时偷偷更换线圈、频率或人工域。

在 P0 数值结果出现前，本文件所称“公式成立”不等于“P0 完成”；**当前 P0=待数值验证，G1=FAIL。**
